"""
季度频率 PeerMeta / SG-AL：固定第一层、累计反馈、剩余池递减及预测拼接。

直接运行本文件；每轮新增反馈上限为初始测试数的 10%，最多预测 10 轮。
使用公司内 ffill/bfill 加季度中位数或众数补齐的 LLM 版 77 因子，结果写入独立的
processed_peerlag_ffill_bfill_quarter_stats 目录。bfill 使用后续季度信息；
季度统计量共享横截面信息。当前预处理文件无因子缺失，模型内部保留填补兜底。

算法组合：
- PeerMeta：两层 stacked generalization、三分类标签，以及基于第二层模型
  分歧度（p_std / s_tilde）的伪标签排序；
- 固定第一层：每个滚动窗口只用真实历史训练集 T0 生成一次历史 OOF 表示
  Z_train^0，并用 T0 的 full-fit 基础模型生成完整测试季度表示 Z_test^0；
- 第二层累计反馈：每轮用历史 OOF 表示加此前累计伪样本拟合，只预测剩余池。
  入选样本先冻结本轮预测，再移出测试池；其固定第一层表示和入选时伪标签
  累计加入下一轮第二层训练集。

数据流：
1. 读取独立预处理后的季度表，时间列为 quarter（例如 2015Q1），每股每季一行；
2. 使用当季 38 项自身因子、38 项同行差值及 Peer_FSFP_lag，共 77 项特征，
   直接预测输入表已有的 FSFP_LLM_next，不再对标签移位或重新分类；
   例如 2018Q1 特征 -> 2018Q2 的 FSFP_LLM；空标签保留在审计中，不参与训练和评价；
3. 每个窗口用连续 12 个“特征季度”训练，随后 1 个特征季度测试，并逐季滚动；
4. 第一层仅在窗口开始时运行一次：T0 -> Z_train^0(OOF)，T0 full-fit -> Z_test^0；
5. 第 1 轮第二层只用 (Z_train^0, y0) 拟合并预测完整 Z_test^0；
6. 每轮从剩余池按 p_std 选择最多 d=ceil(0.1*m) 条新反馈，m 固定为初始测试数；
7. 先归档入选样本的预测，再将其表示和冻结伪标签加入累计反馈集并移出剩余池；
8. 第一层始终固定，第二层用历史 OOF 和全部累计伪样本重新拟合；
9. 最多执行 10 轮预测（含第 1 轮历史基准）；最后一轮或剩余池即将耗尽时
   归档全部剩余预测，不做无后续拟合用途的反馈。最终拼接覆盖全部初始测试样本。

每个测试样本恰有一条进入训练前的最终预测。真实测试标签仅在全部迭代结束后
用于离线评价；不会影响反馈选择、伪标签或训练。
"""

from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

# 所有基础模型的内部数值线程固定为 1。正式运行的窗口级并行和计时入口的
# 模型级并行均由有界进程池控制，避免各进程再次创建 BLAS/OpenMP 线程。
for thread_env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_env_name] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sgal_model_registry import (
    REQUIRED_MODEL_NAMES,
    required_model_factories,
    validate_required_model_names,
)


# ============================================================
# 1. 固定配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
# 先运行 preprocess_quarterly_factors_ffill_bfill.py 生成同目录的预处理数据。
# 公司内 ffill/bfill 后，原始观测的同季度中位数/众数及最近历史季度统计量补齐缺口。
INPUT_FILE = BASE_DIR / "merged_quarterly_fsfp_peer_factors_processed_ffill_bfill.csv"
INPUT_PREPROCESSING = "company_ffill_bfill_then_quarter_median_mode_previous_quarter"

STOCK_COL = "code"
QUARTER_COL = "quarter"
CURRENT_FSFP_COL = "FSFP_LLM"
SOURCE_TARGET_COL = "FSFP_LLM_next"
TARGET_COL = SOURCE_TARGET_COL
PEER_FSFP_FEATURE_COL = "Peer_FSFP_lag"
PREDICTION_HORIZON = "one_quarter_ahead"
TARGET_ALIGNMENT_MODE = "feature_quarter_q_to_input_FSFP_LLM_next"
TARGET_QUARTER_RULE = "exact_next_quarter_per_stock"
QUARTER_DEFINITION = "calendar_quarters_from_input_quarter"
QUARTER_VALUE_RULE = "input_quarter_end_with_within_quarter_field_fallback"

# 三年固定为 12 个季度；每个 OOF 验证块包含连续 4 个季度。
TRAIN_WINDOW_QUARTERS = 12
OOF_FOLDS = 3
QUARTERS_PER_FOLD = TRAIN_WINDOW_QUARTERS // OOF_FOLDS
OOF_SPLIT_POLICY = "contiguous_quarter_blocks"
AL_POLICY = "cumulative_pseudo_labels_shrinking_pool_layer2_only"
LAYER_1_POLICY = "fixed_historical_oof_plus_full_fit"
LAYER_2_POLICY = "refit_on_fixed_historical_oof_and_cumulative_pseudo_labels"
FEEDBACK_SELECTION_POLICY = "low_p_std_new_top_d_from_remaining_pool"
PREDICTION_PROTOCOL = "freeze_before_feedback_then_stitch_full_cohort"
FEEDBACK_SIZE_RULE = "ceil(consensus_fraction * initial_test_rows)"
RESULT_SCHEMA_VERSION = "fixed_layer1_cumulative_al_v2_processed_next_peerlag_ffill_bfill_quarter_stats"
MODEL_LOG_COLUMNS = [
    "layer", "model_name", "fit_stage", "fold_id", "status", "error_message",
    "fit_rows", "predict_rows", "feature_count", "model_parallel_jobs",
    "effective_model_parallel_jobs", "model_inner_max_threads", "scheduler_mode",
    "model_position", "model_count", "worker_pid", "started_at", "elapsed_seconds",
    "finished_at", "window_id", "iteration", "adaptation_role",
]
LABELS = [0, 1, 2]
EXCLUDED_MODEL_NAMES = ("NCA", "ITML", "ExtraTrees", "HGB")

# full 是正式运行；single_quarter_test 仅用于依赖和流程冒烟测试。
SG_AL_RUN_MODE = os.getenv("SG_AL_RUN_MODE", "full").strip().lower()
SG_AL_SINGLE_TEST_QUARTER = os.getenv(
    "SG_AL_SINGLE_TEST_QUARTER", ""
).strip().upper()
SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS = int(
    os.getenv("SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS", "5000")
)
SG_AL_RANDOM_STATE = 42
# m 固定为初始测试数；每轮从剩余池新增最多 ceil(0.1*m) 条，累计反馈。
# 第 1 轮只用真实历史，最多 10 轮预测、9 次有效反馈；空池提前结束。
SG_AL_CONSENSUS_FRACTION = 0.10
SG_AL_MAX_ITERATIONS = 10
SG_AL_VERBOSE = True

# 面向 30 vCPU 服务器：28 个窗口并行，窗口内逐模型串行、模型内部单线程。
# OOF 折与累计反馈轮次保持串行；下方校验要求可用逻辑处理器不少于 28。
# 此处不读取外部环境变量，避免终端中残留的配置改变正式运行方式。
MODEL_PARALLEL_JOBS = 1
MODEL_INNER_MAX_THREADS = 1
NCA_EXCLUSIVE_PROCESS = False
WINDOW_PARALLEL_JOBS = 28
AVAILABLE_CPU_COUNT = os.cpu_count() or 1
# 默认从首窗口运行；LLM 版使用独立输出目录，避免覆盖旧数据集的结果。
RESUME_FROM_TEST_QUARTER = None

if SG_AL_RUN_MODE not in {"single_quarter_test", "full"}:
    raise ValueError(
        "SG_AL_RUN_MODE 只能是 single_quarter_test 或 full。"
    )
if TRAIN_WINDOW_QUARTERS % OOF_FOLDS != 0:
    raise ValueError(
        f"{TRAIN_WINDOW_QUARTERS} 个训练季度必须能被 "
        f"{OOF_FOLDS} 个 OOF 折整除。"
    )
if MODEL_PARALLEL_JOBS < 1:
    raise ValueError("MODEL_PARALLEL_JOBS 必须是大于等于 1 的整数。")
if MODEL_PARALLEL_JOBS > AVAILABLE_CPU_COUNT:
    raise ValueError(
        "MODEL_PARALLEL_JOBS 不能超过当前可用逻辑处理器数 "
        f"{AVAILABLE_CPU_COUNT}。"
    )
if WINDOW_PARALLEL_JOBS < 1:
    raise ValueError("WINDOW_PARALLEL_JOBS 必须是大于等于 1 的整数。")
if WINDOW_PARALLEL_JOBS > AVAILABLE_CPU_COUNT:
    raise ValueError(
        "WINDOW_PARALLEL_JOBS 不能超过当前可用逻辑处理器数 "
        f"{AVAILABLE_CPU_COUNT}。"
    )

# 新版 Windows 常不再提供 wmic；显式给 loky 可用 CPU 上限可避免其
# 物理核心探测警告。真正同时运行的进程数仍由 MODEL_PARALLEL_JOBS 控制。
os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT",
    str(AVAILABLE_CPU_COUNT),
)

RUN_OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / (
        "sg_al_next_quarter_single_quarter_3year_model15_fixed_layer1_meta_cumulative_al_fixed10pct_10rounds_LLM_77factors_processed_peerlag_ffill_bfill_quarter_stats"
        if SG_AL_RUN_MODE == "single_quarter_test"
        else "sg_al_next_quarter_quarterly_full_3year_model15_fixed_layer1_meta_cumulative_al_fixed10pct_10rounds_LLM_77factors_processed_peerlag_ffill_bfill_quarter_stats"
    )
)

# 字段名称与脚本同目录的 merged_quarterly_fsfp_peer_factors_processed.csv 表头对应。
# 小写 peer_ 列为“自身值 - 同行均值”；Peer_FSFP_lag 已由上游按上一自然季度构造。
# 38 列自身因子 + 38 列同行差值 + 1 列 Peer_FSFP_lag = 77 个模型特征。
FEATURE_GROUP_SPECS = [
    ("财务因子", ("fin__", "peer_fin__"), 30),
    ("情绪因子", ("sent__", "peer_sent__"), 6),
    ("市场因子", ("mkt__", "peer_mkt__"), 16),
    ("治理因子", ("gov__", "peer_gov__"), 24),
]
FACTOR_NAMES_BY_PREFIX = {
    "fin__": (
        "Lev", "FCFR", "ROA", "ROE", "EIR", "RIR", "CAR", "FAT",
        "RTR", "RR", "IR", "FSR", "2yLoss", "NCFO", "3yGrw",
    ),
    "sent__": ("RepoSent", "GubaSent", "NewsSent"),
    "mkt__": ("MrkVal", "PE", "Turnover", "Vol", "AnnRet", "Beta", "FF3", "Liu3"),
    "gov__": (
        "3yMC", "Board", "IHR", "IndDre", "NSAO", "TopHold", "MHR",
        "DUAL", "SOE", "3yAFC", "3yBC", "Top4Aud",
    ),
}
EXPECTED_FEATURE_COLS = tuple(
    column
    for prefix, names in FACTOR_NAMES_BY_PREFIX.items()
    for name in names
    for column in (f"{prefix}{name}", f"peer_{prefix}{name}-{name}_均值")
) + (PEER_FSFP_FEATURE_COL,)


# ============================================================
# 2. 直接读取季度面板及已构造的下一季度标签
# ============================================================

def clean_stock_code(series):
    """提取股票代码中的数字并统一为 6 位，避免 000001 被读成 1。"""
    code = series.astype(str).str.extract(r"(\d+)", expand=False)
    if code.isna().any():
        raise ValueError(f"{STOCK_COL} 中存在无法识别的股票代码。")
    return code.str.zfill(6)


def clean_quarter(series):
    """校验季度字段 quarter，并规范为 YYYYQ1-YYYYQ4。"""
    quarter = series.astype("string").str.strip().str.upper()
    valid = quarter.str.fullmatch(r"\d{4}Q[1-4]", na=False)
    if not valid.all():
        examples = series.loc[~valid].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{QUARTER_COL} 必须写成 YYYYQ1-YYYYQ4，例如 2015Q1；"
            f"无效值：{examples}"
        )
    return quarter


def validate_target(series):
    """允许标签缺失，但所有非空标签必须严格属于整数集合 {0, 1, 2}。"""
    numeric = pd.to_numeric(series, errors="coerce")
    provided = series.notna() & series.astype(str).str.strip().ne("")
    unparseable = provided & numeric.isna()
    if unparseable.any():
        examples = (
            series.loc[unparseable]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含无法解析的非空标签：{examples}"
        )

    non_finite = numeric.notna() & ~np.isfinite(numeric)
    if non_finite.any():
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含无穷值，共 {int(non_finite.sum())} 行。"
        )

    non_integer = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
    if non_integer.any():
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含非整数标签，共 {int(non_integer.sum())} 行。"
        )

    observed = sorted(numeric.dropna().astype(int).unique().tolist())
    unexpected = sorted(set(observed) - set(LABELS))
    if unexpected:
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含 0/1/2 之外的标签：{unexpected}"
        )
    return numeric, observed


def prepare_quarterly_panel(quarterly_df):
    """
    直接保留每只股票、每个季度的输入行及上游 FSFP_LLM_next 标签。

    预测口径固定为：
        features(i, q) -> FSFP_LLM_next(i, q) = FSFP_LLM(i, q+1)

    quarter 已是 YYYYQn 季度编码，不做周频转换或季度内抽样。
    (code, quarter) 必须唯一；直接校验输入标签，不再移位、填补或重新分类。
    q+1 的键匹配仅用于覆盖率审计，不从当季标签重建目标。
    测试季度表示“特征信息所在季度”，目标季度始终是其后紧邻的一个季度。
    """
    required = {STOCK_COL, QUARTER_COL, SOURCE_TARGET_COL}
    missing = sorted(required - set(quarterly_df.columns))
    if missing:
        raise ValueError(f"输入文件缺少必要字段：{missing}")

    df = quarterly_df.copy()
    df[STOCK_COL] = clean_stock_code(df[STOCK_COL])
    df[QUARTER_COL] = clean_quarter(df[QUARTER_COL])
    df[TARGET_COL], observed_labels = validate_target(df[SOURCE_TARGET_COL])

    duplicate_count = int(
        df.duplicated([STOCK_COL, QUARTER_COL], keep="first").sum()
    )
    if duplicate_count:
        raise ValueError(
            f"输入存在 {duplicate_count} 个重复 ({STOCK_COL}, {QUARTER_COL}) 键。"
        )

    df["quarter"] = df[QUARTER_COL].astype(str)
    df["quarter_period"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df["year"] = df["quarter_period"].dt.year
    df["quarter_num"] = df["quarter_period"].dt.quarter
    df["feature_quarter"] = df["quarter"]
    # 输入已为季度频率，仅统一行序，不聚合、不抽样。
    quarterly = (
        df.sort_values(["quarter_period", STOCK_COL])
        .reset_index(drop=True)
    )

    # 精确匹配 q+1 的季度键，仅记录目标季度是否存在；标签仍取本行已有 next。
    target_lookup = quarterly[[
        STOCK_COL, "quarter_period", "quarter",
    ]].rename(columns={
        "quarter_period": "target_quarter_period",
        "quarter": "matched_target_quarter",
    })
    quarterly["target_quarter_period"] = quarterly["quarter_period"] + 1
    quarterly["target_quarter"] = quarterly["target_quarter_period"].astype(str)
    quarterly["target_year"] = quarterly["target_quarter_period"].dt.year
    quarterly["target_quarter_num"] = quarterly["target_quarter_period"].dt.quarter
    quarterly = quarterly.merge(
        target_lookup,
        on=[STOCK_COL, "target_quarter_period"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    quarterly["matched_target_quarter"] = quarterly["matched_target_quarter"].astype("string")

    same_feature_target_quarter = quarterly["target_quarter"].eq(
        quarterly["feature_quarter"]
    )
    target_alignment_audit = pd.DataFrame([{
        "frequency": "quarterly_input_quarterly_target",
        "quarter_col": QUARTER_COL,
        "quarter_definition": QUARTER_DEFINITION,
        "quarter_value_rule": QUARTER_VALUE_RULE,
        "prediction_horizon": PREDICTION_HORIZON,
        "source_target_col": SOURCE_TARGET_COL,
        "alignment_mode": TARGET_ALIGNMENT_MODE,
        "target_quarter_rule": TARGET_QUARTER_RULE,
        "quarterly_input_rows": int(len(df)),
        "quarterly_rows": int(len(quarterly)),
        "quarterly_stock_count": int(quarterly[STOCK_COL].nunique()),
        "feature_quarter_count": int(quarterly["quarter_period"].nunique()),
        "rows_with_target": int(quarterly[TARGET_COL].notna().sum()),
        "rows_missing_target": int(quarterly[TARGET_COL].isna().sum()),
        "rows_missing_next_quarter": int(quarterly["matched_target_quarter"].isna().sum()),
        "rows_missing_next_quarter_fsfp": int((
            quarterly["matched_target_quarter"].notna() & quarterly[TARGET_COL].isna()
        ).sum()),
        "target_same_feature_quarter_rate": float(
            same_feature_target_quarter.mean()
        ),
        "target_next_quarter_rate": float(
            quarterly["target_quarter_period"].eq(
                quarterly["quarter_period"] + 1
            ).mean()
        ),
        "observed_labels": ",".join(map(str, observed_labels)),
    }])

    quarter_panel_audit = (
        quarterly.groupby(
            ["quarter_period", "quarter"],
            as_index=False,
            observed=True,
        )
        .agg(
            row_count=(STOCK_COL, "size"),
            stock_count=(STOCK_COL, "nunique"),
        )
        .sort_values("quarter_period")
        .reset_index(drop=True)
    )
    quarter_panel_audit["quarter_period"] = (
        quarter_panel_audit["quarter_period"].astype(str)
    )

    # 逐季审计 q -> q+1 的目标覆盖率；末季、缺季和缺失 FSFP 均保留在审计中。
    target_alignment_by_quarter = (
        quarterly.assign(
            has_target=quarterly[TARGET_COL].notna(),
            has_target_quarter=quarterly["matched_target_quarter"].notna(),
        )
        .groupby(
            ["feature_quarter", "target_quarter"],
            as_index=False,
            observed=True,
        )
        .agg(
            row_count=(STOCK_COL, "size"),
            target_rows=("has_target", "sum"),
            matched_target_quarter_rows=("has_target_quarter", "sum"),
        )
        .sort_values(["feature_quarter", "target_quarter"])
        .reset_index(drop=True)
    )
    target_alignment_by_quarter["missing_target_rows"] = (
        target_alignment_by_quarter["row_count"]
        - target_alignment_by_quarter["target_rows"]
    )

    return {
        "quarterly_df": quarterly,
        "target_alignment_audit": target_alignment_audit,
        "target_alignment_by_quarter": target_alignment_by_quarter,
        "quarter_panel_audit": quarter_panel_audit,
    }


def load_quarterly_panel():
    """读取 processed 季度宽表，校验股票季度键和已有下一季度标签。"""
    print(f"读取季度输入：{INPUT_FILE}")
    input_df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        encoding="utf-8-sig",
        dtype={STOCK_COL: str, QUARTER_COL: str},
        float_precision="round_trip",
    )
    input_shape = tuple(input_df.shape)
    result = prepare_quarterly_panel(input_df)
    del input_df
    gc.collect()

    quarterly_df = result["quarterly_df"]
    print(
        "季度面板校验完成："
        f"{len(quarterly_df):,} 季度行；"
        f"{quarterly_df[STOCK_COL].nunique():,} 只股票；"
        f"{quarterly_df['quarter_period'].nunique()} 个季度。"
    )
    result["quarterly_input_shape"] = input_shape
    return result


# ============================================================
# 3. 特征契约
# ============================================================

def get_feature_group(feature_name):
    """按来源前缀把普通及同行字段映射到对应经济类别。"""
    if feature_name == PEER_FSFP_FEATURE_COL:
        return "上一季度同行 FSFP"
    for group_name, prefixes, _ in FEATURE_GROUP_SPECS:
        if feature_name.startswith(prefixes):
            return group_name
    return None


def build_feature_contract(df):
    """
    固定模型输入字段并阻止未知列静默进入训练。

    仅允许指定的 38 项自身因子、对应 38 项同行差值和 Peer_FSFP_lag。
    两版原始 FSFP、当季/下一季标签、当季同行 FSFP、词频版 lag 及时间索引
    均排除。逐列名称、分组数量及总数同时校验，防止同前缀的额外列混入。
    """
    exclude_cols = {
        STOCK_COL,
        QUARTER_COL,
        CURRENT_FSFP_COL,
        SOURCE_TARGET_COL,
        TARGET_COL,
        "week",
        "fsfp_original",
        "fsfp_original_词频",
        "FSFP_词频",
        "FSFP_词频_next",
        "Peer_FSFP",
        "Peer_FSFP_词频",
        "Peer_FSFP_词频_lag",
        "FSFP_next",
        "Year",
        "year",
        "quarter_num",
        "quarter",
        "quarter_period",
        "feature_quarter",
        "target_year",
        "matched_target_quarter",
        "target_quarter_num",
        "target_quarter",
        "target_quarter_period",
        "oof_fold",
    }

    feature_group_by_col = {}
    unexpected = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        group_name = get_feature_group(col)
        if group_name is None:
            unexpected.append(col)
        else:
            feature_group_by_col[col] = group_name

    feature_cols = [
        col for col in df.columns
        if col in feature_group_by_col
    ]
    if unexpected:
        raise ValueError(f"发现未归类的候选特征列：{unexpected}")
    missing_features = sorted(set(EXPECTED_FEATURE_COLS) - set(feature_cols))
    extra_features = sorted(set(feature_cols) - set(EXPECTED_FEATURE_COLS))
    if missing_features or extra_features:
        raise ValueError(
            "77 项特征字段与 processed 数据契约不一致："
            f"缺少={missing_features}；额外={extra_features}"
        )

    expected_counts = {
        group_name: expected
        for group_name, _, expected in FEATURE_GROUP_SPECS
    }
    expected_counts["上一季度同行 FSFP"] = 1

    group_count_df = pd.DataFrame([
        {
            "feature_group": group_name,
            "expected_count": expected,
            "actual_count": sum(
                feature_group_by_col.get(col) == group_name
                for col in feature_cols
            ),
        }
        for group_name, expected in expected_counts.items()
    ])
    group_count_df["matches_expected"] = (
        group_count_df["expected_count"]
        == group_count_df["actual_count"]
    )
    if not group_count_df["matches_expected"].all():
        raise ValueError(
            "特征分组数量与数据契约不一致：\n"
            + group_count_df.to_string(index=False)
        )
    if len(feature_cols) != 77:
        raise ValueError(f"特征总数应为 77，当前为 {len(feature_cols)}。")

    availability_rows = []
    for col in feature_cols:
        group_name = feature_group_by_col[col]
        # 经济类别合并后仍按字段来源沿用可用性审计口径。
        if col.startswith(("mkt__", "sent__")):
            status, risk = "conditional", "medium"
        else:
            status, risk = "unverified", "high"
        availability_rows.append({
            "feature": col,
            "feature_group": group_name,
            "availability_status": status,
            "leakage_risk": risk,
            "quarter_value_rule": QUARTER_VALUE_RULE,
            "prediction_horizon": PREDICTION_HORIZON,
            "included_in_model": True,
        })

    return {
        "feature_cols": feature_cols,
        "feature_group_by_col": feature_group_by_col,
        "feature_group_count_df": group_count_df,
        "feature_availability_audit_df": pd.DataFrame(availability_rows),
    }


def coerce_features_to_numeric(df, feature_cols):
    """
    在滚动开始前把 77 个特征统一转成数值类型。

    原代码在每个高度重叠的季度窗口、甚至每个 OOF 预处理调用中重复
    pd.to_numeric。同一条历史记录会被转换多次。季度面板确定后只转换
    一次即可；无法解析的值仍变为 NaN，随后由严格折内规则填补，因此
    统计口径不变，也不会用到测试季度的填补统计量。

    同时返回逐列审计，记录有多少“原本非空”的值因无法解析而变成 NaN。
    该函数有意原位替换特征列，以避免再复制一份 77 列特征的季度面板。
    """
    non_missing_before = df.loc[:, feature_cols].notna().sum()
    numeric_features = df.loc[:, feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    non_missing_after = numeric_features.notna().sum()
    df[feature_cols] = numeric_features

    non_numeric = [
        col
        for col in feature_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(f"特征转数值后仍存在非数值列：{non_numeric}")

    coercion_audit = pd.DataFrame({
        "feature": feature_cols,
        "non_missing_before": non_missing_before.reindex(
            feature_cols
        ).to_numpy(),
        "non_missing_after": non_missing_after.reindex(
            feature_cols
        ).to_numpy(),
    })
    coercion_audit["coerced_to_missing"] = (
        coercion_audit["non_missing_before"]
        - coercion_audit["non_missing_after"]
    )
    return df, coercion_audit


# ============================================================
# 4. 十二季度训练、单季度测试的滚动窗口
# ============================================================

def parse_quarter(quarter_text):
    """解析 YYYYQn 环境变量，并把格式错误转换为可读异常。"""
    try:
        return pd.Period(quarter_text, freq="Q")
    except Exception as exc:
        raise ValueError(
            f"季度必须写成 YYYYQ1-YYYYQ4，当前为 {quarter_text!r}。"
        ) from exc


def build_rolling_windows(df_model):
    """
    构造固定长度、逐季度滚动的窗口定义。

    这里的“季度”始终指 feature quarter：测试季度 q 中每个样本使用该股
    在 q 的输入季度特征，预测 q+1 的真实季度 FSFP。
    测试季度 q 只有在此前连续 12 个 feature quarters 全部存在时才可用。
    因而窗口天然满足：
    2015Q1-2017Q4 特征训练 -> 2018Q1 特征测试（预测 2018Q2 末 FSFP），
    下一窗口 2015Q2-2018Q1 特征训练 -> 2018Q2 特征测试（预测 2018Q3 末 FSFP）。
    最后一个训练特征季度 q-1 的真实标签来自 q 末，在本窗口预测时点已可用。
    """
    available_quarters = sorted(
        df_model["quarter_period"].drop_duplicates().tolist()
    )
    available_set = set(available_quarters)
    candidate_test_quarters = []

    for test_quarter in available_quarters:
        train_start = test_quarter - TRAIN_WINDOW_QUARTERS
        train_quarters = list(
            pd.period_range(
                start=train_start,
                periods=TRAIN_WINDOW_QUARTERS,
                freq="Q",
            )
        )
        if all(quarter in available_set for quarter in train_quarters):
            candidate_test_quarters.append(test_quarter)

    if SG_AL_RUN_MODE == "single_quarter_test":
        if SG_AL_SINGLE_TEST_QUARTER:
            requested = parse_quarter(SG_AL_SINGLE_TEST_QUARTER)
            if requested not in candidate_test_quarters:
                raise ValueError(
                    f"指定季度 {requested} 不存在，或此前没有连续 "
                    f"{TRAIN_WINDOW_QUARTERS} 个训练季度。"
                )
            candidate_test_quarters = [requested]
        else:
            candidate_test_quarters = candidate_test_quarters[:1]

    if not candidate_test_quarters:
        raise ValueError("没有可用的季度滚动窗口。")

    windows = []
    for window_id, test_quarter in enumerate(
        candidate_test_quarters,
        start=1,
    ):
        train_start = test_quarter - TRAIN_WINDOW_QUARTERS
        train_end = test_quarter - 1
        train_quarters = list(
            pd.period_range(
                start=train_start,
                end=train_end,
                freq="Q",
            )
        )
        windows.append({
            "window_id": window_id,
            "train_start_quarter": str(train_start),
            "train_end_quarter": str(train_end),
            "test_quarter": str(test_quarter),
            "test_target_quarter": str(test_quarter + 1),
            "train_start_period": train_start,
            "train_end_period": train_end,
            "test_period_obj": test_quarter,
            "train_quarters": train_quarters,
            "train_period": f"{train_start}-{train_end}",
            "test_period": str(test_quarter),
            "prediction_asof": str(test_quarter),
            "prediction_horizon": PREDICTION_HORIZON,
            "source_target_col": SOURCE_TARGET_COL,
            "target_col": TARGET_COL,
            "alignment_mode": TARGET_ALIGNMENT_MODE,
            "target_quarter_rule": TARGET_QUARTER_RULE,
            "train_start_year": int(train_start.year),
            "train_end_year": int(train_end.year),
            "test_start_year": int(test_quarter.year),
            "test_end_year": int(test_quarter.year),
        })

    return windows


def split_windows_for_resume(windows, resume_from_test_quarter):
    """把完整窗口序列拆成断点前已完成部分和待执行部分。"""
    if resume_from_test_quarter is None:
        return [], list(windows)
    for index, window in enumerate(windows):
        if window["test_quarter"] == resume_from_test_quarter:
            return list(windows[:index]), list(windows[index:])
    raise ValueError(
        "断点季度不在滚动窗口定义中："
        f"{resume_from_test_quarter}"
    )


def assign_quarter_folds(train_df, train_quarters):
    """
    把训练季度按时间顺序切成 OOF_FOLDS 个连续验证块。

    同季度的所有股票必须落入同一折，避免同一时点的截面样本同时出现在
    拟合集和验证集中。每折模型使用其余季度拟合；这不是随机 K 折，也不是
    只允许过去数据的 expanding-window 验证。
    """
    if len(train_quarters) != TRAIN_WINDOW_QUARTERS:
        raise ValueError(
            f"训练窗口必须包含 {TRAIN_WINDOW_QUARTERS} 个季度。"
        )
    quarter_to_fold = {
        quarter: position // QUARTERS_PER_FOLD
        for position, quarter in enumerate(train_quarters)
    }
    result = train_df.copy()
    result["oof_fold"] = result["quarter_period"].map(quarter_to_fold)
    if result["oof_fold"].isna().any():
        raise RuntimeError("训练样本中存在无法分配 OOF 折的季度。")
    result["oof_fold"] = result["oof_fold"].astype(int)
    if sorted(result["oof_fold"].unique().tolist()) != list(range(OOF_FOLDS)):
        raise ValueError(
            f"季度训练样本没有完整覆盖 {OOF_FOLDS} 个时间 OOF 折。"
        )
    return result


def stratified_training_cap(train_df, max_rows):
    """
    仅为单季度冒烟测试压缩训练集。

    按 OOF 折 × 标签分层分配配额，至少保留每个已存在分组的一条记录；
    正式 full 模式完全不调用该抽样。
    """
    if max_rows is None or max_rows <= 0 or len(train_df) <= max_rows:
        return train_df

    groups = list(
        train_df.groupby(
            ["oof_fold", TARGET_COL],
            sort=True,
            observed=True,
        )
    )
    if max_rows < len(groups):
        raise ValueError(
            f"训练样本上限 {max_rows} 小于折×标签分组数 {len(groups)}。"
        )

    sizes = np.asarray([len(group) for _, group in groups], dtype=int)
    raw_quotas = max_rows * sizes / sizes.sum()
    quotas = np.maximum(1, np.floor(raw_quotas).astype(int))
    quotas = np.minimum(quotas, sizes)

    while quotas.sum() > max_rows:
        reducible = np.where(quotas > 1)[0]
        if len(reducible) == 0:
            raise RuntimeError("无法在保留每个折×标签组的同时压缩训练集。")
        index = reducible[np.argmax(quotas[reducible])]
        quotas[index] -= 1

    fractional = raw_quotas - np.floor(raw_quotas)
    while quotas.sum() < max_rows:
        addable = np.where(quotas < sizes)[0]
        if len(addable) == 0:
            break
        index = addable[np.argmax(fractional[addable])]
        quotas[index] += 1
        fractional[index] = -1

    rng = np.random.default_rng(SG_AL_RANDOM_STATE)
    selected_indices = []
    for quota, (_, group) in zip(quotas, groups):
        selected_indices.extend(
            rng.choice(
                group.index.to_numpy(),
                size=int(quota),
                replace=False,
            ).tolist()
        )

    return (
        train_df.loc[selected_indices]
        .sort_values(["quarter_period", STOCK_COL])
        .reset_index(drop=True)
    )


def prepare_window_data(df_model, feature_cols, window):
    """
    从季度面板切出连续 12 个特征季度训练集和紧随其后的单特征季度测试集。

    此处只做行选择和元数据拆分。特征已经在滚动前统一转为数值；缺失值
    仍原样保留，必须等到第一层每个 OOF 拟合时才能使用对应训练子集填补。
    """
    train_mask = (
        (df_model["quarter_period"] >= window["train_start_period"])
        & (df_model["quarter_period"] <= window["train_end_period"])
    )
    test_mask = df_model["quarter_period"].eq(window["test_period_obj"])

    train_df = df_model.loc[train_mask].copy()
    test_df = df_model.loc[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"窗口 {window['window_id']} 的训练集或测试集为空。"
        )
    if train_df["quarter_period"].nunique() != TRAIN_WINDOW_QUARTERS:
        raise ValueError(
            f"窗口 {window['window_id']} 未覆盖完整 "
            f"{TRAIN_WINDOW_QUARTERS} 个训练季度。"
        )

    train_df = assign_quarter_folds(
        train_df,
        window["train_quarters"],
    )
    # 测试季度从不参与第一层 OOF。-1 仅作为审计标记，避免误解为某个训练折。
    test_df["oof_fold"] = -1
    rows_before_cap = len(train_df)

    if SG_AL_RUN_MODE == "single_quarter_test":
        train_df = stratified_training_cap(
            train_df,
            SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS,
        )

    # df_model 在面板校验后已按 quarter_period、code 全局排序；布尔筛选、
    # 折号映射和 dropna 都保持行序。单季度抽样函数也会自行恢复该顺序，
    # 因此这里不再为每个高度重叠的窗口重复排序整张宽表。
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    meta_cols = [
        STOCK_COL,
        QUARTER_COL,
        "quarter_num",
        "quarter_period",
        "feature_quarter",
        "matched_target_quarter",
        "target_quarter",
        "target_quarter_period",
        "oof_fold",
    ]

    window_info = {
        key: value
        for key, value in window.items()
        if key != "train_quarters"
        and key not in {
            "train_start_period",
            "train_end_period",
            "test_period_obj",
        }
    }
    window_info.update({
        "train_quarter_count": int(
            train_df["quarter_period"].nunique()
        ),
        "max_iterations": SG_AL_MAX_ITERATIONS,
        "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
        "train_rows_before_cap": int(rows_before_cap),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_stock_count": int(train_df[STOCK_COL].nunique()),
        "test_stock_count": int(test_df[STOCK_COL].nunique()),
        "oof_fold_count": int(train_df["oof_fold"].nunique()),
        "quarters_per_oof_fold": QUARTERS_PER_FOLD,
        "oof_split_policy": OOF_SPLIT_POLICY,
        "al_policy": AL_POLICY,
        "layer_1_policy": LAYER_1_POLICY,
        "layer_2_policy": LAYER_2_POLICY,
        "pseudo_feedback_entry_layer": "layer_2_only",
        "feedback_size_rule": FEEDBACK_SIZE_RULE,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_scope": "stitched_full_pool",
        "feedback_selection_policy": FEEDBACK_SELECTION_POLICY,
        "prediction_protocol": PREDICTION_PROTOCOL,
        "is_capped_trial_sample": (
            SG_AL_RUN_MODE == "single_quarter_test"
        ),
        "feature_count": int(len(feature_cols)),
    })

    return {
        "window_info": window_info,
        "train_X_raw": (
            train_df[feature_cols]
            .reset_index(drop=True)
        ),
        "test_X_raw": (
            test_df[feature_cols]
            .reset_index(drop=True)
        ),
        "y_train": (
            train_df[TARGET_COL].astype(int).reset_index(drop=True)
        ),
        "y_test": (
            test_df[TARGET_COL].astype(int).reset_index(drop=True)
        ),
        "train_meta": train_df[meta_cols].reset_index(drop=True),
        "test_meta": test_df[meta_cols].reset_index(drop=True),
    }


# ============================================================
# 5. 折内预处理、15 模型和评价指标
# ============================================================

MODEL_FACTORIES = required_model_factories()
MODEL_NAMES = validate_required_model_names(REQUIRED_MODEL_NAMES)


def validate_active_model_names(model_names):
    """校验活动模型是默认注册表的非空有序子集。"""
    names = tuple(model_names)
    if not names:
        raise ValueError("活动模型列表不能为空。")
    if len(names) != len(set(names)):
        raise ValueError(f"活动模型列表包含重复项：{list(names)}")

    unknown = [name for name in names if name not in REQUIRED_MODEL_NAMES]
    if unknown:
        raise ValueError(f"活动模型不在默认注册表中：{unknown}")

    expected_order = tuple(
        name for name in REQUIRED_MODEL_NAMES if name in names
    )
    if names != expected_order:
        raise ValueError(
            "活动模型必须保持默认注册表顺序；"
            f"expected={list(expected_order)}, received={list(names)}"
        )
    return list(names)


def configure_active_models(excluded_model_names=()):
    """为当前进程选择模型子集，不修改默认注册表。"""
    excluded = tuple(dict.fromkeys(excluded_model_names))
    unknown = [
        name for name in excluded if name not in REQUIRED_MODEL_NAMES
    ]
    if unknown:
        raise ValueError(f"无法排除未注册模型：{unknown}")

    active_names = validate_active_model_names([
        name for name in REQUIRED_MODEL_NAMES if name not in excluded
    ])
    all_factories = required_model_factories()

    global MODEL_NAMES, MODEL_FACTORIES
    MODEL_NAMES = active_names
    MODEL_FACTORIES = {
        name: all_factories[name] for name in MODEL_NAMES
    }
    return list(MODEL_NAMES)


configure_active_models(EXCLUDED_MODEL_NAMES)


def fit_fold_preprocessor(X_fit, fit_stocks, X_apply, apply_stocks):
    """
    只用当前拟合子集建立“公司中位数 + 全局中位数 + 标准化”参数。

    X_apply 可以是 OOF 验证块，也可以是测试季度。公司在拟合子集中出现
    时优先使用该公司的历史中位数；新公司或公司中位数仍缺失时退回全局
    中位数。此阶段的均值和标准差同样只来自 X_fit。
    输入已经过公司内 ffill/bfill 和季度统计量填补；上游预处理的信息共享不会在此撤销。

    输入已由 coerce_features_to_numeric 统一转为数值，因此这里不再在每个
    折、每次 SG-AL 迭代重复执行字符串解析。
    """
    X_fit = pd.DataFrame(X_fit).reset_index(drop=True)
    X_apply = pd.DataFrame(X_apply).reset_index(drop=True)
    fit_stocks = pd.Series(fit_stocks).reset_index(drop=True)
    apply_stocks = pd.Series(apply_stocks).reset_index(drop=True)

    # groupby 只使用拟合样本。同一股票跨季度的历史记录共同给出公司基准。
    company_medians = X_fit.groupby(fit_stocks).median()
    global_medians = X_fit.median().fillna(0.0)

    # reindex 按每一行的股票代码广播对应公司中位数；测试期新股票会得到
    # 全 NaN 行，随后自然退回 global_medians。
    fit_company_fill = company_medians.reindex(
        fit_stocks
    ).reset_index(drop=True)
    apply_company_fill = company_medians.reindex(
        apply_stocks
    ).reset_index(drop=True)
    fit_company_fill.columns = X_fit.columns
    apply_company_fill.columns = X_apply.columns

    X_fit_imputed = (
        X_fit.fillna(fit_company_fill).fillna(global_medians)
    )
    X_apply_imputed = (
        X_apply.fillna(apply_company_fill).fillna(global_medians)
    )

    # 标准差为 0 的常量列除数改为 1，使其标准化结果为 0，而不是 NaN。
    means = X_fit_imputed.mean()
    stds = X_fit_imputed.std(ddof=0).replace(0, 1).fillna(1)
    return (
        (X_fit_imputed - means) / stds,
        (X_apply_imputed - means) / stds,
    )


def wall_time_text():
    """返回便于终端定位的本地墙钟时间。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def available_physical_memory_gib():
    """在 Windows 返回当前可用物理内存；查询失败时返回 None。"""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            return None
        return status.ullAvailPhys / (1024 ** 3)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def choose_effective_parallel_jobs(requested_jobs):
    """
    根据批次开始时的可用物理内存下调普通模型并发数。

    NCA 不进入该进程池；这里的阈值只用于避免普通模型在用户同时运行其他
    高内存程序时继续固定启动 6 个进程。
    """
    requested_jobs = max(1, min(int(requested_jobs), AVAILABLE_CPU_COUNT))
    available_gib = available_physical_memory_gib()
    if available_gib is None or requested_jobs == 1:
        return requested_jobs, available_gib
    if available_gib >= 16:
        memory_cap = 6
    elif available_gib >= 12:
        memory_cap = 4
    elif available_gib >= 8:
        memory_cap = 2
    else:
        memory_cap = 1
    return min(requested_jobs, memory_cap), available_gib


def shutdown_reusable_loky_workers():
    """
    强制结束当前批次的 loky 工作进程。

    joblib 默认会保留空闲工作进程供下一次 Parallel 复用；这里必须主动结束，
    才能保证下一折或下一层的 NCA 启动前没有普通模型子进程继续驻留。
    """
    try:
        from joblib.externals.loky import reusable_executor

        executor = reusable_executor._executor
        if executor is not None:
            executor.shutdown(wait=True, kill_workers=True)
    except Exception as exc:
        print(
            f"[{wall_time_text()}] loky 工作进程清理警告 | "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def safe_model_fit_predict(
    model_name,
    X_fit,
    y_fit,
    X_predict,
    scheduler_mode="serial",
    requested_parallel_jobs=1,
    effective_parallel_jobs=1,
    model_position=None,
    model_count=None,
):
    """
    创建一个全新模型实例，完成一次拟合和预测，并把异常转成结构化日志。

    每次都从工厂新建实例，防止不同 OOF 折或 SG-AL 轮次共享已拟合状态。
    不在单个模型结束后强制 gc.collect；局部引用退出函数即释放，集中在
    折、迭代和窗口边界回收可避免上万次无必要的全量垃圾回收。
    """
    started_at = time.perf_counter()
    diagnostics = {
        "fit_rows": int(len(X_fit)),
        "predict_rows": int(len(X_predict)),
        "feature_count": int(pd.DataFrame(X_fit).shape[1]),
        "model_parallel_jobs": int(requested_parallel_jobs),
        "effective_model_parallel_jobs": int(effective_parallel_jobs),
        "model_inner_max_threads": MODEL_INNER_MAX_THREADS,
        "scheduler_mode": scheduler_mode,
        "model_position": model_position,
        "model_count": model_count,
        "worker_pid": os.getpid(),
        "started_at": wall_time_text(),
    }
    try:
        model = MODEL_FACTORIES[model_name]()
        model.fit(X_fit, y_fit)
        prediction = np.asarray(model.predict(X_predict)).reshape(-1)
        if len(prediction) != len(X_predict):
            raise ValueError(
                "模型预测长度与待预测样本数不一致："
                f"{len(prediction)} != {len(X_predict)}"
            )
        prediction = prediction.astype(int)
        if not np.isin(prediction, LABELS).all():
            raise ValueError("模型输出包含 0/1/2 之外的类别。")
        diagnostics["elapsed_seconds"] = (
            time.perf_counter() - started_at
        )
        diagnostics["finished_at"] = wall_time_text()
        return prediction, "success", "", diagnostics
    except Exception as exc:
        diagnostics["elapsed_seconds"] = (
            time.perf_counter() - started_at
        )
        diagnostics["finished_at"] = wall_time_text()
        return (
            None,
            "failed",
            f"{type(exc).__name__}: {exc}",
            diagnostics,
        )


def execute_model_task(
    model_name,
    X_fit,
    y_fit,
    X_predict,
    stage_label,
    scheduler_mode,
    requested_parallel_jobs,
    effective_parallel_jobs,
    model_position,
    model_count,
):
    """由实际执行模型的进程打印开始位置，再返回带模型名的结构化结果。"""
    print(
        f"[{wall_time_text()}] [{stage_label}] 模型开始 | "
        f"模型 {model_position}/{model_count} {model_name} | "
        f"调度={scheduler_mode} | pid={os.getpid()} | "
        f"fit={len(X_fit):,}, predict={len(X_predict):,}",
        flush=True,
    )
    result = safe_model_fit_predict(
        model_name,
        X_fit,
        y_fit,
        X_predict,
        scheduler_mode=scheduler_mode,
        requested_parallel_jobs=requested_parallel_jobs,
        effective_parallel_jobs=effective_parallel_jobs,
        model_position=model_position,
        model_count=model_count,
    )
    return model_name, result


def print_model_result(
    stage_label,
    model_name,
    result,
    completed_count,
    model_count,
    stage_started_at,
):
    """模型返回后立即打印成功/失败、样本规模和训练预测总耗时。"""
    _, status, error_message, diagnostics = result
    elapsed_seconds = float(diagnostics["elapsed_seconds"])
    elapsed_minutes = elapsed_seconds / 60
    stage_elapsed_minutes = (
        time.perf_counter() - stage_started_at
    ) / 60
    location = (
        f"模型 {diagnostics.get('model_position')}/{model_count} "
        f"{model_name} | 阶段完成 {completed_count}/{model_count}"
    )
    if status == "success":
        print(
            f"[{wall_time_text()}] [{stage_label}] 模型成功 | "
            f"{location} | "
            f"fit={diagnostics['fit_rows']:,}, "
            f"predict={diagnostics['predict_rows']:,} | "
            f"耗时 {elapsed_seconds:.2f} 秒 "
            f"({elapsed_minutes:.2f} 分钟) | "
            f"阶段累计 {stage_elapsed_minutes:.2f} 分钟 | "
            f"pid={diagnostics.get('worker_pid')}",
            flush=True,
        )
    else:
        print(
            f"[{wall_time_text()}] [{stage_label}] 模型失败 | "
            f"{location} | "
            f"耗时 {elapsed_seconds:.2f} 秒 "
            f"({elapsed_minutes:.2f} 分钟) | "
            f"阶段累计 {stage_elapsed_minutes:.2f} 分钟 | "
            f"{error_message}",
            flush=True,
        )


def build_failed_process_result(
    model_name,
    X_fit,
    X_predict,
    started_at,
    error,
    scheduler_mode,
    effective_parallel_jobs,
    model_position,
    model_count,
):
    """把独占子进程崩溃转换为与普通模型一致的失败结果。"""
    diagnostics = {
        "fit_rows": int(len(X_fit)),
        "predict_rows": int(len(X_predict)),
        "feature_count": int(pd.DataFrame(X_fit).shape[1]),
        "model_parallel_jobs": MODEL_PARALLEL_JOBS,
        "effective_model_parallel_jobs": int(effective_parallel_jobs),
        "model_inner_max_threads": MODEL_INNER_MAX_THREADS,
        "scheduler_mode": scheduler_mode,
        "model_position": model_position,
        "model_count": model_count,
        "worker_pid": None,
        "started_at": None,
        "finished_at": wall_time_text(),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    return (
        None,
        "failed",
        f"{type(error).__name__}: {error}",
        diagnostics,
    )


def run_required_model_batch(
    X_fit,
    y_fit,
    X_predict,
    stage_label,
):
    """
    拟合当前活动模型，并始终按注册表顺序返回结果。

    正式脚本的默认串行模式保持原顺序和失败即停。首窗口安全并行模式把
    NCA 放入独立 spawn 子进程并等待其完全退出，再用最多 6 个 loky 进程
    运行其余模型；OOF 折和 SG-AL 迭代仍由调用方串行执行。普通模型按预计
    耗时从长到短提交，但最终结果重新按 MODEL_NAMES 排列，不改变 stacking
    列契约。
    """
    stage_started_at = time.perf_counter()
    X_fit_array = np.ascontiguousarray(
        pd.DataFrame(X_fit).to_numpy(dtype=float, copy=False)
    )
    X_predict_array = np.ascontiguousarray(
        pd.DataFrame(X_predict).to_numpy(dtype=float, copy=False)
    )
    y_fit_array = np.asarray(y_fit, dtype=int).reshape(-1)

    if len(X_fit_array) != len(y_fit_array):
        raise ValueError("模型批次的 X_fit 和 y_fit 长度不一致。")

    model_count = len(MODEL_NAMES)
    model_positions = {
        model_name: position
        for position, model_name in enumerate(MODEL_NAMES, start=1)
    }
    print(
        f"[{wall_time_text()}] [{stage_label}] 阶段开始 | "
        f"fit={len(X_fit_array):,}, predict={len(X_predict_array):,}, "
        f"features={X_fit_array.shape[1]:,} | "
        f"请求普通模型并发={MODEL_PARALLEL_JOBS} | "
        f"NCA独占={bool('NCA' in MODEL_NAMES and (MODEL_PARALLEL_JOBS > 1 or NCA_EXCLUSIVE_PROCESS))}",
        flush=True,
    )

    use_resource_groups = MODEL_PARALLEL_JOBS > 1 or (
        "NCA" in MODEL_NAMES and NCA_EXCLUSIVE_PROCESS
    )
    if not use_resource_groups:
        results_by_name = {}
        completed_count = 0
        for model_name in MODEL_NAMES:
            returned_name, result = execute_model_task(
                model_name,
                X_fit_array,
                y_fit_array,
                X_predict_array,
                stage_label,
                "serial",
                MODEL_PARALLEL_JOBS,
                1,
                model_positions[model_name],
                model_count,
            )
            results_by_name[returned_name] = result
            completed_count += 1
            print_model_result(
                stage_label,
                returned_name,
                result,
                completed_count,
                model_count,
                stage_started_at,
            )

            # 当前活动模型策略不允许忽略失败模型。串行时立即返回，
            # 让上层保存日志并终止当前折，不再浪费时间运行后续模型。
            if result[1] != "success":
                break
        print(
            f"[{wall_time_text()}] [{stage_label}] 阶段结束 | "
            f"完成={len(results_by_name)}/{model_count} | "
            f"状态={'success' if len(results_by_name) == model_count else 'failed'} | "
            f"阶段耗时={(time.perf_counter() - stage_started_at) / 60:.2f} 分钟",
            flush=True,
        )
        return [
            (model_name, results_by_name[model_name])
            for model_name in MODEL_NAMES
            if model_name in results_by_name
        ]

    results_by_name = {}
    completed_count = 0
    if "NCA" in MODEL_NAMES:
        nca_position = model_positions["NCA"]
        nca_core_gib = (
            len(X_fit_array) ** 2 * (4 * 8 + 1) / (1024 ** 3)
        )
        nca_available_gib = available_physical_memory_gib()
        memory_text = (
            "当前可用物理内存=未知"
            if nca_available_gib is None
            else f"当前可用物理内存={nca_available_gib:.2f} GiB"
        )
        warning_text = ""
        if (
            nca_available_gib is not None
            and nca_core_gib + 4 > nca_available_gib
        ):
            warning_text = (
                " | 警告：NCA核心矩阵加运行余量超过当前可用内存"
            )
        print(
            f"[{wall_time_text()}] [{stage_label}] NCA独占屏障 | "
            f"核心矩阵粗估={nca_core_gib:.2f} GiB | {memory_text}"
            f"{warning_text}",
            flush=True,
        )

        nca_process_started_at = time.perf_counter()
        try:
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=get_context("spawn"),
            ) as executor:
                returned_name, nca_result = executor.submit(
                    execute_model_task,
                    "NCA",
                    X_fit_array,
                    y_fit_array,
                    X_predict_array,
                    stage_label,
                    "exclusive_spawn",
                    MODEL_PARALLEL_JOBS,
                    1,
                    nca_position,
                    model_count,
                ).result()
        except Exception as exc:
            returned_name = "NCA"
            nca_result = build_failed_process_result(
                "NCA",
                X_fit_array,
                X_predict_array,
                nca_process_started_at,
                exc,
                "exclusive_spawn",
                1,
                nca_position,
                model_count,
            )

        results_by_name[returned_name] = nca_result
        completed_count += 1
        print_model_result(
            stage_label,
            returned_name,
            nca_result,
            completed_count,
            model_count,
            stage_started_at,
        )
        gc.collect()
        if nca_result[1] != "success":
            print(
                f"[{wall_time_text()}] [{stage_label}] 阶段中止 | "
                "NCA 独占任务失败，普通模型队列未启动 | "
                f"阶段耗时={(time.perf_counter() - stage_started_at) / 60:.2f} 分钟",
                flush=True,
            )
            return [("NCA", nca_result)]

    normal_execution_order = [
        "SVM-Lin",
        "SVM-Poly",
        "SVM-RBF",
        "FNN3",
        "FNN2",
        "FNN1",
        "Probit",
        "Logit",
        "ITML",
        "KNN",
        "LDA",
        "QDA",
        "NaiveBayes",
        "Tree-ID3",
        "Tree-C4.5",
        "Tree-CART",
    ]
    normal_execution_order = [
        name for name in normal_execution_order if name in MODEL_NAMES
    ]
    expected_normal_models = set(MODEL_NAMES) - {"NCA"}
    if (
        len(normal_execution_order) != len(expected_normal_models)
        or set(normal_execution_order) != expected_normal_models
    ):
        raise RuntimeError("普通模型执行队列与活动模型注册表不一致。")

    effective_jobs, normal_available_gib = choose_effective_parallel_jobs(
        MODEL_PARALLEL_JOBS
    )
    available_text = (
        "未知"
        if normal_available_gib is None
        else f"{normal_available_gib:.2f} GiB"
    )
    print(
        f"[{wall_time_text()}] [{stage_label}] 普通模型队列开始 | "
        f"模型数={len(normal_execution_order)} | "
        f"请求并发={MODEL_PARALLEL_JOBS}, 实际并发={effective_jobs} | "
        f"可用物理内存={available_text} | 内部线程=1",
        flush=True,
    )

    if effective_jobs == 1:
        for model_name in normal_execution_order:
            returned_name, result = execute_model_task(
                model_name,
                X_fit_array,
                y_fit_array,
                X_predict_array,
                stage_label,
                "resource_queue_serial",
                MODEL_PARALLEL_JOBS,
                1,
                model_positions[model_name],
                model_count,
            )
            results_by_name[returned_name] = result
            completed_count += 1
            print_model_result(
                stage_label,
                returned_name,
                result,
                completed_count,
                model_count,
                stage_started_at,
            )
            if result[1] != "success":
                break
    else:
        try:
            with parallel_config(
                backend="loky",
                n_jobs=effective_jobs,
                inner_max_num_threads=MODEL_INNER_MAX_THREADS,
            ):
                completion_stream = Parallel(
                    n_jobs=effective_jobs,
                    backend="loky",
                    return_as="generator_unordered",
                    batch_size=1,
                    pre_dispatch=effective_jobs,
                    max_nbytes="1M",
                    mmap_mode="r",
                )(
                    delayed(execute_model_task)(
                        model_name,
                        X_fit_array,
                        y_fit_array,
                        X_predict_array,
                        stage_label,
                        "parallel_loky",
                        MODEL_PARALLEL_JOBS,
                        effective_jobs,
                        model_positions[model_name],
                        model_count,
                    )
                    for model_name in normal_execution_order
                )
                for returned_name, result in completion_stream:
                    results_by_name[returned_name] = result
                    completed_count += 1
                    print_model_result(
                        stage_label,
                        returned_name,
                        result,
                        completed_count,
                        model_count,
                        stage_started_at,
                    )
        finally:
            shutdown_reusable_loky_workers()

    gc.collect()
    stage_elapsed_minutes = (
        time.perf_counter() - stage_started_at
    ) / 60
    print(
        f"[{wall_time_text()}] [{stage_label}] 阶段结束 | "
        f"完成={len(results_by_name)}/{model_count} | "
        f"阶段耗时={stage_elapsed_minutes:.2f} 分钟",
        flush=True,
    )
    return [
        (model_name, results_by_name[model_name])
        for model_name in MODEL_NAMES
        if model_name in results_by_name
    ]


def persist_required_model_failure(logs, window_id, iteration):
    """在抛出终止异常前保存截至失败点的全部模型诊断。"""
    failure_df = pd.DataFrame(logs)
    failure_df["window_id"] = window_id
    failure_df["iteration"] = iteration
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure_path = RUN_OUTPUT_DIR / (
        "sg_al_required_model_failure_"
        f"window_{window_id}_iteration_{iteration}.csv"
    )
    failure_df.to_csv(
        failure_path,
        mode="a",
        header=not failure_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def run_first_layer_with_oof(
    X_train,
    y_train,
    X_test,
    fold_ids,
    train_stocks,
    test_stocks,
    window_id=None,
    iteration=None,
):
    """
    生成第一层无折内拟合泄漏的训练特征 Z_train 和测试特征 Z_test。

    在当前正式 SG-AL 架构中，本函数每个窗口只对真实历史 T0 调用一次：
    - OOF_FOLDS 个 OOF 拟合分别填充 Z_train^0 中对应历史验证块；
    - 再用完整 T0 拟合一次，预测完整测试季度形成固定 Z_test^0。
    测试季度伪标签不会作为本函数的训练输入。

    原始 77 维特征的填补和标准化必须在每个拟合内部重新估计；返回的
    Z_train/Z_test 均为活动模型固定顺序的硬标签列。
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(int)
    fold_ids = pd.Series(fold_ids).reset_index(drop=True).astype(int)
    train_stocks = pd.Series(train_stocks).reset_index(drop=True)
    test_stocks = pd.Series(test_stocks).reset_index(drop=True)
    layer_name = "layer_1"

    if not (
        len(X_train)
        == len(y_train)
        == len(fold_ids)
        == len(train_stocks)
    ):
        raise ValueError(
            "X_train、y_train、fold_ids 和 train_stocks 长度不一致。"
        )
    if len(X_test) != len(test_stocks):
        raise ValueError("X_test 和 test_stocks 长度不一致。")
    if sorted(fold_ids.unique().tolist()) != list(range(OOF_FOLDS)):
        raise ValueError(
            f"{layer_name} 必须具有完整的 {OOF_FOLDS} 个 OOF 折。"
        )

    validate_active_model_names(MODEL_NAMES)
    oof_predictions = {
        model_name: np.full(len(X_train), np.nan)
        for model_name in MODEL_NAMES
    }
    logs = []

    def abort(message):
        persist_required_model_failure(
            logs,
            window_id,
            iteration,
        )
        raise RuntimeError(message)

    for fold_id in range(OOF_FOLDS):
        validation_mask = fold_ids.eq(fold_id).to_numpy()
        fit_mask = ~validation_mask
        if not validation_mask.any() or not fit_mask.any():
            raise ValueError(
                f"{layer_name} 的第 {fold_id} 折为空。"
            )

        X_fit, X_validation = fit_fold_preprocessor(
            X_train.loc[fit_mask],
            train_stocks.loc[fit_mask],
            X_train.loc[validation_mask],
            train_stocks.loc[validation_mask],
        )

        fold_failures = []
        for model_name, result in run_required_model_batch(
            X_fit,
            y_train.loc[fit_mask],
            X_validation,
            stage_label=(
                f"window={window_id} | iter={iteration} | "
                f"layer_1 OOF fold={fold_id + 1}/{OOF_FOLDS}"
            ),
        ):
            prediction, status, error_message, diagnostics = result
            logs.append({
                "layer": layer_name,
                "model_name": model_name,
                "fit_stage": "oof",
                "fold_id": fold_id,
                "status": status,
                "error_message": error_message,
                **diagnostics,
            })
            if status != "success":
                fold_failures.append(
                    f"{layer_name} 的必需模型 {model_name} "
                    f"在 OOF 折 {fold_id} 失败：{error_message}"
                )
            else:
                oof_predictions[model_name][validation_mask] = prediction

        if fold_failures:
            abort(fold_failures[0])

        del X_fit, X_validation
        gc.collect()

    X_full_fit, X_full_test = fit_fold_preprocessor(
        X_train,
        train_stocks,
        X_test,
        test_stocks,
    )

    train_prediction_df = pd.DataFrame(index=np.arange(len(X_train)))
    test_prediction_df = pd.DataFrame(index=np.arange(len(X_test)))

    for model_name in MODEL_NAMES:
        if np.isnan(oof_predictions[model_name]).any():
            logs.append({
                "layer": layer_name,
                "model_name": model_name,
                "fit_stage": "oof_coverage_check",
                "fold_id": None,
                "status": "failed",
                "error_message": "OOF 预测未完整覆盖训练集。",
            })
            abort(
                f"{layer_name} 的 {model_name} 未完整生成 OOF 预测。"
            )

    full_fit_failures = []
    full_fit_results = run_required_model_batch(
        X_full_fit,
        y_train,
        X_full_test,
        stage_label=(
            f"window={window_id} | iter={iteration} | "
            "layer_1 full_train"
        ),
    )
    for model_name, result in full_fit_results:
        prediction, status, error_message, diagnostics = result
        logs.append({
            "layer": layer_name,
            "model_name": model_name,
            "fit_stage": "full_train",
            "fold_id": None,
            "status": status,
            "error_message": error_message,
            **diagnostics,
        })
        if status != "success":
            full_fit_failures.append(
                f"{layer_name} 的必需模型 {model_name} "
                f"在完整训练集拟合失败：{error_message}"
            )
        else:
            train_prediction_df[model_name] = (
                oof_predictions[model_name].astype(int)
            )
            test_prediction_df[model_name] = prediction.astype(int)

    if full_fit_failures:
        abort(full_fit_failures[0])

    if list(train_prediction_df.columns) != list(MODEL_NAMES):
        raise RuntimeError(
            f"{layer_name} 未生成完整有序的 "
            f"{len(MODEL_NAMES)} 列模型输出。"
        )

    return (
        train_prediction_df,
        test_prediction_df,
        pd.DataFrame(logs),
    )


def run_second_layer_full_fit(
    X_train,
    y_train,
    X_test,
    window_id=None,
    iteration=None,
):
    """
    第二层只在完整 Z_train 上拟合当前活动模型并预测 Z_test。

    当前没有第三层，也不消费第二层训练预测，因此无需再次做 OOF。
    这样每轮第二层只需每个活动模型拟合一次，同时不改变
    最终 P_test 定义。
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(int)
    layer_name = "layer_2"

    if len(X_train) != len(y_train):
        raise ValueError("X_train 和 y_train 长度不一致。")

    validate_active_model_names(MODEL_NAMES)
    test_prediction_df = pd.DataFrame(index=np.arange(len(X_test)))
    logs = []

    def abort(message):
        persist_required_model_failure(
            logs,
            window_id,
            iteration,
        )
        raise RuntimeError(message)

    failures = []
    for model_name, result in run_required_model_batch(
        X_train,
        y_train,
        X_test,
        stage_label=(
            f"window={window_id} | iter={iteration} | "
            "layer_2 full_train"
        ),
    ):
        prediction, status, error_message, diagnostics = result
        logs.append({
            "layer": layer_name,
            "model_name": model_name,
            "fit_stage": "full_train",
            "fold_id": None,
            "status": status,
            "error_message": error_message,
            **diagnostics,
        })
        if status != "success":
            failures.append(
                f"{layer_name} 的必需模型 {model_name} "
                f"在完整训练集拟合失败：{error_message}"
            )
        else:
            test_prediction_df[model_name] = prediction.astype(int)

    if failures:
        abort(failures[0])

    if list(test_prediction_df.columns) != list(MODEL_NAMES):
        raise RuntimeError(
            f"{layer_name} 未生成完整有序的 "
            f"{len(MODEL_NAMES)} 列模型输出。"
        )

    return test_prediction_df, pd.DataFrame(logs)


def ensemble_vote(prediction_df):
    """
    汇总第二层活动模型的硬标签，并以模型间标准差衡量分歧。

    pred_label 不是多数票，而是活动模型类别编号均值最接近的 0/1/2；
    agreement_rate 才是众数占比。NumPy 按列向量化计算，避免逐行
    DataFrame.apply/value_counts。
    """
    prediction_array = prediction_df.to_numpy(dtype=int, copy=False)
    p_mean_array = prediction_array.mean(axis=1)
    p_std_array = prediction_array.std(axis=1, ddof=0)
    label_array = np.asarray(LABELS, dtype=float)
    nearest = np.abs(
        p_mean_array[:, None] - label_array[None, :]
    ).argmin(axis=1)

    index = prediction_df.index
    pred_label = pd.Series(
        label_array[nearest].astype(int),
        index=index,
    )
    p_mean = pd.Series(p_mean_array, index=index)
    p_std = pd.Series(p_std_array, index=index)
    label_counts = np.column_stack([
        (prediction_array == label).sum(axis=1)
        for label in LABELS
    ])
    agreement_rate = pd.Series(
        label_counts.max(axis=1) / prediction_array.shape[1],
        index=index,
    )
    return pred_label, p_mean, p_std, agreement_rate


def compute_multiclass_metrics(y_true, y_pred):
    """计算逐类、总体和混淆矩阵指标；固定输出缺失类别的零值指标。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    metrics = {}
    for index, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = float(precision[index])
        metrics[f"recall_{label}"] = float(recall[index])
        metrics[f"f1_{label}"] = float(f1[index])
        metrics[f"support_{label}"] = int(support[index])
        metrics[f"acc_{label}"] = float(
            accuracy_score(
                (y_true == label).astype(int),
                (y_pred == label).astype(int),
            )
        )

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )
    metrics["weighted_f1"] = float(
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )
    metrics["confusion_matrix"] = json.dumps(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
        ensure_ascii=False,
    )
    return metrics


# ============================================================
# 6. 单季度 SG-AL：固定第一层、第二层累计反馈、剩余池递减
# ============================================================

def build_fixed_first_layer_representation(
    original_train_X,
    original_train_y,
    original_train_meta,
    original_test_X,
    original_test_meta,
    window_info,
):
    """
    每个滚动窗口只运行一次第一层，并冻结其输出供全部 ASL 轮次复用。

    历史训练样本使用严格的季度块 OOF 表示 Z_train^0；测试季度使用仅在 T0
    上 full-fit 的基础模型生成 Z_test^0。测试期伪标签永远不会进入这个步骤，
    因而不会反向改变任何历史样本的 OOF representation。
    """
    Z_train_0, Z_test_0, first_logs = run_first_layer_with_oof(
        original_train_X,
        original_train_y,
        original_test_X,
        original_train_meta["oof_fold"],
        original_train_meta[STOCK_COL],
        original_test_meta[STOCK_COL],
        window_id=window_info["window_id"],
        iteration=0,
    )
    first_logs["window_id"] = window_info["window_id"]
    first_logs["iteration"] = 0
    first_logs["adaptation_role"] = "fixed_historical_representation"

    if len(Z_train_0) != len(original_train_y):
        raise RuntimeError("固定第一层 Z_train^0 与历史标签长度不一致。")
    if len(Z_test_0) != len(original_test_X):
        raise RuntimeError("固定第一层 Z_test^0 与完整测试季度长度不一致。")
    if list(Z_train_0.columns) != list(Z_test_0.columns):
        raise RuntimeError("固定第一层训练/测试表示的模型列不一致。")

    return Z_train_0, Z_test_0, first_logs


def run_adaptive_second_layer_once(
    meta_train_Z,
    meta_train_y,
    fixed_test_Z,
    test_meta,
    window_info,
    iteration,
):
    """
    在冻结的第一层表示上拟合第二层，仅预测当前剩余测试池。

    第一层表示在窗口内保持不变；第二层训练集为真实历史表示加累计伪样本。
    """
    meta_train_Z = pd.DataFrame(meta_train_Z).reset_index(drop=True)
    fixed_test_Z = pd.DataFrame(fixed_test_Z).reset_index(drop=True)
    meta_train_y = pd.Series(meta_train_y).reset_index(drop=True).astype(int)
    test_meta = test_meta.reset_index(drop=True)

    if len(meta_train_Z) != len(meta_train_y):
        raise ValueError("第二层 meta_train_Z 与 meta_train_y 长度不一致。")
    if len(fixed_test_Z) != len(test_meta):
        raise ValueError("固定 Z_test^0 与测试元数据长度不一致。")
    if list(meta_train_Z.columns) != list(fixed_test_Z.columns):
        raise ValueError("第二层训练表示与固定测试表示的列契约不一致。")
    if ("test_row_id" not in test_meta or test_meta["test_row_id"].isna().any()
            or test_meta["test_row_id"].duplicated().any()):
        raise ValueError("剩余测试池必须保留唯一非空的原始 test_row_id。")

    P_test, second_logs = run_second_layer_full_fit(
        meta_train_Z,
        meta_train_y,
        fixed_test_Z,
        window_id=window_info["window_id"],
        iteration=iteration,
    )
    second_logs["window_id"] = window_info["window_id"]
    second_logs["iteration"] = iteration
    second_logs["adaptation_role"] = "adaptive_meta_learner"

    pred_label, p_mean, p_std, agreement_rate = ensemble_vote(P_test)

    round_prediction = test_meta.copy()
    # test_row_id 随剩余池保留，绝不能在每次缩池后从 0 重新编号。
    round_prediction["window_id"] = window_info["window_id"]
    round_prediction["train_period"] = window_info["train_period"]
    round_prediction["test_period"] = window_info["test_period"]
    round_prediction["prediction_asof"] = window_info["prediction_asof"]
    round_prediction["prediction_horizon"] = window_info["prediction_horizon"]
    round_prediction["prediction_target"] = window_info["source_target_col"]
    round_prediction["alignment_mode"] = window_info["alignment_mode"]
    round_prediction["iteration"] = iteration
    round_prediction["prediction_iteration"] = iteration
    round_prediction["prediction_protocol"] = PREDICTION_PROTOCOL
    round_prediction["pred_label"] = pred_label.to_numpy()
    round_prediction["p_mean"] = p_mean.to_numpy()
    round_prediction["p_std"] = p_std.to_numpy()
    round_prediction["s_tilde"] = p_std.to_numpy()
    round_prediction["agreement_rate"] = agreement_rate.to_numpy()
    round_prediction["requested_model_count"] = len(MODEL_NAMES)
    round_prediction["layer_1_active_model_count"] = fixed_test_Z.shape[1]
    round_prediction["layer_2_active_model_count"] = P_test.shape[1]
    round_prediction["layer_1_policy"] = LAYER_1_POLICY
    round_prediction["layer_2_policy"] = LAYER_2_POLICY

    diagnostics = {
        "layer_1_active_model_count": int(fixed_test_Z.shape[1]),
        "layer_2_active_model_count": int(P_test.shape[1]),
        "meta_train_rows": int(len(meta_train_Z)),
        "layer_1_recomputed_this_iteration": False,
    }

    del P_test
    gc.collect()
    return round_prediction, second_logs, diagnostics

def select_consensus_feedback(round_prediction, target_n):
    """从本轮剩余池选新增 top-d；按 p_std、稳定 ID 排序，不接触真实标签。"""
    target_n = int(target_n)
    if target_n <= 0 or target_n > len(round_prediction):
        raise ValueError("反馈数必须在 1 与当前剩余样本数之间。")
    if round_prediction["test_row_id"].duplicated().any():
        raise ValueError("剩余池存在重复 test_row_id。")
    p_std = round_prediction["p_std"].to_numpy(dtype=float)
    if not np.isfinite(p_std).all():
        raise ValueError("模型分歧必须为有限值。")
    row_ids = round_prediction["test_row_id"].to_numpy(dtype=int)
    selected = round_prediction.iloc[np.lexsort((row_ids, p_std))[:target_n]].copy()
    selected["feedback_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    selected["pseudo_label"] = selected["pred_label"].astype(int)
    return selected


def rebuild_meta_training_from_original(
    original_Z_train,
    original_train_y,
    fixed_Z_test_by_id,
    cumulative_feedback,
):
    """使用固定历史表示和全部累计伪样本重建第二层训练集，按原始 ID 对齐。"""
    original_Z_train = pd.DataFrame(original_Z_train).reset_index(drop=True)
    original_train_y = pd.Series(original_train_y).reset_index(drop=True).astype(int)
    selected_ids = cumulative_feedback["test_row_id"].to_numpy(dtype=int)
    if len(np.unique(selected_ids)) != len(selected_ids):
        raise ValueError("累计反馈中的 test_row_id 必须唯一。")
    if len(selected_ids) == 0:
        return original_Z_train, original_train_y
    pseudo_Z = fixed_Z_test_by_id.loc[selected_ids].reset_index(drop=True)
    pseudo_y = cumulative_feedback["pseudo_label"].astype(int).reset_index(drop=True)
    return (
        pd.concat([original_Z_train, pseudo_Z], ignore_index=True),
        pd.concat([original_train_y, pseudo_y], ignore_index=True).astype(int),
    )

def validate_pool_partition(initial_ids, remaining_ids, feedback):
    """累计训练伪样本和剩余池必须无重复、互斥并恰好覆盖原始池。"""
    selected_ids = feedback["test_row_id"].tolist()
    initial = set(initial_ids)
    remaining = set(remaining_ids)
    selected = set(selected_ids)
    if (len(initial) != len(initial_ids)
            or len(remaining) != len(remaining_ids)
            or len(selected) != len(selected_ids)
            or remaining & selected or remaining | selected != initial):
        raise RuntimeError("累计反馈池与剩余测试池不满足唯一、互斥、完整覆盖约束。")


def evaluate_saved_rounds(round_predictions, final_predictions, truth_by_id):
    """全部 AL 结束后再接入真实标签；返回逐轮明细与三种评价口径。"""
    rounds = round_predictions.merge(
        truth_by_id.rename("true_label"), left_on="test_row_id", right_index=True,
        how="left", validate="many_to_one",
    )
    if rounds["true_label"].isna().any():
        raise RuntimeError("逐轮预测存在无法按 ID 匹配的真实标签。")
    baseline = rounds.loc[rounds["iteration"].eq(1)].set_index("test_row_id")
    metric_rows = []
    for iteration, current in rounds.groupby("iteration", sort=True):
        earlier = final_predictions.loc[
            final_predictions["prediction_iteration"].lt(iteration)
        ]
        stitched = pd.concat([earlier, current], ignore_index=True)
        if (len(stitched) != len(truth_by_id)
                or stitched["test_row_id"].duplicated().any()
                or set(stitched["test_row_id"]) != set(truth_by_id.index)):
            raise RuntimeError("逐轮全池拼接的覆盖或唯一性检查失败。")
        comparisons = (
            ("remaining_pool", current["true_label"], current["pred_label"]),
            ("baseline_same_remaining_pool", current["true_label"],
             baseline.loc[current["test_row_id"], "pred_label"]),
            ("stitched_full_pool", stitched["true_label"], stitched["pred_label"]),
        )
        for scope, y_true, y_pred in comparisons:
            metric_rows.append({
                "window_id": int(current["window_id"].iloc[0]),
                "test_period": current["test_period"].iloc[0],
                "iteration": int(iteration),
                "evaluation_scope": scope,
                "sample_count": int(len(y_true)),
                **compute_multiclass_metrics(y_true, y_pred),
            })
    return rounds, pd.DataFrame(metric_rows)


def run_sg_al_single_window(df_model, feature_cols, window):
    """固定第一层；第二层累计反馈，拼接每个初始测试样本入训前的预测。"""
    data = prepare_window_data(df_model, feature_cols, window)
    historical_X = data["train_X_raw"].reset_index(drop=True)
    historical_y = data["y_train"].reset_index(drop=True)
    historical_meta = data["train_meta"].reset_index(drop=True)
    original_test_meta = data["test_meta"].reset_index(drop=True).copy()
    original_test_X = data["test_X_raw"].reset_index(drop=True)
    original_test_y = data["y_test"].reset_index(drop=True)
    window_info = dict(data["window_info"])
    del data

    initial_train_size = len(historical_X)
    initial_test_size = len(original_test_X)
    if initial_test_size <= 0:
        raise ValueError("测试季度不能为空。")
    if (len(original_test_meta) != initial_test_size
            or len(original_test_y) != initial_test_size):
        raise ValueError("测试特征、标签及元数据长度不一致。")
    if "test_row_id" not in original_test_meta:
        original_test_meta["test_row_id"] = np.arange(initial_test_size, dtype=int)
    ids = original_test_meta["test_row_id"]
    if ids.isna().any() or ids.duplicated().any() or not pd.api.types.is_integer_dtype(ids):
        raise ValueError("test_row_id 必须为唯一非空整数，且在整个窗口内保持不变。")
    initial_ids = ids.to_numpy(dtype=int)
    test_meta_by_id = original_test_meta.set_index("test_row_id", drop=False)

    if not 0 < SG_AL_CONSENSUS_FRACTION <= 1:
        raise ValueError("SG_AL_CONSENSUS_FRACTION 必须在 (0, 1] 内。")
    max_iterations = int(SG_AL_MAX_ITERATIONS)
    if max_iterations <= 0:
        raise ValueError("SG_AL_MAX_ITERATIONS 必须为正整数。")
    feedback_size = int(np.ceil(initial_test_size * SG_AL_CONSENSUS_FRACTION))
    remaining_ids = initial_ids.copy()
    feedback = pd.DataFrame({
        "test_row_id": pd.Series(dtype="int64"),
        "pseudo_label": pd.Series(dtype="int64"),
        "selected_iteration": pd.Series(dtype="int64"),
    })
    finalized_parts, round_parts = [], []
    iteration_logs, feedback_audits = [], []
    if SG_AL_VERBOSE:
        print(
            f"\n[{wall_time_text()}] 窗口开始 | window={window_info['window_id']} | "
            f"训练 {window_info['train_period']} | 测试 {window_info['test_period']} | "
            f"历史={initial_train_size:,}, 初始测试={initial_test_size:,}, "
            f"每轮新增上限={feedback_size:,}, 最多{max_iterations}轮",
            flush=True,
        )

    Z_train_0, Z_test_0, first_logs = build_fixed_first_layer_representation(
        historical_X, historical_y, historical_meta,
        original_test_X, original_test_meta, window_info,
    )
    fixed_Z_test_by_id = Z_test_0.copy().set_axis(initial_ids)
    layer_logs = [first_logs]
    del historical_X, original_test_X, Z_test_0
    gc.collect()

    for iteration in range(1, max_iterations + 1):
        iteration_started_at = time.perf_counter()
        validate_pool_partition(initial_ids, remaining_ids, feedback)
        pseudo_count = len(feedback)
        train_Z, train_y = rebuild_meta_training_from_original(
            Z_train_0, historical_y, fixed_Z_test_by_id, feedback,
        )
        if SG_AL_VERBOSE:
            print(
                f"[{wall_time_text()}] 迭代开始 | window={window_info['window_id']} | "
                f"iter={iteration}/{max_iterations} | Layer1=frozen | "
                f"meta_fit={len(train_Z):,} (pseudo={pseudo_count:,}) | "
                f"remaining={len(remaining_ids):,}", flush=True,
            )
        prediction, second_logs, diagnostics = run_adaptive_second_layer_once(
            train_Z, train_y, fixed_Z_test_by_id.loc[remaining_ids],
            test_meta_by_id.loc[remaining_ids], window_info, iteration,
        )
        layer_logs.append(second_logs)
        if (len(prediction) != len(remaining_ids)
                or prediction["test_row_id"].duplicated().any()
                or set(prediction["test_row_id"]) != set(remaining_ids)):
            raise RuntimeError("本轮预测未唯一覆盖剩余测试池。")
        selected = select_consensus_feedback(
            prediction, min(feedback_size, len(remaining_ids)),
        )
        reached_cap = iteration == max_iterations
        would_exhaust_pool = len(selected) == len(remaining_ids)
        applied = not reached_cap and not would_exhaust_pool
        termination_reason = (
            "max_iterations" if reached_cap else
            "pool_exhausted" if would_exhaust_pool else "continue"
        )
        selected_mask = prediction["test_row_id"].isin(selected["test_row_id"])
        prediction["feedback_candidate"] = selected_mask
        prediction["used_in_later_fit"] = selected_mask & applied
        prediction["selected_iteration"] = pd.Series(
            pd.NA, index=prediction.index, dtype="Int64",
        )
        if applied:
            prediction.loc[selected_mask, "selected_iteration"] = iteration
        # 先归档，再更新训练池；最后一轮归档全部剩余样本。
        finalized = prediction.loc[selected_mask].copy() if applied else prediction.copy()
        finalized["finalization_reason"] = "feedback" if applied else "terminal_remaining"
        finalized_parts.append(finalized)
        round_parts.append(prediction)

        audit = selected.copy()
        audit["event_type"] = "entered" if applied else "selected_not_applied"
        audit["selected_iteration"] = iteration
        audit["applied_to_next_fit"] = applied
        audit["used_in_later_fit"] = applied
        audit["first_training_iteration"] = iteration + 1 if applied else pd.NA
        feedback_audits.append(audit)
        iteration_logs.append({
            "window_id": window_info["window_id"],
            "test_period": window_info["test_period"],
            "iteration": iteration,
            "train_rows_used": int(len(train_Z)),
            "meta_train_rows_used": int(len(train_y)),
            "historical_train_rows": initial_train_size,
            "historical_meta_rows": initial_train_size,
            "pseudo_rows_used": pseudo_count,
            "remaining_test_rows": int(len(remaining_ids)),
            "feedback_target_n": feedback_size,
            "feedback_selected_n": int(len(selected)),
            "feedback_applied_n": int(len(selected)) if applied else 0,
            "feedback_applied_to_next_fit": applied,
            "cumulative_feedback_rows_after_round": pseudo_count + (len(selected) if applied else 0),
            "remaining_pool_rows_after_feedback": len(remaining_ids) - (len(selected) if applied else 0),
            "finalized_rows_this_round": int(len(finalized)),
            "unfinalized_rows_after_round": len(remaining_ids) - len(finalized),
            "mean_selected_p_std": float(selected["p_std"].mean()),
            "layer_1_policy": LAYER_1_POLICY,
            "layer_2_policy": LAYER_2_POLICY,
            "termination_reason": termination_reason,
            "elapsed_seconds": time.perf_counter() - iteration_started_at,
            **diagnostics,
        })
        if SG_AL_VERBOSE:
            print(
                f"[{wall_time_text()}] 迭代完成 | iter={iteration} | "
                f"归档={len(finalized):,} | 新增训练={len(selected) if applied else 0:,} | "
                f"状态={termination_reason}", flush=True,
            )
        if not applied:
            break
        new_feedback = selected[["test_row_id", "pseudo_label"]].copy()
        new_feedback["selected_iteration"] = iteration
        feedback = pd.concat([feedback, new_feedback], ignore_index=True)
        remaining_ids = remaining_ids[~np.isin(remaining_ids, selected["test_row_id"])]
        validate_pool_partition(initial_ids, remaining_ids, feedback)
        del train_Z, train_y
        gc.collect()

    predictions_df = pd.concat(finalized_parts, ignore_index=True)
    if (len(predictions_df) != initial_test_size
            or predictions_df["test_row_id"].duplicated().any()
            or set(predictions_df["test_row_id"]) != set(initial_ids)):
        raise RuntimeError("最终入训前预测未唯一覆盖完整初始测试池。")
    # 保持原始池顺序；真实标签到此才接入，按 ID 连接而不是按缩小后的行号。
    predictions_df = predictions_df.set_index("test_row_id").loc[initial_ids].reset_index()
    truth_by_id = pd.Series(
        original_test_y.to_numpy(dtype=int), index=pd.Index(initial_ids, name="test_row_id"),
    )
    predictions_df = predictions_df.merge(
        truth_by_id.rename("true_label"), left_on="test_row_id", right_index=True,
        how="left", validate="one_to_one",
    )
    predictions_df["final_iteration"] = iteration
    round_predictions_df, round_metrics_df = evaluate_saved_rounds(
        pd.concat(round_parts, ignore_index=True), predictions_df, truth_by_id,
    )
    metrics = compute_multiclass_metrics(
        predictions_df["true_label"], predictions_df["pred_label"],
    )
    metrics_row = {
        **window_info,
        "initial_train_rows": initial_train_size,
        "test_rows": initial_test_size,
        "predicted_rows": len(predictions_df),
        "feature_version": "quarterly_numeric_fold_local_impute_standardize",
        "model_names": ",".join(MODEL_NAMES),
        "requested_model_count": len(MODEL_NAMES),
        "feedback_size": feedback_size,
        "d_per_round": feedback_size,
        "max_iterations": max_iterations,
        "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
        "iterations_completed": iteration,
        "effective_feedback_updates": iteration - 1,
        "pseudo_rows_used_final_fit": len(feedback),
        "terminal_test_rows": len(remaining_ids),
        "termination_reason": termination_reason,
        "evaluation_scope": "stitched_full_pool",
        **metrics,
    }
    if SG_AL_VERBOSE:
        print(
            f"[{wall_time_text()}] 窗口完成 | window={window_info['window_id']} | "
            f"全池={initial_test_size:,} | accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}", flush=True,
        )
    return {
        "window_info": window_info,
        "predictions_df": predictions_df,
        "round_predictions_df": round_predictions_df,
        "round_metrics_df": round_metrics_df,
        "iteration_logs_df": pd.DataFrame(iteration_logs),
        "feedback_audit_df": pd.concat(feedback_audits, ignore_index=True),
        "model_logs_df": pd.concat(layer_logs, ignore_index=True).reindex(columns=MODEL_LOG_COLUMNS),
        "metrics_row": metrics_row,
    }


# ============================================================
# 7. 落盘与主程序
# ============================================================

RESULT_FRAME_FILES = {
    "predictions": ("predictions_df", "predictions.csv", "sg_al_predictions.csv"),
    "iterations": ("iteration_logs_df", "iteration_log.csv", "sg_al_iteration_logs.csv"),
    "feedback": ("feedback_audit_df", "feedback_audit.csv", "sg_al_feedback_audit.csv"),
    "models": ("model_logs_df", "model_log.csv", "sg_al_model_logs.csv"),
    "round_predictions": ("round_predictions_df", "round_predictions.csv", "sg_al_round_predictions.csv"),
    "round_metrics": ("round_metrics_df", "round_metrics.csv", "sg_al_round_metrics.csv"),
}


def append_frame(frame, path, write_state, state_key):
    """按窗口增量写顶层汇总，避免把全部预测和逐拟合日志常驻内存。"""
    if frame.empty:
        return
    first_write = not write_state.get(state_key, False)
    with open(path, "w" if first_write else "a", encoding="utf-8-sig", newline="") as file:
        frame.to_csv(file, header=first_write, index=False)
        file.flush()
        os.fsync(file.fileno())
    write_state[state_key] = True


def atomic_write_csv(frame, path):
    """先同步临时文件，再替换 CSV，避免中断留下半份检查点。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8-sig", newline="") as file:
        frame.to_csv(file, index=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def atomic_write_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def find_saved_resume_windows(windows, resume_from_test_quarter, output_dir=None):
    """恢复已提交的分区，包括断点后先于其他季度完成的窗口。"""
    completed, pending = split_windows_for_resume(windows, resume_from_test_quarter)
    if resume_from_test_quarter is None:
        return completed, pending
    target_dir = RUN_OUTPUT_DIR if output_dir is None else Path(output_dir)
    saved = [
        window for window in pending
        if (target_dir / "window_partitions" / window["test_quarter"] / "window_manifest.json").is_file()
    ]
    saved_quarters = {window["test_quarter"] for window in saved}
    return (
        sorted(completed + saved, key=lambda window: window["window_id"]),
        [window for window in pending if window["test_quarter"] not in saved_quarters],
    )


def clear_previous_run_outputs(output_dir=None):
    """只清理本脚本已知产物，避免重跑时混入上一次的部分结果。"""
    target_dir = RUN_OUTPUT_DIR if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    top_level_names = (
        "sg_al_predictions.csv",
        "sg_al_round_predictions.csv",
        "sg_al_round_metrics.csv",
        "sg_al_iteration_logs.csv",
        "sg_al_feedback_audit.csv",
        "sg_al_model_logs.csv",
        "sg_al_window_metrics.csv",
        "sg_al_pooled_metrics.csv",
        "sg_al_window_mean_metrics.csv",
        "sg_al_completed_window_definitions.csv",
        "sg_al_model_status_summary.csv",
        "sg_al_target_alignment_audit.csv",
        "sg_al_target_alignment_by_quarter.csv",
        "sg_al_quarter_sampling_audit.csv",
        "sg_al_quarter_panel_audit.csv",
        "sg_al_feature_group_counts.csv",
        "sg_al_feature_availability_audit.csv",
        "sg_al_feature_numeric_coercion_audit.csv",
        "sg_al_window_definitions.csv",
        "sg_al_run_manifest.json",
        # 首窗口计时入口复用本函数，以下为其固定产物名。
        "predictions.csv",
        "round_predictions.csv",
        "round_metrics.csv",
        "iteration_log.csv",
        "feedback_audit.csv",
        "model_log.csv",
        "metrics.csv",
        "window_definition.json",
        "benchmark_summary.json",
        # 兼容旧版单文件失败日志。
        "sg_al_required_model_failure.csv",
    )
    for name in top_level_names:
        path = target_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()

    for failure_path in target_dir.glob(
        "sg_al_required_model_failure_window_*_iteration_*.csv"
    ):
        if failure_path.is_file() or failure_path.is_symlink():
            failure_path.unlink()

    partition_names = (
        "predictions.csv",
        "round_predictions.csv",
        "round_metrics.csv",
        "iteration_log.csv",
        "feedback_audit.csv",
        "model_log.csv",
        "metrics.csv",
        "window_manifest.json",
    )
    partition_root = target_dir / "window_partitions"
    if partition_root.is_dir() and not partition_root.is_symlink():
        for quarter_dir in partition_root.iterdir():
            if not quarter_dir.is_dir() or quarter_dir.is_symlink():
                continue
            for name in partition_names:
                path = quarter_dir / name
                if path.exists() or path.is_symlink():
                    path.unlink()
            try:
                quarter_dir.rmdir()
            except OSError:
                # 保留用户放入分区目录的未知文件以及对应目录。
                pass


def serializable_window(window):
    """移除仅供内存筛选使用的 Period 对象和季度列表。"""
    return {
        key: value
        for key, value in window.items()
        if key != "train_quarters"
        and key not in {
            "train_start_period",
            "train_end_period",
            "test_period_obj",
        }
    }


def build_execution_metadata():
    """返回与实际窗口、OOF 和并行配置一致的运行清单字段。"""
    return {
        "window_frequency": "quarterly",
        "prediction_horizon": PREDICTION_HORIZON,
        "source_target_col": SOURCE_TARGET_COL,
        "target_col": TARGET_COL,
        "alignment_mode": TARGET_ALIGNMENT_MODE,
        "target_quarter_rule": TARGET_QUARTER_RULE,
        "train_window_quarters": TRAIN_WINDOW_QUARTERS,
        "test_window_quarters": 1,
        "step_quarters": 1,
        "max_iterations": SG_AL_MAX_ITERATIONS,
        "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
        "feedback_size_rule": FEEDBACK_SIZE_RULE,
        "oof_folds": OOF_FOLDS,
        "quarters_per_oof_fold": QUARTERS_PER_FOLD,
        "oof_split_policy": OOF_SPLIT_POLICY,
        "al_policy": AL_POLICY,
        "feedback_selection_policy": FEEDBACK_SELECTION_POLICY,
        "prediction_protocol": PREDICTION_PROTOCOL,
        "window_parallelism": WINDOW_PARALLEL_JOBS,
        "window_parallel_backend": (
            "serial"
            if WINDOW_PARALLEL_JOBS == 1
            else "spawn_process_pool"
        ),
        "layer_1_policy": LAYER_1_POLICY,
        "layer_2_policy": LAYER_2_POLICY,
        "pseudo_feedback_entry_layer": "layer_2_only",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_scope": "stitched_full_pool",
        "max_effective_feedback_updates": max(0, SG_AL_MAX_ITERATIONS - 1),
    }


_WINDOW_WORKER_DF_MODEL = None
_WINDOW_WORKER_FEATURE_COLS = None
_WINDOW_WORKER_RUNNER = None


def initialize_window_worker(df_model, feature_cols, window_runner):
    """每个窗口子进程只接收并常驻一份独立反序列化的只读输入。"""
    global _WINDOW_WORKER_DF_MODEL
    global _WINDOW_WORKER_FEATURE_COLS
    global _WINDOW_WORKER_RUNNER
    _WINDOW_WORKER_DF_MODEL = df_model
    _WINDOW_WORKER_FEATURE_COLS = list(feature_cols)
    _WINDOW_WORKER_RUNNER = window_runner


def execute_window_task(window):
    """在已初始化的窗口子进程中执行单个滚动窗口。"""
    if _WINDOW_WORKER_RUNNER is None:
        raise RuntimeError("窗口子进程尚未初始化。")
    return _WINDOW_WORKER_RUNNER(
        _WINDOW_WORKER_DF_MODEL,
        _WINDOW_WORKER_FEATURE_COLS,
        window,
    )


def load_completed_window_results(completed_windows, output_dir=None):
    """从断点前的季度分区恢复汇总所需的完整窗口结果。"""
    target_dir = RUN_OUTPUT_DIR if output_dir is None else Path(output_dir)
    partition_root = target_dir / "window_partitions"
    results = []
    for expected_window in completed_windows:
        test_period = expected_window["test_quarter"]
        quarter_dir = partition_root / test_period
        paths = {
            **{key: quarter_dir / names[1] for key, names in RESULT_FRAME_FILES.items()},
            "metrics": quarter_dir / "metrics.csv",
            "manifest": quarter_dir / "window_manifest.json",
        }
        missing = [
            path.name for path in paths.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"断点前窗口 {test_period} 的分区不完整：{missing}"
            )

        with open(paths["manifest"], "r", encoding="utf-8") as file:
            window_info = json.load(file)
        if (
            window_info.get("window_id") != expected_window["window_id"]
            or window_info.get("test_quarter") != test_period
        ):
            raise ValueError(
                f"断点前窗口 {test_period} 的清单与当前窗口定义不一致。"
            )

        metrics_df = pd.read_csv(paths["metrics"], encoding="utf-8-sig")
        if len(metrics_df) != 1:
            raise ValueError(
                f"断点前窗口 {test_period} 的 metrics.csv 必须恰好一行。"
            )
        # 分区必须匹配季度输入、标签口径和反馈配置，避免混入旧版结果。
        policies = {
            "prediction_horizon": PREDICTION_HORIZON,
            "source_target_col": SOURCE_TARGET_COL,
            "target_col": TARGET_COL,
            "alignment_mode": TARGET_ALIGNMENT_MODE,
            "target_quarter_rule": TARGET_QUARTER_RULE,
            "max_iterations": SG_AL_MAX_ITERATIONS,
            "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
            "al_policy": AL_POLICY,
            "layer_1_policy": LAYER_1_POLICY,
            "layer_2_policy": LAYER_2_POLICY,
            "feedback_selection_policy": FEEDBACK_SELECTION_POLICY,
            "oof_split_policy": OOF_SPLIT_POLICY,
            "prediction_protocol": PREDICTION_PROTOCOL,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "feedback_size_rule": FEEDBACK_SIZE_RULE,
            "pseudo_feedback_entry_layer": "layer_2_only",
        }
        metrics = metrics_df.iloc[0]
        for saved in (window_info, metrics.to_dict()):
            if any(saved.get(key) != value
                   for key, value in policies.items()):
                raise ValueError(f"断点前窗口 {test_period} 的标签口径或 AL/OOF/预测协议与当前版本不一致。")
        if (window_info.get("quarters_per_oof_fold") != QUARTERS_PER_FOLD
                or window_info.get("oof_fold_count") != OOF_FOLDS
                or metrics.get("model_names") != ",".join(MODEL_NAMES)):
            raise ValueError(f"断点前窗口 {test_period} 的季度 OOF 或活动模型与当前版本不一致。")
        results.append({
            "window_info": window_info,
            **{
                names[0]: pd.read_csv(paths[key], encoding="utf-8-sig", dtype={STOCK_COL: str})
                for key, names in RESULT_FRAME_FILES.items()
            },
            "metrics_row": metrics_df.iloc[0].to_dict(),
        })
    return results


def iter_window_results(
    df_model,
    feature_cols,
    windows,
    window_runner=run_sg_al_single_window,
    max_workers=None,
):
    """并行计算窗口，按实际完成顺序返回，立即允许主进程落盘。"""
    worker_count = (
        WINDOW_PARALLEL_JOBS if max_workers is None else int(max_workers)
    )
    if worker_count < 1:
        raise ValueError("窗口并行数必须大于等于 1。")
    if worker_count > AVAILABLE_CPU_COUNT:
        raise ValueError(
            "窗口并行数不能超过当前可用逻辑处理器数 "
            f"{AVAILABLE_CPU_COUNT}。"
        )
    nested_model_pool = MODEL_PARALLEL_JOBS > 1 or (
        "NCA" in MODEL_NAMES and NCA_EXCLUSIVE_PROCESS
    )
    if worker_count > 1 and nested_model_pool:
        raise ValueError(
            "窗口并行时须保持 MODEL_PARALLEL_JOBS=1，且不得同时启用"
            "活动 NCA 的 NCA_EXCLUSIVE_PROCESS，以避免嵌套进程池。"
        )

    if worker_count == 1:
        for window in windows:
            yield window_runner(df_model, feature_cols, window)
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
        initializer=initialize_window_worker,
        initargs=(df_model, feature_cols, window_runner),
    ) as executor:
        pending = {executor.submit(execute_window_task, window) for window in windows}
        for future in as_completed(pending):
            pending.remove(future)
            yield future.result()


def run_all_windows(
    df_model,
    feature_cols,
    windows,
    completed_results=None,
    window_runner=run_sg_al_single_window,
    max_workers=None,
    checkpoint_callback=None,
):
    """
    以有限进程并行计算窗口，并由主进程按实际完成顺序落盘。

    子进程不写正常结果，避免并发修改顶层汇总；每个完成结果由主进程
    同时写入独立分区和顶层增量汇总，随后立即释放并回收内存。
    """
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partition_root = RUN_OUTPUT_DIR / "window_partitions"
    partition_root.mkdir(parents=True, exist_ok=True)

    aggregate_paths = {
        key: RUN_OUTPUT_DIR / names[2] for key, names in RESULT_FRAME_FILES.items()
    }
    write_state = {}
    metrics_rows = []
    completed_windows = []
    model_status_parts = []
    pooled_true = []
    pooled_pred = []

    def record_window_result(result, persist_partition):
        window_info = result["window_info"]
        quarter_dir = (
            partition_root / window_info["test_period"]
        )
        if persist_partition:
            quarter_dir.mkdir(parents=True, exist_ok=True)
            # 先撤销旧提交标记，避免重算中断后把新旧文件组合视为已完成窗口。
            (quarter_dir / "window_manifest.json").unlink(missing_ok=True)
            for key, filename, _ in RESULT_FRAME_FILES.values():
                atomic_write_csv(result[key], quarter_dir / filename)
            atomic_write_csv(pd.DataFrame([result["metrics_row"]]), quarter_dir / "metrics.csv")
            # 清单最后提交；它存在即表示该季度的七份 CSV 均已完整同步。
            atomic_write_json(window_info, quarter_dir / "window_manifest.json")

        for state_key, (result_key, _, _) in RESULT_FRAME_FILES.items():
            append_frame(result[result_key], aggregate_paths[state_key], write_state, state_key)

        metrics_rows.append(result["metrics_row"])
        completed_windows.append(window_info)
        pooled_true.extend(
            result["predictions_df"]["true_label"].astype(int).tolist()
        )
        pooled_pred.extend(
            result["predictions_df"]["pred_label"].astype(int).tolist()
        )
        if not result["model_logs_df"].empty:
            model_status_parts.append(
                result["model_logs_df"]
                .groupby(
                    ["layer", "model_name", "status"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    count=("status", "size"),
                    elapsed_seconds=("elapsed_seconds", "sum"),
                )
            )

    started_at = time.perf_counter()
    for completed_result in completed_results or ():
        record_window_result(completed_result, persist_partition=False)
        gc.collect()

    def checkpoint():
        summary = persist_window_summaries(
            metrics_rows, completed_windows, model_status_parts,
            pooled_true, pooled_pred, feature_cols,
            time.perf_counter() - started_at,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(summary["completed_df"]["test_quarter"].tolist())
        return summary

    if completed_windows:
        checkpoint()

    for result in iter_window_results(
        df_model,
        feature_cols,
        windows,
        window_runner=window_runner,
        max_workers=max_workers,
    ):
        record_window_result(result, persist_partition=True)
        checkpoint()
        print(
            f"[{wall_time_text()}] 窗口结果已落盘 | "
            f"window={result['window_info']['window_id']} | "
            f"quarter={result['window_info']['test_quarter']} | "
            f"completed={len(completed_windows)} | 分区及全部汇总 CSV 已更新",
            flush=True,
        )
        gc.collect()

    # 汇总明细在运行中按完成顺序追加；结束时从分区恢复为季度顺序。
    for state_key, (_, filename, _) in RESULT_FRAME_FILES.items():
        target = aggregate_paths[state_key]
        temporary = target.with_name(target.name + ".tmp")
        with open(temporary, "w", encoding="utf-8-sig", newline="") as file:
            first_write = True
            for window in sorted(completed_windows, key=lambda item: item["window_id"]):
                frame = pd.read_csv(
                    partition_root / window["test_period"] / filename,
                    encoding="utf-8-sig", dtype={STOCK_COL: str},
                )
                if not frame.empty or first_write:
                    frame.to_csv(file, header=first_write, index=False)
                    first_write = False
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)

    return checkpoint()


def persist_window_summaries(
    metrics_rows, completed_windows, model_status_parts,
    pooled_true, pooled_pred, feature_cols, total_seconds,
):
    """每次提交季度后更新全部小型汇总，行顺序始终按季度排列。"""
    metrics_df = pd.DataFrame(metrics_rows).sort_values("window_id").reset_index(drop=True)
    completed_df = pd.DataFrame(completed_windows).sort_values("window_id").reset_index(drop=True)
    pooled_metrics = compute_multiclass_metrics(
        pooled_true,
        pooled_pred,
    )
    pooled_df = pd.DataFrame([{
        "sample_count": len(pooled_true),
        "window_count": len(completed_windows),
        "frequency": "quarterly",
        **build_execution_metadata(),
        "feature_count": len(feature_cols),
        "model_names": ",".join(MODEL_NAMES),
        "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        **pooled_metrics,
    }])

    metric_cols = [
        col for col in metrics_df.columns
        if col.startswith(
            ("precision_", "recall_", "f1_", "support_", "acc_")
        )
        or col in {"accuracy", "macro_f1", "weighted_f1"}
    ]
    window_mean_df = (
        metrics_df[metric_cols].mean().to_frame().T
    )
    window_mean_df.insert(0, "model_name", "SG-AL")
    window_mean_df.insert(1, "window_count", len(metrics_df))
    window_mean_df.insert(2, "total_seconds", total_seconds)

    if model_status_parts:
        model_status_summary_df = (
            pd.concat(model_status_parts, ignore_index=True)
            .groupby(
                ["layer", "model_name", "status"],
                as_index=False,
                dropna=False,
            )[["count", "elapsed_seconds"]]
            .sum()
        )
    else:
        model_status_summary_df = pd.DataFrame()

    for frame, filename in (
        (metrics_df, "sg_al_window_metrics.csv"),
        (pooled_df, "sg_al_pooled_metrics.csv"),
        (window_mean_df, "sg_al_window_mean_metrics.csv"),
        (completed_df, "sg_al_completed_window_definitions.csv"),
        (model_status_summary_df, "sg_al_model_status_summary.csv"),
    ):
        atomic_write_csv(frame, RUN_OUTPUT_DIR / filename)

    return {
        "total_seconds": total_seconds,
        "metrics_df": metrics_df,
        "pooled_df": pooled_df,
        "window_mean_df": window_mean_df,
        "completed_df": completed_df,
        "model_status_summary_df": model_status_summary_df,
    }


def main(max_windows=None, checkpoint_callback=None):
    """按“季度面板校验→契约→窗口→训练→汇总清单”的顺序运行流程。"""
    if max_windows is not None:
        max_windows = int(max_windows)
        if max_windows < 1:
            raise ValueError("max_windows 必须为正整数或 None。")
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 非断点运行从干净的已知产物集合开始；断点续跑必须保留已完成分区。
    if SG_AL_RUN_MODE != "full" or RESUME_FROM_TEST_QUARTER is None:
        clear_previous_run_outputs(RUN_OUTPUT_DIR)

    panel_result = load_quarterly_panel()
    quarterly_df = panel_result["quarterly_df"]
    feature_result = build_feature_contract(quarterly_df)
    feature_cols = feature_result["feature_cols"]

    # 特征只在这里统一解析一次。NaN 不在全局填补，真正的填补和标准化
    # 仍由每个第一层 OOF 拟合仅依据其训练子集完成。
    quarterly_df, feature_numeric_coercion_audit = (
        coerce_features_to_numeric(
            quarterly_df,
            feature_cols,
        )
    )
    # 末季或缺少下一季度末真实 FSFP 的行仅保留在对齐审计中，不参与训练/评价。
    df_model = (
        quarterly_df.dropna(subset=[TARGET_COL])
        .reset_index(drop=True)
    )
    all_windows = build_rolling_windows(df_model)
    if max_windows is not None:
        all_windows = all_windows[:max_windows]
    if SG_AL_RUN_MODE == "full":
        completed_windows, windows = find_saved_resume_windows(
            all_windows,
            RESUME_FROM_TEST_QUARTER,
        )
        completed_results = load_completed_window_results(
            completed_windows,
        )
    else:
        completed_windows = []
        completed_results = []
        windows = all_windows
    window_definitions_df = pd.DataFrame([
        serializable_window(window)
        for window in all_windows
    ])

    panel_result["target_alignment_audit"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_target_alignment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel_result["target_alignment_by_quarter"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_target_alignment_by_quarter.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel_result["quarter_panel_audit"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_quarter_panel_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_result["feature_group_count_df"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_feature_group_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_result["feature_availability_audit_df"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_feature_availability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_numeric_coercion_audit.to_csv(
        RUN_OUTPUT_DIR / "sg_al_feature_numeric_coercion_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    window_definitions_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_window_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"特征契约通过：{len(feature_cols)} 个特征；"
        f"滚动窗口：{len(all_windows)} 个；"
        f"运行模式：{SG_AL_RUN_MODE}；"
        f"窗口并行：{WINDOW_PARALLEL_JOBS}。"
    )
    if SG_AL_RUN_MODE == "full":
        print(
            f"断点续跑：已加载 {len(completed_windows)} 个窗口；"
            f"从 {RESUME_FROM_TEST_QUARTER} 开始执行 "
            f"{len(windows)} 个窗口。"
        )
    print(
        f"时间配置：训练 {TRAIN_WINDOW_QUARTERS} 个季度；"
        f"OOF {OOF_FOLDS} 折 × 每折 {QUARTERS_PER_FOLD} 个季度。"
    )
    print(
        f"标签口径：本季度特征 -> 输入列 {SOURCE_TARGET_COL}（同股下一季度 FSFP_LLM）；"
        f"因目标缺失排除 {len(quarterly_df) - len(df_model):,} 行。"
    )
    print(
        f"反馈配置：每轮从剩余池新增最多 ceil({SG_AL_CONSENSUS_FRACTION} × 初始测试数) 条；"
        f"最多 {SG_AL_MAX_ITERATIONS} 轮（含第 1 轮历史基准预测），最终拼接全池预测。"
    )
    print(f"结果目录：{RUN_OUTPUT_DIR}")
    print(f"固定 {len(MODEL_NAMES)} 个基础学习器：" + ", ".join(MODEL_NAMES))
    if MODEL_PARALLEL_JOBS == 1:
        print(
            "模型执行：逐模型串行；失败后立即停止当前折。"
        )
    else:
        nca_text = (
            "NCA 独占 spawn 子进程；其余模型"
            if "NCA" in MODEL_NAMES
            else f"{len(MODEL_NAMES)} 个模型"
        )
        print(
            f"模型安全并行：{nca_text}最多 "
            f"{MODEL_PARALLEL_JOBS} 个 loky 进程；"
            f"每进程内部线程上限 {MODEL_INNER_MAX_THREADS}；"
            "OOF 折和 SG-AL 迭代串行。"
        )

    run_result = run_all_windows(
        df_model,
        feature_cols,
        windows,
        completed_results=completed_results,
        checkpoint_callback=checkpoint_callback,
    )

    feature_group_counts = {
        row.feature_group: int(row.actual_count)
        for row in feature_result[
            "feature_group_count_df"
        ].itertuples(index=False)
    }
    with open(
        RUN_OUTPUT_DIR / "sg_al_run_manifest.json",
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            {
                "input_file": str(INPUT_FILE),
                "input_preprocessing": INPUT_PREPROCESSING,
                "input_preprocessing_version": 2,
                "input_bfill_uses_future_quarters": True,
                "input_quarter_statistics_source": "original_nonmissing_observations_only",
                "input_quarter_statistics_fallback": "latest_previous_quarter",
                "quarterly_input_rows": int(
                    panel_result["quarterly_input_shape"][0]
                ),
                "quarterly_rows": int(len(quarterly_df)),
                "model_rows": int(len(df_model)),
                "stock_count": int(df_model[STOCK_COL].nunique()),
                "frequency": "quarterly",
                "quarter_col": QUARTER_COL,
                "quarter_definition": QUARTER_DEFINITION,
                "quarter_value_rule": QUARTER_VALUE_RULE,
                "target_alignment": TARGET_ALIGNMENT_MODE,
                "prediction_horizon": PREDICTION_HORIZON,
                "source_target_col": SOURCE_TARGET_COL,
                "target_col": TARGET_COL,
                "current_fsfp_excluded_from_features": CURRENT_FSFP_COL,
                **build_execution_metadata(),
                "test_window_quarters": 1,
                "step_quarters": 1,
                "run_mode": SG_AL_RUN_MODE,
                "window_limit": max_windows,
                "resume_from_test_quarter": (
                    RESUME_FROM_TEST_QUARTER
                    if SG_AL_RUN_MODE == "full"
                    else None
                ),
                "resumed_completed_window_count": len(
                    completed_windows
                ),
                "single_test_quarter": (
                    windows[0]["test_quarter"]
                    if len(windows) == 1
                    else None
                ),
                "single_quarter_train_max_rows": (
                    SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS
                    if SG_AL_RUN_MODE == "single_quarter_test"
                    else None
                ),
                "feature_count": len(feature_cols),
                "feature_columns": feature_cols,
                "feature_group_counts": feature_group_counts,
                "feature_values_coerced_to_missing": int(
                    feature_numeric_coercion_audit[
                        "coerced_to_missing"
                    ].sum()
                ),
                "Peer_FSFP_included": "Peer_FSFP" in feature_cols,
                "Peer_FSFP_lag_included": PEER_FSFP_FEATURE_COL in feature_cols,
                "model_names": MODEL_NAMES,
                "model_parallel_jobs": MODEL_PARALLEL_JOBS,
                "model_parallel_backend": (
                    "serial"
                    if MODEL_PARALLEL_JOBS == 1
                    else (
                        "nca_exclusive_spawn_plus_loky"
                        if "NCA" in MODEL_NAMES
                        else "bounded_model_level_loky"
                    )
                ),
                "model_inner_max_threads": MODEL_INNER_MAX_THREADS,
                "nca_exclusive_process": (
                    "NCA" in MODEL_NAMES
                    and (
                        MODEL_PARALLEL_JOBS > 1
                        or NCA_EXCLUSIVE_PROCESS
                    )
                ),
                "oof_fold_parallelism": 1,
                "sg_al_iteration_parallelism": 1,
                "base_learner_policy": (
                    "15_models_nca_itml_excluded_no_fallback"
                ),
                "layer_1_frozen_within_window": True,
                "implemented_optimizations": [
                    "direct_quarterly_input_without_resampling",
                    "single_global_quarter_panel_sort",
                    "one_time_feature_numeric_coercion",
                    "vectorized_second_layer_vote",
                    "fixed_layer1_historical_oof_and_test_representation",
                    "pseudo_labels_enter_layer2_meta_training_only",
                    "cumulative_feedback_in_fixed_layer1_representation",
                    "remove_feedback_from_remaining_test_pool",
                    "freeze_predictions_before_feedback_and_stitch_full_cohort",
                    "bounded_window_level_spawn_parallelism",
                    "garbage_collection_at_stage_boundaries",
                    "incremental_window_output",
                ],
                "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
                "max_iterations": SG_AL_MAX_ITERATIONS,
                "feedback_size_rule": FEEDBACK_SIZE_RULE,
                "feedback_selection_policy": FEEDBACK_SELECTION_POLICY,
                "total_seconds": run_result["total_seconds"],
                "output_dir": str(RUN_OUTPUT_DIR),
            },
            manifest_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"季度 SG-AL 完成，用时 "
        f"{run_result['total_seconds'] / 60:.2f} 分钟。"
    )
    print(f"结果目录：{RUN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
