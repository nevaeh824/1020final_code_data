"""
SG-AL 学习器注册表。

注册 19 个候选学习器；主程序默认排除 NCA、ITML、ExtraTrees、HGB，
因此正式 PeerMeta-style stacking 使用 15 个活动模型。参数尽量贴近作者
公开 ``detection_model.py``，但补充显式随机种子以保证跨 OOF 折、ASL 轮次
和完整重跑的可复现性。

NCA、ITML 本身是监督度量学习器，不能直接输出 0/1/2 类别，因此通过
Pipeline 连接 KNN。LMNN 因当前样本规模下完整训练成本过高未注册；
ExtraTrees/HGB 仅作为扩展候选，不属于 15-model reference-code core。

注意：为忠实保留作者公开实现，名为 Probit 的模型仍是 SAG Logistic，
Tree-ID3 与 Tree-C4.5 也都使用 sklearn entropy tree；代码注释明确保留这一
“reference-code implementation”差异，避免把它误称为严格统计 Probit/C4.5。
"""

from functools import partial

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_RANDOM_STATE = 42
QDA_REG_PARAM = 1e-3


REQUIRED_MODEL_NAMES = (
    # 判别分析、广义线性模型和朴素贝叶斯
    "LDA",
    "QDA",
    "Logit",
    "Probit",
    "NaiveBayes",
    # 三种论文命名的决策树
    "Tree-ID3",
    "Tree-C4.5",
    "Tree-CART",
    # 一至三层隐藏层的前馈神经网络
    "FNN1",
    "FNN2",
    "FNN3",
    # 三种核函数的支持向量机
    "SVM-Lin",
    "SVM-Poly",
    "SVM-RBF",
    # 原始 KNN 与两种“度量学习 + KNN”（度量学习模型默认排除）
    "KNN",
    "NCA",
    "ITML",
    # 随机化森林与直方图梯度提升；追加名称以保持已有模型的相对顺序
    "ExtraTrees",
    "HGB",
)


def validate_required_model_names(model_names):
    """要求名称和顺序完全一致，因为每个 stacking 列都按该顺序解释。"""
    names = tuple(model_names)
    if names != REQUIRED_MODEL_NAMES:
        missing = [name for name in REQUIRED_MODEL_NAMES if name not in names]
        extra = [name for name in names if name not in REQUIRED_MODEL_NAMES]
        raise ValueError(
            f"完整模型注册表必须包含 {len(REQUIRED_MODEL_NAMES)} 个学习器，且顺序必须一致；"
            f"missing={missing}, extra={extra}, received={list(names)}"
        )
    return list(names)


def build_required_model(model_name):
    """按名称创建一个尚未拟合的基础学习器。"""
    if model_name == "LDA":
        return LinearDiscriminantAnalysis()
    if model_name == "QDA":
        # 第二层 stacking 输入是多个分类器的离散硬标签，列间高度相关，
        # 新版 sklearn 对奇异类内协方差会直接报错。极小 ridge 正则避免
        # 数值失败，同时尽量贴近未正则化 QDA。
        return QuadraticDiscriminantAnalysis(reg_param=QDA_REG_PARAM)
    if model_name == "Logit":
        # 作者参考代码使用 liblinear，但新版 sklearn 不再允许它直接处理
        # 三分类。用 One-vs-Rest 包装保留 liblinear 二分类器的原始意图，
        # 同时使 PeerMeta 的 0/1/2 标签可以稳定运行。
        return OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                random_state=MODEL_RANDOM_STATE,
            )
        )
    if model_name == "Probit":
        # 忠实保留参考代码：名称虽为 Probit，实际是 SAG Logistic，
        # 并不是带正态分布链接函数的严格 Probit。
        return LogisticRegression(solver="sag", random_state=MODEL_RANDOM_STATE)
    if model_name == "NaiveBayes":
        return GaussianNB()
    if model_name == "Tree-ID3":
        return DecisionTreeClassifier(criterion="entropy", random_state=MODEL_RANDOM_STATE)
    if model_name == "Tree-C4.5":
        # 参考实现与 ID3 一样使用 entropy。sklearn 没有原生实现
        # C4.5 的增益率划分，因此这里不是严格意义的 C4.5。
        return DecisionTreeClassifier(criterion="entropy", random_state=MODEL_RANDOM_STATE)
    if model_name == "Tree-CART":
        return DecisionTreeClassifier(criterion="gini", random_state=MODEL_RANDOM_STATE)
    if model_name == "FNN1":
        return MLPClassifier(
            hidden_layer_sizes=(32,), solver="sgd", alpha=0.01,
            random_state=MODEL_RANDOM_STATE,
        )
    if model_name == "FNN2":
        return MLPClassifier(
            hidden_layer_sizes=(32, 32), solver="sgd", alpha=0.01,
            random_state=MODEL_RANDOM_STATE,
        )
    if model_name == "FNN3":
        return MLPClassifier(
            hidden_layer_sizes=(32, 32, 32), solver="sgd", alpha=0.01,
            random_state=MODEL_RANDOM_STATE,
        )
    if model_name == "SVM-Lin":
        return SVC(kernel="linear", random_state=MODEL_RANDOM_STATE)
    if model_name == "SVM-Poly":
        return SVC(kernel="poly", random_state=MODEL_RANDOM_STATE)
    if model_name == "SVM-RBF":
        return SVC(kernel="rbf", random_state=MODEL_RANDOM_STATE)
    if model_name == "KNN":
        return KNeighborsClassifier()
    if model_name == "ExtraTrees":
        # 外层已控制进程并行；限制树规模和内部线程，控制重复拟合成本。
        return ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=MODEL_RANDOM_STATE,
        )
    if model_name == "HGB":
        # auto 早停只使用当前拟合数据，独立于外层 SG 反馈循环；
        # OpenMP 线程数由两个入口在导入 sklearn 前限制为 1。
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping="auto",
            random_state=MODEL_RANDOM_STATE,
        )
    if model_name == "NCA":
        # 先学习监督距离变换，再在变换后的空间用 KNN 输出类别。
        return Pipeline([
            ("metric", NeighborhoodComponentsAnalysis(random_state=MODEL_RANDOM_STATE)),
            ("classifier", KNeighborsClassifier()),
        ])
    if model_name == "ITML":
        # metric-learn 延迟导入：只有真正创建 ITML 时才要求依赖。
        try:
            from metric_learn import ITML_Supervised
        except ImportError as exc:
            raise ImportError(
                "启用 ITML 需要安装 metric-learn，活动模型拟合失败时不允许静默跳过。"
            ) from exc
        return Pipeline([
            ("metric", ITML_Supervised(random_state=MODEL_RANDOM_STATE)),
            ("classifier", KNeighborsClassifier()),
        ])
    raise KeyError(f"未知基础学习器: {model_name}")


def required_model_factories():
    """为全部注册模型返回零参数工厂；每次调用都得到独立的新模型实例。"""
    return {
        model_name: partial(build_required_model, model_name)
        for model_name in REQUIRED_MODEL_NAMES
    }
