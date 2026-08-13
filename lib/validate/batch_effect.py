from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def compute_eta_squared(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    潜在表現の各次元における効果量（η²）を計算する関数。

    Eta^2 = SS_effect / SS_total

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        グループラベルのデータ。

    Returns
    -------
    Dict[str, float]
        平均値、中央値、および最大値のη²を含む辞書。
    """
    # Xの形状を取得
    # タプルはこうやってアンパックできる
    N, D = X.shape
    # ユニークなラベルとそのカウントを取得
    # np.uniqueは、配列内のユニークな要素を返す関数で、return_counts=Trueを指定すると、それぞれのユニークな要素の出現回数も返す
    unique_labels, counts = np.unique(y, return_counts=True)

    # ラベルによらない総平方和（SS_total）を計算
    grand_mean = np.mean(X, axis=0)

    # 全体の平方和を計算
    ss_total = np.sum((X - grand_mean) ** 2, axis=0)

    # グループごとの平均を計算
    group_means = np.zeros((len(unique_labels), D))
    # グループごとの平均を計算するために、各ラベルに対してXの対応する行を抽出し、その平均を計算
    for i, label in enumerate(unique_labels):
        group_means[i] = np.mean(X[y == label], axis=0)

    # グループ間平方和（SS_between）を計算
    ss_between = np.sum(counts[:, np.newaxis] * (group_means - grand_mean) ** 2, axis=0)

    # ゼロ除算を避けるために、ss_totalがゼロの場合は小さな値に置き換える
    ss_total = np.where(ss_total == 0, 1e-10, ss_total)  # Avoid division by zero

    # η²を計算
    eta_sq_per_dim = ss_between / ss_total

    return {
        "eta_sq_mean": float(np.mean(eta_sq_per_dim)),
        "eta_sq_median": float(np.median(eta_sq_per_dim)),
        "eta_sq_max": float(np.max(eta_sq_per_dim)),
        "eta_sq_var": float(np.var(eta_sq_per_dim)),
    }


def compute_knn_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 15,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    stratified K-Fold クロスバリデーションを使用して、潜在表現のKNN分類器の精度を計算する関数。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        グループラベルのデータ。
    n_neighbors : int, optional
        KNN分類器の近傍数（デフォルトは15）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42）。

    Returns
    -------
    float
        KNN分類器の平均精度。
    """
    # n_jobs=-1 is set on cross_val_score below, which already parallelizes
    # across folds; parallelizing here too would oversubscribe CPUs/memory.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(knn, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

    return float(np.mean(scores))


def compute_logreg_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    L2正則化ロジスティック回帰による二値分類プローブ(concept-erasing-toxpathoの
    同名関数を移植)。

    サンプル数が次元数より少ない(例: パッチをスライド単位で平均プーリングした特徴量は
    サンプル数=スライド数と少なくなりがち)ような高次元・少サンプルの状況でも、
    正則化された線形モデルは安定して線形の手がかりを検出できる。クラス不均衡を
    考慮してclass_weight="balanced"を使い、balanced_accuracyとROC-AUCの両方を返す。

    Parameters
    ----------
    X : np.ndarray
        潜在表現のデータ（各次元の特徴量を含む）。
    y : np.ndarray
        二値ラベル（例: 病理所見の有無）。
    n_splits : int, optional
        クロスバリデーションの分割数（デフォルトは5）。
    random_state : int, optional
        乱数シード（デフォルトは42）。

    Returns
    -------
    Dict[str, float]
        "balanced_accuracy": 平均balanced accuracy。
        "roc_auc": 平均ROC-AUC。
    """
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    balanced_acc = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy", n_jobs=-1)
    roc_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

    return {
        "balanced_accuracy": float(np.mean(balanced_acc)),
        "roc_auc": float(np.mean(roc_auc)),
    }


def stratified_subsample_indices(labels: np.ndarray, max_per_group: int, seed: int) -> np.ndarray:
    """Indices capping each unique value in `labels` to at most `max_per_group` rows.

    Plain global random subsampling would let small groups fall under a downstream
    fold count (StratifiedKFold) or visual sample size by chance; capping per-group
    instead guarantees every group keeps min(group_size, max_per_group) samples.
    Used both to bound KNN CV cost (compute_knn_accuracy) and to pick a legible,
    evenly-represented subsample for scatter-plot visualizations.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        keep.append(idx)
    idx = np.concatenate(keep)
    idx.sort()
    return idx
