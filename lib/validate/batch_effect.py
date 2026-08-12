from typing import Dict

import numpy as np
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
