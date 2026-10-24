from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from starsort.kmeans import SEED, kmeans

CLUSTERS = 6
NOISE = -1
MIN_SAMPLES = 5
EPS_GRID = np.round(np.arange(0.1, 1.51, 0.05), 2)


@dataclass(frozen=True)
class Clustering:
    labels: np.ndarray
    detail: str


def scale(features) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(features, dtype=float))


def by_kmeans(X: np.ndarray, k: int, seed: int) -> Clustering:
    result = kmeans(X, k, seed=seed)
    return Clustering(result.labels, f"k={k}")


def by_hierarchy(X: np.ndarray, k: int, seed: int) -> Clustering:
    return Clustering(AgglomerativeClustering(k, linkage="ward").fit_predict(X), f"k={k}, ward")


def by_mixture(X: np.ndarray, k: int, seed: int) -> Clustering:
    model = GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed)
    return Clustering(model.fit_predict(X), f"k={k}, full covariance")


def by_density(X: np.ndarray, k: int, seed: int) -> Clustering:
    best: tuple[float, float, np.ndarray] | None = None
    for eps in EPS_GRID:
        labels = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
        if len(set(labels) - {NOISE}) < 2:
            continue
        score = silhouette_score(X, labels)
        if best is None or score > best[0]:
            best = (score, eps, labels)
    if best is None:
        return Clustering(np.full(len(X), NOISE), "no eps gives two clusters")
    return Clustering(best[2], f"eps={best[1]:.2f}, min_samples={MIN_SAMPLES}")


@dataclass(frozen=True)
class Algorithm:
    key: str
    name: str
    fit: Callable[[np.ndarray, int, int], Clustering]


ALGORITHMS = (
    Algorithm("kmeans", "K-Means", by_kmeans),
    Algorithm("hierarchical", "Hierarchical", by_hierarchy),
    Algorithm("dbscan", "DBSCAN", by_density),
    Algorithm("mixture", "Gaussian Mixture", by_mixture),
)
BEST_ALGORITHM = "mixture"


def algorithm(key: str) -> Algorithm:
    for item in ALGORITHMS:
        if item.key == key.lower():
            return item
    choices = ", ".join(item.key for item in ALGORITHMS)
    raise ValueError(f"unknown algorithm {key!r}, choose one of: {choices}")


def cluster(features, method: Algorithm, k: int = CLUSTERS, seed: int = SEED) -> Clustering:
    if k < 2:
        raise ValueError("at least two clusters are needed")
    return method.fit(scale(features), k, seed)
