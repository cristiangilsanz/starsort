from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED = 42


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int


def squared_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    cross = X @ centroids.T
    distances = (X**2).sum(axis=1)[:, None] - 2 * cross + (centroids**2).sum(axis=1)[None, :]
    return np.maximum(distances, 0.0)


def plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centroids = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        nearest = squared_distances(X, np.array(centroids)).min(axis=1)
        total = nearest.sum()
        if total == 0:
            centroids.append(X[rng.integers(len(X))])
            continue
        centroids.append(X[rng.choice(len(X), p=nearest / total)])
    return np.array(centroids)


def lloyd(X: np.ndarray, centroids: np.ndarray, max_iter: int, tol: float) -> KMeansResult:
    threshold = tol * X.var(axis=0).mean()
    k = len(centroids)
    centroids = centroids.copy()
    previous = None
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        distances = squared_distances(X, centroids)
        labels = distances.argmin(axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels
        counts = np.bincount(labels, minlength=k)
        sums = np.column_stack(
            [np.bincount(labels, weights=X[:, axis], minlength=k) for axis in range(X.shape[1])]
        )
        updated = centroids.copy()
        filled = counts > 0
        updated[filled] = sums[filled] / counts[filled, None]
        if not filled.all():
            farthest = np.argsort(distances.min(axis=1))[::-1]
            updated[~filled] = X[farthest[: (~filled).sum()]]
        shift = ((updated - centroids) ** 2).sum()
        centroids = updated
        if shift <= threshold:
            break
    distances = squared_distances(X, centroids)
    labels = distances.argmin(axis=1)
    inertia = float(distances[np.arange(len(X)), labels].sum())
    return KMeansResult(labels, centroids, inertia, iterations)


def kmeans(
    X,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-8,
    seed: int = SEED,
) -> KMeansResult:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X):
        raise ValueError("k-means needs a non-empty two-dimensional array")
    if not 1 <= k <= len(X):
        raise ValueError(f"k must be between 1 and {len(X)}, got {k}")
    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive")
    rng = np.random.default_rng(seed)
    runs = (lloyd(X, plus_plus(X, k, rng), max_iter, tol) for _ in range(n_init))
    return min(runs, key=lambda run: run.inertia)
