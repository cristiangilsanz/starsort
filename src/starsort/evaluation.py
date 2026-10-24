from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

from starsort.clustering import ALGORITHMS, CLUSTERS, NOISE, Algorithm, cluster, scale
from starsort.data import TYPE
from starsort.encoding import ENCODINGS, Encoding, radius_magnitude
from starsort.kmeans import SEED, kmeans


@dataclass(frozen=True)
class Result:
    encoding: str
    algorithm: str
    clusters: int
    noise: int
    silhouette: float
    agreement: float
    detail: str


def score(frame: pd.DataFrame, encoding: Encoding, method: Algorithm, k: int = CLUSTERS) -> Result:
    features = encoding.encode(frame)
    found = cluster(features, method, k)
    labels = found.labels
    groups = set(labels) - {NOISE}
    silhouette = silhouette_score(scale(features), labels) if len(groups) > 1 else float("nan")
    agreement = adjusted_rand_score(frame[TYPE], labels) if TYPE in frame else float("nan")
    return Result(
        encoding=encoding.name,
        algorithm=method.name,
        clusters=len(groups),
        noise=int((labels == NOISE).sum()),
        silhouette=float(silhouette),
        agreement=float(agreement),
        detail=found.detail,
    )


def compare(
    frame: pd.DataFrame,
    encodings: Sequence[Encoding] = ENCODINGS,
    methods: Sequence[Algorithm] = ALGORITHMS,
    k: int = CLUSTERS,
) -> list[Result]:
    return [score(frame, encoding, method, k) for encoding in encodings for method in methods]


def ordered(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    found = [label for label in np.unique(labels) if label != NOISE]
    faintest = sorted(found, key=lambda label: -frame["magnitude"][labels == label].median())
    ranks = {label: rank for rank, label in enumerate(faintest, start=1)}
    return np.array([ranks.get(label, NOISE) for label in labels])


def match(labels: np.ndarray, types: pd.Series) -> dict[int, tuple[str, float]]:
    table = pd.crosstab(np.asarray(labels), np.asarray(types))
    return {
        int(label): (str(row.idxmax()), float(row.max() / row.sum()))
        for label, row in table.iterrows()
        if label != NOISE
    }


def profiles(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    grouped = frame.assign(cluster=labels).groupby("cluster")
    summary = grouped.agg(
        stars=("temperature", "size"),
        temperature=("temperature", "median"),
        luminosity=("luminosity", "median"),
        radius=("radius", "median"),
        magnitude=("magnitude", "median"),
        spectral_class=("spectral_class", lambda classes: classes.mode().iloc[0]),
    )
    if TYPE in frame:
        matched = match(labels, frame[TYPE])
        summary["type"] = [matched.get(label, ("Noise", 0.0))[0] for label in summary.index]
        summary["purity"] = [matched.get(label, ("Noise", 0.0))[1] for label in summary.index]
    return summary


@dataclass(frozen=True)
class Benchmark:
    dataset: str
    samples: int
    own_seconds: float
    library_seconds: float
    own_inertia: float
    library_inertia: float
    agreement: float


def timed(run, repeats: int = 3):
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = run()
        seconds.append(time.perf_counter() - started)
    return result, min(seconds)


def benchmark(X: np.ndarray, dataset: str, k: int = CLUSTERS, seed: int = SEED) -> Benchmark:
    own, own_seconds = timed(lambda: kmeans(X, k, n_init=10, seed=seed))
    library, library_seconds = timed(lambda: KMeans(k, n_init=10, random_state=seed).fit(X))
    return Benchmark(
        dataset=dataset,
        samples=len(X),
        own_seconds=own_seconds,
        library_seconds=library_seconds,
        own_inertia=own.inertia,
        library_inertia=float(library.inertia_),
        agreement=float(adjusted_rand_score(library.labels_, own.labels)),
    )


def benchmarks(frame: pd.DataFrame, sizes: Sequence[int] = (2_000, 20_000)) -> list[Benchmark]:
    results = [benchmark(scale(radius_magnitude(frame)), "Stars")]
    for size in sizes:
        X, _ = make_blobs(size, n_features=2, centers=CLUSTERS, random_state=SEED)
        results.append(benchmark(scale(X), "Synthetic blobs"))
    return results
