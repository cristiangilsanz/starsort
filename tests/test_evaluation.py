import numpy as np
import pandas as pd
import pytest

from starsort.clustering import algorithm, cluster
from starsort.data import TYPE, load
from starsort.encoding import encoding, radius_magnitude
from starsort.evaluation import benchmarks, compare, match, ordered, profiles


@pytest.fixture(scope="module")
def results():
    return {(result.encoding, result.algorithm): result for result in compare(load())}


def test_every_encoding_meets_every_algorithm(results):
    assert len(results) == 16


def test_the_radius_magnitude_mixture_recovers_the_star_types(results):
    best = results[("Radius-Magnitude", "Gaussian Mixture")]
    assert best.agreement == pytest.approx(0.961, abs=0.01)
    assert best == max(results.values(), key=lambda result: result.agreement)


def test_the_original_encodings_stay_far_behind(results):
    for name in ("One-Hot", "Ordinal"):
        assert results[(name, "K-Means")].agreement < 0.55


def test_dropping_redundant_features_helps_every_algorithm(results):
    for method in ("K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"):
        assert (
            results[("Radius-Magnitude", method)].agreement > results[("Ordinal", method)].agreement
        )


def test_clusters_are_numbered_from_faintest_to_brightest():
    frame = load()
    labels = ordered(frame, cluster(radius_magnitude(frame), algorithm("mixture")).labels)
    medians = [frame["magnitude"][labels == label].median() for label in range(1, 7)]
    assert medians == sorted(medians, reverse=True)


def test_matching_names_each_cluster_by_its_majority():
    labels = np.array([0, 0, 0, 1, 1, -1])
    types = pd.Series(["a", "a", "b", "c", "c", "a"])
    assert match(labels, types) == {0: ("a", pytest.approx(2 / 3)), 1: ("c", 1.0)}


def test_profiles_describe_every_cluster():
    frame = load()
    labels = ordered(
        frame, cluster(encoding("radius-magnitude").encode(frame), algorithm("mixture")).labels
    )
    summary = profiles(frame, labels)
    assert summary["stars"].sum() == 240
    assert list(summary["type"]) == [
        "Brown Dwarf",
        "Red Dwarf",
        "White Dwarf",
        "Main Sequence",
        "Supergiant",
        "Hypergiant",
    ]
    assert summary.loc[6, "radius"] > 1000


def test_profiles_work_without_catalogued_types():
    frame = load().drop(columns=[TYPE])
    summary = profiles(frame, np.repeat([1, 2], 120))
    assert "type" not in summary


def test_the_own_kmeans_agrees_with_scikit_learn():
    for result in benchmarks(load(), sizes=(2_000,)):
        assert result.agreement > 0.98
        assert result.own_inertia == pytest.approx(result.library_inertia, rel=0.01)
