import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from starsort.kmeans import kmeans, plus_plus, squared_distances


@pytest.fixture(scope="module")
def blobs():
    return make_blobs(600, n_features=2, centers=4, cluster_std=0.6, random_state=0)


def test_separated_blobs_are_recovered(blobs):
    X, truth = blobs
    result = kmeans(X, 4)
    assert adjusted_rand_score(truth, result.labels) == 1.0


def test_inertia_matches_scikit_learn(blobs):
    X, _ = blobs
    own = kmeans(X, 4)
    library = KMeans(4, n_init=10, random_state=0).fit(X)
    assert own.inertia == pytest.approx(library.inertia_, rel=1e-6)


def test_inertia_is_the_sum_of_squared_distances(blobs):
    X, _ = blobs
    result = kmeans(X, 4)
    assert result.inertia == pytest.approx(((X - result.centroids[result.labels]) ** 2).sum())


def test_centroids_are_the_means_of_their_clusters(blobs):
    X, _ = blobs
    result = kmeans(X, 4)
    for cluster, centroid in enumerate(result.centroids):
        assert np.allclose(X[result.labels == cluster].mean(axis=0), centroid)


def test_the_same_seed_gives_the_same_clusters(blobs):
    X, _ = blobs
    assert np.array_equal(kmeans(X, 4, seed=7).labels, kmeans(X, 4, seed=7).labels)


def test_distances_match_the_definition():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    centroids = np.array([[0.0, 0.0]])
    assert squared_distances(X, centroids).ravel().tolist() == [0.0, 25.0]


def test_plus_plus_picks_distinct_points():
    X = np.array([[0.0], [0.0], [10.0], [20.0]])
    seeds = plus_plus(X, 3, np.random.default_rng(0))
    assert sorted(seeds.ravel().tolist()) == [0.0, 10.0, 20.0]


def test_duplicate_points_never_leave_a_cluster_empty():
    X = np.array([[1.0, 1.0]] * 5 + [[9.0, 9.0]])
    result = kmeans(X, 3)
    assert set(result.labels) <= {0, 1, 2}
    assert result.inertia == pytest.approx(0.0)


@pytest.mark.parametrize("k", [0, 7])
def test_impossible_cluster_counts_are_rejected(k):
    with pytest.raises(ValueError, match="between"):
        kmeans(np.zeros((6, 2)), k)
