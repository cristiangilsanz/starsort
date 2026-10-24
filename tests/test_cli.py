import pandas as pd

from starsort.cli import main
from starsort.data import TYPE, load


def test_cluster_describes_the_six_types(capsys):
    assert main(["cluster"]) == 0
    out = capsys.readouterr().out
    assert "Hypergiant" in out
    assert "Adjusted Rand index against the catalogued types: 0.961" in out


def test_cluster_writes_every_star(tmp_path):
    out = tmp_path / "clusters.csv"
    assert main(["cluster", "--algorithm", "kmeans", "--out", str(out)]) == 0
    written = pd.read_csv(out)
    assert len(written) == 240
    assert sorted(written["cluster"].unique()) == [1, 2, 3, 4, 5, 6]


def test_compare_can_be_narrowed(capsys):
    assert main(["compare", "--encodings", "ordinal", "--algorithms", "kmeans", "dbscan"]) == 0
    out = capsys.readouterr().out
    assert out.count("Ordinal") == 3


def test_kmeans_benchmark_runs(capsys):
    assert main(["kmeans"]) == 0
    assert "Synthetic blobs" in capsys.readouterr().out


def test_plot_draws_both_panels(tmp_path):
    chart = tmp_path / "diagram.png"
    assert main(["plot", "--out", str(chart)]) == 0
    assert chart.stat().st_size > 50_000


def test_unlabelled_catalogues_are_clustered(tmp_path, capsys):
    data = tmp_path / "stars.csv"
    load().drop(columns=[TYPE]).to_csv(data, index=False)
    assert main(["cluster", "--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "Closest Type" not in out
    assert main(["plot", "--data", str(data), "--out", str(tmp_path / "chart.png")]) == 0


def test_unknown_algorithms_fail_cleanly(capsys):
    assert main(["cluster", "--algorithm", "magic"]) == 1
    assert "unknown algorithm" in capsys.readouterr().err


def test_a_single_cluster_is_refused(capsys):
    assert main(["cluster", "--clusters", "1"]) == 1
    assert "two clusters" in capsys.readouterr().err
