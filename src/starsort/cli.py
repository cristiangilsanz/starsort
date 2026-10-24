from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sklearn.metrics import adjusted_rand_score

from starsort.clustering import ALGORITHMS, BEST_ALGORITHM, CLUSTERS, algorithm, cluster
from starsort.data import TYPE, load
from starsort.encoding import BEST_ENCODING, ENCODINGS, encoding
from starsort.evaluation import benchmarks, compare, match, ordered, profiles
from starsort.plotting import plot_diagram


def table(header: Sequence[str], rows: list[Sequence[str]]) -> str:
    lines = [header, *rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]

    def render(line: Sequence[str]) -> str:
        cells = [line[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(line[1:], widths[1:], strict=True)]
        return "  ".join(cells)

    return "\n".join(render(line) for line in lines)


def number(value: float) -> str:
    if value >= 1_000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.3g}"


def run_compare(args: argparse.Namespace) -> int:
    encodings = [encoding(key) for key in args.encodings] if args.encodings else ENCODINGS
    methods = [algorithm(key) for key in args.algorithms] if args.algorithms else ALGORITHMS
    frame = load(args.data)
    results = compare(frame, encodings, methods, args.clusters)
    rows = [
        (
            result.encoding,
            result.algorithm,
            str(result.clusters),
            str(result.noise),
            f"{result.silhouette:.3f}",
            f"{result.agreement:.3f}",
            result.detail,
        )
        for result in results
    ]
    header = ("Encoding", "Algorithm", "Clusters", "Noise", "Silhouette", "ARI", "Settings")
    print(table(header, rows))
    if TYPE in frame:
        best = max(results, key=lambda result: result.agreement)
        print(f"\nClosest to the catalogued types: {best.encoding} + {best.algorithm}")
    return 0


def run_kmeans(args: argparse.Namespace) -> int:
    rows = [
        (
            result.dataset,
            f"{result.samples:,}",
            f"{result.own_inertia:.2f}",
            f"{result.library_inertia:.2f}",
            f"{result.agreement:.3f}",
            f"{result.own_seconds * 1000:.1f} ms",
            f"{result.library_seconds * 1000:.1f} ms",
        )
        for result in benchmarks(load(args.data))
    ]
    header = ("Dataset", "Points", "Own Inertia", "Library Inertia", "Agreement", "Own", "Library")
    print(table(header, rows))
    return 0


def found(args: argparse.Namespace):
    frame = load(args.data)
    chosen_encoding = encoding(args.encoding)
    chosen_algorithm = algorithm(args.algorithm)
    result = cluster(chosen_encoding.encode(frame), chosen_algorithm, args.clusters)
    return frame, chosen_encoding, chosen_algorithm, result, ordered(frame, result.labels)


def run_cluster(args: argparse.Namespace) -> int:
    frame, chosen_encoding, chosen_algorithm, result, labels = found(args)
    print(f"{chosen_algorithm.name} on the {chosen_encoding.name} encoding ({result.detail})\n")
    summary = profiles(frame, labels)
    header = ["Cluster", "Stars", "Temperature", "Luminosity", "Radius", "Magnitude", "Class"]
    if TYPE in frame:
        header += ["Closest Type", "Purity"]
    rows = []
    for label, row in summary.iterrows():
        line = [
            "Noise" if label == -1 else str(label),
            str(row["stars"]),
            f"{row['temperature']:,.0f} K",
            number(row["luminosity"]),
            number(row["radius"]),
            f"{row['magnitude']:.2f}",
            row["spectral_class"],
        ]
        if TYPE in frame:
            line += [row["type"], f"{row['purity']:.0%}"]
        rows.append(line)
    print(table(header, rows))
    if TYPE in frame:
        agreement = adjusted_rand_score(frame[TYPE], labels)
        print(f"\nAdjusted Rand index against the catalogued types: {agreement:.3f}")
    if args.out:
        frame.assign(cluster=labels).to_csv(args.out, index=False)
        print(f"Wrote the clusters to {args.out}")
    return 0


def run_plot(args: argparse.Namespace) -> int:
    frame, _, chosen_algorithm, _, labels = found(args)
    names: dict[int, str] = {}
    title = f"Clusters found by {chosen_algorithm.name}"
    if TYPE in frame:
        names = {label: name for label, (name, _) in match(labels, frame[TYPE]).items()}
        title += f" · ARI {adjusted_rand_score(frame[TYPE], labels):.2f}"
    figure = plot_diagram(frame, labels, names, title)
    figure.savefig(args.out, dpi=150, facecolor=figure.get_facecolor())
    print(f"Saved the Hertzsprung-Russell diagram to {args.out}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="star-sort",
        description="Group stars into types from their physical measurements.",
    )
    commands = root.add_subparsers(dest="command", required=True, metavar="command")

    compare_command = commands.add_parser(
        "compare", help="compare every encoding and clustering algorithm"
    )
    compare_command.add_argument("--encodings", nargs="+", metavar="NAME")
    compare_command.add_argument("--algorithms", nargs="+", metavar="NAME")
    compare_command.set_defaults(run=run_compare)

    kmeans_command = commands.add_parser(
        "kmeans", help="benchmark the own k-means against scikit-learn"
    )
    kmeans_command.set_defaults(run=run_kmeans)

    cluster_command = commands.add_parser("cluster", help="group the stars and describe each group")
    cluster_command.add_argument("--out", type=Path, help="CSV to write every star's cluster to")
    cluster_command.set_defaults(run=run_cluster)

    plot = commands.add_parser("plot", help="draw the groups on a Hertzsprung-Russell diagram")
    plot.add_argument("--out", type=Path, default=Path("diagram.png"))
    plot.set_defaults(run=run_plot)

    for command in (cluster_command, plot):
        command.add_argument("--encoding", default=BEST_ENCODING)
        command.add_argument("--algorithm", default=BEST_ALGORITHM)
    for command in (compare_command, cluster_command, plot):
        command.add_argument("--clusters", type=int, default=CLUSTERS)
    for command in (compare_command, kmeans_command, cluster_command, plot):
        command.add_argument(
            "--data", type=Path, help="CSV of stars, the bundled catalogue by default"
        )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"star-sort: {error}", file=sys.stderr)
        return 1


def entry() -> None:
    sys.exit(main())
