from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullFormatter

from starsort.clustering import NOISE
from starsort.data import TYPE, TYPES

BACKGROUND = "#0d1117"
GRID = "#2a3146"
TEXT = "#c9d1e3"
NOISE_COLOR = "#59607a"
PALETTE = ("#c8794a", "#ff6b6b", "#e6edf3", "#ffd166", "#4fb3ff", "#b083f0", "#56d4a8", "#f78fb3")
TEMPERATURES = (3_000, 5_000, 10_000, 20_000, 40_000)


def sizes(radius: pd.Series) -> np.ndarray:
    return 14 + 14 * (np.log10(radius.to_numpy()) + 2.2)


def style(axes: Axes, title: str) -> None:
    axes.set_facecolor(BACKGROUND)
    axes.set_xscale("log")
    axes.xaxis.set_major_locator(FixedLocator(TEMPERATURES))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_xticklabels([f"{value:,}" for value in TEMPERATURES])
    axes.invert_xaxis()
    axes.invert_yaxis()
    axes.grid(color=GRID, linewidth=0.8)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    axes.tick_params(colors=TEXT, labelsize=11)
    axes.set_xlabel("Surface temperature (K)", color=TEXT, fontsize=12)
    axes.set_ylabel("Absolute magnitude", color=TEXT, fontsize=12)
    axes.set_title(title, color=TEXT, fontsize=14, loc="left", pad=10)


def draw(axes: Axes, frame: pd.DataFrame, groups: list[tuple[str, np.ndarray, str]]) -> None:
    for name, mask, color in groups:
        part = frame[mask]
        axes.scatter(
            part["temperature"],
            part["magnitude"],
            s=sizes(part["radius"]),
            color=color,
            alpha=0.85,
            edgecolor=BACKGROUND,
            linewidth=0.6,
            label=f"{name} ({mask.sum()})",
        )
    legend = axes.legend(
        facecolor=BACKGROUND, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper left"
    )
    for handle in legend.legend_handles:
        handle.set_sizes([40])


def cluster_groups(
    frame: pd.DataFrame, labels: np.ndarray, names: dict[int, str]
) -> list[tuple[str, np.ndarray, str]]:
    found = [label for label in np.unique(labels) if label != NOISE]
    order = sorted(found, key=lambda label: -frame["magnitude"][labels == label].median())
    groups = [
        (names.get(label, f"Cluster {label}"), labels == label, PALETTE[index % len(PALETTE)])
        for index, label in enumerate(order)
    ]
    if (labels == NOISE).any():
        groups.append(("Noise", labels == NOISE, NOISE_COLOR))
    return groups


def plot_diagram(
    frame: pd.DataFrame, labels: np.ndarray, names: dict[int, str], title: str
) -> Figure:
    labels = np.asarray(labels)
    panels = 2 if TYPE in frame else 1
    figure, axes = plt.subplots(
        1, panels, figsize=(7.2 * panels, 6.4), sharey=True, facecolor=BACKGROUND, squeeze=False
    )
    if panels == 2:
        catalogued = [
            (name, (frame[TYPE] == name).to_numpy(), PALETTE[index])
            for index, name in enumerate(TYPES)
        ]
        style(axes[0, 0], "Catalogued star types")
        draw(axes[0, 0], frame, catalogued)
    style(axes[0, -1], title)
    draw(axes[0, -1], frame, cluster_groups(frame, labels, names))
    figure.tight_layout()
    return figure
