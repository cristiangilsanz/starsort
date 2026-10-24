from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import pandas as pd

NUMERIC = ("temperature", "luminosity", "radius", "magnitude")
COLUMNS = (*NUMERIC, "color", "spectral_class")
TYPE = "type"
TYPES = ("Brown Dwarf", "Red Dwarf", "White Dwarf", "Main Sequence", "Supergiant", "Hypergiant")
COLORS = (
    "Red",
    "Orange-Red",
    "Orange",
    "Pale-Yellow-Orange",
    "Yellowish",
    "Yellowish-White",
    "White-Yellow",
    "White",
    "Whitish",
    "Blue-White",
    "Blue",
)
SPECTRAL_CLASSES = ("M", "K", "G", "F", "A", "B", "O")
ALIASES = {"Yellow-White": "White-Yellow"}


def bundled() -> Path:
    return Path(str(files("starsort") / "data" / "stars.csv"))


def clean_color(name: str) -> str:
    words = re.split(r"[\s-]+", name.strip())
    canonical = "-".join(word.capitalize() for word in words if word)
    return ALIASES.get(canonical, canonical)


def load(path: Path | str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or bundled())
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    frame["color"] = frame["color"].map(clean_color)
    frame["spectral_class"] = frame["spectral_class"].str.strip().str.upper()
    unknown_colors = sorted(set(frame["color"]) - set(COLORS))
    if unknown_colors:
        raise ValueError(f"unknown colors: {', '.join(unknown_colors)}")
    unknown_classes = sorted(set(frame["spectral_class"]) - set(SPECTRAL_CLASSES))
    if unknown_classes:
        raise ValueError(f"unknown spectral classes: {', '.join(unknown_classes)}")
    if (frame[["temperature", "luminosity", "radius"]] <= 0).any().any():
        raise ValueError("temperature, luminosity and radius must be positive")
    return frame
