from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from starsort.data import COLORS, NUMERIC, SPECTRAL_CLASSES


def ranks(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": frame["color"].map({name: rank for rank, name in enumerate(COLORS)}),
            "spectral_class": frame["spectral_class"].map(
                {name: rank for rank, name in enumerate(SPECTRAL_CLASSES)}
            ),
        },
        index=frame.index,
    )


def one_hot(frame: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(frame[["color", "spectral_class"]], dtype=float)
    return pd.concat([frame[list(NUMERIC)], dummies], axis=1)


def ordinal(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame[list(NUMERIC)], ranks(frame)], axis=1)


def logarithmic(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_temperature": np.log10(frame["temperature"]),
            "log_luminosity": np.log10(frame["luminosity"]),
            "log_radius": np.log10(frame["radius"]),
            "magnitude": frame["magnitude"],
        },
        index=frame.index,
    )


def log_ordinal(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([logarithmic(frame), ranks(frame)], axis=1)


def radius_magnitude(frame: pd.DataFrame) -> pd.DataFrame:
    return logarithmic(frame)[["log_radius", "magnitude"]]


@dataclass(frozen=True)
class Encoding:
    key: str
    name: str
    encode: Callable[[pd.DataFrame], pd.DataFrame]


ENCODINGS = (
    Encoding("one-hot", "One-Hot", one_hot),
    Encoding("ordinal", "Ordinal", ordinal),
    Encoding("log-ordinal", "Log Ordinal", log_ordinal),
    Encoding("radius-magnitude", "Radius-Magnitude", radius_magnitude),
)
BEST_ENCODING = "radius-magnitude"


def encoding(key: str) -> Encoding:
    for item in ENCODINGS:
        if item.key == key.lower():
            return item
    choices = ", ".join(item.key for item in ENCODINGS)
    raise ValueError(f"unknown encoding {key!r}, choose one of: {choices}")
