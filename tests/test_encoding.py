import numpy as np
import pytest

from starsort.data import load
from starsort.encoding import ENCODINGS, encoding, log_ordinal, one_hot, ordinal, radius_magnitude


def test_one_hot_adds_a_column_per_category():
    frame = load()
    encoded = one_hot(frame)
    categories = frame["color"].nunique() + frame["spectral_class"].nunique()
    assert encoded.shape == (240, 4 + categories)
    assert (encoded.filter(like="spectral_class_").sum(axis=1) == 1).all()


def test_ordinal_ranks_follow_energy():
    frame = load()
    encoded = ordinal(frame)
    hottest = encoded.loc[frame["spectral_class"] == "O", "spectral_class"]
    coolest = encoded.loc[frame["spectral_class"] == "M", "spectral_class"]
    assert (hottest == 6).all()
    assert (coolest == 0).all()
    assert encoded.loc[frame["color"] == "Blue", "color"].eq(10).all()


def test_log_scales_tame_luminosity():
    encoded = log_ordinal(load())
    assert encoded["log_luminosity"].max() - encoded["log_luminosity"].min() < 11


def test_radius_and_magnitude_are_the_only_features_kept():
    encoded = radius_magnitude(load())
    assert list(encoded.columns) == ["log_radius", "magnitude"]
    assert np.isfinite(encoded.to_numpy()).all()


def test_stefan_boltzmann_makes_temperature_redundant():
    encoded = log_ordinal(load())
    implied = 2 * encoded["log_radius"] + 4 * encoded["log_temperature"]
    assert np.corrcoef(implied, encoded["log_luminosity"])[0, 1] > 0.95


def test_encodings_are_found_by_key():
    assert encoding("Radius-Magnitude").name == "Radius-Magnitude"
    assert len(ENCODINGS) == 4
    with pytest.raises(ValueError, match="one-hot"):
        encoding("pca")
