import pandas as pd
import pytest

from starsort.data import COLORS, TYPE, TYPES, clean_color, load


def test_the_catalogue_has_forty_stars_of_each_type():
    frame = load()
    assert len(frame) == 240
    assert frame[TYPE].value_counts().to_dict() == dict.fromkeys(TYPES, 40)


def test_color_spellings_are_unified():
    assert clean_color("Blue white") == "Blue-White"
    assert clean_color(" blue-White ") == "Blue-White"
    assert clean_color("yellow-white") == "White-Yellow"
    assert clean_color("Pale yellow orange") == "Pale-Yellow-Orange"
    assert set(load()["color"]) <= set(COLORS)


def write(tmp_path, **changes):
    frame = load().drop(columns=[TYPE]).head(3).assign(**changes)
    path = tmp_path / "stars.csv"
    frame.to_csv(path, index=False)
    return path


def test_custom_catalogues_load_without_types(tmp_path):
    frame = load(write(tmp_path))
    assert TYPE not in frame
    assert len(frame) == 3


def test_unknown_colors_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Purple"):
        load(write(tmp_path, color="Purple"))


def test_unknown_spectral_classes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Z"):
        load(write(tmp_path, spectral_class="Z"))


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "stars.csv"
    pd.DataFrame({"temperature": [3000]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="radius"):
        load(path)


def test_sizes_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        load(write(tmp_path, radius=0.0))
