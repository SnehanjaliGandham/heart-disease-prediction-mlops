"""Data loading, validation, cleaning and splitting (src/data.py)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (
    build_clean_dataset,
    clean,
    file_sha256,
    load_clean,
    load_raw,
    split,
    validate_schema,
)

# ------------------------------------------------------------------ loading


def test_raw_file_has_expected_shape(raw_df):
    assert raw_df.shape == (303, 14)
    assert list(raw_df.columns[-1:]) == [config.RAW_TARGET]


def test_load_raw_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw(tmp_path / "nope.csv")


# ------------------------------------------------------------------ validation


def test_real_data_passes_validation(raw_df):
    validate_schema(raw_df, config.RAW_TARGET)   # must not raise


def test_validation_allows_missing_values(raw_df):
    assert raw_df[["ca", "thal"]].isna().sum().sum() == 6
    validate_schema(raw_df, config.RAW_TARGET)


def test_missing_column_is_rejected(raw_df):
    with pytest.raises(ValueError, match="Missing columns.*chol"):
        validate_schema(raw_df.drop(columns=["chol"]), config.RAW_TARGET)


@pytest.mark.parametrize("column,bad_value", [("cp", 5), ("thal", 4), ("sex", 2), ("slope", 0)])
def test_unknown_category_code_is_rejected(raw_df, column, bad_value):
    df = raw_df.copy()
    df.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=f"Unexpected codes in '{column}'"):
        validate_schema(df, config.RAW_TARGET)


@pytest.mark.parametrize("column,bad_value", [("age", 200), ("chol", 5), ("oldpeak", -1.0)])
def test_out_of_range_value_is_rejected(raw_df, column, bad_value):
    df = raw_df.copy()
    df.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=f"'{column}' outside"):
        validate_schema(df, config.RAW_TARGET)


# ------------------------------------------------------------------ cleaning


@pytest.mark.parametrize("num,expected", [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1)])
def test_target_is_binarised(synthetic_raw, num, expected):
    df = synthetic_raw.copy()
    df["num"] = num
    assert set(clean(df)[config.TARGET]) == {expected}


def test_exact_duplicates_are_removed(synthetic_raw):
    assert synthetic_raw.duplicated().sum() == 1
    out = clean(synthetic_raw)
    assert len(out) == len(synthetic_raw) - 1
    assert not out.duplicated().any()


def test_missing_values_are_kept_for_the_pipeline(synthetic_raw):
    out = clean(synthetic_raw)
    assert out["ca"].isna().sum() == 2 and out["thal"].isna().sum() == 1


def test_clean_drops_raw_target_and_keeps_column_order(raw_df):
    out = clean(raw_df)
    assert config.RAW_TARGET not in out.columns
    assert list(out.columns) == [c for c in raw_df.columns if c != config.RAW_TARGET] + [
        config.TARGET]


def test_real_class_balance(clean_df):
    counts = clean_df[config.TARGET].value_counts().to_dict()
    assert counts == {0: 164, 1: 139}


def test_committed_clean_csv_matches_cleaning_code(raw_df):
    """Regression guard: the committed clean file is exactly what the code produces."""
    committed = pd.read_csv(config.CLEAN_DATA_PATH)
    pd.testing.assert_frame_equal(committed, clean(raw_df))


def test_build_clean_dataset_writes_file(tmp_path):
    out_path = tmp_path / "clean.csv"
    frame = build_clean_dataset(clean_path=out_path)
    assert out_path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(out_path), frame)


def test_load_clean_creates_file_when_missing(tmp_path):
    path = tmp_path / "sub" / "clean.csv"
    df = load_clean(path)
    assert path.exists() and df.shape == (303, 14)


# ------------------------------------------------------------------ splitting


def test_split_sizes_and_stratification(clean_df):
    X_train, X_test, y_train, y_test = split(clean_df)
    assert (len(X_train), len(X_test)) == (242, 61)
    assert abs(y_train.mean() - y_test.mean()) < 0.01   # same disease share


def test_split_is_reproducible_and_disjoint(clean_df):
    a = split(clean_df)
    b = split(clean_df)
    pd.testing.assert_index_equal(a[0].index, b[0].index)
    assert set(a[0].index).isdisjoint(a[1].index)


def test_split_uses_float_features_only(clean_df):
    X_train, *_ = split(clean_df)
    assert list(X_train.columns) == config.FEATURES
    assert all(dtype == np.float64 for dtype in X_train.dtypes)


def test_file_sha256(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"heart")
    assert file_sha256(f) == hashlib.sha256(b"heart").hexdigest()


def test_file_sha256_ignores_windows_line_endings(tmp_path):
    unix, windows = tmp_path / "unix.csv", tmp_path / "windows.csv"
    unix.write_bytes(b"a,b\n1,2\n")
    windows.write_bytes(b"a,b\r\n1,2\r\n")
    assert file_sha256(unix) == file_sha256(windows)
