"""Preprocessing ColumnTransformer (src/features.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features import build_preprocessor


def test_output_has_25_named_features(split_data):
    X_train, *_ = split_data
    pre = build_preprocessor().fit(X_train)
    names = list(pre.get_feature_names_out())
    assert len(names) == 25
    assert names[:6] == [f"num__{c}" for c in config.NUMERIC_FEATURES]
    assert "cat__thal_7.0" in names and "cat__cp_4.0" in names


def test_no_missing_values_after_transform(split_data):
    X_train, X_test, *_ = split_data
    pre = build_preprocessor().fit(X_train)
    gappy = X_test.copy()
    gappy.loc[gappy.index[:5], ["ca", "thal", "chol"]] = np.nan
    assert not np.isnan(pre.transform(gappy)).any()


def test_numeric_features_are_standardised(split_data):
    X_train, *_ = split_data
    Xt = build_preprocessor().fit_transform(X_train)[:, :6]
    np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(Xt.std(axis=0), 1, atol=1e-9)


def test_one_hot_block_has_one_active_level_per_feature(split_data):
    X_train, *_ = split_data
    onehot = build_preprocessor().fit_transform(X_train)[:, 6:]
    np.testing.assert_array_equal(onehot.sum(axis=1), len(config.CATEGORICAL_FEATURES))


def test_unknown_category_does_not_crash(split_data):
    """An unseen code at prediction time must not break the API."""
    X_train, X_test, *_ = split_data
    pre = build_preprocessor().fit(X_train)
    odd = X_test.iloc[[0]].copy()
    odd["restecg"] = 9.0                      # code never seen in training
    row = pre.transform(odd)[0]
    names = list(pre.get_feature_names_out())
    restecg_cols = [i for i, n in enumerate(names) if n.startswith("cat__restecg_")]
    assert row[restecg_cols].sum() == 0       # encoded as "none of the known levels"


def test_imputers_learn_from_training_data_only(split_data):
    """Leakage guard: imputation statistics come from the training split, not all data."""
    X_train, X_test, *_ = split_data
    pre = build_preprocessor().fit(X_train)
    num_imputer = pre.named_transformers_["num"].named_steps["impute"]
    ca_idx = config.NUMERIC_FEATURES.index("ca")
    assert num_imputer.statistics_[ca_idx] == X_train["ca"].median()

    scaler = pre.named_transformers_["num"].named_steps["scale"]
    chol_idx = config.NUMERIC_FEATURES.index("chol")
    assert np.isclose(scaler.mean_[chol_idx], X_train["chol"].mean())
    full = pd.concat([X_train, X_test])
    assert not np.isclose(scaler.mean_[chol_idx], full["chol"].mean())


def test_each_call_returns_a_fresh_unfitted_transformer():
    a, b = build_preprocessor(), build_preprocessor()
    assert a is not b
    assert not hasattr(a, "transformers_")
