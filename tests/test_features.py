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


# ---------------------------------------------------------------- engineered feature

def test_hr_pct_predicted_formula():
    from src.features import ClinicalFeatures
    X = pd.DataFrame({"age": [40, 70], "thalach": [180, 120]})
    out = ClinicalFeatures(add_hr_ratio=True).fit_transform(X)
    # 180 / (220 - 40) = 1.0 ; 120 / (220 - 70) = 0.8
    np.testing.assert_allclose(out["hr_pct_predicted"], [1.0, 0.8])
    assert list(X.columns) == ["age", "thalach"]          # input is not modified


def test_engineered_feature_is_optional(split_data):
    from src.features import ClinicalFeatures
    X_train, *_ = split_data
    off = ClinicalFeatures(add_hr_ratio=False).fit_transform(X_train)
    on = ClinicalFeatures(add_hr_ratio=True).fit_transform(X_train)
    assert list(off.columns) == config.FEATURES
    assert list(on.columns) == config.FEATURES + config.ENGINEERED_FEATURES
    assert on["hr_pct_predicted"].between(0.3, 1.5).all()    # plausible clinical range


def test_engineered_feature_is_stateless_so_it_cannot_leak(split_data):
    """Fitting on different data must not change the output: nothing is learned."""
    from src.features import ClinicalFeatures
    X_train, X_test, *_ = split_data
    a = ClinicalFeatures(add_hr_ratio=True).fit(X_train).transform(X_test)
    b = ClinicalFeatures(add_hr_ratio=True).fit(X_test).transform(X_test)
    pd.testing.assert_frame_equal(a, b)


def test_preprocessor_scales_the_engineered_feature_too(split_data):
    from src.features import ClinicalFeatures
    X_train, *_ = split_data
    X = ClinicalFeatures(add_hr_ratio=True).fit_transform(X_train)
    pre = build_preprocessor().fit(X)
    names = list(pre.get_feature_names_out())
    assert len(names) == 26 and names[6] == "num__hr_pct_predicted"
    Xt = pre.transform(X)[:, :7]
    np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=1e-9)


def test_feature_names_out():
    from src.features import ClinicalFeatures
    assert list(ClinicalFeatures(True).get_feature_names_out()) == \
        config.FEATURES + config.ENGINEERED_FEATURES
    assert list(ClinicalFeatures(False).get_feature_names_out(["a"])) == ["a"]


def test_every_real_model_tunes_the_engineered_feature():
    from src.models import get_candidates
    for name, (_, grid) in get_candidates().items():
        if name != "baseline_majority":
            assert grid["features__add_hr_ratio"] == [False, True], name


def test_model_with_engineered_feature_takes_the_13_raw_values(split_data):
    """The API sends only the 13 UCI attributes; the pipeline adds the feature itself."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    from src.models import make_pipeline
    X_train, X_test, y_train, _ = split_data
    pipe = make_pipeline(LogisticRegression(max_iter=2000))
    pipe.set_params(features__add_hr_ratio=True).fit(X_train, y_train)
    assert pipe.named_steps["model"].coef_.shape[1] == 26
    path = config.ROOT_DIR / "models" / "_tmp_test_model.joblib"
    try:
        joblib.dump(pipe, path)
        reloaded = joblib.load(path)
        np.testing.assert_allclose(reloaded.predict_proba(X_test[config.FEATURES]),
                                   pipe.predict_proba(X_test))
    finally:
        path.unlink(missing_ok=True)
