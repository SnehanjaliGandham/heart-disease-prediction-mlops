"""Feature engineering: one ColumnTransformer used by every model.

Putting preprocessing inside the sklearn Pipeline means:
* cross-validation refits imputers/scaler/encoder on each training fold (no leakage);
* the saved model file contains the preprocessing, so the API sends raw values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Add engineered clinical features to the 13 raw attributes.

    hr_pct_predicted = thalach / (220 - age): the peak heart rate reached in the
    exercise test as a fraction of the age-predicted maximum heart rate (Fox et al.,
    1971). A low value (chronotropic incompetence) is a known marker of heart disease,
    and it combines two raw features in a way a linear model cannot learn by itself.

    Whether the feature helps is NOT assumed: `add_hr_ratio` is a hyper-parameter, and
    the grid search keeps it only where it improves cross-validated ROC-AUC.
    The transformer is stateless (nothing is learned from data), so it cannot leak.
    """

    def __init__(self, add_hr_ratio: bool = False):
        self.add_hr_ratio = add_hr_ratio

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.add_hr_ratio:
            max_hr = (220 - X["age"].astype(float)).clip(lower=1)
            X["hr_pct_predicted"] = np.round(X["thalach"].astype(float) / max_hr, 4)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None else config.FEATURES)
        return np.asarray(names + (config.ENGINEERED_FEATURES if self.add_hr_ratio else []),
                          dtype=object)


# Numeric columns = the raw numeric features plus any engineered feature present. A
# selector (not a fixed list) lets the same preprocessor work with and without it.
NUMERIC_SELECTOR = make_column_selector(
    pattern="^(?:" + "|".join(config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES) + ")$")


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_SELECTOR),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=True,
    )
