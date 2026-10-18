"""Inference helpers and the committed packaged model (src/predict.py)."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from src import config
from src.predict import SAMPLE_PATIENT, load_model, predict, to_frame, version_mismatch
from tests.conftest import HEALTHY_PATIENT, SICK_PATIENT

# ------------------------------------------------------------------ input handling


def test_single_record_becomes_one_ordered_float_row():
    df = to_frame(SAMPLE_PATIENT)
    assert df.shape == (1, 13)
    assert list(df.columns) == config.FEATURES
    assert all(dtype == np.float64 for dtype in df.dtypes)


def test_list_of_records():
    assert to_frame([SAMPLE_PATIENT, HEALTHY_PATIENT, SICK_PATIENT]).shape == (3, 13)


def test_missing_feature_is_reported_by_name():
    record = {k: v for k, v in SAMPLE_PATIENT.items() if k != "thalach"}
    with pytest.raises(ValueError, match="thalach"):
        to_frame(record)


def test_extra_fields_are_ignored():
    df = to_frame({**SAMPLE_PATIENT, "patient_name": "x"})
    assert "patient_name" not in df.columns


# ------------------------------------------------------------------ predictions


def test_prediction_output_contract(fitted_lr):
    out = predict(SAMPLE_PATIENT, model=fitted_lr)
    assert len(out) == 1
    res = out[0]
    assert set(res) == {"prediction", "label", "probability_disease", "confidence"}
    assert res["prediction"] in (0, 1)
    assert 0 <= res["probability_disease"] <= 1
    assert res["confidence"] == pytest.approx(max(res["probability_disease"],
                                                  1 - res["probability_disease"]), abs=1e-4)
    assert res["label"] == ("Heart disease" if res["prediction"] else "No heart disease")


def test_threshold_controls_the_decision(fitted_lr):
    records = [SAMPLE_PATIENT, HEALTHY_PATIENT, SICK_PATIENT]
    assert {r["prediction"] for r in predict(records, model=fitted_lr, threshold=0.0)} == {1}
    assert {r["prediction"] for r in predict(records, model=fitted_lr, threshold=1.01)} == {0}


def test_missing_ca_and_thal_are_imputed(fitted_lr):
    out = predict({**SAMPLE_PATIENT, "ca": None, "thal": None}, model=fitted_lr)
    assert 0 <= out[0]["probability_disease"] <= 1


def test_risk_ordering_makes_clinical_sense(fitted_lr):
    healthy, sick = predict([HEALTHY_PATIENT, SICK_PATIENT], model=fitted_lr)
    assert sick["probability_disease"] > healthy["probability_disease"]
    assert sick["prediction"] == 1 and healthy["prediction"] == 0


# ------------------------------------------------------------------ packaged model

# The committed model file only loads with the scikit-learn version that trained it
# (recorded in model_metadata.json). On other versions these tests are skipped; the
# CI training job, which installs the locked versions, always runs them.
needs_training_sklearn = pytest.mark.skipif(
    version_mismatch() is not None, reason=f"version mismatch: {version_mismatch()}")


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model.__wrapped__(str(tmp_path / "none.joblib"))


@needs_training_sklearn
def test_committed_model_matches_its_metadata():
    model = load_model.__wrapped__(str(config.MODEL_PATH))
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    assert type(model.named_steps["model"]).__name__ == meta["model_class"]
    assert meta["features"]["numeric"] + meta["features"]["categorical"] == config.FEATURES


@needs_training_sklearn
def test_committed_model_beats_chance_on_test_split(split_data):
    from sklearn.metrics import roc_auc_score
    _, X_test, _, y_test = split_data
    model = load_model.__wrapped__(str(config.MODEL_PATH))
    assert roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) > 0.85


@needs_training_sklearn
def test_command_line_scores_a_json_file(tmp_path):
    f = tmp_path / "patients.json"
    f.write_text(json.dumps([HEALTHY_PATIENT, SICK_PATIENT]))
    out = subprocess.run([sys.executable, "-m", "src.predict", "--json", str(f)],
                         cwd=config.ROOT_DIR, capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert [r["prediction"] for r in result] == [0, 1]


def test_version_mismatch_is_detected_and_warned(tmp_path, fitted_lr):
    import joblib
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    meta["sklearn_version"] = "0.0.1"
    (tmp_path / config.MODEL_METADATA_PATH.name).write_text(json.dumps(meta))
    joblib.dump(fitted_lr, tmp_path / "heart_disease_model.joblib")
    assert "0.0.1" in version_mismatch(tmp_path / config.MODEL_METADATA_PATH.name)
    with pytest.warns(RuntimeWarning, match="trained with scikit-learn 0.0.1"):
        load_model.__wrapped__(str(tmp_path / "heart_disease_model.joblib"))


def test_no_mismatch_without_metadata(tmp_path):
    assert version_mismatch(tmp_path / "missing.json") is None
