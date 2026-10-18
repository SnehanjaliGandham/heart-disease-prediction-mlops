"""Helper scripts: data download/verification and the environment check."""
from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import download_data as dl
from src import config


@pytest.fixture
def data_copy(tmp_path):
    """A private copy of the raw CSV (and its manifest) to tamper with."""
    shutil.copy(config.RAW_DATA_PATH, tmp_path / dl.RAW_NAME)
    shutil.copy(config.DATA_DIR / dl.MANIFEST_NAME, tmp_path / dl.MANIFEST_NAME)
    return tmp_path


def test_committed_manifest_matches_committed_data():
    manifest = json.loads((config.DATA_DIR / dl.MANIFEST_NAME).read_text())
    assert manifest["sha256"] == dl.sha256(config.RAW_DATA_PATH)
    assert (manifest["rows"], manifest["columns"]) == dl.EXPECTED_SHAPE


def test_download_reuses_and_verifies_existing_file(data_copy, capsys):
    path = dl.download(data_dir=data_copy)       # no network needed
    assert path == data_copy / dl.RAW_NAME
    assert "Verified" in capsys.readouterr().out


def test_windows_line_endings_still_verify(data_copy):
    """A Git checkout on Windows turns LF into CRLF - the data has not changed."""
    raw = data_copy / dl.RAW_NAME
    raw.write_bytes(raw.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert b"\r\n" in raw.read_bytes()
    assert dl.verify(raw)


def test_tampered_file_is_detected(data_copy):
    raw = data_copy / dl.RAW_NAME
    raw.write_text(raw.read_text().replace("63,1,1,145", "64,1,1,145", 1))
    with pytest.raises(ValueError, match="changed"):
        dl.verify(raw)


def test_manifest_is_created_when_missing(data_copy):
    (data_copy / dl.MANIFEST_NAME).unlink()
    assert dl.verify(data_copy / dl.RAW_NAME)
    manifest = json.loads((data_copy / dl.MANIFEST_NAME).read_text())
    assert manifest["rows"] == 303 and manifest["uci_id"] == dl.UCI_ID


def test_verify_command_line():
    out = subprocess.run([sys.executable, "scripts/download_data.py", "--verify"],
                         cwd=config.ROOT_DIR, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Verified" in out.stdout


def test_fresh_download_writes_data_and_manifest(tmp_path, monkeypatch):
    """The download path, with the UCI client replaced by a local stand-in (no network)."""
    raw = pd.read_csv(config.RAW_DATA_PATH)
    fake = SimpleNamespace(
        data=SimpleNamespace(features=raw.drop(columns="num"), targets=raw[["num"]]),
        variables=pd.read_csv(config.DATA_DIR / "heart_disease_variables.csv"),
        metadata={"name": "Heart Disease"},
    )
    monkeypatch.setitem(sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda id: fake))
    path = dl.download(data_dir=tmp_path, force=True)
    assert pd.read_csv(path).shape == dl.EXPECTED_SHAPE
    manifest = json.loads((tmp_path / dl.MANIFEST_NAME).read_text())
    assert manifest["sha256"] == dl.sha256(path)
    assert (tmp_path / "heart_disease_variables.csv").exists()


def test_download_rejects_unexpected_shape(tmp_path, monkeypatch):
    raw = pd.read_csv(config.RAW_DATA_PATH).head(10)
    fake = SimpleNamespace(data=SimpleNamespace(features=raw.drop(columns="num"),
                                                targets=raw[["num"]]),
                           variables=pd.DataFrame(), metadata={})
    monkeypatch.setitem(sys.modules, "ucimlrepo", SimpleNamespace(fetch_ucirepo=lambda id: fake))
    with pytest.raises(ValueError, match="Unexpected shape"):
        dl.download(data_dir=tmp_path, force=True)


def test_environment_check_reports_every_section(capsys):
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(config.ROOT_DIR / "scripts" / "check_env.py"), run_name="__main__")
    out = capsys.readouterr().out
    for section in ["== Python", "== Network", "== Python packages", "== Project files",
                    "== Tools for later steps", "Summary:"]:
        assert section in out
    assert "[PASS] scikit-learn" in out
    assert exit_info.value.code in (0, 1)    # 1 only when e.g. the network is blocked


# ------------------------------------------------------------------ model quality gate


def _gate_inputs(tmp_path, **overrides):
    from scripts import quality_gate  # noqa: F401  (import check)
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    meta["test_metrics"].update(overrides)
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    return meta_path, config.REPORTS_DIR / "model_comparison.csv"


def test_quality_gate_passes_for_the_committed_model(tmp_path, monkeypatch):
    from scripts import quality_gate
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert quality_gate.main(*_gate_inputs(tmp_path)) == 0
    assert "| test_recall |" in summary.read_text()


def test_quality_gate_blocks_a_weak_model(tmp_path, capsys, monkeypatch):
    from scripts import quality_gate
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)   # keep the real CI summary clean
    assert quality_gate.main(*_gate_inputs(tmp_path, test_recall=0.55)) == 1
    assert "QUALITY GATE FAILED: test_recall" in capsys.readouterr().out
