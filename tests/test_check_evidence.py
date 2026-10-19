"""scripts/check_evidence.py: screenshot register parsing and the missing-file report."""
from __future__ import annotations

from scripts import check_evidence as ce

REGISTER = """# register
| File | Task | What it shows | Where |
|---|---|---|---|
| `a.png` | 1 | first | PG |
| `b.png` | 7 | second | Laptop |
"""


def test_parse_reads_every_row(tmp_path):
    (tmp_path / "README.md").write_text(REGISTER)
    rows = ce.parse(tmp_path / "README.md")
    assert [(r["file"], r["task"], r["where"]) for r in rows] == \
        [("a.png", "1", "PG"), ("b.png", "7", "Laptop")]


def test_missing_and_unlisted_files_are_reported(tmp_path, capsys):
    (tmp_path / "README.md").write_text(REGISTER)
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "extra.png").write_bytes(b"x")
    assert ce.main([], folder=tmp_path) == 1
    out = capsys.readouterr().out
    assert "b.png" in out and "extra.png" in out and "1/2 screenshots present" in out


def test_complete_register_passes(tmp_path, capsys):
    (tmp_path / "README.md").write_text(REGISTER)
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"x")
    assert ce.main([], folder=tmp_path) == 0
    assert "[OK  ] Task 1" in capsys.readouterr().out


def test_the_real_register_is_well_formed():
    rows = ce.parse(ce.ROOT / "screenshots" / "README.md")
    assert len(rows) > 60
    assert {r["task"] for r in rows} == set(ce.TASKS)
    assert len({r["file"] for r in rows}) == len(rows), "duplicate file names"
