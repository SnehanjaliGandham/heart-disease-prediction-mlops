"""List which screenshots in screenshots/README.md exist and which are still missing.

    python scripts/check_evidence.py          # summary per task + missing files
Exits with code 1 while anything listed in the register is missing.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "1": "Data acquisition & EDA", "2": "Feature engineering & models",
    "3": "Experiment tracking (MLflow)", "4": "Packaging & reproducibility",
    "5": "CI/CD & tests", "6": "Docker & API", "7": "Kubernetes deployment",
    "8": "Monitoring & logging", "9": "Documentation",
}
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*$")


def parse(register: Path) -> list[dict]:
    rows = []
    for line in register.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m:
            rows.append({"file": m.group(1), "task": m.group(2), "what": m.group(3),
                         "where": m.group(4)})
    return rows


def main(argv=None, folder: Path = ROOT / "screenshots") -> int:
    rows = parse(folder / "README.md")
    by_task = defaultdict(list)
    for row in rows:
        row["present"] = (folder / row["file"]).exists()
        by_task[row["task"]].append(row)
    missing = [r for r in rows if not r["present"]]
    for task in sorted(by_task):
        items = by_task[task]
        done = sum(r["present"] for r in items)
        mark = "OK  " if done == len(items) else "TODO"
        print(f"[{mark}] Task {task} {TASKS.get(task, ''):32s} {done}/{len(items)}")
    unlisted = sorted({p.name for p in folder.glob("*.png")} - {r["file"] for r in rows})
    if missing:
        print("\nStill missing:")
        for r in missing:
            print(f"  - {r['file']:40s} ({r['where']}) {r['what']}")
    if unlisted:
        print("\nIn the folder but not in the register (rename, add or delete):")
        for name in unlisted:
            print(f"  - {name}")
    print(f"\n{len(rows) - len(missing)}/{len(rows)} screenshots present.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
