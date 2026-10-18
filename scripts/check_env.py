"""Check whether this machine (e.g. the BITS Prayogshala lab) can run the project.

Uses only the Python standard library, so it runs BEFORE anything is installed:
    python3 scripts/check_env.py

Prints a PASS / WARN / FAIL line per check and what to do about each problem.
"""
import importlib.metadata as md
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []


def report(status, check, detail="", fix=""):
    results.append(status)
    print(f"[{status:4}] {check:32} {detail}")
    if fix and status != "PASS":
        print(f"{'':7}-> {fix}")


def can_reach(host, port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def tool_version(cmd):
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f"found but failed to run ({exc.__class__.__name__})"


print(f"Project: {ROOT}\n")
print("== Python")
v = sys.version_info
ver = f"{v.major}.{v.minor}.{v.micro}"
if v >= (3, 12):
    report("PASS", "Python version",
           ver + "  (requirements.txt and requirements-lock.txt both work)")
elif v >= (3, 9):
    report("PASS", "Python version", ver,
           "use requirements.txt (requirements-lock.txt needs Python >= 3.12)")
else:
    report("FAIL", "Python version", ver, "need Python >= 3.9; try `python3.11` or `python3.12`")
print(f"{'':7}OS: {platform.platform()} | CPUs: {os.cpu_count()}")

in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
report("PASS" if in_venv else "WARN", "Virtual environment",
       "active" if in_venv else "not active",
       "python3 -m venv .venv && source .venv/bin/activate")

pip_ok = subprocess.run([sys.executable, "-m", "pip", "--version"],
                        capture_output=True).returncode == 0
report("PASS" if pip_ok else "FAIL", "pip", "available" if pip_ok else "missing",
       "python3 -m ensurepip --upgrade   (or: sudo apt install python3-pip python3-venv)")

print("\n== Network")
report("PASS" if can_reach("pypi.org") else "FAIL", "PyPI (pip install)",
       "reachable" if can_reach("pypi.org") else "NOT reachable",
       "packages cannot be downloaded; ask lab support for a proxy / pip mirror")
uci = can_reach("archive.ics.uci.edu")
report("PASS" if uci else "WARN", "UCI repository (dataset)",
       "reachable" if uci else "NOT reachable",
       "fine: data/heart_disease_raw.csv is committed, the download step is skipped")
gh = can_reach("github.com")
report("PASS" if gh else "WARN", "GitHub (push / CI)", "reachable" if gh else "NOT reachable",
       "needed to push the repo and see GitHub Actions runs")

print("\n== Python packages (after `pip install -r requirements.txt`)")
needed = {"numpy": "1.26", "pandas": "2.1", "scikit-learn": "1.6", "joblib": "1.3",
          "skops": "0.10", "mlflow": "3.1", "matplotlib": "3.8", "seaborn": "0.13",
          "ucimlrepo": "0.0.7", "pytest": "8", "jupyter": "0"}


def vtuple(s):
    return tuple(int(p) for p in s.split(".")[:3] if p.isdigit())


for pkg, minimum in needed.items():
    try:
        have = md.version(pkg)
        ok = vtuple(have) >= vtuple(minimum)
        report("PASS" if ok else "FAIL", pkg, have,
               f"needs >= {minimum}: pip install -U '{pkg}>={minimum}'")
    except md.PackageNotFoundError:
        report("WARN", pkg, "not installed", "pip install -r requirements.txt")

print("\n== Project files")
for rel in ["data/heart_disease_raw.csv", "src/train.py", "requirements.txt"]:
    p = ROOT / rel
    report("PASS" if p.exists() else "FAIL", rel, "found" if p.exists() else "MISSING",
           "copy the complete project folder (or git clone the repo)")

print("\n== Tools for later steps (CI/CD, Docker, Kubernetes)")
for name, cmd, why in [
    ("git", ["git", "--version"], "version control / pushing to GitHub"),
    ("docker", ["docker", "--version"], "Step 6: build and run the API container"),
    ("kubectl", ["kubectl", "version", "--client"], "Step 7: deploy to Kubernetes"),
    ("minikube", ["minikube", "version"], "Step 7: local Kubernetes cluster"),
    ("kind", ["kind", "version"], "Step 7: alternative local cluster"),
]:
    out = tool_version(cmd)
    report("PASS" if out else "WARN", name, out or "not found", f"needed for {why}")

if shutil.which("docker"):
    daemon = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    report("PASS" if daemon else "WARN", "docker daemon", "running" if daemon else "not usable",
           "try `sudo docker info`, or ask whether your user can be added to the docker group")

free_gb = shutil.disk_usage(ROOT).free / 1e9
report("PASS" if free_gb > 3 else "WARN", "Free disk space", f"{free_gb:.1f} GB",
       "MLflow + packages need ~1.5 GB, Docker images ~2 GB more")

print("\nSummary:", {s: results.count(s) for s in ("PASS", "WARN", "FAIL")})
sys.exit(1 if "FAIL" in results else 0)
