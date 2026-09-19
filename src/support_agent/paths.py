"""Folders the scripts read and write. `outputs/` is git-ignored; `reports/` is committed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
