"""Append one JSON line per run to results/runs.jsonl (loop rule 2)."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs.jsonl"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def log_run(task_id: str, config: dict, metrics: dict, seed: int = 0,
            model_hash: str | None = None, wall_s: float | None = None) -> None:
    entry = {
        "task": task_id,
        "config": config,
        "seed": seed,
        "metrics": metrics,
        "git_sha": git_sha(),
        "wall_s": wall_s,
        "hardware": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "model_hash": model_hash,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    RUNS.parent.mkdir(exist_ok=True)
    with RUNS.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
