from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

from liars_poker.core import GameSpec, ARTIFACTS_ROOT
from liars_poker.serialization import save_policy


def basic_eta_control(episodes: int) -> float:
    return 1 / (episodes + 2)


def plot_exploitability_series(logs: Dict | Iterable[Dict], *, figsize: tuple[int, int] = (12, 6)):
    """Plot exploitability over iterations using the logs dict returned by fsp_loop/dense_fsp_loop."""
    import matplotlib.pyplot as plt

    series_list = logs if isinstance(logs, Iterable) and not isinstance(logs, Dict) else [logs]

    fig, ax = plt.subplots(figsize=figsize)
    for idx, entry in enumerate(series_list, start=1):
        series = entry.get("exploitability_series", [])
        if not series:
            continue
        vals = [
            pt.get("rollout_avg", pt.get("predicted_avg", 0.0))
            for pt in series
        ]
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=f"run {idx}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploitability")
    ax.set_title("Exploitability over FSP iterations")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    return ax


def save_fsp_run(run_id: str, policy, info: dict, spec: GameSpec, root: Path | None = None) -> None:
    """Save a policy and its FSP run metrics to a run directory."""
    root = root or Path(ARTIFACTS_ROOT)
    run_dir = root / "benchmark_runs" / run_id
    (run_dir / "policy").mkdir(parents=True, exist_ok=True)

    # Save policy
    save_policy(policy, run_dir / "policy")

    # Save metrics
    metrics = {
        "spec": spec.to_json(),
        "exploitability_series": info.get("exploitability_series", []),
        "p_values": info.get("p_values", []),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved run to {run_dir}")
