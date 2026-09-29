"""Read-only summary of the September 29 bridge and cumulative experiments."""

import json
from pathlib import Path
from statistics import median


ROOT = Path("artifacts")


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_curve(label, data, field="exploitability", time_field="snapshot_min"):
    print("\n", label, "points", len(data))
    if not data:
        return
    for target in (15, 60, 120, 180, 240, 300, 330, 345, 390, 450):
        row = min(data, key=lambda r: abs(r[time_field] - target))
        if abs(row[time_field] - target) <= 1.0:
            value = row[field] if field in row else row["average"]["exploitability"]
            print(f"  {target:3d}m: {value:.6f} iter={row['iteration']} eval_s={row.get('evaluation_s', row.get('exact_eval_s', 0)):.1f}")


old = rows(ROOT / "cfr_plus_18_parallel_cpu/long_20260928/live_exact.jsonl")
selected_curve("old /t 4096", [r for r in old if r.get("arm") == "trav4096__aggregate_then_clip__seed17"])
cumulative = rows(ROOT / "cfr_plus_18_cumulative_regret/main_20260929/live_exact.jsonl")
for arm in ("nk1024", "nk4096", "conditional4096"):
    arm_curve = [r for r in cumulative if r.get("arm") == arm]
    selected_curve(arm, arm_curve)
    if arm_curve:
        best = min(arm_curve, key=lambda r: r["exploitability"])
        print("  best", best["snapshot_min"], best["iteration"], best["exploitability"])
rescue_root = ROOT / "cfr_plus_18_oens_followups/main_20260929/exact_g"
selected_curve("exact g rescue", rows(rescue_root / "monitors.jsonl"), time_field="training_min")
rescue_train = rows(rescue_root / "training.jsonl")
if rescue_train:
    last = rescue_train[-min(100, len(rescue_train)):]
    print("rescue timing median", {k: round(median([r["timing"][k] for r in last]), 3) for k in ("traversal_s", "regret_training_s", "strategy_training_s")}, "iter_s", round(median([r["iteration_s"] for r in last]), 3))

bridge = ROOT / "cfr_plus_18_tabular_bridge/main_20260929"
print("\nBridge per-iteration medians (90-120 training min; latest 100 rows too)")
for k in (128, 256, 512, 1024):
    for arm in ("sample_both", "conditional"):
        base = bridge / (f"low_roots/k{k:04d}" if k < 1024 else "") / arm
        data = rows(base / "training.jsonl")
        matched = [r for r in data if 90 <= r["measured_training_min"] <= 120]
        if len(matched) < 10:
            matched = data[-min(100, len(data)):]
        if not matched:
            continue
        timing = {}
        for name in ("iteration_s", "sampling_s", "update_s"):
            timing[name] = round(median(r[name] for r in matched), 3)
        timing["other_s"] = round(median(r["iteration_s"] - r["sampling_s"] - r["update_s"] for r in matched), 3)
        timing["rows"] = round(median(r["sample_unique_rows"] for r in matched))
        ev = rows(base / "evaluations.jsonl")
        latest = ev[-1] if ev else {}
        print(k, arm, "n", len(matched), "max_min", round(data[-1]["measured_training_min"], 1), "max_iter", data[-1]["iteration"], timing, "latest_eval", round(latest.get("exploitability", -1), 6), "latest_eval_iter", latest.get("iteration"), "eval_s", round(median(r["evaluation_s"] for r in ev), 1) if ev else None)
        print("  curve", [(round(r["measured_training_min"]), r["iteration"], round(r["exploitability"], 5)) for r in ev if round(r["measured_training_min"]) in (15, 60, 90, 105, 120, 135, 150, 180)])

old_curve = [r for r in old if r.get("arm") == "trav4096__aggregate_then_clip__seed17"]
new_curve = [r for r in cumulative if r.get("arm") == "conditional4096"]
for target in (1000, 2000, 3000, 4000):
    print("iter comparison", target, [(label, r["iteration"], round(r["exploitability"], 6), r["snapshot_min"]) for label, curve in (("old", old_curve), ("cumulative", new_curve)) for r in [min(curve, key=lambda x: abs(x["iteration"] - target))]])
