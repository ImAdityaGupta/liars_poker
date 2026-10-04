#!/usr/bin/env python3
"""Live, read-only comparison of six discount arms and two batched bridge controls."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = {
    "V_cfr_uniform": ("V · CFR, uniform", "#0072B2"),
    "A_cfr_plus_linear": ("A · CFR+, linear", "#D55E00"),
    "B_cfr_plus_quadratic": ("B · CFR+, quadratic", "#009E73"),
    "C_dcfr_plus_quadratic": ("C · DCFR+, quadratic", "#CC79A7"),
    "D_dcfr_exact_quadratic": ("D · DCFR, exact decay", "#E69F00"),
    "E_dcfr_visited_quadratic": ("E · DCFR, visited decay", "#56B4E9"),
}
CONTROLS = {
    "exact4096": ("Batched K=4096 | exact average", "#111827"),
    "neural1024": ("Batched K=1024 | neural average", "#8B5CF6"),
}
NEURAL_CONTROLS = {
    "neural_o4_k1024": ("Neural K=1024 | O4 refit", "#E11D48"),
    "neural_o4_k4096": ("Neural K=4096 | O4 refit", "#0D9488"),
}
REGRET_ARMS = {
    "c0": ("C0 baseline", "#2563EB"),
    "c_batch": ("C batch 8192", "#D97706"),
    "c_anneal": ("C cosine per update", "#059669"),
    "c_low": ("C learning rate 3e-4", "#DB2777"),
}
ROOT_ARMS = {
    "k0256": ("Constant K=256", "#2563EB"),
    "k0512": ("Constant K=512", "#D97706"),
    "k1024": ("Constant K=1,024", "#059669"),
    "k2048": ("Constant K=2,048", "#DB2777"),
    "k8192": ("Constant K=8,192", "#7C3AED"),
    "k16384": ("Constant K=16,384", "#DC2626"),
    "ramp_up": ("Ramp 512 → 7,680 → 32,768", "#0891B2"),
    "ramp_down": ("Ramp 7,680 → 512", "#B45309"),
    "step_late": ("Step 512 → 7,680", "#4D7C0F"),
    "step_early": ("Step 7,680 → 512", "#BE185D"),
}
EXACT_DISCOUNT_ARMS = {
    "V_cfr_uniform": ("V · CFR, uniform average", "#7C3AED"),
    "B_cfr_plus_quadratic": ("B · CFR+, quadratic average", "#D55E00"),
    "C_dcfr_plus_quadratic": ("C · DCFR+, quadratic average", "#009E73"),
    "D_dcfr_exact_quadratic": ("D · DCFR, exact decay", "#CC79A7"),
    "E_dcfr_visited_quadratic": ("E · DCFR, visited decay", "#E69F00"),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def last_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        for line in reversed(handle.read().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {}


def status(root: Path, controls_root: Path, neural_root: Path | None = None,
           regret_root: Path | None = None, schedule_root: Path | None = None,
           exact_discount_root: Path | None = None) -> dict:
    rows = []
    for name, (label, _) in ARMS.items():
        directory = root / name
        state = read_json(directory / "state.json")
        summary = read_json(directory / "summary.json")
        evaluations = read_jsonl(directory / "evaluations.jsonl")
        latest = evaluations[-1] if evaluations else {}
        checkpoint = directory / "latest_checkpoint.pt"
        summary_path = directory / "summary.json"
        training_path = directory / "training.jsonl"
        training = last_jsonl(training_path)
        run_status = summary.get("status", "running" if state else "waiting")
        if (run_status in {"paused", "target_reached"} and training_path.exists() and summary_path.exists()
                and training_path.stat().st_mtime > summary_path.stat().st_mtime):
            run_status = "running"
        rows.append({
            "name": name, "label": label,
            "status": run_status,
            "minutes": training.get("measured_training_min", state.get("measured_training_s", 0) / 60),
            "iteration": training.get("iteration", state.get("iteration", 0)),
            "exact": latest.get("exploitability"),
            "evaluations": len(evaluations),
            "checkpoint_gib": checkpoint.stat().st_size / 2**30 if checkpoint.exists() else 0,
        })
    for name, (label, _) in CONTROLS.items():
        directory = controls_root / name
        evaluations = read_jsonl(directory / "evaluations.jsonl")
        training = last_jsonl(directory / "training.jsonl")
        state = read_json(directory / "state.json")
        summary = read_json(directory / "summary.json")
        checkpoint = directory / "latest_checkpoint.pt"
        training_path = directory / "training.jsonl"
        summary_path = directory / "summary.json"
        run_status = summary.get("status", "running" if state else "waiting")
        if (run_status in {"paused", "target_reached"}
                and training_path.exists() and summary_path.exists()
                and training_path.stat().st_mtime > summary_path.stat().st_mtime):
            run_status = "running"
        rows.append({
            "name": name, "label": label,
            "status": run_status,
            "minutes": training.get("measured_training_min", state.get("measured_training_s", 0) / 60),
            "iteration": training.get("iteration", state.get("iteration", 0)),
            "exact": evaluations[-1].get("exploitability") if evaluations else None,
            "evaluations": len(evaluations),
            "checkpoint_gib": checkpoint.stat().st_size / 2**30 if checkpoint.exists() else 0,
        })
    if neural_root is not None:
        for name, (label, _) in NEURAL_CONTROLS.items():
            directory = neural_root / name
            evaluations = read_jsonl(directory / "evaluations.jsonl")
            training = last_jsonl(directory / "training.jsonl")
            summary = read_json(directory / "summary.json")
            checkpoint = directory / "latest_checkpoint.pt"
            fit_ready = len(list((directory / "policy_snapshots").glob("*/READY.json")))
            latest = next((row for row in reversed(evaluations)
                           if row.get("policy_kind") == "o4"), {})
            run_status = summary.get("status", "running" if training else "waiting")
            if (run_status in {"paused", "target_reached"} and training
                    and (directory / "training.jsonl").stat().st_mtime
                    > (directory / "summary.json").stat().st_mtime):
                run_status = "running"
            rows.append({
                "name": name, "label": label, "status": run_status,
                "minutes": training.get("measured_training_min", 0),
                "iteration": training.get("iteration", 0),
                "exact": latest.get("exploitability"),
                "evaluations": len(evaluations), "fit_ready": fit_ready,
                "checkpoint_gib": checkpoint.stat().st_size / 2**30 if checkpoint.exists() else 0,
            })
    for extra_root, extra_arms, kind in (
        (regret_root, REGRET_ARMS, "o4"),
        (schedule_root, ROOT_ARMS, None),
    ):
        if extra_root is None:
            continue
        for name, (label, _) in extra_arms.items():
            directory = extra_root / name
            training = last_jsonl(directory / "training.jsonl")
            summary = read_json(directory / "summary.json")
            evaluations = read_jsonl(directory / "evaluations.jsonl")
            latest = next((row for row in reversed(evaluations)
                           if kind is None or row.get("policy_kind") == kind), {})
            ckpt = directory / "latest_checkpoint.pt"
            run_status = summary.get("status", "running" if training else "waiting")
            if (run_status in {"paused", "target_reached"}
                    and (directory / "training.jsonl").exists()
                    and (directory / "summary.json").exists()
                    and (directory / "training.jsonl").stat().st_mtime
                    > (directory / "summary.json").stat().st_mtime):
                run_status = "running"
            rows.append({"name": name, "label": label,
                         "status": run_status,
                         "minutes": training.get("measured_training_min", 0),
                         "iteration": training.get("iteration", 0),
                         "exact": latest.get("exploitability"),
                         "evaluations": len(evaluations),
                         "checkpoint_gib": ckpt.stat().st_size / 2**30 if ckpt.exists() else 0})
    if exact_discount_root is not None:
        for name, (label, _) in EXACT_DISCOUNT_ARMS.items():
            directory = exact_discount_root / name
            training = last_jsonl(directory / "training.jsonl")
            summary = read_json(directory / "summary.json")
            evaluations = read_jsonl(directory / "evaluations.jsonl")
            checkpoint = directory / "latest_checkpoint.pt"
            rows.append({
                "name": name, "label": f"Exact-average rerun · {label}",
                "status": summary.get("status", "running" if training else "waiting"),
                "minutes": training.get("measured_training_min", 0),
                "iteration": training.get("iteration", 0),
                "exact": evaluations[-1].get("exploitability") if evaluations else None,
                "evaluations": len(evaluations),
                "checkpoint_gib": checkpoint.stat().st_size / 2**30 if checkpoint.exists() else 0,
            })
    disk = shutil.disk_usage(root)
    return {"arms": rows, "free_gib": disk.free / 2**30,
            "has_regret": regret_root is not None,
            "has_schedules": schedule_root is not None}


def plot_followup(root: Path, arms: dict, *, regret: bool,
                  reference_root: Path | None = None) -> bytes:
    if regret:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
        x_keys = ("iteration", "iteration")
        x_labels = ("CFR iteration", "CFR iteration")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
        x_keys = ("measured_training_min", "iteration", "cumulative_roots_per_player")
        x_labels = ("Measured training minutes", "CFR iteration", "Cumulative roots per player")
    for name, (label, color) in arms.items():
        rows = read_jsonl(root / name / "evaluations.jsonl")
        for kind, style, suffix in (("o4", "-", ""), ("online", ":", " online"),
                                    ("current", "--", " current")) if regret else ((None, "-", ""),):
            selected = sorted((row for row in rows
                               if (kind is None or row.get("policy_kind") == kind)
                               and row.get("exploitability", 0) > 0),
                              key=lambda row: row["iteration"])
            if not selected:
                continue
            for ax, key in zip(axes, x_keys):
                x = [row.get(key) for row in selected]
                if all(value is not None for value in x):
                    ax.plot(x, [row["exploitability"] for row in selected],
                            color=color, linestyle=style, marker="o", markersize=3,
                            label=label + suffix)
    if not regret and reference_root is not None:
        rows = read_jsonl(reference_root / "exact4096" / "evaluations.jsonl")
        rows = [r for r in rows if r.get("exploitability", 0) > 0]
        for ax, key in zip(axes, x_keys):
            if key == "cumulative_roots_per_player":
                x = [r.get(key, 4096 * r["iteration"]) for r in rows]
            else:
                x = [r.get(key, r["iteration"]) for r in rows]
            ax.plot(x, [r["exploitability"] for r in rows], color="#111827",
                    linestyle="--", label="Historical constant K=4,096")
    for ax, xlabel in zip(axes, x_labels):
        ax.set_xlabel(xlabel)
    for ax in axes:
        ax.set(ylabel="Exact exploitability", yscale="log")
        ax.grid(alpha=.25)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center",
               ncol=3, fontsize=8, frameon=False)
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=145)
    plt.close(fig)
    return out.getvalue()


def plot(root: Path, bridge_root: Path, controls_root: Path,
         neural_root: Path | None = None,
         exact_discount_root: Path | None = None) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    # The current top chart is a focused discount-rule comparison. Arm A is
    # the existing batched exact-average K=4096 control; retain only the two
    # requested O4 neural curves as external context.
    exact_a = [row for row in read_jsonl(controls_root / "exact4096" / "evaluations.jsonl")
               if row.get("exploitability", 0) > 0]
    series = [("A · CFR+, linear average (K=4,096)", "#111827", "-", exact_a)]
    if neural_root is not None:
        for name, (label, color) in NEURAL_CONTROLS.items():
            if name not in {"neural_o4_k1024", "neural_o4_k4096"}:
                continue
            rows = read_jsonl(neural_root / name / "evaluations.jsonl")
            selected = [row for row in rows if row.get("policy_kind") == "o4"
                        and row.get("exploitability", 0) > 0]
            series.append((label, color, "--", selected))
    if exact_discount_root is not None:
        for name, (label, color) in EXACT_DISCOUNT_ARMS.items():
            rows = [row for row in read_jsonl(exact_discount_root / name / "evaluations.jsonl")
                    if row.get("exploitability", 0) > 0]
            series.append((label, color, "-", rows))
    for label, color, style, rows in series:
        rows.sort(key=lambda row: row["iteration"])
        if not rows:
            continue
        for ax, key in zip(axes, ("measured_training_min", "iteration")):
            ax.plot([row[key] for row in rows], [row["exploitability"] for row in rows],
                    color=color, linestyle=style, marker="o", markersize=3,
                    linewidth=2 if style == "-" else 1.7, label=label)
    for ax, xlabel in zip(axes, ("Measured training minutes", "CFR iteration")):
        ax.set(xlabel=xlabel, ylabel="Exact exploitability of average policy", yscale="log")
        ax.grid(alpha=0.25, which="both")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center",
               ncol=3, fontsize=8, frameon=False)
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=145)
    plt.close(fig)
    return output.getvalue()


def plot_neural(neural_root: Path) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for name, (label, color) in NEURAL_CONTROLS.items():
        rows = read_jsonl(neural_root / name / "evaluations.jsonl")
        for kind, style, marker, suffix in (
            ("o4", "-", "o", "O4 refit"),
            ("online", ":", "x", "online fit"),
        ):
            selected = sorted((row for row in rows if row.get("policy_kind") == kind
                               and row.get("exploitability", 0) > 0),
                              key=lambda row: row["iteration"])
            if not selected:
                continue
            for ax, key in zip(axes, ("measured_training_min", "iteration")):
                ax.plot([row[key] for row in selected],
                        [row["exploitability"] for row in selected],
                        color=color, linestyle=style, marker=marker,
                        markersize=4, linewidth=2, label=f"{label.split('|')[0].strip()} | {suffix}")
    for ax, xlabel in zip(axes, ("Measured training minutes", "CFR iteration")):
        ax.set(xlabel=xlabel, ylabel="Exact exploitability", yscale="log")
        ax.grid(alpha=0.25, which="both")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center",
               ncol=2, fontsize=9, frameon=False)
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=145)
    plt.close(fig)
    return output.getvalue()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>18-claim CFR+ comparisons</title><style>
body{font:15px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;
background:#f5f7fb;color:#1c2738}.card{background:white;border:1px solid #dfe5eb;
border-radius:10px;padding:1rem;margin:1rem 0}img{width:100%}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:.5rem;border-bottom:1px solid #e5e9ef}small{color:#59687b}
</style></head><body><h1>18-claim CFR+ comparisons</h1>
<p>The top chart compares the exact-average sampled discount rerun: arm A is the existing
batched K=4,096 exact-average control, with V and B–E added as their runs progress.
The two K=1,024/4,096 O4-refit policies are contextual neural references.
All exploitability values are exact and the vertical axes are logarithmic.
The lower chart retains the online-versus-O4 neural comparison.
The original six discounting arms remain in the progress table, but are omitted from the top chart.
Those earlier arms used a learned neural average. The new exact-average arms save and evaluate
their policies every 15 measured training minutes. O4 reference fitting runs on its separate
schedule; only its refitted curve appears in the top chart.</p>
<div class="card"><h2>Average-policy exact exploitability</h2>
<small>Lower is better; logarithmic vertical axes. Training and evaluation time are separate.</small>
<img id="curve" alt="Exploitability by training time and CFR iteration"></div>
<div class="card"><h2>Neural K=1024 and K=4096: online versus O4</h2>
<small>Same training trajectory for each solid/dotted pair. O4 is fit from a frozen
reservoir on separate CPUs; its fitting time is excluded from measured training minutes.</small>
<img id="neural" alt="Exact exploitability of online and O4 average policies"></div>
<div class="card" id="regret-card" style="display:none"><h2>GPU regret fit arms</h2>
<img id="regret" alt="Regret fit comparison by iteration"></div>
<div class="card" id="schedules-card" style="display:none"><h2>Root-count sweep and schedules (exact average)</h2>
<small>Root-schedule arms completed 9 measured training hours; six fixed-K arms and the ramp-up arm are continuing for 10 more hours (19 hours total). The ramp-up continues smoothly from K=7,680 at hour 9 to K=32,768 at hour 19. Exact exploitability snapshots are scheduled every 15 measured training minutes. Constant K values are 256, 512, 1,024, 2,048, 8,192 and 16,384.
Dynamic schedules change K over measured training time; their total root counts can differ
because iteration costs depend on K. The dashed reference is the existing K=4,096 run; it is
not retrained. Evaluations are scheduled every 15 measured training minutes, so points need
not align to iteration counts.</small>
<img id="schedules" alt="Exact exploitability by training time, iteration and cumulative roots"></div>
<div class="card"><h2>Progress</h2><p id="disk"></p><table><thead><tr>
<th>Arm</th><th>Status</th><th>Training minutes</th><th>Iteration</th>
<th>Latest exact value</th><th>Evaluations</th><th>Checkpoint GiB</th>
</tr></thead><tbody id="rows"></tbody></table></div>
<script>async function refresh(){try{const r=await fetch('/status',{cache:'no-store'});
if(!r.ok)throw Error(r.status);const s=await r.json();
document.getElementById('disk').textContent=`Disk available: ${s.free_gib.toFixed(1)} GiB`;
const body=document.getElementById('rows');body.replaceChildren();for(const a of s.arms){
const tr=document.createElement('tr');for(const v of [a.label,a.status,a.minutes.toFixed(1),
a.iteration,a.exact==null?'pending':a.exact.toFixed(6),a.evaluations,a.checkpoint_gib.toFixed(2)]){
const td=document.createElement('td');td.textContent=String(v);tr.append(td)}body.append(tr)}
document.getElementById('curve').src='/plot.png?t='+Date.now();
document.getElementById('neural').src='/neural.png?t='+Date.now();
if(s.has_regret){document.getElementById('regret-card').style.display='block';document.getElementById('regret').src='/regret.png?t='+Date.now()}
if(s.has_schedules){document.getElementById('schedules-card').style.display='block';document.getElementById('schedules').src='/schedules.png?t='+Date.now()}
}catch(e){document.getElementById('disk').textContent='Dashboard error: '+e}}
refresh();setInterval(refresh,30000);</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--neural-root", type=Path,
                        help="Additional neural K=1024/4096 O4-refit runs")
    parser.add_argument("--regret-root", type=Path)
    parser.add_argument("--schedule-root", type=Path)
    parser.add_argument("--exact-discount-root", type=Path,
                        help="New exact-average reruns of V and B–E")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    root = args.output_root.resolve()
    bridge_root = args.bridge_root.resolve()
    controls_root = args.controls_root.resolve()
    neural_root = args.neural_root.resolve() if args.neural_root else None
    regret_root = args.regret_root.resolve() if args.regret_root else None
    schedule_root = args.schedule_root.resolve() if args.schedule_root else None
    exact_discount_root = (args.exact_discount_root.resolve()
                           if args.exact_discount_root else None)
    root.mkdir(parents=True, exist_ok=True)
    plot_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    body, mime = PAGE.encode(), "text/html; charset=utf-8"
                elif path == "/status":
                    body, mime = json.dumps(status(root, controls_root, neural_root,
                                                   regret_root, schedule_root,
                                                   exact_discount_root)).encode(), "application/json"
                elif path == "/plot.png":
                    with plot_lock:
                        body = plot(root, bridge_root, controls_root, neural_root,
                                    exact_discount_root)
                    mime = "image/png"
                elif path == "/neural.png" and neural_root is not None:
                    with plot_lock:
                        body = plot_neural(neural_root)
                    mime = "image/png"
                elif path == "/regret.png" and regret_root is not None:
                    with plot_lock:
                        body = plot_followup(regret_root, REGRET_ARMS, regret=True)
                    mime = "image/png"
                elif path == "/schedules.png" and schedule_root is not None:
                    with plot_lock:
                        body = plot_followup(schedule_root, ROOT_ARMS, regret=False,
                                             reference_root=controls_root)
                    mime = "image/png"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(500, str(exc))

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
