#!/usr/bin/env python3
"""CPU-only exact evaluator and local dashboard for the 18-claim GPU fit forks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_cfr_plus_18_fit_snapshot.py"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>18-claim regret fitting sweep</title><style>
body{font:15px system-ui,sans-serif;max-width:1450px;margin:2rem auto;padding:0 1rem;color:#1b2736;background:#f4f6fa}
.card{background:white;border:1px solid #dde3eb;border-radius:10px;padding:1rem;margin:1rem 0}
h1{font-size:1.7rem}img{width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
.metric{padding:.6rem;background:#f2f5fa;border-radius:7px}.metric b{display:block;font-size:1.25rem}
table{border-collapse:collapse;width:100%}td,th{padding:.5rem;text-align:left;border-bottom:1px solid #e5e9ef}
small{color:#57677a}#error{color:#b31b3b}
</style></head><body><h1>18-claim cumulative CFR+: regret fitting</h1>
<p>CPU S24 continues from the beginning. GPU S96 and S384, plus the CPU tabular-regret fork, branch from the same frozen cumulative checkpoint.</p>
<div class="card"><div class="grid" id="metrics"></div><p id="error"></p></div>
<div class="card"><h2>Exact average-policy exploitability</h2>
<small>Left: total measured training minutes. A GPU fork's time is CPU trunk time plus GPU branch time; machines have different iteration rates. Right: CFR+ iteration. Both vertical axes are logarithmic. The dashed line marks the common fork point.</small>
<img id="exact" alt="Exact exploitability by training time and iteration"></div>
<div class="card"><h2>Fork progress</h2><table><thead><tr><th>Arm</th><th>Branch training</th><th>Iteration</th><th>Latest exact average</th><th>Latest exact current</th><th>State</th></tr></thead><tbody id="arms"></tbody></table></div>
<div class="card"><h2>GPU use</h2><img id="gpu" alt="GPU utilization and memory history"></div>
<small>Snapshots every 15 training minutes. GPU fork evaluations run in separate low-priority CPU processes; the tabular fork evaluates its saved average policy. Refreshes every 20 seconds.</small>
<script>
async function refresh(){try{
const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();
const items=[['CPU S24',`${s.cpu.iteration??'—'} iter; ${Number(s.cpu.training_min||0).toFixed(1)}m`],
['GPU',`${s.gpu.utilization??'—'}% util; ${s.gpu.used_mib??'—'} / ${s.gpu.total_mib??'—'} MiB`],
['Disk available',`${Number(s.disk_free_gib).toFixed(1)} GiB`],
['Exact evaluation',`${s.evaluation.running||'idle'}; ${s.evaluation.queued} queued`]];
document.getElementById('metrics').replaceChildren(...items.map(([name,value])=>{const d=document.createElement('div');d.className='metric';const b=document.createElement('b');b.textContent=value;d.append(b,name);return d}));
const body=document.getElementById('arms');body.replaceChildren();for(const a of s.arms){const tr=document.createElement('tr');
for(const value of [a.name,`${Number(a.branch_min||0).toFixed(1)} / ${s.target_min} min`,a.iteration??'—',
a.average==null?'pending':Number(a.average).toFixed(6),a.current==null?'pending':Number(a.current).toFixed(6),a.status]){
const td=document.createElement('td');td.textContent=String(value);tr.append(td)}body.append(tr)}
if(s.tabular){const a=s.tabular;const tr=document.createElement('tr');
for(const value of [a.name,`${Number(a.branch_min||0).toFixed(1)} / ${s.tabular_target_min} min`,a.iteration??'—',
a.average==null?'pending':Number(a.average).toFixed(6),'not evaluated',a.status]){
const td=document.createElement('td');td.textContent=String(value);tr.append(td)}body.append(tr)}
const t=Date.now();document.getElementById('exact').src='/exact.png?t='+t;document.getElementById('gpu').src='/gpu.png?t='+t;
document.getElementById('error').textContent='';}catch(e){document.getElementById('error').textContent=String(e)}}
refresh();setInterval(refresh,20000);
</script></body></html>"""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def last_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 65536))
        for line in reversed(handle.read().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {}


def write_png(fig, path: Path) -> None:
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, path)


def plot_exact(root: Path, cpu_root: Path, manifest: dict,
               tabular_fork: Path | None = None) -> None:
    source_min = manifest["source_training_s"] / 60
    source_iteration = manifest["source_iteration"]
    cpu = sorted((row for row in read_jsonl(cpu_root / "live_exact.jsonl")
                  if row.get("arm") == "conditional4096"),
                 key=lambda row: row["iteration"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    series = [("CPU S24 (cumulative, from start)", "#505b6b", cpu)]
    for arm, color in ((96, "#0072b2"), (384, "#d55e00")):
        results = [row for row in read_jsonl(root / f"s{arm}" / "exact_evaluations.jsonl")
                   if row.get("kind") == "average"]
        results.sort(key=lambda row: row["iteration"])
        source = [row for row in cpu if row["iteration"] == source_iteration]
        series.append((f"GPU S{arm} (fork)", color, source + results))
    if tabular_fork is not None:
        results = sorted(read_jsonl(tabular_fork / "evaluations.jsonl"),
                         key=lambda row: row.get("iteration", 0))
        source = [row for row in cpu if row["iteration"] == source_iteration]
        if not source:
            source = [{**row, "total_training_min": source_min}
                      for row in results if row.get("measured_fork_min") == 0][:1]
        branch = [{**row,
                   "total_training_min": source_min + row["measured_fork_min"]}
                  for row in results if row.get("measured_fork_min", 0) > 0]
        series.append(("CPU tabular regrets (fork)", "#009e73", source + branch))
    for label, color, rows in series:
        if not rows:
            continue
        for axis, xkey in zip(axes, ("total_training_min", "iteration")):
            x = [row.get(xkey, row.get("snapshot_min")) for row in rows]
            axis.plot(x, [row["exploitability"] for row in rows],
                      marker="o", markersize=3.5, linewidth=2,
                      color=color, label=label)
    for axis, point in zip(axes, (source_min, source_iteration)):
        axis.axvline(point, color="#9299a3", linestyle="--", linewidth=1.2)
        axis.set_yscale("log")
        axis.grid(alpha=.2, which="both")
        axis.set_ylabel("Exact average-policy exploitability")
    axes[0].set_xlabel("Total measured training minutes (CPU trunk + branch)")
    axes[1].set_xlabel("CFR+ iteration")
    axes[1].legend(fontsize=9)
    write_png(fig, root / "exact.png")


def gpu_reading() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True, timeout=8,
        )
        util, used, total = [int(piece.strip()) for piece in result.stdout.splitlines()[0].split(",")]
        return {"utilization": util, "used_mib": used, "total_mib": total}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def plot_gpu(root: Path) -> None:
    rows = read_jsonl(root / "gpu_history.jsonl")[-720:]
    if not rows:
        return
    x = [(datetime.fromisoformat(row["utc"]) -
          datetime.fromisoformat(rows[0]["utc"])).total_seconds() / 60 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), layout="constrained")
    axes[0].plot(x, [row.get("utilization") for row in rows], color="#0072b2")
    axes[0].set_ylabel("GPU utilization (%)")
    axes[0].set_ylim(0, 100)
    axes[1].plot(x, [row.get("used_mib", 0) / 1024 for row in rows], color="#d55e00")
    axes[1].set_ylabel("GPU memory used (GiB)")
    for axis in axes:
        axis.set_xlabel("Minutes since dashboard started")
        axis.grid(alpha=.2)
    write_png(fig, root / "gpu.png")


def evaluation_tasks(root: Path) -> list[dict]:
    tasks = []
    for arm in (96, 384):
        directory = root / f"s{arm}"
        done = {(row["label"], row["kind"]) for row in
                read_jsonl(directory / "exact_evaluations.jsonl")}
        for event in read_jsonl(directory / "events.jsonl"):
            if event.get("event") != "policy_snapshot":
                continue
            for kind in ("average", "current"):
                if (event["label"], kind) not in done:
                    tasks.append({"arm": arm, "kind": kind, **event,
                                  "policy_dir": directory / "snapshots" / event["label"] /
                                                f"{kind}_policy"})
    return tasks


def evaluate(task: dict) -> dict:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"})
    command = [sys.executable, str(EVALUATOR), str(task["policy_dir"])]
    if shutil.which("nice"):
        command = ["nice", "-n", "10", *command]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, check=True, timeout=300)
    value = json.loads(result.stdout.splitlines()[-1])
    return {key: task[key] for key in ("arm", "kind", "label", "iteration",
                                      "branch_training_s", "total_training_s")} | {
        "total_training_min": task["total_training_s"] / 60,
        "utc": datetime.now(timezone.utc).isoformat(), **value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--tabular-fork", type=Path,
                        help="Overlay exact average-policy evaluations from a tabular regret fork")
    parser.add_argument("--tabular-target-min", type=float, default=180.0)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    root = args.run_root.resolve()
    cpu_root = args.cpu_root.resolve()
    tabular_fork = args.tabular_fork.resolve() if args.tabular_fork else None
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    shared = {"gpu": {}, "running": None, "failures": {}}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                payload, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api/status":
                cpu_arm = (cpu_root / "conditional4096" / "aggregate_then_clip__seed_17")
                cpu_train = last_jsonl(cpu_arm / "training.jsonl")
                cpu_exact = [row for row in read_jsonl(cpu_root / "live_exact.jsonl")
                             if row.get("arm") == "conditional4096"]
                arms = []
                for arm in (96, 384):
                    directory = root / f"s{arm}"
                    state_file = directory / "state.json"
                    state = json.loads(state_file.read_text()) if state_file.exists() else {}
                    latest = last_jsonl(directory / "training.jsonl")
                    exact = read_jsonl(directory / "exact_evaluations.jsonl")
                    by_kind = {kind: next((row["exploitability"] for row in reversed(exact)
                                           if row["kind"] == kind), None)
                               for kind in ("average", "current")}
                    arms.append({"name": f"GPU S{arm}", "status": state.get("status", "waiting"),
                                 "branch_min": latest.get("branch_training_s", state.get("branch_training_s", 0))/60,
                                 "iteration": latest.get("iteration", state.get("iteration")),
                                 **by_kind})
                tabular = None
                if tabular_fork is not None:
                    latest = last_jsonl(tabular_fork / "training.jsonl")
                    state_file = tabular_fork / "state.json"
                    state = json.loads(state_file.read_text()) if state_file.exists() else {}
                    summary_file = tabular_fork / "summary.json"
                    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
                    exact = read_jsonl(tabular_fork / "evaluations.jsonl")
                    latest_eval = next((row for row in reversed(exact)
                                        if row.get("measured_fork_min", 0) > 0),
                                       exact[-1] if exact else {})
                    status = summary.get("status")
                    if status is None:
                        status = "running" if latest else "starting"
                    tabular = {
                        "name": "CPU tabular regrets",
                        "status": status,
                        "branch_min": latest.get("measured_fork_min",
                                                  state.get("measured_fork_s", 0) / 60),
                        "iteration": latest.get("iteration", state.get("iteration")),
                        "average": latest_eval.get("exploitability"),
                    }
                with lock:
                    gpu = dict(shared["gpu"])
                    running = shared["running"]
                    failed = dict(shared["failures"])
                value = {"cpu": {"iteration": cpu_train.get("iteration"),
                                  "training_min": cpu_train.get("measured_training_min"),
                                  "latest_exact": cpu_exact[-1]["exploitability"] if cpu_exact else None},
                         "gpu": gpu, "arms": arms, "tabular": tabular,
                         "target_min": manifest["hours_per_arm"] * 60,
                         "tabular_target_min": args.tabular_target_min,
                         "disk_free_gib": shutil.disk_usage(root).free / 1024**3,
                         "evaluation": {"running": running,
                                        "queued": len(evaluation_tasks(root)),
                                        "failures": failed}}
                payload, mime = json.dumps(value).encode(), "application/json"
            elif path in ("/exact.png", "/gpu.png"):
                file = root / path[1:]
                if not file.exists():
                    self.send_error(404)
                    return
                payload, mime = file.read_bytes(), "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"dashboard http://127.0.0.1:{args.port}", flush=True)
    last_plot = 0.0
    while True:
        now = time.monotonic()
        if now - last_plot > 25:
            gpu = gpu_reading()
            gpu["utc"] = datetime.now(timezone.utc).isoformat()
            with lock:
                shared["gpu"] = gpu
            with (root / "gpu_history.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(gpu) + "\n")
            plot_gpu(root)
            plot_exact(root, cpu_root, manifest, tabular_fork)
            last_plot = now
        tasks = evaluation_tasks(root)
        with lock:
            failures = dict(shared["failures"])
        task = next((task for task in tasks
                     if failures.get(f"{task['arm']}:{task['label']}:{task['kind']}", 0) < 3), None)
        if task:
            key = f"{task['arm']}:{task['label']}:{task['kind']}"
            with lock:
                shared["running"] = key
            try:
                result = evaluate(task)
                with (root / f"s{task['arm']}" / "exact_evaluations.jsonl").open(
                    "a", encoding="utf-8") as out:
                    out.write(json.dumps(result) + "\n")
                print(f"exact {key}: {result['exploitability']:.6f}", flush=True)
                plot_exact(root, cpu_root, manifest, tabular_fork)
            except Exception as exc:
                with lock:
                    shared["failures"][key] = shared["failures"].get(key, 0) + 1
                with (root / "evaluation_errors.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                          "task": key, "error": repr(exc)}) + "\n")
                print(f"evaluation failed {key}: {exc}", flush=True)
                time.sleep(5)
            finally:
                with lock:
                    shared["running"] = None
        time.sleep(5 if task else 15)


if __name__ == "__main__":
    main()
