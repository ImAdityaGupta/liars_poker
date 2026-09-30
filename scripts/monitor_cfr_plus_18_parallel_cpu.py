#!/usr/bin/env python3
"""Low-impact live exact evaluation and local-only dashboard for the CPU study.

This is independent of the trainer. It watches completed snapshot events,
evaluates saved average policies in low-priority subprocesses, and serves the
training progress and exact curves on 127.0.0.1. Stopping this process cannot
stop or alter the training processes. Restarting it skips completed evaluations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
SINGLE_EVAL = ROOT / "scripts" / "evaluate_cfr_plus_18_single_policy.py"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>18-claim CFR+ experiment</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#182132;background:#f5f7fb}
h1{font-size:1.65rem}.card{background:white;border:1px solid #dfe5ef;border-radius:10px;padding:1rem;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #e7eaf0}th{background:#f7f9fc}
.bar{height:8px;width:120px;background:#e6eaf2;border-radius:6px;overflow:hidden}.fill{height:100%;background:#4263eb}
img{width:100%;height:auto}small{color:#5d6b80}#error{color:#a21d31}.metrics{display:flex;flex-wrap:wrap;gap:1.5rem}.metric{min-width:170px}.metric b{font-size:1.35rem;display:block}
</style></head><body>
<h1>18-claim CFR+: cumulative regret and tabular fork</h1><p id="summary">Loading...</p><p id="error"></p>
<div class="card"><h2>Machine use</h2><small>CPU is measured in core equivalents, not a percentage of 128 logical threads. This VM has 64 physical cores; exact evaluation normally uses up to two additional cores.</small><div class="metrics" id="metrics"></div><img id="resources" alt="CPU and memory history"></div>
<div class="card"><h2>New training arms</h2><table><thead><tr><th>Traversals</th><th>Target</th><th>Seed</th><th>Training</th><th>Iteration</th><th>Latest exact</th><th>Snapshot evaluation</th></tr></thead><tbody id="arms"></tbody></table></div>
<div class="card"><h2>Tabular regret fork</h2><p id="fork">Waiting for fork data...</p></div>
<div class="card"><h2>Exact average-policy exploitability</h2><small>Two log-scale graphs: total measured training time and CFR+ iterations. The tabular fork starts at the source checkpoint's 300-minute mark. Lower is better.</small><img id="exact" alt="Exact exploitability by time and iteration"></div>
<div class="card"><h2>Training progress</h2><img id="progress" alt="Iteration progress plot"></div>
<small>Refreshes every 20 seconds. Exact evaluations run in separate low-priority CPU processes. The training jobs are independent of this page.</small>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/status',{cache:'no-store'}); if(!r.ok) throw Error(r.status);
    const s=await r.json();
    document.getElementById('summary').textContent=`${s.running} running · ${s.stopped} stopped · ${s.finished} finished · ${s.exact_count} exact snapshots evaluated · ${s.evaluations.running} evaluating now · ${s.evaluations.queued} queued · updated ${s.updated_utc}`;
    const m=s.resources||{};
    const items=[['Training CPU',`${(m.training_cores||0).toFixed(1)} cores`],['Exact evaluation CPU',`${(m.evaluation_cores||0).toFixed(1)} cores`],['Machine CPU',`${(m.machine_busy_cores||0).toFixed(1)} / ${m.logical_cpus||'?'} logical`],['Available RAM',`${(m.memory_available_gib||0).toFixed(1)} / ${(m.memory_total_gib||0).toFixed(1)} GiB`]];
    const metrics=document.getElementById('metrics');metrics.replaceChildren();
    for(const [label,value] of items){const div=document.createElement('div');div.className='metric';const bold=document.createElement('b');bold.textContent=value;div.appendChild(bold);div.append(label);metrics.appendChild(div)}
    const body=document.getElementById('arms');body.replaceChildren();
    for(const a of s.arms){
      const tr=document.createElement('tr');
      const parts=[a.traversals,a.label||a.mode,a.seed];
      for(const v of parts){const td=document.createElement('td');td.textContent=String(v);tr.appendChild(td)}
      const td=document.createElement('td');
      const n=Number(a.measured_min||0), goal=Number(a.target_minutes||s.target_min||150);td.textContent=`${n.toFixed(1)} / ${goal} min`;
      const bar=document.createElement('div');bar.className='bar';const fill=document.createElement('div');fill.className='fill';fill.style.width=`${Math.min(100,n/goal*100)}%`;bar.appendChild(fill);td.appendChild(bar);tr.appendChild(td);
      for(const v of [a.iteration??'—',a.latest_exact==null?'—':Number(a.latest_exact).toFixed(5),a.evaluation_status]){const c=document.createElement('td');c.textContent=String(v);tr.appendChild(c)}
      body.appendChild(tr);
    }
    const f=s.tabular_fork;
    document.getElementById('fork').textContent=f?`Source: ${f.source_min.toFixed(0)} min, iteration ${f.source_iteration} · fork training: ${f.measured_min.toFixed(1)} / ${f.target_min.toFixed(0)} min · current iteration: ${f.iteration} · latest exact: ${f.latest_exact==null?'pending':f.latest_exact.toFixed(6)} · ${f.status}`:'No fork directory configured';
    const stamp=Date.now();
    document.getElementById('exact').src='/live_exploitability.png?t='+stamp;
    document.getElementById('progress').src='/live_iterations.png?t='+stamp;
    document.getElementById('resources').src='/live_resources.png?t='+stamp;
    document.getElementById('error').textContent='';
  }catch(e){document.getElementById('error').textContent='Dashboard refresh failed: '+e}
}
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
                pass  # A writer may be in the middle of appending its final line.
    return rows


def last_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        n = handle.tell()
        handle.seek(max(0, n - 65_536))
        for line in reversed(handle.read().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def target_minutes(root: Path, arms: list[dict]) -> float:
    target = float(json.loads((root / "parallel_manifest.json").read_text())["minutes_per_arm"])
    for arm in arms:
        for event in read_jsonl(root / arm["name"] / "resume_events.jsonl"):
            target = max(target, float(event.get("target_hours_per_arm", 0)) * 60.0)
    return target


def arm_dir(root: Path, arm: dict) -> Path:
    return root / arm["name"] / f'{arm["mode"]}__seed_{arm["seed"]}'


def resource_counters(root: Path) -> dict:
    """Read cumulative CPU seconds without starting another profiling process."""
    ticks = os.sysconf("SC_CLK_TCK")
    cpus = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            name, value, *_ = line.split()
            memory[name.removesuffix(":")] = int(value) / (1024 * 1024)
    processes = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if str(root) not in command:
                continue
            if "run_cfr_plus_18_target_order_cpu_overnight.py" in command:
                kind = "training"
            elif ("evaluate_cfr_plus_18_single_policy.py" in command
                  or "evaluate_cfr_plus_18_target_order_cpu.py" in command):
                kind = "evaluation"
            else:
                continue
            stat = (directory / "stat").read_text().split(") ", 1)[1].split()
            processes[int(directory.name)] = (kind, (int(stat[11]) + int(stat[12])) / ticks)
        except (OSError, IndexError, ValueError):
            continue  # A process can exit while /proc is being scanned.
    return {"at": time.monotonic(), "cpu_total": sum(cpus),
            "cpu_idle": cpus[3] + cpus[4], "processes": processes,
            "memory_total_gib": memory.get("MemTotal", 0.0),
            "memory_available_gib": memory.get("MemAvailable", 0.0)}


def resource_sample(previous: dict, current: dict) -> dict:
    elapsed = max(current["at"] - previous["at"], 1e-6)
    cores = {"training": 0.0, "evaluation": 0.0}
    for pid, (kind, seconds) in current["processes"].items():
        old = previous["processes"].get(pid)
        if old and old[0] == kind:
            cores[kind] += max(0.0, seconds - old[1]) / elapsed
    total = current["cpu_total"] - previous["cpu_total"]
    idle = current["cpu_idle"] - previous["cpu_idle"]
    logical_cpus = len(os.sched_getaffinity(0))
    busy = logical_cpus * (1 - idle / total) if total > 0 else 0.0
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "training_cores": round(cores["training"], 2),
            "evaluation_cores": round(cores["evaluation"], 2),
            "machine_busy_cores": round(max(0.0, busy), 2),
            "logical_cpus": logical_cpus,
            "memory_total_gib": round(current["memory_total_gib"], 2),
            "memory_available_gib": round(current["memory_available_gib"], 2)}


def plot_resources(root: Path) -> None:
    rows = read_jsonl(root / "resource_history.jsonl")[-180:]
    if not rows:
        return
    minutes = [(datetime.fromisoformat(row["utc"]) -
                datetime.fromisoformat(rows[0]["utc"])).total_seconds() / 60 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    for key, label, colour in (("training_cores", "Trainers", "#0072B2"),
                               ("evaluation_cores", "Exact evaluators", "#D55E00"),
                               ("machine_busy_cores", "Machine busy", "#666666")):
        axes[0].plot(minutes, [row[key] for row in rows], label=label,
                     color=colour, linewidth=1.8)
    axes[0].set_ylabel("Logical CPU cores used")
    axes[0].legend(fontsize=8, ncol=3, loc="upper center")
    axes[1].plot(minutes, [row["memory_available_gib"] for row in rows],
                 color="#009E73", linewidth=2)
    axes[1].set_ylabel("Available RAM (GiB)")
    axes[1].set_ylim(0, max(row["memory_total_gib"] for row in rows))
    for axis in axes:
        axis.set_xlabel("Minutes since monitor started")
        axis.grid(True, alpha=0.2)
    fig.tight_layout()
    path = root / "live_resources.png"
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    tmp.replace(path)


def plot_comparison(root: Path, arms: list[dict], rows: list[dict], reference_root: Path,
                    extra_run: Path | None = None,
                    tabular_fork: Path | None = None,
                    fork_start_min: float = 300) -> None:
    reference_manifest = json.loads((reference_root / "parallel_manifest.json").read_text())
    reference_rows = read_jsonl(reference_root / "live_exact.jsonl")
    reference_colours = {
        (1024, "clip_each_record"): "#7baec9",
        (1024, "aggregate_then_clip"): "#e6a06d",
        (4096, "clip_each_record"): "#87b9a4",
        (4096, "aggregate_then_clip"): "#b49ac4",
    }
    new_colours = {1024: "#173c6b", 4096: "#bf205d"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharey=True)

    def draw(group: list[dict], label: str, color: str, *, new: bool, seed: int) -> None:
        if not group:
            return
        group = sorted(group, key=lambda r: r["snapshot_min"])
        for axis, xfield in zip(axes, ("snapshot_min", "iteration")):
            axis.plot([r[xfield] for r in group], [r["exploitability"] for r in group],
                      color=color, linestyle="-" if seed == 17 else "--",
                      marker="D" if new else ("o" if seed == 17 else "s"),
                      markersize=5 if new else 3, linewidth=2.6 if new else 1.3,
                      alpha=1.0 if new else 0.65, label=label)

    for arm in reference_manifest["arms"]:
        key = (arm["traversals"], arm["mode"])
        color = reference_colours[key]
        mode_label = "aggregate" if arm["mode"] == "aggregate_then_clip" else "clip each"
        draw([r for r in reference_rows if r["arm"] == arm["name"]],
             f"Old {arm['traversals']} {mode_label} s{arm['seed']}", color,
             new=False, seed=arm["seed"])
    for arm in arms:
        if arm.get("reach_mode") == "visit_fraction":
            continue
        draw([r for r in rows if r["arm"] == arm["name"]],
             arm.get("label", f"New {arm['traversals']} s{arm['seed']}"),
             arm.get("color", new_colours.get(arm["traversals"], "#222222")),
             new=True, seed=arm["seed"])
    if extra_run is not None:
        extras = sorted(read_jsonl(extra_run / "monitors.jsonl"),
                        key=lambda row: row["training_min"])
        for axis, key in zip(axes, ("training_min", "iteration")):
            axis.plot([r[key] for r in extras],
                      [r["average"]["exploitability"] for r in extras],
                      color="#663399", marker="*", markersize=8, linewidth=2.5,
                      label="Exact g at visited sets · normalized s17, fork at 330m")
    if tabular_fork is not None:
        fork_rows = sorted(read_jsonl(tabular_fork / "evaluations.jsonl"),
                           key=lambda row: row["measured_fork_min"])
        for axis, x in zip(axes,
                           ([fork_start_min + r["measured_fork_min"] for r in fork_rows],
                            [r["iteration"] for r in fork_rows])):
            axis.plot(x, [r["exploitability"] for r in fork_rows],
                      color="#00858a", marker="P", markersize=8, linewidth=3,
                      label="Tabular regret fork from 300m OENS checkpoint")
    for axis, title, xlabel in zip(
        axes, ("Equal measured training time", "Equal CFR+ iteration"),
        ("Measured training minutes", "CFR+ iteration"),
    ):
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=.22)
    axes[0].set_ylabel("Exact average-policy exploitability")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .15, 1, 1))
    path = root / "live_exploitability.png"
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    tmp.replace(path)


def plot_rows(root: Path, arms: list[dict], rows: list[dict], *, exact: bool,
              reference_root: Path | None = None,
              extra_run: Path | None = None,
              tabular_fork: Path | None = None,
              fork_start_min: float = 300) -> None:
    if exact and reference_root is not None:
        plot_comparison(root, arms, rows, reference_root, extra_run,
                        tabular_fork, fork_start_min)
        return
    budgets = sorted({arm["traversals"] for arm in arms})
    palette = ("#0072B2", "#D55E00", "#009E73", "#884EA0",
               "#CC79A7", "#56B4E9", "#E69F00", "#444444")
    colours = {(budget, mode): palette[(2 * index + offset) % len(palette)]
               for index, budget in enumerate(budgets)
               for mode, offset in (("clip_each_record", 0),
                                    ("aggregate_then_clip", 1),
                                    ("clip_on_read", 2))}
    styles = {17: ("-", "o"), 23: ("--", "s")}
    fig, axes = plt.subplots(1, max(1, len(budgets)),
                             figsize=(7 * max(1, len(budgets)), 5.4),
                             sharey=exact, squeeze=False)
    axes = axes[0]
    for arm in arms:
        if exact:
            sub = sorted((r for r in rows if r["arm"] == arm["name"]),
                         key=lambda r: r["snapshot_min"])
            x_values = ([r["snapshot_min"] for r in sub],
                        [r["iteration"] for r in sub])
            values = [r["exploitability"] for r in sub]
        else:
            data = read_jsonl(arm_dir(root, arm) / "training.jsonl")
            sub = data[::max(1, len(data) // 300)]
            if data and (not sub or sub[-1] is not data[-1]):
                sub.append(data[-1])
            x_values = ([r["measured_training_min"] for r in sub],)
            values = [r["iteration"] for r in sub]
        if not sub:
            continue
        line, marker = styles.get(arm["seed"], (":", "^"))
        colour = arm.get("color", colours[(arm["traversals"], arm["mode"])])
        if exact:
            for axis, x in zip(axes, x_values):
                axis.plot(x, values, color=colour, linestyle=line, marker=marker,
                          markersize=5, linewidth=2, alpha=0.9)
        else:
            axes[budgets.index(arm["traversals"])].plot(
                x_values[0], values, color=colour, linestyle=line,
                linewidth=1.8, alpha=0.9)
    if exact:
        for axis, title, xlabel in zip(axes,
                                       ("By measured training time", "By CFR+ iteration"),
                                       ("Measured training minutes", "CFR+ iteration")):
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.22)
        axes[0].set_ylabel("Exact average-policy exploitability")
    else:
        for axis, budget in zip(axes, budgets):
            axis.set_title(f"{budget:,} traversals per player")
            axis.set_xlabel("Measured training minutes")
            axis.grid(True, alpha=0.22)
        axes[0].set_ylabel("CFR+ iteration")
    colour_handles = [Line2D(
        [0], [0], color=arm.get("color", colours[(arm["traversals"], arm["mode"])]),
        linewidth=2.5,
        label=arm.get("label", f"{arm['traversals']:,} · "
                               f"{'clip each' if arm['mode'] == 'clip_each_record' else 'aggregate first'}"),
    ) for arm in arms]
    seed_handles = [Line2D([0], [0], color="#333333", linestyle=styles[seed][0],
                           marker=styles[seed][1] if exact else None, linewidth=2,
                           label=f"seed {seed}") for seed in sorted({a["seed"] for a in arms})]
    fig.legend(handles=colour_handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.095), fontsize=9)
    fig.legend(handles=seed_handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = root / ("live_exploitability.png" if exact else "live_iterations.png")
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    tmp.replace(path)


def status(root: Path, arms: list[dict], activity: dict, activity_lock: threading.Lock,
           tabular_fork: Path | None = None, fork_start_min: float = 300) -> dict:
    target_min = target_minutes(root, arms)
    latest = {}
    rows = read_jsonl(root / "live_exact.jsonl")
    for row in rows:
        if row["arm"] not in latest or row["snapshot_min"] > latest[row["arm"]][0]:
            latest[row["arm"]] = (row["snapshot_min"], row["exploitability"])
    tasks = completed_snapshots(root, arms)
    done = {(row["arm"], row["label"]) for row in rows}
    with activity_lock:
        active = dict(activity["active"])
        failures = dict(activity["failures"])
    newest = {}
    for task in tasks:
        newest[task["arm"]] = task
    queued = [task for task in tasks
              if (task["arm"], task["label"]) not in done
              and (task["arm"], task["label"]) not in active
              and failures.get((task["arm"], task["label"]), 0) < 3]
    out = []
    for arm in arms:
        directory = arm_dir(root, arm)
        training = last_jsonl(directory / "training.jsonl") or {}
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        task = newest.get(arm["name"])
        if task is None:
            evaluation_status = "Awaiting first snapshot"
        else:
            key = (task["arm"], task["label"])
            if key in active:
                evaluation_status = f"Running {task['label']} ({int(time.monotonic() - active[key])}s)"
            elif key in done:
                evaluation_status = f"Done {task['label']}"
            elif failures.get(key, 0) >= 3:
                evaluation_status = f"Failed {task['label']}"
            else:
                evaluation_status = f"Queued {task['label']}"
        stopped = (directory / "STOPPED.json").exists()
        if stopped:
            evaluation_status = f"Stopped; {evaluation_status}"
        arm_target = float(arm.get("target_minutes", target_min))
        out.append({**arm,
                    "measured_min": training.get("measured_training_min", state.get("measured_training_min", 0)),
                    "iteration": training.get("iteration", state.get("iteration")),
                    "finished": state.get("status") == "complete" and
                                float(training.get("measured_training_min",
                                                   state.get("measured_training_min", 0))) >= arm_target,
                    "stopped": stopped,
                    "latest_exact": latest.get(arm["name"], (None, None))[1],
                    "evaluation_status": evaluation_status})
    resources = last_jsonl(root / "resource_history.jsonl") or {}
    fork_status = None
    if tabular_fork is not None:
        fork_manifest = tabular_fork / "manifest.json"
        manifest = json.loads(fork_manifest.read_text()) if fork_manifest.exists() else {}
        state = last_jsonl(tabular_fork / "training.jsonl") or {}
        checkpoint_state = tabular_fork / "state.json"
        if not state and checkpoint_state.exists():
            state = json.loads(checkpoint_state.read_text())
        fork_evals = read_jsonl(tabular_fork / "evaluations.jsonl")
        summary_path = tabular_fork / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        fork_status = {
            "source_min": fork_start_min,
            "source_iteration": manifest.get("source_iteration", 0),
            "measured_min": float(state.get("measured_fork_min",
                                            state.get("measured_fork_s", 0) / 60)),
            "target_min": 600.0,
            "iteration": state.get("iteration", manifest.get("source_iteration", 0)),
            "latest_exact": fork_evals[-1]["exploitability"] if fork_evals else None,
            "status": summary.get("status", "training"),
        }
    return {"updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target_min": target_min,
            "running": sum(not a["finished"] and not a["stopped"] for a in out),
            "stopped": sum(a["stopped"] for a in out),
            "finished": sum(a["finished"] for a in out),
            "exact_count": len(rows), "arms": out, "resources": resources,
            "tabular_fork": fork_status,
            "evaluations": {"running": len(active), "queued": len(queued),
                            "active": [{"arm": arm, "snapshot": label,
                                        "elapsed_s": int(time.monotonic() - started)}
                                       for (arm, label), started in active.items()]}}


def serve(root: Path, arms: list[dict], port: int, activity: dict,
          activity_lock: threading.Lock, tabular_fork: Path | None = None,
          fork_start_min: float = 300) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                payload, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api/status":
                payload = json.dumps(status(root, arms, activity, activity_lock,
                                            tabular_fork, fork_start_min)).encode()
                mime = "application/json"
            elif path in ("/live_exploitability.png", "/live_iterations.png",
                          "/live_resources.png"):
                file = root / path[1:]
                if not file.exists():
                    self.send_error(404, "Waiting for first graph")
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

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def completed_snapshots(root: Path, arms: list[dict]) -> list[dict]:
    tasks = []
    for arm in arms:
        for event in read_jsonl(arm_dir(root, arm) / "events.jsonl"):
            if event.get("event") == "policy_snapshot":
                tasks.append({"arm": arm["name"], "mode": arm["mode"],
                              "traversals": arm["traversals"], "seed": arm["seed"],
                              "label": event["label"],
                              "snapshot_min": int(event["label"].removesuffix("m")),
                              "iteration": event["iteration"],
                              "policy_dir": str(arm_dir(root, arm) / "snapshots" / event["label"] / "average_policy")})
    return sorted(tasks, key=lambda task: (task["snapshot_min"], task["arm"]))


def hand_off_completed_evaluations(root: Path, arms: list[dict], rows: list[dict],
                                   final_minute: float) -> None:
    """Let the original end-of-run evaluator skip already checked snapshots.

    Keep the final snapshot for that evaluator. This avoids both writing to the
    same result file at the end of training and repeating all earlier work.
    """
    for arm in arms:
        path = root / arm["name"] / "exact_evaluations.jsonl"
        existing = {(r["mode"], r["seed"], r["snapshot_min"])
                    for r in read_jsonl(path)}
        additions = []
        for row in rows:
            key = (row["mode"], row["seed"], row["snapshot_min"])
            if row["arm"] != arm["name"] or row["snapshot_min"] >= final_minute or key in existing:
                continue
            additions.append({field: row[field] for field in
                              ("mode", "seed", "snapshot_min", "p_first", "p_second",
                               "exploitability", "evaluation_s", "policy_dir")})
            existing.add(key)
        if additions:
            with path.open("a", encoding="utf-8") as handle:
                for row in additions:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()


def evaluate_one(task: dict) -> dict:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    cmd = [sys.executable, str(SINGLE_EVAL), task["policy_dir"]]
    if shutil.which("nice"):
        cmd = ["nice", "-n", "10", *cmd]
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True,
                            text=True, timeout=300)
    return {**task, **json.loads(result.stdout.strip().splitlines()[-1]),
            "utc": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--reference-root", type=Path,
                        help="Overlay exact curves from this completed run")
    parser.add_argument("--extra-run", type=Path,
                        help="Overlay exact-g continuation evaluations from this run directory")
    parser.add_argument("--tabular-fork", type=Path,
                        help="Overlay saved exact evaluations from a tabular regret fork")
    parser.add_argument("--fork-start-min", type=float, default=300,
                        help="Source checkpoint's measured training minute")
    args = parser.parse_args()
    if args.eval_workers < 1 or args.eval_workers > 4:
        parser.error("eval-workers must be between 1 and 4")
    root = args.run_root.resolve()
    reference_root = args.reference_root.resolve() if args.reference_root else None
    extra_run = args.extra_run.resolve() if args.extra_run else None
    tabular_fork = args.tabular_fork.resolve() if args.tabular_fork else None
    manifest = json.loads((root / "parallel_manifest.json").read_text())
    extra_arms_path = root / "extra_arms.json"
    extra_arms = (json.loads(extra_arms_path.read_text(encoding="utf-8"))
                  if extra_arms_path.exists() else [])
    # The N/K arms are archived in the experiment report and omitted from this
    # live dashboard; keep the visit-count and conditional runs visible.
    arms = [arm for arm in [*manifest["arms"], *extra_arms]
            if arm.get("reach_mode") != "visit_fraction"]
    final_minute = target_minutes(root, arms)
    activity = {"active": {}, "failures": {}}
    activity_lock = threading.Lock()
    server = serve(root, arms, args.port, activity, activity_lock,
                   tabular_fork, args.fork_start_min)
    print(f"dashboard: http://127.0.0.1:{args.port}/", flush=True)
    result_path = root / "live_exact.jsonl"
    if result_path.exists() or reference_root is not None:
        plot_rows(root, arms, read_jsonl(result_path), exact=True,
                  reference_root=reference_root, extra_run=extra_run,
                  tabular_fork=tabular_fork, fork_start_min=args.fork_start_min)
    if (root / "resource_history.jsonl").exists():
        plot_resources(root)
    active = {}
    failures = {}
    last_progress = 0.0
    last_extra_plot = 0.0
    last_resources = time.monotonic() - 25
    previous_counters = resource_counters(root)
    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
        try:
            while True:
                rows = read_jsonl(result_path)
                hand_off_completed_evaluations(root, arms, rows, final_minute)
                done = {(r["arm"], r["label"]) for r in rows}
                for future, task in list(active.items()):
                    if not future.done():
                        continue
                    del active[future]
                    key = (task["arm"], task["label"])
                    with activity_lock:
                        activity["active"].pop(key, None)
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures[key] = failures.get(key, 0) + 1
                        with activity_lock:
                            activity["failures"][key] = failures[key]
                        print(f"evaluation failed {key}, retry {failures[key]}: {exc}", flush=True)
                        continue
                    with result_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result) + "\n")
                        handle.flush()
                    rows.append(result)
                    hand_off_completed_evaluations(root, arms, rows, final_minute)
                    done.add(key)
                    plot_rows(root, arms, rows, exact=True, reference_root=reference_root,
                              extra_run=extra_run, tabular_fork=tabular_fork,
                              fork_start_min=args.fork_start_min)
                    print(f"exact {task['arm']} {task['label']}: {result['exploitability']:.6f} "
                          f"in {result['evaluation_s']:.1f}s", flush=True)
                running = {(t["arm"], t["label"]) for t in active.values()}
                for task in completed_snapshots(root, arms):
                    key = (task["arm"], task["label"])
                    if len(active) >= args.eval_workers:
                        break
                    if key not in done and key not in running and failures.get(key, 0) < 3:
                        active[pool.submit(evaluate_one, task)] = task
                        running.add(key)
                        with activity_lock:
                            activity["active"][key] = time.monotonic()
                if time.monotonic() - last_resources > 30:
                    current_counters = resource_counters(root)
                    resource = resource_sample(previous_counters, current_counters)
                    previous_counters = current_counters
                    with (root / "resource_history.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(resource) + "\n")
                    plot_resources(root)
                    last_resources = time.monotonic()
                if time.monotonic() - last_progress > 45:
                    plot_rows(root, arms, rows, exact=False)
                    last_progress = time.monotonic()
                if (extra_run is not None or tabular_fork is not None) and time.monotonic() - last_extra_plot > 45:
                    plot_rows(root, arms, rows, exact=True, reference_root=reference_root,
                              extra_run=extra_run, tabular_fork=tabular_fork,
                              fork_start_min=args.fork_start_min)
                    last_extra_plot = time.monotonic()
                time.sleep(5)
        except KeyboardInterrupt:
            print("monitor stopped; training continues independently", flush=True)
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
