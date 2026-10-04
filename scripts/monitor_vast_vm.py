#!/usr/bin/env python3
"""Small overview dashboard for VM health and recently active experiments.

Read-only with respect to experiment artifacts. It samples /proc and nvidia-smi,
and serves on loopback so the page is reachable only through an SSH tunnel.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VM and experiment monitor</title><style>
body{font:14px system-ui,sans-serif;max-width:1700px;margin:1.4rem auto;padding:0 1rem;color:#182132;background:#f4f6fa}
h1{font-size:1.6rem;margin:.3rem 0}.card{background:#fff;border:1px solid #dce3ec;border-radius:10px;padding:1rem;margin:1rem 0}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:.7rem}.metric{background:#f5f7fb;padding:.7rem;border-radius:8px;color:#536176}
.metric b{display:block;font-size:1.25rem;color:#182132;font-variant-numeric:tabular-nums}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.55rem .45rem;border-bottom:1px solid #e8ecf1;white-space:nowrap}th{background:#f7f9fc;position:sticky;top:0}
.scroll{overflow:auto;max-height:65vh}small,.muted{color:#66758a}.running{color:#087f5b;font-weight:650}.stopped{color:#66758a}.error{color:#ad2635}
</style></head><body><h1>VM resource and experiment monitor</h1>
<p id="summary" class="muted">Connecting…</p><p id="error" class="error"></p>
<div class="card"><div class="metrics" id="metrics"></div><h2>CPU, RAM and disk</h2><img id="host" alt="VM CPU, available RAM and disk history"></div>
<div class="card"><h2>GPU</h2><img id="gpu" alt="GPU utilization and memory history"></div>
<div class="card"><h2>Recent experiments</h2><small>Active jobs and runs updated in the last 48 hours. Exact exploitability excludes approximate-BR results. Checkpoint size is the largest resumable .pt checkpoint found for that arm.</small>
<div class="scroll"><table><thead><tr><th>Experiment</th><th>Arm</th><th>Status</th><th>Training min</th><th>Iteration</th><th>Latest exact</th><th>Evaluations</th><th>Checkpoint GiB</th><th>Updated</th></tr></thead><tbody id="runs"></tbody></table></div></div>
<small>Refreshes every 15 seconds. This page is served on VM loopback and should be reached through SSH local forwarding.</small>
<script>
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();
document.getElementById('summary').textContent=`${s.host} · ${s.active} trainers · ${s.fit_workers} fit workers alive · ${s.recent} recent arms · updated ${s.updated_utc}`;
const items=[['CPU',`${s.cpu_percent.toFixed(1)}% · ${s.logical_cpus} logical`],['Load average',s.load_average.map(x=>x.toFixed(1)).join(' / ')],['RAM available',`${s.mem_available_gib.toFixed(1)} / ${s.mem_total_gib.toFixed(1)} GiB`],['Disk free',`${s.disk_free_gib.toFixed(1)} / ${s.disk_total_gib.toFixed(1)} GiB`],['GPU',s.gpu.name||'not detected'],['GPU utilization',s.gpu.utilization==null?'—':`${s.gpu.utilization}%`],['GPU memory',s.gpu.used_gib==null?'—':`${s.gpu.used_gib.toFixed(1)} / ${s.gpu.total_gib.toFixed(1)} GiB`]];
const m=document.getElementById('metrics');m.replaceChildren(...items.map(([k,v])=>{const d=document.createElement('div');d.className='metric';const b=document.createElement('b');b.textContent=v;d.append(b,k);return d}));
const body=document.getElementById('runs');body.replaceChildren();for(const a of s.runs){const tr=document.createElement('tr');
for(const v of [a.experiment,a.arm,a.status,fmt(a.training_min,1),a.iteration??'—',fmt(a.latest_exact,6),a.evaluations,fmt(a.checkpoint_gib,2),a.updated_utc||'—']){const td=document.createElement('td');td.textContent=String(v);if(v==='running')td.className='running';tr.append(td)}body.append(tr)}
const t=Date.now();document.getElementById('host').src='/host.png?t='+t;document.getElementById('gpu').src='/gpu.png?t='+t;document.getElementById('error').textContent='';
}catch(e){document.getElementById('error').textContent='Dashboard refresh failed: '+e}}
function fmt(x,n){return x==null?'—':Number(x).toFixed(n)}refresh();setInterval(refresh,15000);
</script></body></html>"""

INK, MUTED, GRID = "#1f2937", "#64748b", "#e2e8f0"
HISTORY_LOCK = threading.Lock()
COUNTS: dict[str, tuple[int, int]] = {}


def tail_jsonl(path: Path) -> list[dict]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 131072))
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return out


def count_lines(path: Path) -> int:
    key = str(path)
    try:
        size = path.stat().st_size
        old_size, old_count = COUNTS.get(key, (-1, 0))
        if old_size < 0 or size < old_size:
            count = 0
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    count += chunk.count(b"\n")
            COUNTS[key] = (size, count)
            return count
        if size > old_size:
            count = old_count
            with path.open("rb") as f:
                f.seek(old_size)
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    count += chunk.count(b"\n")
            COUNTS[key] = (size, count)
        return COUNTS[key][1]
    except OSError:
        return 0


def last_row(path: Path) -> dict:
    rows = tail_jsonl(path)
    return rows[-1] if rows else {}


def proc_snapshot() -> tuple[dict, list[dict]]:
    stat = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ticks = [int(v) for v in stat]
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            k, v, *_ = line.split()
            mem[k[:-1]] = int(v) * 1024
    procs = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            if not cmd:
                continue
            cwd = os.readlink(proc / "cwd")
            pstat = proc.joinpath("stat").read_text().split(") ", 1)[1].split()
            procs.append({"pid": int(proc.name), "cmd": cmd, "cwd": cwd,
                          "cpu_ticks": int(pstat[11]) + int(pstat[12])})
        except (OSError, IndexError, ValueError):
            continue
    return {"ticks": ticks, "mem_total": mem.get("MemTotal:", mem.get("MemTotal", 0)),
            "mem_available": mem.get("MemAvailable:", mem.get("MemAvailable", 0))}, procs


def gpu_reading() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=True, timeout=5)
        name, util, used, total = [x.strip() for x in result.stdout.splitlines()[0].split(",", 3)]
        return {"name": name, "utilization": int(util), "used_gib": int(used) / 1024,
                "total_gib": int(total) / 1024}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"name": None, "utilization": None, "used_gib": None, "total_gib": None}


def is_eval_file(path: Path) -> bool:
    name = path.name.lower()
    return ("eval" in name or "exact" in name) and "br_" not in name and "approx" not in name


def active_process_roles(processes: list[dict], artifacts: list[Path]) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    train_files = list(training_files(artifacts))
    for p in processes:
        cmd = p["cmd"]
        if not any(token in cmd for token in ("run_cfr", "train", "trainer", "deep_cfr")):
            continue
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        script_index = next((i for i, token in enumerate(tokens) if token.endswith(".py")), None)
        script_name = Path(tokens[script_index]).name.lower() if script_index is not None else ""
        if "audit" in script_name:
            mode = "audit"
        elif "eval" in script_name:
            mode = "eval"
        elif "monitor" in script_name:
            mode = "monitor"
        elif script_index is not None and script_index + 1 < len(tokens) and tokens[script_index + 1] in {"fit", "eval", "audit", "train"}:
            mode = tokens[script_index + 1]
        else:
            mode = "train"
        if mode in {"eval", "audit"}:
            continue
        arm_arg = None
        output_roots = []
        for i, token in enumerate(tokens):
            if token == "--arm" and i + 1 < len(tokens):
                arm_arg = tokens[i + 1]
            elif token.startswith("--arm="):
                arm_arg = token.split("=", 1)[1]
            elif token in {"--output-root", "--run-dir", "--output-dir"} and i + 1 < len(tokens):
                output_roots.append(Path(tokens[i + 1]))
            elif token.startswith(("--output-root=", "--run-dir=", "--output-dir=")):
                output_roots.append(Path(token.split("=", 1)[1]))
        for candidate in train_files:
            arm_dir = candidate.parent
            try:
                artifact_root = next(root for root in artifacts if arm_dir.is_relative_to(root))
                rel = str(arm_dir.relative_to(artifact_root))
            except ValueError:
                continue
            exact_path = rel and (rel in cmd or rel.replace("\\", "/") in cmd)
            arm_root_match = (arm_arg == arm_dir.name and any(
                root.resolve() == arm_dir.parent.resolve() for root in output_roots))
            run_dir_match = any(root.resolve() == arm_dir.resolve() for root in output_roots)
            experiment_root_match = (arm_arg is None and any(
                arm_dir.is_relative_to(root.resolve()) for root in output_roots))
            if exact_path or arm_root_match or run_dir_match or experiment_root_match:
                role = "fit worker" if mode == "fit" else "training"
                roles.setdefault(str(arm_dir), set()).add(role)
    return roles


def training_files(artifacts: list[Path]):
    # Policy snapshots and checkpoints can contain many nested files; the
    # training logs live at arm roots, so don't walk those bulk directories.
    skip = {"snapshots", "policy_snapshots", "average_policy", "current_policy",
            "final_policy", "__pycache__"}
    seen = set()
    for artifact_root in artifacts:
        if not artifact_root.exists():
            continue
        for base, dirs, files in os.walk(artifact_root):
            dirs[:] = [name for name in dirs if name not in skip]
            if "training.jsonl" in files:
                path = (Path(base) / "training.jsonl").resolve()
                if str(path) not in seen:
                    seen.add(str(path))
                    yield path


def checkpoint_size(arm_dir: Path, latest: dict) -> float | None:
    claimed = latest.get("checkpoint_path") or latest.get("checkpoint")
    candidates = [Path(claimed)] if claimed else []
    for directory in (arm_dir, arm_dir.parent):
        for name in ("latest_checkpoint.pt", "checkpoint.pt", "trainer_state.pt"):
            candidates.append(directory / name)
    sizes = []
    for path in candidates:
        try:
            if path.is_file():
                sizes.append(path.stat().st_size / 1024**3)
        except OSError:
            pass
    return max(sizes) if sizes else None


def run_rows(artifacts: list[Path], processes: list[dict]) -> list[dict]:
    active = active_process_roles(processes, artifacts)
    now = time.time()
    rows = []
    for train_file in training_files(artifacts):
        arm_dir = train_file.parent
        try:
            age_h = (now - train_file.stat().st_mtime) / 3600
        except OSError:
            continue
        process_roles = active.get(str(arm_dir), set())
        running = "training" in process_roles
        train = last_row(train_file)
        summary = {}
        for filename in ("summary.json", "state.json", "controller_state.json"):
            try:
                summary.update(json.loads((arm_dir / filename).read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        stored_status = str(summary.get("status") or "inactive")
        if running:
            status = "running"
        elif "fit worker" in process_roles:
            status = f"{stored_status} · fit worker alive" if stored_status != "inactive" else "fit worker alive"
        else:
            status = stored_status
        eval_paths = []
        try:
            # Keep the count arm-specific. Shared run-level evaluation logs
            # often contain every sibling arm and would otherwise be double-counted.
            eval_paths.extend(p for p in arm_dir.iterdir()
                              if p.is_file() and p.suffix == ".jsonl" and is_eval_file(p))
        except OSError:
            pass
        eval_paths = list({str(p): p for p in eval_paths}.values())
        eval_count = sum(count_lines(path) for path in eval_paths)
        eval_rows = [row for path in eval_paths for row in tail_jsonl(path)]
        exact_rows = [r for r in eval_rows if isinstance(r.get("exploitability"), (int, float))
                      and r.get("kind", "average") in ("average", "avg", None)
                      and not r.get("approximate", False)]
        if exact_rows:
            exact_rows.sort(key=lambda r: (r.get("measured_training_min", r.get("measured_fork_min", r.get("snapshot_min", 0))),
                                           r.get("iteration", 0)))
            latest_exact = exact_rows[-1]["exploitability"]
        else:
            latest_exact = None
        minutes = next((train.get(k) for k in ("measured_training_min", "measured_fork_min", "training_min")
                        if isinstance(train.get(k), (int, float))), None)
        if minutes is None:
            seconds = next((train.get(k) for k in ("measured_training_s", "measured_fork_s", "training_s")
                            if isinstance(train.get(k), (int, float))), None)
            minutes = seconds / 60 if seconds is not None else None
        iteration = train.get("iteration", summary.get("iteration"))
        try:
            artifact_root = next(root for root in artifacts if arm_dir.is_relative_to(root))
            experiment = f"{artifact_root.parent.name}/{arm_dir.parent.relative_to(artifact_root)}"
        except ValueError:
            experiment = arm_dir.parent.name
        updated = datetime.fromtimestamp(train_file.stat().st_mtime, timezone.utc).isoformat(timespec="minutes")
        terminal = any(word in stored_status.lower() for word in
                       ("complete", "target_reached", "finished", "stopped", "failed"))
        if not (running or "fit worker" in process_roles or (not terminal and age_h <= 6)):
            continue
        rows.append({"experiment": experiment, "arm": arm_dir.name, "status": status,
                     "training_min": minutes, "iteration": iteration,
                     "latest_exact": latest_exact, "evaluations": eval_count,
                     "checkpoint_gib": checkpoint_size(arm_dir, {**summary, **train}), "updated_utc": updated,
                     "age_h": age_h})
    rows.sort(key=lambda r: (r["status"] != "running", r["experiment"], r["arm"]))
    return rows


def host_percent(prev: dict, cur: dict, elapsed: float) -> float:
    before, after = prev["ticks"], cur["ticks"]
    total = sum(after) - sum(before)
    idle = (after[3] + after[4]) - (before[3] + before[4])
    return max(0.0, min(100.0, 100 * (1 - idle / total))) if total > 0 else 0.0


def append_history(path: Path, row: dict) -> None:
    with HISTORY_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        # At a 15-second sample interval this is several days of history. Trim
        # only occasionally; do not rewrite the history file on every poll.
        if path.stat().st_size > 8 * 1024 * 1024:
            with path.open(encoding="utf-8") as f:
                recent = deque(f, maxlen=10080)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(recent), encoding="utf-8")
            tmp.replace(path)


def read_history(path: Path, count: int = 2880) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=count)
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def plot_history(state_dir: Path, rows: list[dict]) -> None:
    if len(rows) < 2:
        return
    dates = [datetime.fromisoformat(r["utc"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), layout="constrained")
    axes[0].plot(dates, [r["cpu_percent"] for r in rows], color="#0072b2", lw=1.8)
    axes[0].set_ylabel("CPU busy (%)")
    axes[1].plot(dates, [r["mem_available_gib"] for r in rows], color="#009e73", lw=1.8)
    axes[1].set_ylabel("Available RAM (GiB)")
    axes[2].plot(dates, [r["disk_free_gib"] for r in rows], color="#cc79a7", lw=1.8)
    axes[2].set_ylabel("Free disk (GiB)")
    for ax in axes:
        ax.grid(alpha=.22); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle("VM CPU, memory and disk")
    tmp = state_dir / "host.png.tmp.png"; fig.savefig(tmp, dpi=125); plt.close(fig); tmp.replace(state_dir / "host.png")

    gpu_rows = [r for r in rows if r.get("gpu_utilization") is not None]
    if gpu_rows:
        gd = [datetime.fromisoformat(r["utc"]) for r in gpu_rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.1), layout="constrained")
        axes[0].plot(gd, [r["gpu_utilization"] for r in gpu_rows], color="#0072b2", lw=1.8)
        axes[0].set_ylabel("GPU utilization (%)"); axes[0].set_ylim(0, 100)
        axes[1].plot(gd, [r["gpu_used_gib"] for r in gpu_rows], color="#d55e00", lw=1.8)
        axes[1].set_ylabel("GPU memory used (GiB)")
        for ax in axes:
            ax.grid(alpha=.22); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.suptitle("GPU utilization and memory")
        tmp = state_dir / "gpu.png.tmp.png"; fig.savefig(tmp, dpi=125); plt.close(fig); tmp.replace(state_dir / "gpu.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, nargs="+", default=[Path("artifacts")])
    parser.add_argument("--state-dir", type=Path, default=Path("artifacts/vm_overview"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    artifacts = [path.resolve() for path in args.artifacts]; state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    latest: dict = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                payload, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api/status":
                with lock: payload_obj = dict(latest)
                payload, mime = json.dumps(payload_obj).encode(), "application/json"
            elif path in ("/host.png", "/gpu.png"):
                image_path = state_dir / path[1:]
                if not image_path.exists(): self.send_error(404); return
                payload, mime = image_path.read_bytes(), "image/png"
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"VM overview dashboard: http://127.0.0.1:{args.port}", flush=True)
    previous, _ = proc_snapshot()
    last_run_scan = 0.0
    last_plot = 0.0
    cached_runs: list[dict] = []
    while True:
        time.sleep(args.interval)
        current, processes = proc_snapshot()
        elapsed = args.interval
        cpu = host_percent(previous, current, elapsed); previous = current
        memory = current["mem_available"] / 1024**3
        total_memory = current["mem_total"] / 1024**3
        disk = shutil.disk_usage(artifacts[0])
        gpu = gpu_reading()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {"utc": stamp, "cpu_percent": cpu, "mem_available_gib": memory,
               "disk_free_gib": disk.free / 1024**3,
               "gpu_utilization": gpu["utilization"], "gpu_used_gib": gpu["used_gib"]}
        append_history(state_dir / "resource_history.jsonl", row)
        if time.monotonic() - last_plot >= 60:
            history = read_history(state_dir / "resource_history.jsonl")
            plot_history(state_dir, history)
            last_plot = time.monotonic()
        now = time.monotonic()
        if now - last_run_scan >= 60:
            cached_runs = run_rows(artifacts, processes)
            last_run_scan = now
        runs = cached_runs
        try: load = [round(float(x), 2) for x in os.getloadavg()]
        except (AttributeError, OSError): load = [0.0, 0.0, 0.0]
        value = {"host": os.uname().nodename, "updated_utc": stamp,
                 "cpu_percent": cpu, "logical_cpus": os.cpu_count() or 0,
                 "load_average": load, "mem_available_gib": memory,
                 "mem_total_gib": total_memory, "disk_free_gib": disk.free / 1024**3,
                 "disk_total_gib": disk.total / 1024**3, "gpu": gpu,
                 "active": sum(r["status"] == "running" for r in runs),
                 "fit_workers": sum("fit worker alive" in r["status"] for r in runs),
                 "recent": len(runs), "runs": runs}
        with lock: latest.clear(); latest.update(value)


if __name__ == "__main__":
    main()
