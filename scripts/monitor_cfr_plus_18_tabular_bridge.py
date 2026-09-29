#!/usr/bin/env python3
"""Read-only live dashboard for the tabular bridge."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_cfr_plus_18_tabular_bridge import ARMS, read_jsonl


COLORS = {
    "exact": "#232a3d", "sample_reach": "#0077b6", "ignore_reach": "#d97706",
    "sample_value": "#a23d7c", "sample_both": "#168348", "conditional": "#ba3a2c",
    "exact_reach_gated": "#795548", "unit_reach_gated": "#d42f93",
}
LABELS = {
    "exact": "0 exact", "sample_reach": "1a sampled reach", "ignore_reach": "1b ignore reach",
    "sample_value": "2 sampled value", "sample_both": "3 sampled reach + value",
    "conditional": "4 conditional sampled",
    "exact_reach_gated": "2-control exact qg, visit gate",
    "unit_reach_gated": "1c exact g, visit gate",
}
VIEWS = {
    "all": "All 14 arms",
    "original": "Original eight arms · K=1,024",
    "roots": "Arms 3 and 4 · K=128, 256, 512, 1,024",
}


def discovered_arms(root: Path) -> list[dict]:
    """Include the original bridge and any smaller-root continuations."""
    sources = [root, *(root / "low_roots").glob("k*/")]
    found = []
    for source in sources:
        run_path = source / "run.json"
        if not run_path.exists():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        roots = int(run.get("roots", 1024))
        for arm in run.get("arms", ARMS):
            small = source != root
            found.append({
                "name": f"{arm}_k{roots}" if small else arm,
                "label": f"{LABELS[arm]} · K={roots}" if small else LABELS[arm],
                "arm": arm,
                "roots": roots,
                "small": small,
                "dir": source / arm,
                "target_minutes": float(run.get("target_minutes", 300)),
            })
    return found
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>18-claim tabular bridge</title>
<style>body{font:15px system-ui,sans-serif;background:#f5f7fb;color:#1c2738;margin:1.5rem auto;
max-width:1500px;padding:0 1rem}.card{background:white;border:1px solid #dfe5eb;border-radius:10px;
padding:1rem;margin:1rem 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:.55rem;border-bottom:1px solid #e4e9ef}th{background:#f6f8fb}
img{max-width:100%;width:100%}small{color:#59687b}.warn{color:#a72121}
.plots{display:grid;grid-template-columns:1fr;gap:1rem}.plots .card{margin:0}
select{font:inherit;padding:.3rem;border:1px solid #aebccb;border-radius:5px}
</style></head><body>
<h1>18-claim tabular CFR+ bridge</h1><p id="summary">Loading...</p>
<div class="card"><label>Policy shown: <select id="kind">
<option value="average">Average policy</option><option value="current">Current policy</option>
</select></label><small> Exact exploitability is lower when the policy is better. Both x-axis panels use a logarithmic y-axis.</small></div>
<div class="plots">
<div class="card"><h2>All 14 arms</h2><small>Overview. The two highlighted method families are easier to compare in the third view.</small><img id="plot-all" alt="All bridge arms"></div>
<div class="card"><h2>Original eight arms</h2><small>The first six bridge updates and the two visit-gated controls, all at K=1,024.</small><img id="plot-original" alt="Original bridge arms"></div>
<div class="card"><h2>Sampled updates by root count</h2><small>Arm 3: sampled reach and value; arm 4: conditional sampled value. Each has K=128, 256, 512, and 1,024.</small><img id="plot-roots" alt="Arms 3 and 4 at four root counts"></div>
</div>
<div class="card"><h2>Arms</h2><table><thead><tr><th>Arm</th><th>Status</th>
<th>Training minutes</th><th>Iteration</th><th>Latest average</th><th>Latest current</th>
<th>Checkpoint</th><th>Last update</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card"><h2>Machine</h2><div id="machine"></div></div>
<small>Refreshes every 20 seconds. Only numeric evaluation results and a rolling checkpoint are stored per arm.</small>
<script>
async function refresh(){try{const r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);
const s=await r.json();document.getElementById('summary').textContent=`${s.complete} complete · ${s.running} running · ${s.evaluations} exact evaluations · ${s.updated_utc}`;
const body=document.getElementById('rows');body.replaceChildren();for(const a of s.arms){const tr=document.createElement('tr');
for(const v of [a.label,a.status,a.minutes.toFixed(1)+' / '+a.target_minutes,a.iteration,
a.average==null?'—':a.average.toFixed(6),a.current==null?'—':a.current.toFixed(6),
a.checkpoint_gib.toFixed(2)+' GiB',a.updated_utc||'—']){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}body.appendChild(tr)}
const m=s.machine;document.getElementById('machine').innerHTML=`Disk free: <b class="${m.disk_free_gib<12?'warn':''}">${m.disk_free_gib.toFixed(1)} GiB</b> / ${m.disk_total_gib.toFixed(1)} GiB · RAM available: ${m.ram_available_gib.toFixed(1)} GiB`;
for(const view of ['all','original','roots'])document.getElementById('plot-'+view).src='/plot?view='+view+'&kind='+document.getElementById('kind').value+'&t='+Date.now();
}catch(e){document.getElementById('summary').textContent='Dashboard error: '+e}}
document.getElementById('kind').onchange=refresh;refresh();setInterval(refresh,20000);
</script></body></html>"""


def machine_status(root: Path) -> dict:
    disk = shutil.disk_usage(root)
    available_kib = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break
    except OSError:
        pass
    return {"disk_free_gib": disk.free / 1024**3,
            "disk_total_gib": disk.total / 1024**3,
            "ram_available_gib": available_kib / 1024**2}


def status(root: Path) -> dict:
    arms = []
    evaluations = 0
    for config in discovered_arms(root):
        arm_dir = config["dir"]
        state_path = arm_dir / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        rows = read_jsonl(arm_dir / "evaluations.jsonl")
        evaluations += len(rows)
        latest = {kind: next((float(r["exploitability"]) for r in reversed(rows)
                              if r["kind"] == kind), None) for kind in ("average", "current")}
        cp = arm_dir / "latest_checkpoint.npz"
        arms.append({"name": config["name"], "label": config["label"],
                     "target_minutes": config["target_minutes"],
                     "status": state.get("status", "starting"),
                     "minutes": float(state.get("measured_training_min", 0)),
                     "iteration": int(state.get("iteration", 0)),
                     "average": latest["average"], "current": latest["current"],
                     "checkpoint_gib": cp.stat().st_size / 1024**3 if cp.exists() else 0.0,
                     "updated_utc": state.get("updated_utc")})
    return {"arms": arms,
            "complete": sum(a["status"] == "complete" for a in arms),
            "running": sum(a["status"] == "running" for a in arms),
            "evaluations": evaluations, "machine": machine_status(root),
            "updated_utc": datetime.now(timezone.utc).isoformat()}


def plot(root: Path, kind: str, view: str = "all") -> bytes:
    if kind not in ("average", "current"):
        raise ValueError(kind)
    if view not in VIEWS:
        raise ValueError(view)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained")
    configs = discovered_arms(root)
    if view == "original":
        configs = [c for c in configs if not c["small"]]
    elif view == "roots":
        configs = [c for c in configs if c["arm"] in {"sample_both", "conditional"}]
    for config in configs:
        arm = config["arm"]
        rows = read_jsonl(config["dir"] / "evaluations.jsonl")
        for policy_kind in (kind,):
            group = sorted((r for r in rows if r["kind"] == policy_kind),
                           key=lambda r: r["measured_training_min"])
            if not group:
                continue
            y = [max(float(r["exploitability"]), 1e-8) for r in group]
            style = "--" if arm == "conditional" else "-"
            label = config["label"]
            small = config["small"]
            color = ({128: "#8ac8b2", 256: "#48a382", 512: "#1b795c"}
                     if arm == "sample_both" else
                     {128: "#f3aa83", 256: "#e4754b", 512: "#b9452a"}).get(config["roots"], COLORS[arm]) if small else COLORS[arm]
            if view == "all" and arm not in {"sample_both", "conditional"}:
                alpha, linewidth = .52, 1.35
            else:
                alpha, linewidth = .96, 2.0
            for ax, field in zip(axes, ("measured_training_min", "iteration")):
                ax.plot([r[field] for r in group], y, color=color, linestyle=style,
                        marker="^" if arm == "sample_both" else "s" if arm == "conditional" else "o",
                        markersize=3.1, linewidth=linewidth, alpha=alpha,
                        label=label)
    for ax, xlabel in zip(axes, ("Measured training minutes", "CFR+ iteration")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Exact exploitability")
        ax.set_yscale("log")
        ax.grid(True, alpha=.25, which="both")
    fig.suptitle(f"{VIEWS[view]} · {kind} policy", fontsize=14)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=7.8 if view == "all" else 8.8,
                   loc="outside lower center", ncol=4 if view == "all" else 2,
                   frameon=False)
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=145)
    plt.close(fig)
    return out.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/cfr_plus_18_tabular_bridge"))
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    body, content_type = PAGE.encode(), "text/html; charset=utf-8"
                elif request.path == "/status":
                    body = json.dumps(status(root), allow_nan=False).encode()
                    content_type = "application/json"
                elif request.path == "/plot":
                    kind = parse_qs(request.query).get("kind", ["average"])[0]
                    view = parse_qs(request.query).get("view", ["all"])[0]
                    body, content_type = plot(root, kind, view), "image/png"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(500, str(exc))

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
