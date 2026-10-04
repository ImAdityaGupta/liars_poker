#!/usr/bin/env python3
"""Read-only dashboard for the seven 18-claim root-schedule/O4 arms."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlparse

ARMS = ("k1024", "k4096", "k16384", "k32768", "ramp", "ramp8m", "ramp_exact")
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Root schedules + O4</title>
<style>body{font:14px system-ui;color:#172033;background:#f4f6fa;max-width:1700px;margin:1.5rem auto;padding:0 1rem}
.card{background:white;padding:1rem;border-radius:10px;margin:1rem 0}img{width:100%}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.55rem;border-bottom:1px solid #e3e6eb}th{background:#f8fafc}small{color:#64748b}</style></head>
<body><h1>18-claim root schedules with O4 averaging</h1><p id="status">Loading…</p>
<div class="card"><h2>All seven arms</h2><table><thead><tr><th>Arm</th><th>Status</th><th>Training min</th><th>Iteration</th><th>K now</th><th>Roots (M)</th><th>O4 exact</th><th>Online exact</th><th>Pending fits</th><th>Checkpoint GiB</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card"><h2>Exact exploitability</h2><small>Solid: O4 refit. Dotted: online network. Dashed: exact average observer. Logarithmic vertical scale; evaluations begin after each arm's first 15 training minutes.</small><img id="plot" src="/comparison.png"></div>
<script>function n(x,d=3){return x==null?'—':Number(x).toFixed(d)}async function refresh(){try{let r=await fetch('/api',{cache:'no-store'});if(!r.ok)throw Error(r.status);let s=await r.json();document.getElementById('status').textContent=`${s.updated} · ${s.active} active trainers · ${s.evaluations} exact evaluations`;
document.getElementById('rows').innerHTML=s.arms.map(a=>`<tr><td>${a.arm}</td><td>${a.status}</td><td>${n(a.minute,1)}</td><td>${a.iteration??'—'}</td><td>${a.k??'—'}</td><td>${n(a.roots_m,1)}</td><td>${n(a.o4,6)}</td><td>${n(a.online,6)}</td><td>${a.pending}</td><td>${n(a.checkpoint_gib,2)}</td></tr>`).join('');document.getElementById('plot').src='/comparison.png?t='+Date.now()}catch(e){document.getElementById('status').textContent='Refresh failed: '+e}}refresh();setInterval(refresh,15000)</script></body></html>"""


def last_row(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 65536))
            for line in reversed(file.read().splitlines()):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return {}


def evals(path: Path) -> tuple[dict, int]:
    latest = {}
    count = 0
    try:
        with path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[row.get("policy_kind")] = row.get("exploitability")
                count += 1
    except OSError:
        pass
    return latest, count


def snapshot(root: Path) -> dict:
    rows = []
    total_evals = 0
    for arm in ARMS:
        directory = root / arm
        train = last_row(directory / "training.jsonl")
        scores, count = evals(directory / "evaluations.jsonl")
        total_evals += count
        summary = {}
        try:
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        checkpoint = directory / "latest_checkpoint.pt"
        pending = len(list((directory / "policy_snapshots").glob("*/FIT_INPUT.pt")))
        age = time.time() - (directory / "training.jsonl").stat().st_mtime if train else None
        status = summary.get("status") or ("running" if train and age is not None and age < 300 else "starting" if checkpoint.exists() else "queued")
        rows.append({
            "arm": arm, "status": status,
            "minute": train.get("measured_training_min", summary.get("measured_training_min")),
            "iteration": train.get("iteration", summary.get("iteration")),
            "k": train.get("roots_per_player"),
            "roots_m": train.get("cumulative_roots_per_player", summary.get("cumulative_roots_per_player", 0)) / 1e6,
            "o4": scores.get("o4"), "online": scores.get("online"),
            "pending": pending,
            "checkpoint_gib": checkpoint.stat().st_size / 1024**3 if checkpoint.exists() else None,
        })
    return {"arms": rows, "active": sum(row["status"] == "running" for row in rows),
            "evaluations": total_evals, "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    root = args.output_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data, mime = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/api":
                data, mime = json.dumps(snapshot(root)).encode("utf-8"), "application/json"
            elif path == "/comparison.png":
                image = root / "comparison.png"
                if not image.exists():
                    self.send_error(404, "First O4 evaluations are pending")
                    return
                data, mime = image.read_bytes(), "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f"Root/O4 dashboard: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
