#!/usr/bin/env python3
"""Read-only live dashboard for the fresh O/E/S/N longitudinal run."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>18-claim O/E/S/N audit</title>
<style>body{font:15px system-ui,sans-serif;max-width:1350px;margin:auto;padding:1rem;color:#202a38;background:#f5f7fb}
.card{background:#fff;border:1px solid #dce2ec;border-radius:9px;padding:1rem;margin:1rem 0}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:.35rem .5rem;text-align:left;border-bottom:1px solid #e5e8ee}small{color:#5d6570}</style></head>
<body><h1>18-claim: O/E/S/N through training</h1><p id="status">Loading...</p>
<div class="card"><h2>Exact exploitability</h2><small>Average and current strategy at each 15-minute checkpoint; lower is better. Measured training time excludes audit/evaluation overhead.</small><img id="exploit"></div>
<div class="card"><h2>One-step policy distances</h2><small>O=old, E=exact conditional update, S=sampled target, N=fitted. E–N below O–E means the fit is closer to the exact one-step target under that weighting. These are diagnostics, not exploitability.</small><img id="distances"></div>
<div class="card"><h2>Does a better one-step fit predict subsequent improvement?</h2><small>Each point pairs E–N / O–E at one checkpoint with the change in average-policy exploitability at the next. Negative vertical change is improvement. Correlation is descriptive, not causal.</small><img id="predict"></div>
<div class="card"><h2>Recent checkpoints</h2><table><thead><tr><th>Train min</th><th>Iteration</th><th>Avg exploit.</th><th>Current exploit.</th><th>Visited E–N / O–E</th><th>Next avg change</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>
async function refresh(){
 try{
  const r=await fetch('/api/status',{cache:'no-store'});const data=await r.json();
  document.getElementById('status').textContent=data.message;
  const body=document.getElementById('rows');body.replaceChildren();
  for(const x of data.rows.slice(-12).reverse()){
   const tr=document.createElement('tr');
   for(const v of [x.training_min.toFixed(1),x.iteration,x.average.toFixed(5),x.current.toFixed(5),x.ratio==null?'—':x.ratio.toFixed(2),x.next_change==null?'—':x.next_change.toFixed(5)]){
    const td=document.createElement('td');td.textContent=v;tr.appendChild(td)
   }body.appendChild(tr)
  }
  for(const name of ['exploit','distances','predict'])document.getElementById(name).src='/'+name+'.png?t='+Date.now();
 }catch(e){document.getElementById('status').textContent='Waiting for results: '+e}
}refresh();setInterval(refresh,20000);
</script></body></html>"""


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def plot(root: Path, monitors: list[dict]) -> None:
    if not monitors:
        return
    x = [r["training_min"] for r in monitors]
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    for policy, color in (("average", "#1765a3"), ("current", "#c45e18")):
        ax.plot(x, [r[policy]["exploitability"] for r in monitors],
                marker="o", label=policy, color=color)
    ax.set(xlabel="Measured training minutes", ylabel="Exact exploitability")
    ax.set_yscale("log"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(root / "exploit.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    weightings = (("mean_tv", "Equal among visited sets"),
                  ("inv_neg_log_reach_weighted_tv", "1 / [-log q], all sets"),
                  ("reach_weighted_tv", "Exact q, all sets"))
    for ax, (metric, title) in zip(axes, weightings):
        scope = "visited" if metric == "mean_tv" else "all"
        for pair, label, color in (
            ("old_vs_exact_g", "O–E signal", "#444444"),
            ("exact_g_vs_sampled", "E–S sampling", "#16816e"),
            ("sampled_vs_fitted", "S–N fitting", "#d66a1d"),
            ("exact_g_vs_fitted", "E–N final", "#7a43a4")):
            ax.plot(x, [r["audit"][scope][pair][metric] for r in monitors],
                    marker="o", label=label, color=color)
        ax.set(xlabel="Measured training minutes", title=title)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean total-variation distance")
    axes[-1].legend(fontsize=8)
    fig.savefig(root / "distances.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    for row, nxt in zip(monitors, monitors[1:]):
        audit = row["audit"]["visited"]
        signal = audit["old_vs_exact_g"]["mean_tv"]
        if signal <= 0:
            continue
        ratio = audit["exact_g_vs_fitted"]["mean_tv"] / signal
        delta = nxt["average"]["exploitability"] - row["average"]["exploitability"]
        ax.scatter(ratio, delta, color="#7a43a4")
        ax.annotate(str(round(row["training_min"])), (ratio, delta), xytext=(3, 3),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#555", linewidth=1)
    ax.axvline(1, color="#999", linestyle="--", linewidth=1)
    ax.set(xlabel="Visited E–N / O–E TV at this checkpoint",
           ylabel="Next-checkpoint average exploitability change")
    ax.grid(alpha=.25)
    fig.savefig(root / "predict.png", dpi=130); plt.close(fig)


def status(root: Path) -> dict:
    monitors = sorted(read_rows(root / "monitors.jsonl"), key=lambda r: r["training_min"])
    training = read_rows(root / "training.jsonl")
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    rows = []
    for row, nxt in zip(monitors, monitors[1:] + [None]):
        audit = row["audit"]["visited"]
        signal = audit["old_vs_exact_g"]["mean_tv"]
        rows.append({"training_min": row["training_min"], "iteration": row["iteration"],
                     "average": row["average"]["exploitability"],
                     "current": row["current"]["exploitability"],
                     "ratio": audit["exact_g_vs_fitted"]["mean_tv"] / signal if signal else None,
                     "next_change": (nxt["average"]["exploitability"]
                                     - row["average"]["exploitability"]) if nxt else None})
    return {"message": f"{len(monitors)} audited checkpoints · "
            f"training {training[-1]['training_min']:.1f} min / iter {training[-1]['iteration']}"
            if training else "Waiting for training",
            "state": state.get("status", "starting"), "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--port", type=int, default=8767)
    args = p.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data, mime = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api/status":
                data, mime = json.dumps(status(root)).encode(), "application/json"
            elif path in ("/exploit.png", "/distances.png", "/predict.png"):
                file = root / path[1:]
                if not file.exists():
                    self.send_error(404, "Awaiting first audit")
                    return
                data, mime = file.read_bytes(), "image/png"
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard http://127.0.0.1:{args.port}", flush=True)
    def update():
        while True:
            try:
                rows = sorted(read_rows(root / "monitors.jsonl"), key=lambda r: r["training_min"])
                if rows:
                    plot(root, rows)
            except Exception as exc:
                print("plot refresh failed:", exc, flush=True)
            time.sleep(30)
    threading.Thread(target=update, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
