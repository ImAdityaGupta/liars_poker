#!/usr/bin/env python3
"""Read-only dashboard for the 30-claim two-arm run."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def last_jsonl(path: Path):
    if not path.exists():
        return None
    with path.open('rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 8192))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            pass
    return None


def gpu():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
    except Exception:
        return 'unavailable'


def page(root: Path) -> bytes:
    from html import escape
    rows = []
    for arm, width in (('w512', 512), ('w2048', 2048)):
        run = root / arm
        last = last_jsonl(run / 'training.jsonl') or {}
        snap_count = len(list((run / 'snapshots').glob('*/READY.json')))
        ckpt = run / 'latest_checkpoint.pt'
        ckpt_gib = ckpt.stat().st_size / 1024**3 if ckpt.exists() else 0
        summary = json.loads((run / 'summary.json').read_text()) if (run / 'summary.json').exists() else {}
        status = summary.get('status', 'training' if last else 'waiting')
        rows.append(f'<tr><td>{arm}</td><td>{width}</td><td>{status}</td>'
                    f'<td>{last.get("measured_training_min", 0):.1f}</td>'
                    f'<td>{last.get("iteration", 0)}</td>'
                    f'<td>{last.get("roots_per_player", 0)}</td>'
                    f'<td>{last.get("cumulative_roots", 0):,}</td>'
                    f'<td>{snap_count}</td><td>{ckpt_gib:.2f}</td></tr>')
    evaluations = []
    for path in sorted((root / 'evaluations').glob('*.json')):
        try:
            evaluations.append(json.loads(path.read_text()))
        except (ValueError, OSError):
            pass
    table = ''.join(f'<tr><td>{escape(e["arm"])}</td><td>{escape(e["snapshot"])}</td>'
                    f'<td>{escape(e["policy_kind"])}</td><td>{e["depth"]}</td>'
                    f'<td>{e["discovered_exploitability"]:.5f}</td>'
                    f'<td>{e["elapsed_s"]:.1f}</td></tr>' for e in evaluations[-50:])
    disk = shutil.disk_usage(root)
    failed = len(list((root / 'eval_failures').glob('*.json')))
    precise_root = root / 'precise_evaluations'
    try:
        precise = json.loads((precise_root / 'summary.json').read_text())
    except (OSError, ValueError):
        precise = []
    try:
        paired = json.loads((precise_root / 'paired.json').read_text())
    except (OSError, ValueError):
        paired = []
    precise_rows = ''.join(
        f'<tr><td>{escape(r["arm"])}</td><td>{escape(r["snapshot"])}</td>'
        f'<td>{r["discovered_exploitability"]:.5f} ± {r["half_width_95"]:.5f}</td>'
        f'<td>{r["lower_confidence_bound"]:.5f}</td>'
        f'<td>{r["games_per_seat"]:,}</td>'
        f'<td>{r["completed_shards"]}/{r["target_shards"]}</td></tr>'
        for r in sorted(precise, key=lambda r: (int(r['snapshot'][:-1]), r['arm']))[-60:]
    )
    width_pairs = [r for r in paired
                   if r['left'].startswith('w512_') and r['right'].startswith('w2048_')]
    pair_rows = ''.join(
        f'<tr><td>{escape(r["left"].split("_")[1])}</td>'
        f'<td>{r["difference_left_minus_right"]:+.5f} ± {r["half_width_95"]:.5f}</td>'
        f'<td>{r["paired_games_per_seat"]:,}</td></tr>'
        for r in width_pairs[-30:]
    )
    text = f'''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="45">
<title>30-claim CFR+</title><style>body{{font:16px system-ui;max-width:1200px;margin:2em auto;color:#17202a}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}
img{{width:100%;max-width:1100px}}code{{background:#eee;padding:3px}}</style></head><body>
<h1>30-claim CFR+ first run</h1><p>GPU: <code>{escape(gpu())} MiB used/free, utilization %</code>.
Disk free: {disk.free/1024**3:.1f} GiB. Evaluation failures: {failed}. Auto refresh: 45 s.</p>
<table><thead><tr><th>Arm</th><th>Width</th><th>Status</th><th>Train min</th><th>Iteration</th><th>K</th><th>Cumulative roots</th><th>O4 snapshots</th><th>Checkpoint GiB</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Precise depth-2 BR screen</h2><p>Each estimate averages terminal outcomes over the opponent's posterior hand,
and cycles through every responder hand. Intervals are Monte Carlo 95% intervals. Shard counts show progress toward the planned sample size.</p>
<img src="/plot.png">
<table><thead><tr><th>Arm</th><th>Snapshot</th><th>Estimate ± 95% half-width</th><th>LCB</th><th>Games / seat</th><th>Shards</th></tr></thead>
<tbody>{precise_rows}</tbody></table>
<h3>Paired width comparison</h3><p>W512 minus W2048 at the same snapshot. Positive means W512 is more exploitable; an interval containing zero leaves the comparison unresolved.</p>
<table><thead><tr><th>Snapshot</th><th>Difference ± 95% half-width</th><th>Paired games / seat</th></tr></thead><tbody>{pair_rows}</tbody></table>
<h2>Original low-sample screens</h2>
<table><thead><tr><th>Arm</th><th>Snapshot</th><th>Policy</th><th>Depth</th><th>Discovered exploitability</th><th>Eval s</th></tr></thead><tbody>{table}</tbody></table></body></html>'''
    return text.encode()


def plot(root: Path) -> bytes:
    try:
        rows = json.loads((root / 'precise_evaluations' / 'summary.json').read_text())
    except (OSError, ValueError):
        rows = []
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for arm, color in (('w512', '#2467a4'), ('w2048', '#d76b19')):
        selected = sorted((r for r in rows if r['arm'] == arm and r['kind'] == 'o4'
                           and r['completed_shards'] == r['target_shards']),
                          key=lambda r: int(r['snapshot'][:-1]))
        if selected:
            ax.errorbar([int(r['snapshot'][:-1])/60 for r in selected],
                        [r['discovered_exploitability'] for r in selected],
                        yerr=[r['half_width_95'] for r in selected],
                        marker='o', capsize=3, color=color, label=f'{arm} depth 2')
    june = [r['discovered_exploitability'] for r in rows
            if r['arm'].startswith('june_')
            and r['completed_shards'] == r['target_shards']]
    if june:
        ax.axhline(statistics.median(june), color='#6b7280', ls=':',
                   label=f'June 60m median ({len(june)} policies)')
    ax.set(xlabel='CFR+ training hours', ylabel='Discovered exploitability',
           title='O4 average policy, depth-2 responder with sampled full games')
    ax.grid(alpha=.25)
    if rows:
        ax.legend(ncol=2)
    fig.tight_layout()
    data = io.BytesIO()
    fig.savefig(data, format='png', dpi=125)
    plt.close(fig)
    return data.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8774)
    args = parser.parse_args()
    root = args.output_root.resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/plot.png':
                body, kind = plot(root), 'image/png'
            else:
                body, kind = page(root), 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    print(f'dashboard http://127.0.0.1:{args.port} for {root}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
