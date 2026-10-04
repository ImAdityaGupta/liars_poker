#!/usr/bin/env python3
"""Live dashboard for the 18-claim depth-limited BR calibration."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

LABELS = (
    '01_k4096_o4_0015', '02_k4096_online_0015', '03_k4096_o4_0060',
    '04_k4096_o4_0180', '05_k4096_online_0180', '06_k4096_o4_0450',
    '07_k4096_o4_0795', '08_k4096_o4_1140', '09_k4096_online_1140',
    '10_k1024_o4_0600', '11_exact4096_o4_1080', '12_exact4096_table_1080',
)
LAZY_ROOT = Path('/root/liars_poker/artifacts/cfr_plus_18_approx_br_lazy/main_20261003')
PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>18-claim BR calibration</title>
<style>body{font:14px system-ui;color:#172033;background:#f4f6fa;max-width:1500px;margin:1.5rem auto;padding:0 1rem}
.card{background:white;padding:1rem;border-radius:10px;margin:1rem 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:.5rem;border-bottom:1px solid #e3e6eb}th:first-child,td:first-child{text-align:left}th{background:#f8fafc}
svg{width:100%;height:420px}small{color:#64748b}.legend{display:flex;gap:1.5rem}</style></head>
<body><h1>18-claim approximate BR calibration</h1><p id="status">Loading…</p>
<div class="card"><h2>Exact value against discovered value</h2><small>Raw discovered score is p(first) + p(second) − 1; it can be negative for a weak responder. The usable lower bound is max(0, raw score). The diagonal means the responder found the exact BR. Reported values use full opponent support.</small><svg id="plot" viewBox="0 0 800 420"></svg><div class="legend" id="legend"></div></div>
<div class="card"><h2>Policy and responder results</h2><table><thead><tr><th>Policy</th><th>Exact</th><th>M1</th><th>M2 d2 ε=.001</th><th>M2 d2 ε=.0001</th><th>M2 d3 ε=.001</th><th>M2 d3 ε=.0001</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card"><h2>Lazy neural queries against the dense reference</h2><small>Only d=2 and d=3 at ε=.0001. Difference is the largest absolute difference in either seat's win probability. Times exclude policy loading.</small><table><thead><tr><th>Policy</th><th>d2 value</th><th>d2 max diff</th><th>d2 seconds</th><th>d2 network queries</th><th>d3 value</th><th>d3 max diff</th><th>d3 seconds</th><th>d3 network queries</th></tr></thead><tbody id="lazyrows"></tbody></table></div>
<div class="card"><h2>Completed job costs</h2><table><thead><tr><th>Policy</th><th>Method</th><th>Seconds</th><th>Plan nodes</th><th>Eval nodes</th><th>Pruned / search branches</th></tr></thead><tbody id="costs"></tbody></table></div>
<script>
const settings=[['d1_e0','M1','#2563eb'],['d2_e0.001','M2 d2 .001','#16a34a'],['d2_e0.0001','M2 d2 .0001','#22c55e'],['d3_e0.001','M2 d3 .001','#dc2626'],['d3_e0.0001','M2 d3 .0001','#f97316']];
function num(v,d=5){return v==null?'—':Number(v).toFixed(d)}
async function refresh(){try{const s=await (await fetch('/api',{cache:'no-store'})).json();
 document.getElementById('status').textContent=`${s.done}/60 dense settings and ${s.lazy_done}/24 lazy settings complete · refreshed ${new Date().toLocaleTimeString()}`;
 document.getElementById('rows').innerHTML=s.policies.map(p=>'<tr><td>'+p.label+'</td><td>'+num(p.exact?.exploitability)+'</td>'+settings.map(z=>'<td>'+num(p.results[z[0]]?.discovered_exploitability)+'</td>').join('')+'</tr>').join('');
 document.getElementById('lazyrows').innerHTML=s.policies.map(p=>{const cells=[p.lazy.d2,p.lazy.d3].map(r=>'<td>'+num(r?.discovered_exploitability)+'</td><td>'+num(r?Math.max(...r.seat_differences.map(Math.abs)):null,8)+'</td><td>'+num(r?.elapsed_s,1)+'</td><td>'+(r?.network_queries??'—')+'</td>').join('');return '<tr><td>'+p.label+'</td>'+cells+'</tr>'}).join('');
 document.getElementById('costs').innerHTML=s.policies.flatMap(p=>settings.filter(z=>p.results[z[0]]).map(z=>{const r=p.results[z[0]], seats=r.seats;return `<tr><td>${p.label}</td><td>${z[1]}</td><td>${num(r.elapsed_s,1)}</td><td>${seats.reduce((a,b)=>a+b.plan_nodes,0)}</td><td>${seats.reduce((a,b)=>a+b.eval_nodes,0)}</td><td>${seats.reduce((a,b)=>a+b.skipped_search_branches,0)} / ${seats.reduce((a,b)=>a+b.search_branches,0)}</td></tr>`})).join('');
 const points=s.policies.flatMap(p=>settings.filter(z=>p.exact&&p.results[z[0]]).map(z=>({x:p.exact.exploitability,y:p.results[z[0]].discovered_exploitability,color:z[2],label:p.label+' '+z[1]})));
 const max=Math.max(.005,...points.flatMap(p=>[p.x,p.y]))*1.08, min=Math.min(0,...points.map(p=>p.y))*1.1;
 const X=x=>60+680*x/max,Y=y=>360-320*(y-min)/(max-min);
 let svg=`<path d="M60 40 V360 H740" fill="none" stroke="#94a3b8"/><path d="M60 ${Y(0)} H740" fill="none" stroke="#cbd5e1"/><path d="M60 ${Y(0)} L740 ${Y(max)}" fill="none" stroke="#cbd5e1" stroke-dasharray="6 4"/><text x="300" y="405">Exact exploitability</text><text x="0" y="25">Raw discovered</text><text x="680" y="385">${num(max,3)}</text><text x="5" y="${Y(0)}">0</text>`;
 for(const p of points)svg+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="5" fill="${p.color}"><title>${p.label}: exact ${num(p.x)}, found ${num(p.y)}</title></circle>`;
 document.getElementById('plot').innerHTML=svg;document.getElementById('legend').innerHTML=settings.map(z=>`<span style="color:${z[2]}">● ${z[1]}</span>`).join('');
}catch(e){document.getElementById('status').textContent='Refresh failed: '+e}}refresh();setInterval(refresh,15000);
</script></body></html>'''


def snapshot(root: Path) -> dict:
    policies = []
    for label in LABELS:
        def read(name):
            try:
                return json.loads((root / name).read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                return None
        results = {}
        for depth, eps in ((1, '0'), (2, '0.001'), (2, '0.0001'),
                           (3, '0.001'), (3, '0.0001')):
            key = f'd{depth}_e{eps}'
            results[key] = read(f'{label}_{key}.json')
        def read_lazy(depth):
            try:
                return json.loads((LAZY_ROOT / f'{label}_d{depth}.json').read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                return None
        lazy = {'d2': read_lazy(2), 'd3': read_lazy(3)}
        policies.append({'label': label, 'exact': read(f'{label}_exact.json'),
                         'results': results, 'lazy': lazy})
    return {'policies': policies, 'exact_done': sum(p['exact'] is not None for p in policies),
            'done': sum(v is not None for p in policies for v in p['results'].values()),
            'lazy_done': sum(v is not None for p in policies for v in p['lazy'].values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8772)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                data, mime = PAGE.encode(), 'text/html; charset=utf-8'
            elif path == '/api':
                data, mime = json.dumps(snapshot(args.output_root)).encode(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print(f'Calibration dashboard on port {args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
