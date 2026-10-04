#!/usr/bin/env python3
"""Focused live dashboard for the regret distillation and N/T experiments."""
from __future__ import annotations
import argparse,json,time,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAGE='''<!doctype html><meta charset="utf-8"><title>18-claim regret diagnostics</title><style>body{font:15px system-ui;max-width:1500px;margin:2rem auto;color:#172033;background:#f4f6fa}.card{background:white;padding:1rem;margin:1rem 0;border-radius:10px}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}</style><h1>18-claim regret diagnostics</h1><p id="status">Loading…</p><div class="card"><img src="/plot.png"></div><div class="card"><h2>Latest distillation fits and exact evaluations</h2><table id="tbl"></table></div><script>async function f(){let s=await(await fetch('/api')).json();document.querySelector('#status').textContent=s.status;let t=document.querySelector('#tbl');t.innerHTML='<tr><th>Source</th><th>Arm</th><th>Fit progress</th><th>Exact exploitability</th><th>Ratio to table</th></tr>'+s.fits.map(x=>`<tr><td>${x.source}</td><td>${x.arm}</td><td>${x.progress}</td><td>${x.exploitability??'—'}</td><td>${x.ratio??'—'}</td></tr>`).join('');document.querySelector('img').src='/plot.png?'+Date.now()}f();setInterval(f,15000)</script>'''
def read(path):
 try:return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
 except (OSError,json.JSONDecodeError):return []
def snapshot(root):
 fits=[]; curves={}
 dist=root/'cfr_plus_18_regret_table_distillation'/'main_20261002'
 for tr in sorted(dist.glob('*/*/training.jsonl')):
  rows=read(tr);res=tr.parent/'result.json';rr={}
  if res.exists():
   try:rr=json.loads(res.read_text())
   except json.JSONDecodeError:pass
  fits.append({'source':tr.parent.parent.name,'arm':tr.parent.name,'progress':f"{rows[-1].get('fit_player','?')}/2 · {rows[-1].get('fit_step','?')}/{rows[-1].get('fit_steps','?')}" if rows else 'queued',**rr})
 for arm in ('N','T'):
  rows=read(root/'cfr_plus_18_regret_bootstrap_teacher_forced'/'main_20261002'/arm/'evaluations.jsonl')
  curves[arm]=rows
 curves['exact4096']=[{**r,'kind':'average'} for r in read(root/'cfr_plus_18_batched_bridge_controls'/'main_20260930'/'exact4096'/'evaluations.jsonl')]
 return fits,curves
def plot(path,curves):
 fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');ax=axes[0]
 colors={'N':'#0072b2','T':'#d55e00','exact4096':'#333333'}
 for arm,rows in curves.items():
  for kind in ('average','current'):
   r=[x for x in rows if x.get('kind')==kind]
   if r:ax.plot([x['iteration'] for x in r],[x['exploitability'] for x in r],marker='o',ms=3,color=colors[arm],linestyle='-' if kind=='average' else ':',label=f'{arm} {kind}')
 ax.set_yscale('log');ax.set_xlabel('CFR+ iteration');ax.set_ylabel('Exact exploitability');ax.grid(alpha=.25);ax.legend();ax.set_title('N/T: exact exploitability (solid average, dotted current)')
 dist=Path('/root/liars_poker/artifacts/cfr_plus_18_regret_table_distillation/main_20261002');bx=axes[1];order=['0030m','0045m','0120m','1080m'];palette={'R-visit':'#0072b2','R-mix':'#d55e00','P-visit':'#009e73','R-large':'#cc79a7'}
 for arm,color in palette.items():
  xs=[];ys=[]
  for i,src in enumerate(order):
   rp=dist/src/arm/'result.json'
   if rp.exists():
    try:r=json.loads(rp.read_text());xs.append(i);ys.append(r['ratio'])
    except (OSError,KeyError,json.JSONDecodeError):pass
  if xs:bx.plot(xs,ys,marker='o',label=arm,color=color)
 bx.axhline(1,color='#444',ls='--',lw=1);bx.set_yscale('log');bx.set_xticks(range(4),order);bx.set_ylabel('Network / table exploitability');bx.set_xlabel('Source table checkpoint');bx.grid(alpha=.25);bx.legend();bx.set_title('Offline regret distillation')
 fig.savefig(path,dpi=140);plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--port',type=int,default=8770);p.add_argument('--state-dir',type=Path,required=True);a=p.parse_args();a.state_dir.mkdir(parents=True,exist_ok=True)
 class H(BaseHTTPRequestHandler):
  def do_GET(self):
   fits,curves=snapshot(a.artifacts)
   if self.path=='/':data,mime=PAGE.encode(),'text/html; charset=utf-8'
   elif self.path.startswith('/api'):data,mime=json.dumps({'status':f'{len(fits)} distillation fits · N/T exact evaluations refreshed every 15 seconds','fits':fits}).encode(),'application/json'
   elif self.path.startswith('/plot.png'):plot(a.state_dir/'plot.png',curves);data=(a.state_dir/'plot.png').read_bytes();mime='image/png'
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
  def log_message(self,*args):pass
 print(f'regret dashboard http://127.0.0.1:{a.port}',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),H).serve_forever()
if __name__=='__main__':main()
