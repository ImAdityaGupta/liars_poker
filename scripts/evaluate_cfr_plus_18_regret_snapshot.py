#!/usr/bin/env python3
"""Exact exploitability evaluator for a saved 18-claim policy snapshot."""
import argparse,json,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy
p=argparse.ArgumentParser();p.add_argument('--policy-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
t=time.perf_counter();pol,spec=load_policy(str(a.policy_dir));dense=pol if isinstance(pol,DenseTabularPolicy) else compile_neural_to_dense(pol,batch_size=65536);_,meta=best_response_dense(spec,dense,store_state_values=False);p1,p2=meta['computer'].exploitability();row={'policy_dir':str(a.policy_dir),'exploitability':float(p1+p2-1),'p_first':float(p1),'p_second':float(p2),'evaluation_s':time.perf_counter()-t,'utc':time.time()}
a.output.write_text(json.dumps(row),encoding='utf8');print(json.dumps(row),flush=True)
arm_dir=a.output.parents[2]; train=arm_dir/'training.jsonl'; measured=None
if train.exists():
 try: measured=json.loads(train.read_text(encoding='utf8').splitlines()[-1]).get('measured_training_min')
 except (IndexError,json.JSONDecodeError): pass
log=arm_dir/'evaluations.jsonl'
with log.open('a',encoding='utf8') as f:f.write(json.dumps({**row,'kind':a.output.stem.split('_')[0],'iteration':int(a.output.parent.name),'measured_training_min':measured})+'\n')
shutil.rmtree(a.policy_dir,ignore_errors=True)
