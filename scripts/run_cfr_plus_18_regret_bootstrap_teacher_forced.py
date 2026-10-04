#!/usr/bin/env python3
"""Compare bootstrapped and shadow-table teacher-forced neural CFR+ on GPU."""
from __future__ import annotations
import argparse,json,os,signal,sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.core import GameSpec
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import save_policy

SPEC=GameSpec(ranks=4,suits=4,hand_size=2,claim_kinds=("RankHigh","Pair","TwoPair","Trips"),suit_symmetry=True)
K=4096; MAX_ITER=24000; H=1<<18; NH=10; A=19

class NTTrainer(DeepCFRPlusTrainer):
 def __init__(self,*args,target_source='bootstrap',**kwargs):
  super().__init__(*args,**kwargs); self.target_source=target_source
  self.shadow=torch.zeros((H*NH,A),dtype=torch.float32,device=self.device)
  self.visit_counts=torch.zeros((2,H*NH),dtype=torch.int64,device=self.device)
  self.powers=3**torch.arange(4,device=self.device); self.hp=2**torch.arange(18,device=self.device)
  from itertools import combinations_with_replacement
  self.hand_counts=torch.tensor([[sum(1 for r in h if r==q) for q in range(4)] for h in combinations_with_replacement(range(4),2)],dtype=torch.float32,device=self.device)
  lut=torch.full((81,),-1,dtype=torch.long,device=self.device)
  for i,row in enumerate(self.hand_counts.long().to(self.device)): lut[int((row*self.powers).sum())]=i
  self.hand_lut=lut
 def keys(self,x):
  hc=(x[:,:4].long()*self.powers).sum(1); hist=(x[:,4:].long()*self.hp).sum(1)
  return hist*NH+self.hand_lut[hc]
 def _train_regret(self,pid,roots):
  b=self.regret_buffers[pid]; n=b.size
  if not n:return 0.
  x=b.features[:n]; key=self.keys(x)
  with torch.inference_mode(): old=torch.relu(self.regret_nets[pid](x)).float()
  if self.iteration==1: old=old.clone().zero_()
  g=b.targets[:n]-old
  unique,inv=torch.unique(key,return_inverse=True); sums=torch.zeros((len(unique),A),device=self.device); counts=torch.zeros(len(unique),device=self.device)
  sums.index_add_(0,inv,g); counts.index_add_(0,inv,torch.ones_like(inv,dtype=torch.float32))
  self.visit_counts[pid,unique]+=counts.long()
  updated=torch.relu(self.shadow[unique]+sums/counts.clamp_min(1)[:,None]); self.shadow[unique]=updated
  if self.target_source=='teacher_forced': b.targets[:n]=updated[inv]*b.legal_masks[:n]
  return super()._train_regret(pid,roots)
 def checkpoint_dict(self):
  s=super().checkpoint_dict();s['nt_shadow']=self.shadow.detach().cpu();s['nt_visit_counts']=self.visit_counts.detach().cpu();s['nt_target_source']=self.target_source;return s

def make_dense_average(trainer,acc):
 d=trainer.dense_policy
 Hn,N,A=d.S.shape; device=trainer.device
 S=torch.zeros((Hn,N,A),dtype=torch.float32,device=device);R=torch.zeros_like(S)
 hbits=np.arange(Hn,dtype=np.uint32)
 hbits=((hbits[:,None]>>np.arange(18,dtype=np.uint32))&1).astype(np.float32)
 hand_features=trainer.encoder.encode_hands(d.hands,())
 legal_all=torch.from_numpy(d.legal_mask).to(device)
 pop=torch.from_numpy(d.popcount.astype(np.int64)).to(device)
 with torch.inference_mode():
  for pid in (0,1):
   actor=np.flatnonzero((d.popcount&1)==pid)
   for start in range(0,len(actor),max(1,32768//N)):
    ids=actor[start:start+max(1,32768//N)];n=len(ids)
    xx=np.empty((n,N,trainer.encoder.input_dim),dtype=np.float32);xx[:,:,:4]=hand_features[None,:,:4];xx[:,:,4:]=hbits[ids,None,:]
    x=torch.from_numpy(xx.reshape(-1,trainer.encoder.input_dim)).to(device)
    values=trainer.regret_nets[pid](x).float().reshape(n,N,A)
    R[torch.as_tensor(ids,device=device)]=torch.relu(values)
    mask=legal_all[torch.as_tensor(ids,device=device)]
    positive=torch.relu(values)*mask[:,None,:];tot=positive.sum(2,keepdim=True)
    fallback=mask[:,None,:].float()/mask.sum(1).clamp_min(1)[:,None,None]
    S[torch.as_tensor(ids,device=device)]=torch.where(tot>0,positive/tot.clamp_min(1e-12),fallback)
 # Exact own-reach recurrence: a history's parent removes its highest claim bit.
 L0=torch.ones((Hn,N),device=device);L1=torch.ones_like(L0)
 for c in range(18):
  lo=1<<c; hi=lo<<1; par=torch.arange(lo,device=device);hid=par+lo
  maker0=(pop[par]&1)==0; prob=S[par,:,c+1]
  L0[hid]=torch.where(maker0[:,None],L0[par]*prob,L0[par])
  L1[hid]=torch.where(maker0[:,None],L1[par],L1[par]*prob)
 own=torch.where(((pop&1)==0)[:,None],L0,L1)
 acc.add_((trainer.iteration+1)*own[:,:,None].double()*S.double())
 d.S[:]=S.cpu().numpy();d.L_pid0[:]=L0.cpu().numpy();d.L_pid1[:]=L1.cpu().numpy()
 return d,R

def diagnostics(trainer,d,R):
 dev=trainer.device; pop=torch.as_tensor(d.popcount,device=dev); shadow=trainer.shadow
 hmap=[];counts=trainer.hand_counts.cpu().numpy().astype(int)
 from liars_poker.core import card_rank
 for hand in d.hands:
  row=[0]*4
  for card in hand:row[card_rank(card,SPEC)-1]+=1
  hmap.append(next(i for i,x in enumerate(counts) if np.array_equal(x,row)))
 hmap=torch.tensor(hmap,device=dev);chance=[]
 from math import comb
 for row in counts:chance.append(math.prod(comb(4,int(v)) for v in row)/comb(16,2))
 chance=torch.tensor(chance,dtype=torch.float32,device=dev)[hmap]
 rows=[]
 for pid in (0,1):
  hs=torch.nonzero((pop&1)==pid).squeeze(1);hs_cpu=hs.cpu().numpy();idx=hs[:,None]*NH+hmap[None,:];tab=shadow.index_select(0,idx.reshape(-1)).reshape(len(hs),len(hmap),A)
  legal=torch.from_numpy(d.legal_mask).to(dev).index_select(0,hs)[:,None,:];tp=torch.relu(tab)*legal;ts=tp.sum(2,keepdim=True)
  uniform=legal.float()/legal.sum(2,keepdim=True).clamp_min(1);tp=torch.where(ts>0,tp/ts.clamp_min(1e-12),uniform)
  npol=d.S[hs_cpu].astype(np.float32);np_t=torch.as_tensor(npol,device=dev);tv=.5*(np_t-tp).abs().sum(2)
  pred=R[hs];err=(pred-tab).abs()*legal;scale=tab.sum(2)
  reach=torch.as_tensor((d.L_pid0 if pid==0 else d.L_pid1)[hs_cpu],device=dev)*chance[None,:]
  visits=trainer.visit_counts[pid].index_select(0,idx.reshape(-1)).reshape_as(idx)/max(trainer.iteration,1)
  bins=[visits<.1,(visits>=.1)&(visits<1),(visits>=1)&(visits<10),visits>=10]
  names=['lt_0.1','0.1_1','1_10','ge_10']
  for name,mask in zip(names,bins):
   m=mask&legal.any(dim=2)
   if not m.any():continue
   w=reach[m];wm=w/w.sum().clamp_min(1e-20)
   rows.append({'pid':pid+1,'visit_bin':name,'infosets':int(m.sum()),'expected_visits_mean':float(visits[m].mean()),'tv_uniform':float(tv[m].mean()),'tv_reach':float((tv[m]*wm).sum()),'regret_scale_reach':float((scale[m]*wm).sum()),'regret_abs_error_reach':float((err[m].sum(1)*wm).sum()),'relative_regret_error_reach':float((err[m].sum(1)*wm).sum()/(scale[m]*wm).sum().clamp_min(1e-20))})
 return rows

def save_ckpt(tr,root,acc,measured,step):
 state=tr.checkpoint_dict();state['nt_shadow']=tr.shadow.detach().cpu();state['nt_average_sum']=acc.detach().cpu();state['nt_progress']={'measured_training_s':measured,'snapshot_step':step}
 tmp=root/'latest_checkpoint.pt.tmp';torch.save(state,tmp);os.replace(tmp,root/'latest_checkpoint.pt')

def evaluate_async(root,label,kind,policy):
 d=root/'snapshots'/label;d.mkdir(parents=True,exist_ok=True);pd=d/f'{kind}_policy';save_policy(policy,str(pd))
 import subprocess
 log=(d/f'{kind}_eval.log').open('ab')
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate_cfr_plus_18_regret_snapshot.py'),'--policy-dir',str(pd),'--output',str(d/f'{kind}_exact.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--arm',choices=['N','T'],required=True);p.add_argument('--iterations',type=int,default=MAX_ITER);p.add_argument('--snapshot-every',type=int,default=500);p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.set_num_threads(a.threads);root=a.output_root/a.arm;root.mkdir(parents=True,exist_ok=True); target='bootstrap' if a.arm=='N' else 'teacher_forced'
 cfg=dict(device='cuda',seed=17,regret_hidden_sizes=(512,512),strategy_hidden_sizes=(256,256),learning_rate=1e-3,batch_size=1024,regret_batch_size=1024,regret_train_steps=24,strategy_train_steps=0,use_strategy_network=False,regret_buffer_capacity=4_000_000,strategy_buffer_capacity=0,regret_target_mode='aggregate_then_clip',regret_increment_reach_mode='none',regret_accumulation_mode='cumulative',regret_positive_weight=0.,traversal_backend='gpu_native',traversal_batch_size=512,device_replay=True,fused_optimizer=False,validation_fraction=0.)
 ck=root/'latest_checkpoint.pt'
 if ck.exists():
  s=torch.load(ck,map_location='cpu',weights_only=False); tr=NTTrainer.load_checkpoint(ck,device='cuda'); tr.target_source=target; tr.shadow=s['nt_shadow'].to('cuda'); tr.visit_counts=s['nt_visit_counts'].to('cuda'); acc=s['nt_average_sum'].to('cuda'); measured=s['nt_progress']['measured_training_s'];tr.dense_policy=DenseTabularPolicy(SPEC)
 else:
  tr=NTTrainer(SPEC,target_source=target,**cfg);tr.dense_policy=DenseTabularPolicy(SPEC);acc=torch.zeros((H,len(tr.dense_policy.hands),A),dtype=torch.float64,device='cuda');measured=0.
  torch.save({**tr.checkpoint_dict(),'nt_average_sum':acc.detach().cpu(),'nt_progress':{'measured_training_s':0}},ck)
  (root/'manifest.json').write_text(json.dumps({'arm':a.arm,'target_source':target,'roots_per_player':K,'seed':17,'regret_target_mode':'aggregate_then_clip','regret_accumulation_mode':'cumulative','regret_increment_reach_mode':'none','regret_train_steps':24,'regret_batch_size':1024,'regret_buffer_capacity':4000000,'regret_hidden_sizes':[512,512],'traversal_backend':'gpu_native','traversal_batch_size':512,'averaging':'exact own-reach linear average, GPU recurrence','created_utc':time.time()},indent=2),encoding='utf8')
 stop=[False]
 def sig(*_):stop[0]=True
 signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
 end=min(a.iterations,tr.iteration+(50 if a.smoke else a.iterations));reach_checked=False
 while tr.iteration<end and not stop[0]:
  t=time.perf_counter(); dense,R=make_dense_average(tr,acc)
  if a.smoke and not reach_checked:
   l0=dense.L_pid0.copy();l1=dense.L_pid1.copy();dense.recompute_likelihoods();err=max(float(np.max(np.abs(l0-dense.L_pid0))),float(np.max(np.abs(l1-dense.L_pid1))))
   if err>2e-6:raise AssertionError(f'GPU reach recurrence disagrees with dense CPU recurrence: {err}')
   print(f'[reach parity] max_abs_error={err:.3g}',flush=True);reach_checked=True
  rec=tr.run_iteration(traversals_per_player=K); elapsed=time.perf_counter()-t;measured+=elapsed
  with (root/'training.jsonl').open('a') as f:f.write(json.dumps({'iteration':tr.iteration,'measured_training_min':measured/60,'iteration_s':elapsed,'timing':rec['timing'],'arm':a.arm})+'\n')
  if tr.iteration%25==0:print(f'[{a.arm}] iter={tr.iteration} train={measured/60:.2f}m iter_s={elapsed:.2f}',flush=True)
  if tr.iteration%a.snapshot_every==0 or tr.iteration==end:
   label=f'{tr.iteration:06d}';avg=DenseTabularPolicy(SPEC); sums=acc.detach().cpu().numpy(); den=sums.sum(2,keepdims=True);np.divide(sums,den,out=avg.S,where=den>0);evaluate_async(root,label,'average',avg);evaluate_async(root,label,'current',dense);save_ckpt(tr,root,acc,measured,tr.iteration)
   with (root/'diagnostics.jsonl').open('a') as f:f.write(json.dumps({'iteration':tr.iteration,'measured_training_min':measured/60,'bins':diagnostics(tr,dense,R)})+'\n')
  del dense
 if tr.iteration%a.snapshot_every:save_ckpt(tr,root,acc,measured,tr.iteration)
 (root/'summary.json').write_text(json.dumps({'status':'smoke_complete' if a.smoke else ('paused' if stop[0] else 'target_reached'),'iteration':tr.iteration,'measured_training_min':measured/60}),encoding='utf8')
 print(f'[{a.arm}] finished iter={tr.iteration}',flush=True)
if __name__=='__main__':main()
