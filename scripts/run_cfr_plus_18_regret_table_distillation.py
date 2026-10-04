#!/usr/bin/env python3
"""Fresh regret/policy distillation from compact tabular checkpoints."""
from __future__ import annotations
import argparse, json, math, os, time, sys
from pathlib import Path
from itertools import combinations_with_replacement
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.core import GameSpec
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.policies.neural import InfosetEncoder, NeuralPolicy, compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import save_policy

SPEC=GameSpec(ranks=4,suits=4,hand_size=2,claim_kinds=("RankHigh","Pair","TwoPair","Trips"),suit_symmetry=True)
SOURCES={"0030m":"cfr_plus_18_offline_average_study/exact4096_0030m_checkpoint.pt","0045m":"cfr_plus_18_offline_average_study/exact4096_0045m_checkpoint.pt","0120m":"cfr_plus_18_offline_average_study/exact4096_0120m_checkpoint.pt","1080m":"cfr_plus_18_batched_bridge_controls/main_20260930/exact4096/latest_checkpoint.pt"}
ARMS=[(a,s,(512,512)) for a in ("R-visit","R-mix","P-visit") for s in ("0030m","0045m","0120m","1080m")]+[("R-large",s,(1024,1024,1024)) for s in ("0120m","1080m")]
STEPS=40000; BATCH=16384

def main():
 p=argparse.ArgumentParser(); p.add_argument('--artifacts',type=Path,required=True); p.add_argument('--output-root',type=Path); p.add_argument('--threads',type=int,default=4); p.add_argument('--steps',type=int,default=STEPS); p.add_argument('--source',choices=[*SOURCES,'all'],default='all'); p.add_argument('--arm',choices=['all',*(x[0] for x in ARMS)],default='all'); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
 torch.set_num_threads(a.threads); out=(a.output_root or (a.artifacts/'cfr_plus_18_regret_table_distillation/main_20261002')); out.mkdir(parents=True,exist_ok=True)
 if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is not available')
 device=torch.device(a.device)
 jobs=[j for j in ARMS if (a.source=='all' or j[1]==a.source) and (a.arm=='all' or j[0]==a.arm)]
 for source in dict.fromkeys(j[1] for j in jobs):
  path=a.artifacts/SOURCES[source]; state=torch.load(path,map_location='cpu',weights_only=False)
  tr=ExactAverageTabularDiscountTrainer.load_fork_checkpoint(path); table=tr.regret_table.numpy().reshape(1<<18,10,19).copy(); policy=tr.current_policy_exact_dense();
  if policy.spec!=SPEC: raise ValueError('source spec mismatch')
  ev=best_response_dense(SPEC,policy,store_state_values=False)[1]['computer']; p1,p2=ev.exploitability(); source_x=float(p1+p2-1)
  hbits=torch.from_numpy(((np.arange(1<<18,dtype=np.uint32)[:,None]>>np.arange(18))&1).astype(np.float32)); hands=tr.hand_counts.float(); encoder=InfosetEncoder(SPEC); dense=policy
  # Map compact rank-count hand rows to dense physical-hand columns for opponent reach.
  hand_map=[]
  for hand in dense.hands:
   cnt=[0]*4
   from liars_poker.core import card_rank
   for c in hand: cnt[card_rank(c,SPEC)-1]+=1
   hand_map.append(next(i for i,r in enumerate(hands.tolist()) if r==cnt))
  # Chance probability of each own/opponent rank-count pair; opponent reach comes from the table policy.
  chance=np.zeros((10,10),dtype=np.float64)
  from math import comb
  for i,hi in enumerate(hands.int().tolist()):
   for j,hj in enumerate(hands.int().tolist()):
    n=1
    for x,y in zip(hi,hj): n*=comb(4,x)*comb(4-x,y)
    chance[i,j]=n
  chance/=chance.sum()
  reach=np.zeros((1<<18,10),dtype=np.float64)
  # Dense likelihood arrays indexed by physical hand; project them back to compact canonical hands.
  oppL=[np.zeros_like(reach),np.zeros_like(reach)]
  for pid in (0,1):
   for i,j in enumerate(hand_map): oppL[pid][:,j]=dense.L_pid1[:,i] if pid==0 else dense.L_pid0[:,i]
  weights=np.zeros((2,1<<18,10),dtype=np.float32)
  for pid in (0,1):
   for own in range(10): weights[pid,:,own]=(chance[own]@oppL[pid].T).astype(np.float32)
   weights[pid,(dense.popcount&1)!=pid,:]=0
  visit_prob=weights.copy();visit=weights.reshape(2,-1); visit/=np.maximum(visit.sum(axis=1,keepdims=True),1e-30)
  legal=torch.from_numpy(dense.legal_mask.copy()); targets=torch.from_numpy(table); handbits=hands
  for arm,src,hidden in (j for j in jobs if j[1]==source):
   d=out/src/arm; d.mkdir(parents=True,exist_ok=True); result=d/'result.json'; trainlog=d/'training.jsonl'
   if result.exists(): continue
   (d/'manifest.json').write_text(json.dumps({'source_checkpoint':str(path),'source_iteration':int(state['iteration']),'arm':arm,'hidden_sizes':hidden,'steps_per_player':a.steps,'batch_size':BATCH,'learning_rate_schedule':'cosine 1e-3 to 1e-5','device':a.device,'row_sampling':'50% visits + 50% uniform' if arm in ('R-mix','R-large') else 'visits','seed':17030+list(SOURCES).index(src)*31+len(arm)},indent=2),encoding='utf8')
   torch.manual_seed(17030+list(SOURCES).index(src)*31+len(arm)); nact=19; xbuf=[]; ybuf=[]
   hids=torch.arange(1<<18).repeat_interleave(10); kids=torch.arange(10).repeat(1<<18)
   # Exact target rows remain compact; batches materialize features only when sampled.
   features=torch.cat((handbits[kids],hbits[hids]),dim=1); masks=legal[hids]
   actor_rows=torch.from_numpy((dense.popcount[hids]&1).astype(np.int64))
   legal_rows=masks.any(1)
   rng=torch.Generator().manual_seed(41017)
   policy_class=NeuralPolicy if arm=='P-visit' else NeuralRegretMatchingPolicy
   model=policy_class(SPEC,hidden_sizes=hidden,device=device); models=(model.model_p1,model.model_p2); opts=[torch.optim.Adam(m.parameters(),lr=1e-3) for m in models]
   # P-visit fits normalized regret matching targets; regression arms fit raw positive cumulative regrets.
   target=targets.clone(); target.clamp_(min=0); target*=legal[:,None,:].float()
   if arm=='P-visit':
    sums=target.sum(2,keepdim=True); target=torch.where(sums>0,target/sums.clamp_min(1e-12),legal[:,None,:].float()/legal.sum(1).clamp_min(1)[:,None,None])
   started=time.perf_counter(); losses=[]; resume=d/'fit_state.pt'; pid0=0; step0=0
   if resume.exists():
    rs=torch.load(resume,map_location='cpu',weights_only=False)
    for net,sd in zip(models,rs['models']):net.load_state_dict(sd)
    for opt,sd in zip(opts,rs['optimizers']):
     opt.load_state_dict(sd)
     for opt_state in opt.state.values():
      for key,value in opt_state.items():
       if isinstance(value,torch.Tensor):opt_state[key]=value.to(device)
    pid0=rs['pid'];step0=rs['step'];torch.set_rng_state(rs['rng']);rng.set_state(rs['sample_rng'])
   fit_features=features.to(device); fit_masks=masks.to(device); fit_target=target.view(-1,19).to(device)
   sampling=[]
   for fit_pid in (0,1):
    valid=legal_rows&(actor_rows==fit_pid)
    q=torch.from_numpy(visit[fit_pid].copy());q[~valid]=0;q/=q.sum().clamp_min(1e-30)
    sampling.append((q.to(device),valid.nonzero(as_tuple=True)[0].to(device)))
   gpu_rng=torch.Generator(device=device).manual_seed(41017+list(SOURCES).index(src)*31)
   if resume.exists() and rs.get('sample_rng_device')==a.device:gpu_rng.set_state(rs['sample_rng_gpu'])
   for pid,(net,opt) in enumerate(zip(models,opts)):
    if pid<pid0: continue
    net.train()
    q,uniform_ix=sampling[pid]
    for step in range(step0 if pid==pid0 else 0,a.steps):
     if a.smoke and step>=8: break
     lr=1e-5+0.5*(1e-3-1e-5)*(1+math.cos(math.pi*step/max(a.steps-1,1)))
     for group in opt.param_groups:group['lr']=lr
     if arm in ('R-mix','R-large'):
      choose=torch.rand(BATCH,device=device,generator=gpu_rng)<.5; vi=torch.multinomial(q,BATCH,replacement=True,generator=gpu_rng); ui=uniform_ix[torch.randint(len(uniform_ix),(BATCH,),device=device,generator=gpu_rng)]; ix=torch.where(choose,vi,ui)
     else: ix=torch.multinomial(q,BATCH,replacement=True,generator=gpu_rng)
     xx=fit_features[ix]; yy=fit_target[ix]; mm=fit_masks[ix]
     pred=net(xx).masked_fill(~mm,-1e9)
     if arm=='P-visit': loss=-(yy*torch.log_softmax(pred,1)).sum(1).mean()
     else: loss=((pred-yy).square()*mm).sum(1).div(mm.sum(1).clamp_min(1)).mean()
     opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
     if step%1000==0: print(f'[fit] {src}/{arm} p{pid+1} {step}/{a.steps} loss={loss.item():.5g}',flush=True)
     if (step+1)%500==0 or (a.smoke and step==7):
      tmp=resume.with_suffix('.tmp');torch.save({'models':[m.state_dict() for m in models],'optimizers':[o.state_dict() for o in opts],'pid':pid,'step':step+1,'rng':torch.get_rng_state(),'sample_rng':rng.get_state(),'sample_rng_device':a.device,'sample_rng_gpu':gpu_rng.get_state()},tmp);os.replace(tmp,resume)
      with trainlog.open('a',encoding='utf8') as f:f.write(json.dumps({'arm':arm,'source':src,'iteration':int(state['iteration']),'measured_training_min':(time.perf_counter()-started)/60,'fit_player':pid+1,'fit_step':step+1,'fit_steps':a.steps,'utc':time.time()})+'\n')
    net.eval()
    if 'loss' in locals():losses.append(float(loss.item()))
    next_pid=pid+1
    tmp=resume.with_suffix('.tmp');torch.save({'models':[m.state_dict() for m in models],'optimizers':[o.state_dict() for o in opts],'pid':next_pid,'step':0,'rng':torch.get_rng_state(),'sample_rng':rng.get_state(),'sample_rng_device':a.device,'sample_rng_gpu':gpu_rng.get_state()},tmp);os.replace(tmp,resume)
   del fit_features,fit_masks,fit_target,sampling
   policy_n=compile_neural_to_dense(model,batch_size=16384)
   br=best_response_dense(SPEC,policy_n,store_state_values=False)[1]['computer']; x1,x2=br.exploitability()
   delta=.5*np.abs(policy.S-policy_n.S).sum(axis=2); legal_hist=policy.legal_counts>0;tv_uniform=float(delta[legal_hist].mean())
   chance_by_hand=np.asarray([math.prod(comb(4,int(v)) for v in row)/comb(16,2) for row in hands.int().tolist()],dtype=np.float64)
   own=np.where((policy.popcount&1)[:,None]==0,policy.L_pid0,policy.L_pid1); tvw=own*chance_by_hand[hand_map][None,:];tv_reach=float((delta*tvw).sum()/max(float(tvw.sum()),1e-30))
   regret_abs=regret_scale=0.0;tv_bins={}
   if not a.smoke:
    for pid in (0,1):
     if arm!='P-visit':
      net=models[pid];qmat=visit_prob[pid][:,hand_map].reshape(-1);flat_ix=np.flatnonzero((policy.popcount[hids]&1)==pid)
      for st in range(0,len(flat_ix),BATCH):
       z=flat_ix[st:st+BATCH]; pred=net(features[z]).detach().numpy(); tar=targets.view(-1,19)[z].numpy(); mk=masks[z].numpy(); q=qmat[z]
       er=np.abs(pred-tar)*mk;regret_abs+=float((er.sum(1)*q).sum());regret_scale+=float(((tar*mk).sum(1)*q).sum())
     expected=visit_prob[pid][:,hand_map]*4096
     for name,cond in [('lt_0.1',expected<.1),('0.1_1',(expected>=.1)&(expected<1)),('1_10',(expected>=1)&(expected<10)),('ge_10',expected>=10)]:
      cond &= ((policy.legal_counts>0)&((policy.popcount&1)==pid))[:,None]
      if cond.any():
       tv_bins[f'p{pid+1}_{name}_uniform']=float(delta[cond].mean());bw=tvw[cond];tv_bins[f'p{pid+1}_{name}_reach']=float((delta[cond]*bw).sum()/max(float(bw.sum()),1e-30))
   row={'source':src,'arm':arm,'iteration':int(state['iteration']),'source_exploitability':source_x,'network_exploitability':float(x1+x2-1),'ratio':float((x1+x2-1)/source_x),'tv_uniform':tv_uniform,'tv_reach':tv_reach,'relative_regret_error_reach':None if arm=='P-visit' else regret_abs/max(regret_scale,1e-30),'tv_by_expected_visit_bin':tv_bins,'steps_per_player':a.steps,'batch':BATCH,'hidden':hidden,'loss_last_by_player':losses,'elapsed_s':time.perf_counter()-started,'smoke':a.smoke}
   save_policy(model.eval(),str(d/'policy')); result.write_text(json.dumps(row,indent=2),encoding='utf8');
   with (d/'evaluations.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'kind':'average','iteration':int(state['iteration']),'measured_training_min':row['elapsed_s']/60,'exploitability':row['network_exploitability'],'source_exploitability':source_x,'utc':time.time()})+'\n')
   resume.unlink(missing_ok=True);(d/'summary.json').write_text(json.dumps({'status':'complete','iteration':int(state['iteration']),'exploitability':row['network_exploitability'],'updated_utc':time.time()}),encoding='utf8');print('[done]',json.dumps(row),flush=True)
  del tr,state,table,policy,dense,weights,features,masks,targets
if __name__=='__main__': main()
