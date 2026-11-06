import os
"""Reuse existing matching fixed-LR and overnight results; never retrain old runs."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
r=Path(str(Path(os.environ['PROJECT_ROOT']) / 'bridge_20261106/optimization'));p=Path(str(Path(os.environ['PROJECT_ROOT']) / 'atlas_scale_20261105'));old=Path(str(Path(os.environ['PROJECT_ROOT']) / 'pooling_20261105'));night=Path(str(Path(os.environ['PROJECT_ROOT']) / 'overnight_20261106/explore'));d=pd.read_parquet(p/'data/samples.parquet').set_index('sample_id');y=np.load(p/'data/p.npy');pos=pd.Series(np.arange(len(d)),index=d.index)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(folder):
 z=json.loads((folder/'receipt.json').read_text());roles=json.loads((folder/'roles.json').read_text());pred=np.load(folder/'evaluation_prediction.npy').astype('float64');truth=y[pos.loc[roles['evaluation_ids']].to_numpy()].astype('float64');assert pred.shape==truth.shape and np.isfinite(pred).all() and (pred>=0).all() and max(abs(pred.sum(1)-1))<1e-6;assert sha(folder/'evaluation_prediction.npy')==z['prediction_sha256']
 t=d.loc[roles['evaluation_ids'],['group_id','protein_id']].copy();t['H']=np.sqrt(.5*((truth**.5-pred**.5)**2).sum(1));t['CDF']=((truth.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(truth[:,[0,-1]]-pred[:,[0,-1]]).mean(1);g=t.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean();assert abs(g.H.mean()-z['result']['group_macro_H'])<1e-7
 return z,roles,g
def compare(a,b,label):
 za,ra,ga=read(a);zb,rb,gb=read(b);assert ra==rb and set(ga.index)==set(gb.index);ca=za['config'];cb=zb['config'];assert ca['release_sha256']==cb['release_sha256'] and ca['cache_receipt_sha256']==cb['cache_receipt_sha256']
 aa=torch.load(a/'best.pt',map_location='cpu',weights_only=False);bb=torch.load(b/'best.pt',map_location='cpu',weights_only=False)
 for k in ['embedding_mean','embedding_std','charge_median','charge_mean','charge_std']:assert np.array_equal(aa[k],bb[k])
 delta=ga.loc[gb.index].H.to_numpy()-gb.H.to_numpy();rng=np.random.default_rng(1701);boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
 return {'family':ca['mode'],'fold':ca['fold'],'seed':ca['seed'],'comparison':label,'new_run_id':za['run_id'],'reference_run_id':zb['run_id'],'new_stop_H':za['best_stop_H'],'reference_stop_H':zb['best_stop_H'],'new_H':float(ga.H.mean()),'reference_H':float(gb.H.mean()),'delta_H':float(delta.mean()),'descriptive_group_95':np.quantile(boots,[.025,.975]).tolist(),'new_CDF':float(ga.CDF.mean()),'reference_CDF':float(gb.CDF.mean()),'new_tail':float(ga['tail'].mean()),'reference_tail':float(gb['tail'].mean()),'roles_and_scalers_identical':True,'not_blind':True,'selection_adjusted':False}
oldrows=[(json.loads(q.read_text()),q.parent) for q in (old/'runs').glob('*/receipt.json')];nightrows=[]
for base in [night,night/'confirmation']:
 for q in (base/'runs').glob('*/receipt.json'):
  z=json.loads(q.read_text())
  if z['config']['mode']=='mean_continue' and z['config']['lr']==.0001:nightrows.append((z,q.parent))
out=[]
for stage in [r,r/'confirmation']:
 for q in (stage/'runs').glob('*/receipt.json'):
  z=json.loads(q.read_text());c=z['config']
  if c['policy']!='plateau':continue
  if c['mode']!='charge_deep':
   matches=[(x,folder) for x,folder in oldrows if x['config']['mode']==c['mode'] and x['config']['fold']==c['fold'] and x['config']['seed']==c['seed'] and x['config']['lr']==.0003];assert len(matches)==1;out.append(compare(q.parent,matches[0][1],'plateau minus existing fixed LR3e-4'))
  if c['mode']=='residue_mean':
   matches=[(x,folder) for x,folder in nightrows if x['config']['fold']==c['fold'] and x['config']['seed']==c['seed']];assert len(matches)==1;out.append(compare(q.parent,matches[0][1],'plateau minus existing overnight reset continuation; RNG/control history differs'))
res={'status':'complete','existing_constant_results_reused':sum(x['comparison']=='plateau minus existing fixed LR3e-4' for x in out),'existing_overnight_continuations_reused':sum('overnight' in x['comparison'] for x in out),'comparisons':out,'scope':'public aggregate only; existing results freshly recomputed, no old checkpoint treated as having optimizer state'};(r/'results/reused_reference_comparisons.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
