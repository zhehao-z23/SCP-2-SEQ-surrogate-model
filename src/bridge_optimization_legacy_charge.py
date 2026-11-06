import os
"""Preserve the previous strong charge baseline across different training runners."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
r=Path(str(Path(os.environ['PROJECT_ROOT']) / 'bridge_20261106/optimization'));parent=Path(str(Path(os.environ['PROJECT_ROOT']) / 'atlas_scale_20261105'));sys.path.insert(0,str(r/'code'));sys.path.insert(0,str(Path(os.environ['PROJECT_ROOT']) / 'package/src'));from bridge_optimization_model import PoolingModel;from seq2scp_training.models import SequenceModel
torch.set_num_threads(2);torch.manual_seed(1701);a=PoolingModel('charge_deep');torch.manual_seed(1701);b=SequenceModel(input_dim=78,bins=63,kind='vector',pool='mean',hidden=[1024,512,256],dropout=.3)
pa=list(a.parameters());pb=list(b.parameters());assert len(pa)==len(pb) and all(torch.equal(x,y) for x,y in zip(pa,pb));init_identical=True
d=pd.read_parquet(parent/'data/samples.parquet').set_index('sample_id');y=np.load(parent/'data/p.npy');pos=pd.Series(np.arange(len(d)),index=d.index)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(folder):
 z=json.loads((folder/'receipt.json').read_text());roles=json.loads((folder/'roles.json').read_text());pred=np.load(folder/'evaluation_prediction.npy').astype('float64');truth=y[pos.loc[roles['evaluation_ids']].to_numpy()].astype('float64');assert sha(folder/'evaluation_prediction.npy')==z['prediction_sha256'];t=d.loc[roles['evaluation_ids'],['group_id','protein_id']].copy();t['H']=np.sqrt(.5*((truth**.5-pred**.5)**2).sum(1));t['CDF']=((truth.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(truth[:,[0,-1]]-pred[:,[0,-1]]).mean(1);g=t.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean();assert abs(g.H.mean()-z['result']['group_macro_H'])<1e-7;return z,roles,g
out=[]
for fold in [0,1]:
 old=[]
 for q in (parent/'runs').glob('*/receipt.json'):
  z=json.loads(q.read_text());c=z['config']
  if c['family']=='composition_charge_deep' and c['fold']==fold and c['seed']==1701:old.append((z,q.parent))
 oz,op=min(old,key=lambda z:(z[0]['best_stop_H'],z[0]['config']['lr']));stage=r if fold==0 else r/'confirmation';new=[]
 for q in (stage/'runs').glob('*/receipt.json'):
  z=json.loads(q.read_text());c=z['config']
  if c['mode']=='charge_deep' and c['policy']=='plateau' and c['fold']==fold and c['seed']==1701:new.append(q.parent)
 assert len(new)==1;npth=new[0];nz,nroles,ng=read(npth);oz,oroles,og=read(op);assert nroles==oroles and nz['config']['release_sha256']==oz['config']['data_release_sha256']
 nc=torch.load(npth/'best.pt',map_location='cpu',weights_only=False);oc=torch.load(op/'best.pt',map_location='cpu',weights_only=False)
 for nk,ok in [('charge_median','median'),('charge_mean','mean'),('charge_std','std')]:assert np.array_equal(nc[nk],oc[ok])
 delta=ng.loc[og.index].H.to_numpy()-og.H.to_numpy();rng=np.random.default_rng(1701);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]);out.append({'fold':fold,'seed':1701,'new_run_id':nz['run_id'],'legacy_run_id':oz['run_id'],'legacy_lr_selected_on_stop':oz['config']['lr'],'legacy_stop_H':oz['best_stop_H'],'new_stop_H':nz['best_stop_H'],'legacy_H':float(og.H.mean()),'new_plateau_H':float(ng.H.mean()),'new_minus_legacy_H':float(delta.mean()),'descriptive_group_95':np.quantile(boot,[.025,.975]).tolist(),'legacy_CDF':float(og.CDF.mean()),'new_CDF':float(ng.CDF.mean()),'legacy_tail':float(og['tail'].mean()),'new_tail':float(ng['tail'].mean()),'roles_and_scalers_identical':True})
res={'status':'complete','comparison':'new common pooling runner plateau versus actual previous capacity charge baseline; not a single-factor schedule contrast','same_seed_initial_parameter_tensors_identical_CPU':init_identical,'runner_differences':{'old_batching':'uniform permutation of all training rows; consecutive128; one forward/backward per optimizer batch','new_batching':'random chunks1024, stable length-sort into128 then shuffle batches; 16384-token microbatch accumulation even for vector heads','gradient_clipping':'both global norm1.0 once per optimizer update; clipping threshold is NOT a difference','seed_path':'seed1701 and initial parameter tensors equal, but batching consumes numpy RNG differently and dropout/gradient accumulation follow different paths','budget':'old max600sec per fit vs new900; all runs in this comparison early-stop','lr':'old best of two constant LRs selected on stop; new prespecified continuous plateau'},'comparisons':out,'decision':'retain legacy strong charge baseline; new plateau improves its own fresh runner but does not beat the earlier baseline on evaluation H'};(r/'results/legacy_charge_reference.json').write_text(json.dumps(res,indent=2)+'\n');pub=r/'results/public_summary.json';x=json.loads(pub.read_text());x['legacy_charge_reference']=res;pub.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(res,indent=2))
