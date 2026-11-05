"""Audit each replicated continuation against its own fold/seed source model."""
import os

import hashlib,json,itertools
from pathlib import Path
import numpy as np,pandas as pd
r=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'overnight_20261106/explore/confirmation'));p=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'atlas_scale_20261105'));d=pd.read_parquet(p/'data/samples.parquet').set_index('sample_id');y=np.load(p/'data/p.npy');position=pd.Series(np.arange(len(d)),index=d.index)
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def read(q):
 x=json.loads((q/'receipt.json').read_text());roles=json.loads((q/'roles.json').read_text());ids=roles['evaluation_ids'];truth=y[position.loc[ids].to_numpy()].astype('float64');pred=np.load(q/'evaluation_prediction.npy').astype('float64');fold=x['config']['fold']
 assert pred.shape==truth.shape and np.isfinite(pred).all() and (pred>=0).all() and np.max(abs(pred.sum(1)-1))<1e-6
 assert sha(q/'evaluation_prediction.npy')==x['prediction_sha256'] and sha(q/'best.pt')==x['checkpoint_sha256']
 groups={k:set(d.loc[v].group_id) for k,v in roles.items()};seqs={k:set(d.loc[v].sequence_sha256) for k,v in roles.items()}
 for a,b in itertools.combinations(roles,2):assert not groups[a]&groups[b] and not seqs[a]&seqs[b]
 assert set(d.loc[roles['train_ids']].fold)==set(range(5))-{fold,(fold+1)%5} and set(d.loc[roles['stop_ids']].fold)=={(fold+1)%5} and set(d.loc[ids].fold)=={fold}
 t=d.loc[ids,['group_id','protein_id']].copy();t['H']=np.sqrt(.5*((np.sqrt(truth)-np.sqrt(pred))**2).sum(1));t['CDF']=((truth.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(truth[:,[0,-1]]-pred[:,[0,-1]]).mean(1)
 group=t.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean()
 assert abs(group.H.mean()-x['result']['group_macro_H'])<1e-7
 return x,group,roles
out=[]
for f in sorted((r/'runs').glob('*/receipt.json')):
 x,g,roles=read(f.parent);b,bg,br=read(Path(x['config']['warmstart_path']).parent);assert roles==br and set(g.index)==set(bg.index)
 delta=g.loc[bg.index].H.to_numpy()-bg.H.to_numpy();rng=np.random.default_rng(1701);boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
 out.append({'run_id':x['run_id'],'fold':x['config']['fold'],'seed':x['config']['seed'],'lr':x['config']['lr'],'checkpoint':str(f.parent/'best.pt'),'source_checkpoint':x['config']['warmstart_path'],'source_stop_H':b['best_stop_H'],'continued_stop_H':x['best_stop_H'],'source_H':float(bg.H.mean()),'continued_H':float(g.H.mean()),'delta_H':float(delta.mean()),'descriptive_group_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'source_CDF':float(bg.CDF.mean()),'continued_CDF':float(g.CDF.mean()),'source_tail':float(bg['tail'].mean()),'continued_tail':float(g['tail'].mean()),'best_epoch':x['best_epoch'],'epochs':x['epochs'],'stop_reason':x['stop_reason'],'fit_seconds':x['fit_wall_seconds'],'source_fit_seconds':b['fit_wall_seconds'],'group_and_exact_sequence_isolation':True,'source_roles_identical':True,'checkpoint_and_prediction_hashes_verified':True,'prediction_independently_recomputed':True})
assert len(out)==2 and {(z['fold'],z['seed']) for z in out}=={(1,1701),(0,3407)}
result={'status':'complete','fits':2,'plan_sha256':sha(r/'plan.json'),'comparisons':out,'fit_seconds_total':sum(z['fit_seconds'] for z in out),'scope':'Two additional development configurations; not a complete 2fold x 2seed factorial; no new-C claim'}
(r/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
