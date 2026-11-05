"""Independent prediction audit and paired group comparison after eight fits."""
import os

import hashlib,json,itertools
from pathlib import Path
import numpy as np,pandas as pd

r=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'overnight_20261106/explore'))
p=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'atlas_scale_20261105'));old=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'pooling_20261105'))
plan=json.loads((r/'plan.json').read_text());d=pd.read_parquet(p/'data/samples.parquet').set_index('sample_id');y=np.load(p/'data/p.npy');position=pd.Series(np.arange(len(d)),index=d.index)
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def read(q):
 x=json.loads((q/'receipt.json').read_text());roles=json.loads((q/'roles.json').read_text());ids=roles['evaluation_ids'];truth=y[position.loc[ids].to_numpy()].astype('float64');pred=np.load(q/'evaluation_prediction.npy').astype('float64')
 assert pred.shape==truth.shape and np.isfinite(pred).all() and (pred>=0).all() and np.max(abs(pred.sum(1)-1))<1e-6
 assert sha(q/'evaluation_prediction.npy')==x['prediction_sha256'] and sha(q/'best.pt')==x['checkpoint_sha256']
 groups={k:set(d.loc[v].group_id) for k,v in roles.items()};seqs={k:set(d.loc[v].sequence_sha256) for k,v in roles.items()}
 for a,b in itertools.combinations(roles,2):assert not groups[a]&groups[b] and not seqs[a]&seqs[b]
 assert set(d.loc[roles['train_ids']].fold)=={2,3,4} and set(d.loc[roles['stop_ids']].fold)=={1} and set(d.loc[ids].fold)=={0}
 t=d.loc[ids,['group_id','protein_id']].copy();t['H']=np.sqrt(.5*((np.sqrt(truth)-np.sqrt(pred))**2).sum(1));t['CDF']=((truth.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(truth[:,[0,-1]]-pred[:,[0,-1]]).mean(1)
 group=t.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean()
 assert abs(group.H.mean()-x['result']['group_macro_H'])<1e-7
 return x,group

rows=[];tables={};checks=[]
for q in sorted((r/'runs').glob('*/receipt.json')):
 x,g=read(q.parent);c=x['config'];tables[x['run_id']]=g
 rows.append(dict(run_id=x['run_id'],mode=c['mode'],lr=c['lr'],stop_H=x['best_stop_H'],H=float(g.H.mean()),CDF=float(g.CDF.mean()),tail=float(g['tail'].mean()),initial_stop_H=x['initial_stop_H'],best_epoch=x['best_epoch'],epochs=x['epochs'],stop_reason=x['stop_reason'],fit_seconds=x['fit_wall_seconds'],parameters=x['parameters'],attention=x['attention_evaluation_diagnostics'],gate=x['final_gate'],checkpoint=str(q.parent/'best.pt')))
 checks.append({'run_id':x['run_id'],'prediction_recomputed':True,'group_and_exact_sequence_isolation':True})
assert len(rows)==8 and {(z['mode'],z['lr']) for z in rows}=={(z['mode'],z['lr']) for z in plan['fits']}
selected={m:min([z for z in rows if z['mode']==m],key=lambda z:(z['stop_H'],z['lr'])) for m in set(z['mode'] for z in rows)}
base,bg=read(Path(plan['warmstart']).parent);tables['original_mean']=bg
attention_rows=[]
for q in (old/'runs').glob('*/receipt.json'):
 x=json.loads(q.read_text());c=x['config']
 if c['mode']=='residue_attention' and c['fold']==0 and c['seed']==1701:attention_rows.append((x,q.parent))
oldatt,q=min(attention_rows,key=lambda z:(z[0]['best_stop_H'],z[0]['config']['lr']));_,ag=read(q);tables['original_attention']=ag
comparisons=[]
for mode,row in selected.items():
 for name,baseline in [('original_mean',bg),('continued_mean',tables[selected['mean_continue']['run_id']])]:
  a=tables[row['run_id']];assert set(a.index)==set(baseline.index)
  delta=a.loc[baseline.index].H.to_numpy()-baseline.H.to_numpy();rng=np.random.default_rng(1701)
  boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
  comparisons.append({'mode':mode,'minus':name,'groups':len(delta),'delta_H':float(delta.mean()),'descriptive_group_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'selection_adjusted':False,'seed_uncertainty_included':False})
summary={'status':'complete','fits':len(rows),'plan_sha256':sha(r/'plan.json'),'all_fits':rows,'selected_by_stop':selected,'paired':comparisons,'original_mean':{'run_id':base['run_id'],'H':float(bg.H.mean()),'stop_H':base['best_stop_H'],'best_epoch':base['best_epoch'],'fit_seconds':base['fit_wall_seconds']},'original_attention':{'run_id':oldatt['run_id'],'H':float(ag.H.mean()),'stop_H':oldatt['best_stop_H']},'fit_seconds_total':sum(z['fit_seconds'] for z in rows),'checks':checks,'scope':'exposed historical T/vertex-p fold0 seed1701; no blind/new-C/AFDB claim; warmstart requires prerequisite cost and matched continued-mean control'}
(r/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');pd.DataFrame(rows).to_csv(r/'results/all_fits.csv',index=False)
print(json.dumps({k:summary[k] for k in ['status','fits','selected_by_stop','paired','fit_seconds_total']},indent=2))
