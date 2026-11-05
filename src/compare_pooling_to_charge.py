"""Compare saved residue-readout predictions with a fixed historical charge baseline.

This is a practical historical-baseline contrast, not a parameter- or optimization-
matched mechanism experiment. Export only aggregate observations and artifact hashes.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def macro(t): return t.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--parent',required=True);ap.add_argument('--project',required=True);a=ap.parse_args()
 root=Path(a.root);parent=Path(a.parent);project=Path(a.project)
 d=pd.read_parquet(parent/'data/samples.parquet');assert d.sample_id.is_unique
 pos=pd.Series(np.arange(len(d)),index=d.sample_id);d=d.set_index('sample_id',drop=False)
 y=np.load(parent/'data/p.npy',allow_pickle=False)
 old=pd.read_parquet(project/'data/qualification/release_v1/sample_manifest.parquet').set_index('sample_id');j=old.loc[d.index]
 assert np.array_equal(j.teacher_sequence_sha256.to_numpy(),d.sequence_sha256.to_numpy())
 d['strict_U']=j.sequence.notna()&j.sequence_sha256.notna()&(j.teacher_sequence_sha256==j.sequence_sha256)
 src_old=parent/'results/selected_fits.csv';src_new=root/'results/selected_fits.csv'
 source_hashes={'parent_selected':sha(src_old),'new_selected':sha(src_new),'data_release':sha(parent/'data/release.json'),'data':sha(parent/'data/samples.parquet'),'target':sha(parent/'data/p.npy'),'qualification':sha(project/'data/qualification/release_v1/sample_manifest.parquet')}
 bs=pd.read_csv(src_old);bs=bs[(bs.family=='composition_charge_deep')&(bs.scale=='all')]
 ns=pd.read_csv(src_new);ns=ns[ns['mode']=='residue_mean']
 assert len(bs)==2 and set(bs.fold)=={0,1}
 assert len(ns)==4 and set(ns.seed)=={1701,3407}
 role_sets={};checks=[]
 def load(run,kind,seed,fold):
  z=read(run/'receipt.json');c=z['config'];roles=read(run/'roles.json');ids=roles['evaluation_ids']
  assert c['seed']==seed and c['fold']==fold and (d.loc[ids].fold==fold).all()
  key='release_sha256' if kind=='residue_mean' else 'data_release_sha256'
  assert c[key]==source_hashes['data_release']
  if kind=='charge':role_sets[fold]={k:set(v) for k,v in roles.items()}
  else: assert {k:set(v) for k,v in roles.items()}==role_sets[fold], 'role identities changed'
  q=np.load(run/'evaluation_prediction.npy',allow_pickle=False).astype('float64');truth=y[pos.loc[ids]].astype('float64')
  assert sha(run/'evaluation_prediction.npy')==z['prediction_sha256']
  assert q.shape==truth.shape and np.isfinite(q).all() and (q>=0).all() and abs(q.sum(1)-1).max()<1e-6
  t=d.loc[ids].copy();t['H']=np.sqrt(.5*((np.sqrt(q)-np.sqrt(truth))**2).sum(1));t['CDF']=((q.cumsum(1)[:,:-1]-truth.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(q[:,[0,-1]]-truth[:,[0,-1]]).mean(1)
  assert abs(macro(t).H.mean()-z['result']['group_macro_H'])<1e-7
  checks.append({'model':kind,'seed':int(seed),'fold':int(fold),'prediction_sha256':z['prediction_sha256'],'roles_sha256':sha(run/'roles.json'),'H_recomputed':True})
  return t
 baseline=pd.concat([load(parent/'runs'/z.run_id,'charge',1701,int(z.fold)) for z in bs.itertuples()]);assert baseline.index.is_unique and len(baseline)==8756
 rows=[]
 for seed,part in ns.groupby('seed'):
  assert set(part.fold)=={0,1};cur=pd.concat([load(root/'runs'/z.run_id,'residue_mean',int(seed),int(z.fold)) for z in part.itertuples()])
  assert cur.index.is_unique and set(cur.index)==set(baseline.index)
  masks={'all':np.ones(len(cur),dtype=bool),'strict_T_equals_U':cur.strict_U.to_numpy(),'length_gt_600':(cur.length>600).to_numpy(),'strict_U_and_length_gt_600':(cur.strict_U&(cur.length>600)).to_numpy()}
  for name,mask in masks.items():
   one=cur.loc[mask];two=baseline.loc[one.index];g1=macro(one);g2=macro(two).loc[g1.index];assert g1.index.is_unique and set(g1.index)==set(g2.index)
   delta=(g1-g2).to_numpy();rng=np.random.default_rng(1701);bootstrap=np.asarray([delta[rng.integers(0,len(delta),len(delta))].mean(0) for _ in range(2000)])
   metrics={}
   for i,m in enumerate(['H','CDF','tail']):
    metrics[m]={'residue_mean':float(g1[m].mean()),'historical_charge_deep':float(g2[m].mean()),'paired_delta':float(delta[:,i].mean()),'descriptive_group_bootstrap_95':np.quantile(bootstrap[:,i],[.025,.975]).tolist()}
   rows.append({'seed':int(seed),'baseline_seed':1701,'stratum':name,'rows':len(one),'unique_T':int(one.protein_id.nunique()),'groups':len(g1),'metrics':metrics})
 assert sha(src_old)==source_hashes['parent_selected'] and sha(src_new)==source_hashes['new_selected'], 'selection changed during analysis'
 out={'schema':'pooling.historical_charge.aggregate.v1','status':'complete','comparison':'new residue_mean minus historical composition_charge_deep','folds':[0,1],'new_seeds':[1701,3407],'historical_baseline_seed':1701,'observations_per_seed':8756,'selection':'Existing stop-only selected configurations; no new tuning, fitting or selection','metrics':{'H':'Per-record H, then within unique T, then within group, then equal groups','CDF':'Mean squared cumulative probability error over 62 internal cutpoints; not continuous Wasserstein','tail':'Mean absolute error of first and last probability-bin masses'},'limitations':['Practical comparison to an existing historical baseline, not the new parameter- and batch-matched pooling mechanism contrast.','Historical charge baseline is a single fixed seed; both new seeds reuse its same predictions. Do not pool seeds as independent protein/group observations.','New and historical effective batching/optimization paths and parameter counts differ. Improvement cannot be attributed solely to pooling.','Retrospective exposed development; conditional group-bootstrap intervals exclude model selection, multiplicity and seed uncertainty.','Full-length and long-sequence strata remain conditional subsets, not independent full-U deployment qualification.'], 'source_sha256':source_hashes,'script_sha256':sha(__file__),'checks':checks,'strata':rows}
 destination=root/'results/pooling_vs_charge_public.json';destination.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':out['status'],'strata':rows},indent=2))

if __name__=='__main__': main()
