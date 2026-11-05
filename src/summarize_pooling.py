"""Independent prediction reduction, stop-only selection and descriptive strata."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def macro(d):
 z=d.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean();g=z.groupby('group_id').mean()
 return {'rows':len(d),'proteins':len(z),'groups':len(g),**{'group_macro_'+k:float(v) for k,v in g.mean().items()},**{'protein_macro_'+k:float(v) for k,v in z.mean().items()}},g
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--parent',required=True);ap.add_argument('--project',required=True);ap.add_argument('--expected',type=int,default=12);a=ap.parse_args();r=Path(a.root);p=Path(a.parent);d=pd.read_parquet(p/'data/samples.parquet');pos=pd.Series(np.arange(len(d)),index=d.sample_id);y=np.load(p/'data/p.npy');d=d.set_index('sample_id',drop=False)
 old=pd.read_parquet(Path(a.project)/'data/qualification/release_v1/sample_manifest.parquet').set_index('sample_id');j=old.loc[d.index];known=j.sequence.notna()&j.sequence_sha256.notna();eq=known&(j.teacher_sequence_sha256==j.sequence_sha256)
 d['relation']=np.where(eq,'strict_T_equals_U',np.where(known,'T_differs_from_U','U_unavailable'));d['length_bin']=pd.cut(d.length,[0,150,300,600,np.inf]).astype(str);d['coverage_bin']=pd.cut(d.coverage,[-np.inf,.7,.9,.98,np.inf]).astype(str)
 d['relation_length']=d.relation+' | '+d.length_bin
 rows=[];tables={};checks=[]
 for f in (r/'runs').glob('*/receipt.json'):
  z=json.loads(f.read_text());cfg=z['config'];roles=json.loads((f.parent/'roles.json').read_text());ids=roles['evaluation_ids'];truth=y[pos.loc[ids]];q=np.load(f.parent/'evaluation_prediction.npy')
  assert q.shape==truth.shape and np.isfinite(q).all() and (q>=0).all() and abs(q.sum(1)-1).max()<1e-6
  assert sha(f.parent/'evaluation_prediction.npy')==z['prediction_sha256'] and sha(f.parent/'best.pt')==z['checkpoint_sha256']
  gs={k:set(d.loc[v].group_id) for k,v in roles.items()};ss={k:set(d.loc[v].sequence_sha256) for k,v in roles.items()}
  for left,right in [('train_ids','stop_ids'),('train_ids','evaluation_ids'),('stop_ids','evaluation_ids')]:assert not gs[left]&gs[right] and not ss[left]&ss[right]
  assert (d.loc[ids].fold==cfg['fold']).all() and (d.loc[roles['stop_ids']].fold==(cfg['fold']+1)%5).all()
  t=d.loc[ids].copy();t['H']=np.sqrt(.5*((np.sqrt(truth.astype('float64'))-np.sqrt(q.astype('float64')))**2).sum(1));t['CDF']=((truth.cumsum(1)[:,:-1]-q.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(truth[:,[0,-1]]-q[:,[0,-1]]).mean(1);m,_=macro(t);assert abs(m['group_macro_H']-z['result']['group_macro_H'])<1e-7
  tables[z['run_id']]=t;checks.append({'run_id':z['run_id'],'prediction_recomputed':True,'group_and_exact_sequence_disjoint':True})
  rows.append({'run_id':z['run_id'],'mode':cfg['mode'],'fold':cfg['fold'],'seed':cfg['seed'],'lr':cfg['lr'],'stop_H':z['best_stop_H'],'H':m['group_macro_H'],'best_epoch':z['best_epoch'],'epochs':z['epochs'],'stop_reason':z['stop_reason'],'fit_wall_seconds':z['fit_wall_seconds'],'parameters':z['parameters'],'cached_readout_sequences_per_second':z['cached_readout_sequences_per_second'],'peak_GPU_bytes':z['peak_GPU_bytes'],'peak_RSS_bytes':z['peak_RSS_bytes']})
 if not rows:print('no completed fits');return
 allrows=pd.DataFrame(rows);selected=allrows.sort_values(['stop_H','lr']).groupby(['mode','fold','seed'],sort=False).head(1);selected.to_csv(r/'results/selected_fits.csv',index=False);allrows.to_csv(r/'results/all_fits.csv',index=False)
 full={};scores=[];strata=[]
 # OOF observations are unique within a seed. Different seeds evaluate the
 # same proteins and must never be concatenated as independent observations.
 for (mode,seed),part in selected.groupby(['mode','seed']):
  seed=int(seed);frame=pd.concat([tables[x] for x in part.run_id]);assert frame.index.is_unique
  full[mode,seed]=frame;m,_=macro(frame);scores.append(dict(mode=mode,seed=seed,folds=sorted(part.fold.tolist()),**m,selected_run_ids=part.run_id.tolist(),per_fold_H={str(z.fold):z.H for z in part.itertuples()},selected_budget_limited=int((part.stop_reason!='early_stop').sum())))
  for field in ['relation','length_bin','coverage_bin','relation_length']:
   for value,sub in frame.groupby(field):met,_=macro(sub);strata.append(dict(mode=mode,seed=seed,folds=sorted(part.fold.tolist()),field=field,value=str(value),**met))
 pairs=[]
 contrasts=[('residue_mean','mean_pre'),('max_pre','mean_pre'),('residue_attention','residue_mean'),('residue_attention','mean_pre')]
 paired_membership={}
 for seed in sorted(int(x) for x in selected.seed.unique()):
  for left,right in contrasts:
   lk=(left,seed);rk=(right,seed)
   if lk not in full or rk not in full or set(full[lk].index)!=set(full[rk].index):continue
   folds=sorted(int(x) for x in full[lk].fold.unique())
   for field in ['all','relation','length_bin','coverage_bin','relation_length']:
    values=['all'] if field=='all' else sorted(full[lk][field].unique())
    for value in values:
     one=full[lk] if field=='all' else full[lk][full[lk][field]==value];two=full[rk].loc[one.index];_,g1=macro(one);_,g2=macro(two)
     assert set(g1.index)==set(g2.index)
     delta=(g1.H-g2.loc[g1.index].H).to_numpy();rng=np.random.default_rng(1701);bs=np.asarray([rng.choice(delta,len(delta),True).mean() for _ in range(2000)])
     comparison=left+' minus '+right
     pairs.append({'comparison':comparison,'seed':seed,'folds':folds,'field':field,'value':str(value),'groups':len(delta),'rows':len(one),'delta_H':float(delta.mean()),'descriptive_95':np.quantile(bs,[.025,.975]).tolist(),'selection_adjusted':False,'seed_uncertainty_included':False})
     paired_membership[comparison,field,str(value),seed]=frozenset(one.index)
 # Optional seed summary averages already reduced paired effects, never
 # duplicates proteins/groups or creates a pooled bootstrap across seeds.
 seed_summaries=[]
 if pairs:
  for (comparison,field,value),part in pd.DataFrame(pairs).groupby(['comparison','field','value']):
   seeds=sorted(int(x) for x in part.seed)
   if len(seeds)<2:continue
   memberships=[paired_membership[comparison,field,value,s] for s in seeds]
   if any(z!=memberships[0] for z in memberships[1:]):continue
   byseed={str(int(z.seed)):float(z.delta_H) for z in part.itertuples()};vals=np.asarray(list(byseed.values()))
   seed_summaries.append({'comparison':comparison,'field':field,'value':value,'seeds':seeds,'n_seeds':len(seeds),'per_seed_delta_H':byseed,'mean_delta_H_across_seeds':float(vals.mean()),'min_delta_H':float(vals.min()),'max_delta_H':float(vals.max()),'same_evaluation_observations':True,'groups_per_seed':int(part.groups.iloc[0]),'rows_per_seed':int(part.rows.iloc[0]),'interpretation':'Descriptive mean of per-seed paired group-macro differences on identical observations; not independent sample replication, ensemble performance, or sufficient characterization of seed uncertainty.'})
 summary={'stage':'pooling_20261105','status':'complete' if len(rows)==a.expected else 'partial_or_extra','fits_completed':len(rows),'expected_fits':a.expected,'selected':len(selected),'seeds_present':sorted(int(x) for x in allrows.seed.unique()),'scores':scores,'strata':strata,'paired':pairs,'paired_seed_summary':seed_summaries,'fit_wall_seconds_total':float(allrows.fit_wall_seconds.sum()),'all_fit_stop_counts':allrows.stop_reason.value_counts().to_dict(),'prediction_checks':checks,'qualification':'Historical T/vertex-p exposed development; all OOF/stratum/paired metrics reduced separately by seed; no blind/new-C/full-U claim; source PDB only; nearest homology identity unavailable from existing scored search'}
 (r/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['status','fits_completed','scores','fit_wall_seconds_total']},indent=2))
if __name__=='__main__':main()
