"""Independent probability/metric audit and ID-free aggregate reporting."""
import os

import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def group_scores(frame):
 return frame.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean()
def measure(frame):
 g=group_scores(frame)
 return dict(rows=len(frame),unique_T=frame.protein_id.nunique(),groups=len(g),H=float(g.H.mean()),CDF=float(g.CDF.mean()),tail=float(g['tail'].mean()))
def contrast(a,b,label):
 assert set(a.index)==set(b.index)
 delta=(a.H-b.loc[a.index].H).to_numpy();rng=np.random.default_rng(1701)
 boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
 return dict(contrast=label,groups=len(delta),delta_H=float(delta.mean()),delta_CDF=float((a.CDF-b.loc[a.index].CDF).mean()),delta_tail=float((a['tail']-b.loc[a.index,'tail']).mean()),descriptive_95_group_bootstrap=np.quantile(boots,[.025,.975]).tolist(),selection_adjusted=False,seed_variation_included=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--atlas',required=True);a=ap.parse_args()
 r=Path(a.root);atlas=Path(a.atlas);out=r/'results';out.mkdir(exist_ok=True);public=out/'public';public.mkdir(exist_ok=True)
 meta=pd.read_parquet(atlas/'data/samples.parquet');y=np.load(atlas/'data/p.npy').astype('float64');m=meta.set_index('sample_id');positions=pd.Series(np.arange(len(meta)),index=meta.sample_id)
 old=pd.read_parquet(str(Path(os.environ["PROJECT_ROOT"]) / 'data/qualification/release_v1/sample_manifest.parquet')).set_index('sample_id').loc[meta.sample_id]
 relation=np.where(old.sequence_sha256.isna(),'U_unavailable',np.where(old.sequence_sha256==old.teacher_sequence_sha256,'strict_T_equals_U','T_differs_from_U'))
 relation_map=dict(zip(meta.sample_id,relation));receipts=[];data={};audit=[]
 for p in sorted((r/'runs').glob('*/receipt.json')):
  rec=json.loads(p.read_text());cfg=rec['config'];run=p.parent
  assert sha(run/'evaluation_prediction.npy')==rec['prediction_sha256'] and sha(run/'model.joblib')==rec['checkpoint_sha256']
  roles=json.loads((run/'roles.json').read_text());ids=roles['evaluation_ids'];idx=positions.loc[ids].to_numpy();frame=m.loc[ids].copy().reset_index();pred=np.load(run/'evaluation_prediction.npy')
  assert pred.shape==(len(ids),63) and np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max()<1e-6
  assert (frame.fold==cfg['fold']).all() and (m.loc[roles['stop_ids']].fold==(cfg['fold']+1)%5).all()
  gs={k:set(m.loc[v].group_id) for k,v in roles.items()}
  assert not(gs['train_ids']&gs['stop_ids'] or gs['train_ids']&gs['evaluation_ids'] or gs['stop_ids']&gs['evaluation_ids'])
  assert set(roles['train_ids'])==set(meta.loc[~meta.fold.isin([cfg['fold'],(cfg['fold']+1)%5]),'sample_id'])
  frame['H']=np.sqrt(.5*((np.sqrt(y[idx])-np.sqrt(pred))**2).sum(1));frame['CDF']=((y[idx].cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);frame['tail']=abs(y[idx][:,[0,-1]]-pred[:,[0,-1]]).mean(1)
  frame['T_U_relation']=frame.sample_id.map(relation_map);frame['length_bin']=pd.cut(frame.length,[0,150,300,600,np.inf]).astype(str)
  score=measure(frame);error=abs(score['H']-rec['result']['group_macro_H']);assert error<1e-12
  entry=dict(candidate=cfg['candidate'],fold=cfg['fold'],seed=cfg['seed'],stop_H=rec['stop']['group_macro_H'],**score,
   fit_seconds=rec['fit_seconds'],wall_seconds=rec['wall_seconds'],peak_RSS_GiB=rec['peak_RSS_bytes']/2**30,head_sequences_per_second=rec['head_sequences_per_second'],negative_raw_coordinate_fraction=rec['negative_raw_coordinate_fraction'])
  receipts.append(entry);data[(cfg['candidate'],cfg['fold'])]=frame
  replay=json.loads((run/'independent_replay.json').read_text()) if (run/'independent_replay.json').exists() else {}
  audit.append(dict(candidate=cfg['candidate'],fold=cfg['fold'],probability_valid=True,group_isolation=True,all_eligible_train_rows_present=True,metric_max_H_error=error,checkpoint_replay_rows=rec['checkpoint_replay_rows'],checkpoint_replay_error=rec['checkpoint_replay_max_abs_error'],fresh_process_raw_feature_replay=replay.get('status')=='complete',fresh_process_max_probability_difference=replay.get('max_probability_difference')))
 table=pd.DataFrame(receipts);table.to_csv(public/'fits.csv',index=False)
 screen=table[table.fold==0].sort_values('stop_H');screen.to_csv(public/'screen_by_stop.csv',index=False)
 confirmed=json.loads((r/'records/promotion.json').read_text())['candidates'] if (r/'records/promotion.json').exists() else []
 pooled=[];strata=[];comparisons=[];neural_paired=[];reference_data={}
 confirmation=r.parent/'confirmation/results/all_fits.csv';confirmation_sha=sha(confirmation) if confirmation.exists() else None
 if confirmation.exists():
  nfits=pd.read_csv(confirmation)
  for family in ['composition_charge_deep','mean_pre','residue_mean']:
   fits=nfits[(nfits.family==family)&(nfits.seed==1701)]
   if set(fits.fold)!=set(range(5)):continue
   parts=[]
   for fit in fits.to_dict('records'):
    run=Path(fit['artifact_path']);roles=json.loads((run/'roles.json').read_text());ids=roles['evaluation_ids'];idx=positions.loc[ids].to_numpy();part=m.loc[ids].copy().reset_index();pred=np.load(run/'evaluation_prediction.npy').astype('float64')
    assert pred.shape==(len(ids),63) and np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max()<1e-6
    part['H']=np.sqrt(.5*((np.sqrt(y[idx])-np.sqrt(pred))**2).sum(1));part['CDF']=((y[idx].cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);part['tail']=abs(y[idx][:,[0,-1]]-pred[:,[0,-1]]).mean(1)
    part['T_U_relation']=part.sample_id.map(relation_map);parts.append(part)
   current=pd.concat(parts);assert current.sample_id.is_unique and set(current.sample_id)==set(meta.sample_id)
   reference_data[family]=current
 for candidate in confirmed:
  parts=[v for (k,f),v in data.items() if k==candidate];frame=pd.concat(parts)
  assert frame.sample_id.is_unique
  folds=sorted(table.loc[table.candidate==candidate,'fold'].tolist())
  if len(folds)==5:assert set(frame.sample_id)==set(meta.sample_id)
  pooled.append(dict(candidate=candidate,folds=folds,**measure(frame)))
  for axis in ['T_U_relation','length_bin']:
   for label,part in frame.groupby(axis):strata.append(dict(candidate=candidate,axis=axis,stratum=label,**measure(part)))
  if len(folds)==5:
   strata.append(dict(candidate=candidate,axis='intersection',stratum='strict_U_and_length_gt600',**measure(frame[(frame.T_U_relation=='strict_T_equals_U')&(frame.length>600)])))
   for family,reference_frame in reference_data.items():
    for label in ['all','strict_T_equals_U','length_gt600','strict_U_and_length_gt600']:
     def subset(d):
      if label=='all':return d
      if label=='strict_T_equals_U':return d[d.T_U_relation=='strict_T_equals_U']
      if label=='length_gt600':return d[d.length>600]
      return d[(d.T_U_relation=='strict_T_equals_U')&(d.length>600)]
     current=subset(frame);ref=subset(reference_frame);assert set(current.sample_id)==set(ref.sample_id)
     v=contrast(group_scores(current),group_scores(ref),candidate+' minus '+family);v.update(stratum=label,seed=1701,folds=[0,1,2,3,4],tree_score=measure(current),neural_score=measure(ref));neural_paired.append(v)
  # Old neural comparison only on its two actually available folds, not a
  # mismatched fivefold mean; prediction-level common sample check is required.
  selected=pd.read_csv(atlas/'results/selected_fits.csv')
  for reference in ['composition_charge_deep','esmc_charge']:
   refs=[];tree=[]
   for fit in selected[(selected.family==reference)&(selected.scale=='all')].to_dict('records'):
    if (candidate,fit['fold']) not in data:continue
    ref=pd.read_parquet(Path(fit['artifact_path'])/'evaluation_metrics.parquet').rename(columns={'cutpoint_CDF_MSE':'CDF','edge_bin_mass_MAE':'tail'})
    current=data[(candidate,fit['fold'])];assert set(ref.sample_id)==set(current.sample_id);refs.append(ref);tree.append(current)
   if refs:
    aa=group_scores(pd.concat(tree));bb=group_scores(pd.concat(refs));comparisons.append(contrast(aa,bb,candidate+' minus old_'+reference+' on common folds0/1'))
 if len(confirmed)==2 and all((k,f) in data for k in confirmed for f in range(5)):
  aa=group_scores(pd.concat([data[(confirmed[0],f)] for f in range(5)]));bb=group_scores(pd.concat([data[(confirmed[1],f)] for f in range(5)]));comparisons.append(contrast(aa,bb,confirmed[0]+' minus '+confirmed[1]))
 pd.DataFrame(pooled).to_csv(public/'pooled.csv',index=False);pd.DataFrame(strata).to_csv(public/'strata.csv',index=False)
 assert confirmation_sha is None or confirmation_sha==sha(confirmation),'neural fit index changed during paired audit; rerun on stable snapshot'
 (public/'neural_fivefold_paired.json').write_text(json.dumps(dict(confirmation_fits_sha256=confirmation_sha,available_complete_seed1701_families=list(reference_data),paired=neural_paired),indent=2)+'\n')
 summary=dict(status='complete' if confirmed and all((k,f) in data for k in confirmed for f in range(5)) else 'partial',successful_fits=len(receipts),screened=len(screen),promoted=confirmed,pooled=pooled,paired=comparisons,
  sum_fit_wall_seconds=float(table.wall_seconds.sum()),max_peak_RSS_GiB=float(table.peak_RSS_GiB.max()),
  independent_prediction_recomputations=len(audit),checkpoint_replays=sum(v['checkpoint_replay_rows'] for v in audit),
  independent_fresh_process_replay_fits=sum(v['fresh_process_raw_feature_replay'] for v in audit),
  qualification='Historical Atlas T->vertex p, retrospective exposed development; fold0 stop-selected global configs reused across folds, not nested unbiased generalization estimate; one seed; no new C or full U deployment proof',
  output='Forests regress p or sqrt(p); sqrt outputs squared and normalized; HGB negative coordinates clipped. No claim that regression normalization equals softmax likelihood training.',
  audit=audit)
 (public/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='audit'},indent=2))
if __name__=='__main__':main()
