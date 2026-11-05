"""Independently recalculate saved predictions and select LR using stop only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--expected-fits',type=int,default=48);a=ap.parse_args();r=Path(a.root);d=pd.read_parquet(r/'data/samples.parquet').set_index('sample_id');y=np.load(r/'data/p.npy');receipts=[];tables={};checks=[]
 for path in sorted((r/'runs').glob('*/receipt.json')):
  x=json.loads(path.read_text());cfg=x['config'];out=path.parent;roles=json.loads((out/'roles.json').read_text());ids=roles['evaluation_ids'];meta=d.loc[ids].copy()
  loc=pd.Series(np.arange(len(d)),index=d.index).loc[ids].to_numpy();truth=y[loc];pred=np.load(out/'evaluation_prediction.npy')
  assert pred.shape==truth.shape and np.isfinite(pred).all() and (pred>=0).all() and np.max(abs(pred.sum(1)-1))<1e-6
  assert sha(out/'evaluation_prediction.npy')==x['prediction_sha256'] and sha(out/'best.pt')==x['checkpoint_sha256']
  gs={k:set(d.loc[v].group_id) for k,v in roles.items()}
  assert not gs['train_ids']&gs['stop_ids'] and not gs['train_ids']&gs['evaluation_ids'] and not gs['stop_ids']&gs['evaluation_ids']
  assert (d.loc[roles['stop_ids']].fold==(cfg['fold']+1)%5).all() and (meta.fold==cfg['fold']).all()
  meta['H']=np.sqrt(((np.sqrt(truth.astype('float64'))-np.sqrt(pred.astype('float64')))**2).sum(1)/2)
  meta['CDF']=((truth.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1)
  meta['tail']=abs(truth[:,[0,-1]]-pred[:,[0,-1]]).mean(1)
  group=meta.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean()
  assert abs(group.H.mean()-x['result']['group_macro_H'])<1e-7
  checks.append({'run_id':x['run_id'],'prediction_recomputed':True,'max_H_summary_error':abs(group.H.mean()-x['result']['group_macro_H']),'partition_groups_disjoint':True})
  row={'run_id':x['run_id'],'family':cfg['family'],'scale':cfg['train_groups_requested'],'fold':cfg['fold'],'lr':cfg['lr'],'stop_H':x['best_stop_H'],'evaluation_H':float(group.H.mean()),'training_samples':x['training_samples'],'training_groups':x['training_groups'],'training_proteins':x['training_proteins'],'epochs':x['epochs'],'best_epoch':x['best_epoch'],'stop_reason':x['stop_reason'],'wall_seconds':x['wall_seconds'],'peak_gpu_bytes':x['peak_gpu_allocated_bytes'],'head_per_second':x['head_inference_sequences_per_second'],'artifact_path':str(out)}
  receipts.append(row);tables[x['run_id']]=group
 if not receipts:return
 allrows=pd.DataFrame(receipts);allrows.to_csv(r/'results/all_fits.csv',index=False)
 selected=allrows.sort_values(['stop_H','lr']).groupby(['family','scale','fold'],sort=False).head(1)
 selected.to_csv(r/'results/selected_fits.csv',index=False)
 summaries=[];group_tables={}
 for (family,scale),part in selected.groupby(['family','scale']):
  g=pd.concat([tables[z] for z in part.run_id]);assert g.index.is_unique
  key=family+':'+str(scale);group_tables[key]=g
  summaries.append({'family':family,'scale':scale,'folds':sorted(part.fold.tolist()),'group_macro_H':float(g.H.mean()),'group_macro_CDF_MSE':float(g.CDF.mean()),'group_macro_edge_mass_MAE':float(g['tail'].mean()),'evaluation_groups':len(g),'train_samples_range':[int(part.training_samples.min()),int(part.training_samples.max())],'train_groups_range':[int(part.training_groups.min()),int(part.training_groups.max())],'selected_run_ids':part.run_id.tolist(),'budget_limited_selected':int((part.stop_reason!='early_stop').sum())})
 comparisons=[];rng=np.random.default_rng(1701)
 for scale in ['200','1000','all']:
  base=group_tables.get('composition_charge:'+scale)
  if base is None:continue
  for family in ['composition_length','esmc_mean','esmc_charge','composition_charge_deep']:
   other=group_tables.get(family+':'+scale)
   if other is None or set(other.index)!=set(base.index):continue
   delta=other.loc[base.index].H.to_numpy()-base.H.to_numpy()
   boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
   comparisons.append({'family_minus_charge':family,'scale':scale,'groups':len(delta),'delta_H':float(delta.mean()),'descriptive_group_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'selection_adjusted':False,'seed_variation_included':False})
 if 'composition_charge_deep:all' in group_tables and 'esmc_charge:all' in group_tables:
  base=group_tables['composition_charge_deep:all'];other=group_tables['esmc_charge:all'];assert set(base.index)==set(other.index)
  delta=other.loc[base.index].H.to_numpy()-base.H.to_numpy();boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
  comparisons.append({'comparison':'esmc_charge minus composition_charge_deep','scale':'all','groups':len(delta),'delta_H':float(delta.mean()),'descriptive_group_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'selection_adjusted':False,'seed_variation_included':False})
 summary={'status':'complete' if len(receipts)==a.expected_fits else 'partial_or_extra','fits_complete':len(receipts),'expected_fits':a.expected_fits,'all_recomputed':len(checks),'learning_curves':summaries,'paired':comparisons,'fit_wall_seconds_total':float(allrows.wall_seconds.sum()),'qualification':'retrospective development; LR selected on stop, model/scale comparison on exposed evaluation; no blind or new-C claim'}
 (r/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');(r/'results/audit.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
