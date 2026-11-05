"""Full OOF reduction across reused and new trials, separately for each seed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reduce(d):
 z=d.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean();g=z.groupby('group_id').mean()
 return {'rows':len(d),'proteins':len(z),'groups':len(g),**{'group_macro_'+k:float(v) for k,v in g.mean().items()},**{'protein_macro_'+k:float(v) for k,v in z.mean().items()}},g
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--project',required=True);a=ap.parse_args();r=Path(a.root);plan=json.loads((r/'records/confirmation_plan.json').read_text());parent=Path(plan['atlas_source']);d=pd.read_parquet(parent/'data/samples.parquet');pos=pd.Series(np.arange(len(d)),index=d.sample_id);y=np.load(parent/'data/p.npy');d=d.set_index('sample_id',drop=False);old=pd.read_parquet(Path(a.project)/'data/qualification/release_v1/sample_manifest.parquet').set_index('sample_id').loc[d.index]
 known=old.sequence.notna()&old.sequence_sha256.notna();eq=known&(old.teacher_sequence_sha256==old.sequence_sha256);d['relation']=np.where(eq,'strict_T_equals_U',np.where(known,'T_differs_from_U','U_unavailable'));d['length_bin']=pd.cut(d.length,[0,150,300,600,np.inf]).astype(str);d['coverage_bin']=pd.cut(d.coverage,[-np.inf,.7,.9,.98,np.inf]).astype(str);d['relation_length']=d.relation+' | '+d.length_bin
 sources=[]
 for x in plan['reuse']:
  path=Path(x['path']);assert sha(path/'receipt.json')==x['receipt_sha256'];sources.append((path,True))
 sources += [(f.parent,False) for f in (r/'runs').glob('*/receipt.json')]
 rows=[];tables={};identities=set()
 for path,reused in sources:
  z=json.loads((path/'receipt.json').read_text());cfg=z['config'];family=cfg.get('mode',cfg.get('family'));key=(family,cfg['fold'],cfg['seed']);assert key not in identities;identities.add(key)
  assert family in ['mean_pre','residue_mean','composition_charge_deep'] and cfg['lr']==.0003 and cfg['seed'] in [1701,3407] and cfg.get('release_sha256',cfg.get('data_release_sha256'))==sha(parent/'data/release.json')
  roles=json.loads((path/'roles.json').read_text());ids=roles['evaluation_ids'];meta=d.loc[ids].copy();pred=np.load(path/'evaluation_prediction.npy');target=y[pos.loc[ids]];assert pred.shape==target.shape and np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max()<1e-6
  assert sha(path/'evaluation_prediction.npy')==z['prediction_sha256'] and sha(path/'best.pt')==z['checkpoint_sha256']
  sets={k:set(d.loc[v].group_id) for k,v in roles.items()};seqs={k:set(d.loc[v].sequence_sha256) for k,v in roles.items()}
  for l,h in [('train_ids','stop_ids'),('train_ids','evaluation_ids'),('stop_ids','evaluation_ids')]:assert not sets[l]&sets[h] and not seqs[l]&seqs[h]
  assert set(ids)==set(d.index[d.fold==cfg['fold']]) and set(roles['stop_ids'])==set(d.index[d.fold==(cfg['fold']+1)%5]) and set(roles['train_ids'])==set(d.index[~d.fold.isin([cfg['fold'],(cfg['fold']+1)%5])])
  meta['H']=np.sqrt(.5*((np.sqrt(target.astype('float64'))-np.sqrt(pred.astype('float64')))**2).sum(1));meta['CDF']=((target.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);meta['tail']=abs(target[:,[0,-1]]-pred[:,[0,-1]]).mean(1);score,_=reduce(meta);assert abs(score['group_macro_H']-z['result']['group_macro_H'])<1e-7
  tables[key]=meta;rows.append({'family':family,'fold':cfg['fold'],'seed':cfg['seed'],'run_id':z['run_id'],'reused':reused,'artifact_path':str(path),'H':score['group_macro_H'],'stop_H':z['best_stop_H'],'epochs':z['epochs'],'best_epoch':z['best_epoch'],'stop_reason':z['stop_reason'],'fit_wall_seconds':z.get('fit_wall_seconds',z.get('wall_seconds')),'trainer_sha256':cfg['script_sha256'],'model_sha256':cfg['model_sha256']})
 table=pd.DataFrame(rows);table.to_csv(r/'results/all_fits.csv',index=False);scores=[];strata=[];frames={}
 for (family,seed),part in table.groupby(['family','seed']):
  frame=pd.concat([tables[(family,int(f),int(seed))] for f in part.fold]);assert frame.index.is_unique;frames[family,int(seed)]=frame;score,_=reduce(frame);scores.append(dict(family=family,seed=int(seed),folds=sorted(int(x) for x in part.fold),per_fold_H={str(z.fold):z.H for z in part.itertuples()},**score))
  for field in ['relation','length_bin','coverage_bin','relation_length']:
   for value,sub in frame.groupby(field):score,_=reduce(sub);strata.append(dict(family=family,seed=int(seed),field=field,value=str(value),**score))
 pairs=[]
 for seed in [1701,3407]:
  for left,right in [('residue_mean','mean_pre'),('residue_mean','composition_charge_deep'),('mean_pre','composition_charge_deep')]:
   if (left,seed) not in frames or (right,seed) not in frames:continue
   one=frames[left,seed];two=frames[right,seed]
   if set(one.index)!=set(two.index):continue
   for field in ['all','relation','length_bin','coverage_bin','relation_length']:
    for value in (['all'] if field=='all' else sorted(one[field].unique())):
     x=one if field=='all' else one[one[field]==value];yframe=two.loc[x.index];_,gx=reduce(x);_,gy=reduce(yframe);diff=gx-gy.loc[gx.index];delta=diff.H.to_numpy();rng=np.random.default_rng(1701);bs=np.array([rng.choice(delta,len(delta),True).mean() for _ in range(2000)])
     pairs.append(dict(comparison=left+' minus '+right,seed=seed,field=field,value=str(value),rows=len(x),groups=len(delta),delta_H=float(delta.mean()),descriptive_95=np.quantile(bs,[.025,.975]).tolist(),delta_CDF=float(diff.CDF.mean()),delta_tail=float(diff['tail'].mean()),per_fold_delta_H={str(k):reduce(x[x.fold==k])[0]['group_macro_H']-reduce(yframe[yframe.fold==k])[0]['group_macro_H'] for k in sorted(x.fold.unique())},selection_adjusted=False))
 expected={(f,k,s) for f in ['mean_pre','residue_mean','composition_charge_deep'] for k in range(5) for s in [1701,3407]};complete=identities==expected
 summary={'stage':'overnight_20261106_confirmation','status':'complete' if complete else 'partial','expected_total':30,'fits_total':len(rows),'reused_fits':int(table.reused.sum()),'new_fits':int((~table.reused).sum()),'missing':[list(k) for k in sorted(expected-identities)],'all_predictions_recomputed':len(rows),'scores':scores,'strata':strata,'paired':pairs,'new_fit_wall_seconds_sum':float(table.loc[~table.reused,'fit_wall_seconds'].sum()),'all_stop_counts':table.stop_reason.value_counts().to_dict(),'qualification':'Retrospective exposed development OOF. Seeds use the same observations and are reduced separately. Existing sequence/identity closure, not profile/family-independent final confirmation. Charge pipeline retained from its validated historical runner; mean/residue use matched pooling runner.'}
 (r/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['status','fits_total','new_fits','missing','scores']},indent=2))
if __name__=='__main__':main()
