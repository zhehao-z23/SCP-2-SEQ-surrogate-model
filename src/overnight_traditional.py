"""Bounded CPU tree baselines; unchanged Atlas T->vertex-p development release.

Native forests predict 63 conditional sqrt-probability (or probability) means.
sqrt predictions are squared then normalized; these are not softmax logits.
Histogram boosting fits the 63 sqrt coordinates separately, clips negative
coordinates, and uses the same square/normalize map. All transforms fit train.
"""
import argparse, hashlib, json, os, resource, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

CANDIDATES={
 'et_q_leaf1':dict(kind='et',features='charge',target='sqrt_p',leaf=1,max_features=1.0,trees=256),
 'et_q_leaf3':dict(kind='et',features='charge',target='sqrt_p',leaf=3,max_features=1.0,trees=256),
 'rf_q_leaf3':dict(kind='rf',features='charge',target='sqrt_p',leaf=3,max_features=1.0,trees=256),
 'et_p_leaf3':dict(kind='et',features='charge',target='p',leaf=3,max_features=1.0,trees=256),
 'et_q_esmc_charge':dict(kind='et',features='esmc_charge',target='sqrt_p',leaf=3,max_features=.3,trees=256),
 'hgb_q':dict(kind='hgb',features='charge',target='sqrt_p',iterations=200,leaves=15,learning_rate=.08,l2=1.0),
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def metrics(y,p,meta):
 y=y.astype('float64');p=p.astype('float64')
 assert y.shape==p.shape and np.isfinite(p).all() and (p>=0).all() and np.max(abs(p.sum(1)-1))<1e-6
 d=meta[['sample_id','protein_id','group_id','length','coverage','source']].copy().reset_index(drop=True)
 d['H']=np.sqrt(.5*((np.sqrt(y)-np.sqrt(p))**2).sum(1))
 d['cutpoint_CDF_MSE']=((y.cumsum(1)[:,:-1]-p.cumsum(1)[:,:-1])**2).mean(1)
 d['edge_bin_mass_MAE']=abs(y[:,[0,-1]]-p[:,[0,-1]]).mean(1)
 cols=['H','cutpoint_CDF_MSE','edge_bin_mass_MAE']
 protein=d.groupby(['group_id','protein_id'])[cols].mean();group=protein.groupby('group_id').mean()
 result={'group_macro_'+k:float(v) for k,v in group.mean().items()}
 result.update({'protein_macro_'+k:float(v) for k,v in protein.mean().items()})
 result.update(samples=len(d),proteins=len(protein),groups=len(group))
 return result,d
def probability(raw,target):
 p=np.maximum(np.asarray(raw,dtype='float64'),0)
 if target=='sqrt_p':p=p*p
 sums=p.sum(1,keepdims=True)
 assert np.isfinite(p).all() and (sums>0).all()
 return p/sums
def load_features(atlas,meta,kind):
 d=atlas/'data';paths=[d/'composition_charge.npy'];x=np.load(paths[0])
 if kind=='esmc_charge':
  cache=d/'esmc_300m_mean';rec=json.loads((cache/'receipt.json').read_text());assert rec['status']=='complete'
  idx=pd.read_parquet(cache/'index.parquet');emb=np.load(cache/'mean.npy')
  assert idx.sequence_sha256.is_unique and meta.sequence_sha256.isin(idx.sequence_sha256).all()
  positions=pd.Series(np.arange(len(idx)),index=idx.sequence_sha256).loc[meta.sequence_sha256].to_numpy()
  x=np.concatenate([emb[positions],x],1);paths.extend([cache/'receipt.json',cache/'index.parquet',cache/'mean.npy'])
 assert len(x)==len(meta) and not np.isinf(x).any()
 return x,{str(p):sha(p) for p in paths}
def raw_predict(model,x,kind):
 return np.column_stack([m.predict(x) for m in model]) if kind=='hgb' else model.predict(x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--atlas',required=True)
 ap.add_argument('--candidate',choices=CANDIDATES,required=True);ap.add_argument('--fold',type=int,required=True)
 ap.add_argument('--seed',type=int,default=1701);a=ap.parse_args()
 root=Path(a.root);atlas=Path(a.atlas);root.mkdir(parents=True,exist_ok=True)
 resource.setrlimit(resource.RLIMIT_AS,(32<<30,32<<30));threadpool_limits(limits=8)
 started=time.monotonic();meta=pd.read_parquet(atlas/'data/samples.parquet');y=np.load(atlas/'data/p.npy')
 release=json.loads((atlas/'data/release.json').read_text())
 for name in ['samples.parquet','p.npy','composition_charge.npy']:assert sha(atlas/'data'/name)==release['artifacts'][name]
 assert y.shape==(len(meta),63) and np.isfinite(y).all() and (y>=0).all() and np.max(abs(y.sum(1)-1))<1e-6
 tr=np.flatnonzero(~meta.fold.isin([a.fold,(a.fold+1)%5]));va=np.flatnonzero(meta.fold==(a.fold+1)%5);te=np.flatnonzero(meta.fold==a.fold)
 roles={k:meta.iloc[v].sample_id.tolist() for k,v in [('train_ids',tr),('stop_ids',va),('evaluation_ids',te)]}
 for left,right in [(tr,va),(tr,te),(va,te)]:
  assert not set(meta.iloc[left].group_id)&set(meta.iloc[right].group_id)
  assert not set(meta.iloc[left].sequence_sha256)&set(meta.iloc[right].sequence_sha256)
 spec=CANDIDATES[a.candidate];x,feature_hashes=load_features(atlas,meta,spec['features'])
 median=np.nan_to_num(np.nanmedian(x[tr],axis=0));missing=np.isnan(x)
 x=np.concatenate([np.where(missing,median,x),missing[:,-39:].astype('float32')],1).astype('float32')
 assert np.isfinite(x).all()
 cfg=dict(candidate=a.candidate,spec=spec,fold=a.fold,seed=a.seed,input_dim=x.shape[1],threads=8,max_address_space_GiB=32,
  target_to_p='q=max(raw,0); p=q*q/sum(q*q)' if spec['target']=='sqrt_p' else 'p=max(raw,0)/sum(max(raw,0))',
  probability_note='Regression estimates, not logits/softmax; p forest is a convex probability average; sqrt forest is an extrinsic mean followed by positive-sphere projection. HGB may need clipping.',
  selection='screening uses stop group-macro H; chosen fold0 config fixed across further development folds; not nested blind evaluation',
  train_weight='inverse multiplicity of identical T protein within train, normalized mean one',
  script_sha256=sha(__file__),data_release_sha256=sha(atlas/'data/release.json'),feature_hashes=feature_hashes,
  environment=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__))
 identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();out=root/'runs'/identity;out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.json').exists():print('existing',identity);return
 with (out/'started.json').open('x') as f:json.dump(dict(pid=os.getpid(),start_epoch=time.time()),f)
 put(out/'config.json',cfg);put(out/'roles.json',roles)
 counts=meta.iloc[tr].protein_id.value_counts();weights=1/meta.iloc[tr].protein_id.map(counts).to_numpy();weights/=weights.mean()
 target=np.sqrt(y[tr].astype('float64')) if spec['target']=='sqrt_p' else y[tr].astype('float64')
 fit_start=time.monotonic()
 if spec['kind']=='hgb':
  model=[]
  for k in range(63):
   m=HistGradientBoostingRegressor(loss='squared_error',learning_rate=spec['learning_rate'],max_iter=spec['iterations'],max_leaf_nodes=spec['leaves'],l2_regularization=spec['l2'],max_bins=127,early_stopping=False,random_state=a.seed)
   m.fit(x[tr],target[:,k],sample_weight=weights);model.append(m)
   put(out/'progress.json',dict(targets_complete=k+1,targets=63,elapsed_seconds=time.monotonic()-started))
 else:
  constructor=ExtraTreesRegressor if spec['kind']=='et' else RandomForestRegressor
  model=constructor(n_estimators=spec['trees'],min_samples_leaf=spec['leaf'],max_features=spec['max_features'],criterion='squared_error',random_state=a.seed,n_jobs=8)
  model.fit(x[tr],target,sample_weight=weights)
 fit_seconds=time.monotonic()-fit_start
 va_raw=raw_predict(model,x[va],spec['kind']);te_raw=raw_predict(model,x[te],spec['kind'])
 valid,_=metrics(y[va],probability(va_raw,spec['target']),meta.iloc[va]);pred=probability(te_raw,spec['target'])
 result,per=metrics(y[te],pred,meta.iloc[te]);np.save(out/'evaluation_prediction.npy',pred);per.to_parquet(out/'evaluation_metrics.parquet',index=False)
 np.save(out/'stop_prediction.npy',probability(va_raw,spec['target']))
 joblib.dump(dict(model=model,median=median,config=cfg),out/'model.joblib',compress=1)
 # Reload 128 deterministic evaluation rows; checkpoint replay is end-to-end
 # from the fit-only transformed inputs, and verifies the persisted forest.
 saved=joblib.load(out/'model.joblib');pick=np.unique(np.linspace(0,len(te)-1,128,dtype=int))
 replay=probability(raw_predict(saved['model'],x[te[pick]],spec['kind']),spec['target'])
 replay_error=float(abs(replay-pred[pick]).max());assert replay_error<1e-12
 t0=time.monotonic();_ = probability(raw_predict(model,x[te],spec['kind']),spec['target']);inference=time.monotonic()-t0
 receipt=dict(status='complete',run_id=identity,config=cfg,stop=valid,result=result,fit_seconds=fit_seconds,wall_seconds=time.monotonic()-started,
  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,training_rows=len(tr),stop_rows=len(va),evaluation_rows=len(te),
  prediction_sha256=sha(out/'evaluation_prediction.npy'),checkpoint_sha256=sha(out/'model.joblib'),roles_sha256=sha(out/'roles.json'),
  checkpoint_replay_rows=len(pick),checkpoint_replay_max_abs_error=replay_error,negative_raw_coordinate_fraction=float((te_raw<0).mean()),
  head_sequences_per_second=len(te)/inference,not_blind=True,new_C_transfer_permitted=False)
 put(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['run_id','stop','result','wall_seconds','peak_RSS_bytes']}),flush=True)
if __name__=='__main__':main()
