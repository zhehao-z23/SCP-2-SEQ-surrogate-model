"""Fresh-process joblib recovery from raw features, without trainer imports."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--atlas',required=True);a=ap.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(32<<30,32<<30));threadpool_limits(limits=8);start=time.monotonic()
 run=Path(a.run);atlas=Path(a.atlas);rec=json.loads((run/'receipt.json').read_text());cfg=rec['config'];assert sha(run/'model.joblib')==rec['checkpoint_sha256']
 for p,h in cfg['feature_hashes'].items():assert sha(p)==h
 saved=joblib.load(run/'model.joblib');assert saved['config']==cfg
 meta=pd.read_parquet(atlas/'data/samples.parquet');loc=pd.Series(np.arange(len(meta)),index=meta.sample_id)
 roles=json.loads((run/'roles.json').read_text());tr=loc.loc[roles['train_ids']].to_numpy();te=loc.loc[roles['evaluation_ids']].to_numpy()
 x=np.load(atlas/'data/composition_charge.npy')
 if cfg['spec']['features']=='esmc_charge':
  cache=atlas/'data/esmc_300m_mean';idx=pd.read_parquet(cache/'index.parquet');emb=np.load(cache/'mean.npy');pos=pd.Series(np.arange(len(idx)),index=idx.sequence_sha256).loc[meta.sequence_sha256].to_numpy();x=np.concatenate([emb[pos],x],1)
 median=np.nan_to_num(np.nanmedian(x[tr],axis=0));assert np.array_equal(median,saved['median'])
 order=np.argsort(meta.iloc[te].length.to_numpy());pick=order[np.unique(np.linspace(0,len(te)-1,128,dtype=int))]
 raw=x[te[pick]];mask=np.isnan(raw);xx=np.concatenate([np.where(mask,saved['median'],raw),mask[:,-39:].astype('float32')],1).astype('float32')
 if cfg['spec']['kind']=='hgb':values=np.column_stack([m.predict(xx) for m in saved['model']])
 else:values=saved['model'].predict(xx)
 pp=np.maximum(values.astype('float64'),0)
 if cfg['spec']['target']=='sqrt_p':pp=pp**2
 assert (pp.sum(1)>0).all();pp=pp/pp.sum(1,keepdims=True)
 expected=np.load(run/'evaluation_prediction.npy')[pick]
 error=float(np.max(abs(pp-expected)));assert error<1e-12 and np.isfinite(pp).all() and (pp>=0).all() and abs(pp.sum(1)-1).max()<1e-12
 result=dict(status='complete',candidate=cfg['candidate'],fold=cfg['fold'],seed=cfg['seed'],fresh_process=True,raw_feature_reload=True,fit_only_median_verified=True,
  rows=len(pick),min_length=int(meta.iloc[te[pick]].length.min()),max_length=int(meta.iloc[te[pick]].length.max()),max_probability_difference=error,
  checkpoint_sha256=rec['checkpoint_sha256'],prediction_sha256=rec['prediction_sha256'],script_sha256=sha(__file__),seconds=time.monotonic()-start)
 (run/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
