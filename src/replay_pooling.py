"""Fresh CPU checkpoint restoration on 16 length-spanning evaluation proteins."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from pooling_model import PoolingModel
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',required=True);a=p.parse_args();r=Path(a.root);parent=Path(a.parent);torch.set_num_threads(2)
d=pd.read_parquet(parent/'data/samples.parquet').set_index('sample_id',drop=False);ix=pd.read_parquet(r/'cache/index.parquet');lookup=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256);flat=np.load(r/'cache/flat.npy',mmap_mode='r');means=np.load(r/'cache/mean.npy');maximum=np.load(r/'cache/max.npy');charge=np.load(parent/'data/composition_charge.npy');dpos=pd.Series(np.arange(len(d)),index=d.index);checks=[]
for row in pd.read_csv(r/'results/selected_fits.csv').itertuples():
 out=r/'runs'/row.run_id;ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);roles=json.loads((out/'roles.json').read_text());ids=roles['evaluation_ids'];seq=d.loc[ids];order=np.argsort(seq.length.to_numpy());positions=order[np.linspace(0,len(order)-1,16).astype(int)];m=PoolingModel(row.mode).eval();m.load_state_dict(ck['state_dict']);pred=[]
 for k in positions:
  name=ids[k];z=d.loc[name];j=int(lookup[z.sequence_sha256]);raw=charge[int(dpos[name])][None];flag=np.isnan(raw).astype('float32');ex=np.concatenate([(np.where(np.isnan(raw),ck['charge_median'],raw)-ck['charge_mean'])/ck['charge_std'],flag],1)
  if row.mode.endswith('_pre'):
   x=(means[j] if row.mode=='mean_pre' else maximum[j])[None];x=(x-ck['embedding_mean'])/ck['embedding_std'];mask=None
  else:
   z=ix.iloc[j];x=np.array(flat[int(z.offset):int(z.offset+z.length)],dtype='float32')[None];x=(x-ck['embedding_mean'])/ck['embedding_std'];mask=torch.ones(1,x.shape[1],dtype=torch.bool)
  with torch.inference_mode():pred.append(m(torch.tensor(x),torch.tensor(ex),mask).softmax(1).numpy()[0])
 truth=np.load(out/'evaluation_prediction.npy')[positions];diff=float(abs(np.asarray(pred)-truth).max());assert np.allclose(pred,truth,rtol=1e-4,atol=2e-6),(row.run_id,diff)
 checks.append({'run_id':row.run_id,'mode':row.mode,'fold':row.fold,'rows':16,'length_min':int(seq.iloc[positions].length.min()),'length_max':int(seq.iloc[positions].length.max()),'max_absolute_error':diff,'passed':True})
(r/'results/checkpoint_replay.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
