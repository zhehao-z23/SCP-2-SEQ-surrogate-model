"""Load selected on-disk checkpoints in a fresh process and verify saved predictions."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--source',required=True);args=a.parse_args()
r=Path(args.root);s=Path(args.source);sys.path.insert(0,str(s/'package/src'))
from seq2scp_training.models import SequenceModel
torch.set_num_threads(2)
d=pd.read_parquet(r/'data/samples.parquet');index=pd.Series(np.arange(len(d)),index=d.sample_id)
selected=pd.read_csv(r/'results/selected_fits.csv');checks=[]
for row in selected.itertuples():
 out=Path(row.artifact_path);ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config'];family=cfg['family']
 roles=json.loads((out/'roles.json').read_text());ids=index.loc[roles['evaluation_ids'][:128]].to_numpy()
 if family.startswith('composition'):x=np.load(r/'data'/(family.removesuffix('_deep')+'.npy'))[ids]
 else:
  ei=pd.read_parquet(r/'data/esmc_300m_mean/index.parquet');arr=np.load(r/'data/esmc_300m_mean/mean.npy');lookup=pd.Series(np.arange(len(ei)),index=ei.sequence_sha256)
  x=arr[lookup.loc[d.iloc[ids].sequence_sha256].to_numpy()];assert np.isfinite(x).all()
  if family=='esmc_charge':x=np.concatenate([x,np.load(r/'data/composition_charge.npy')[ids]],1)
 flags=np.isnan(x).astype('float32');x=(np.where(np.isnan(x),ck['median'],x)-ck['mean'])/ck['std']
 if family.startswith('composition'):x=np.concatenate([x,flags],1)
 elif family=='esmc_charge':x=np.concatenate([x,flags[:,-39:]],1)
 model=SequenceModel(input_dim=cfg['input_dim'],bins=63,kind='vector',pool='mean',hidden=cfg['hidden'],dropout=cfg['dropout']).eval();model.load_state_dict(ck['state_dict'])
 with torch.inference_mode():p=model(torch.tensor(x,dtype=torch.float32))['logits'].softmax(1).numpy()
 truth=np.load(out/'evaluation_prediction.npy')[:len(ids)]
 delta=float(abs(p-truth).max());assert np.allclose(p,truth,rtol=1e-4,atol=1e-6),(row.run_id,delta)
 checks.append({'run_id':row.run_id,'rows':len(ids),'CPU_fresh_reload_vs_saved_GPU_max_abs_error':delta,'passed':True})
(r/'results/checkpoint_replay.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({'selected_checkpoints':len(checks),'passed':True,'max_abs_error':max(x['CPU_fresh_reload_vs_saved_GPU_max_abs_error'] for x in checks)}))
