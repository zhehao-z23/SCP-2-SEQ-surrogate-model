"""Fresh-process evaluation replay using each checkpoint's saved transformations."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();r=Path(args.root);sys.path.insert(0,str(r/'code'));from bridge_optimization_model import PoolingModel
plan=json.loads((r/'plan.json').read_text());p=Path(plan['parent']);cache=Path(plan['cache_root'])/'cache';torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert os.environ['CUDA_VISIBLE_DEVICES'] in plan['gpus']
d=pd.read_parquet(p/'data/samples.parquet').set_index('sample_id');pos=pd.Series(np.arange(len(d)),index=d.index);charge=np.load(p/'data/composition_charge.npy');ix=pd.read_parquet(cache/'index.parquet');lookup=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256);means=np.load(cache/'mean.npy');flat=None;checks=[];start=time.monotonic()
for f in sorted((r/'runs').glob('*/best.pt')):
 b=torch.load(f,map_location='cpu',weights_only=False);c=b['config'];m=PoolingModel(c['mode']).cuda().eval();m.load_state_dict(b['state_dict']);ids=json.loads(f.with_name('roles.json').read_text())['evaluation_ids'];sub=d.loc[ids];ri=lookup.loc[sub.sequence_sha256].to_numpy();length=ix.length.to_numpy()[ri];offset=ix.offset.to_numpy()[ri];raw=charge[pos.loc[ids].to_numpy()];flags=np.isnan(raw).astype('float32');v=np.where(np.isnan(raw),b['charge_median'],raw);extra=torch.tensor(np.concatenate([(v-b['charge_mean'])/b['charge_std'],flags],1),device='cuda');em=torch.tensor(b['embedding_mean'],device='cuda');es=torch.tensor(b['embedding_std'],device='cuda')
 if c['mode']=='residue_mean' and flat is None:flat=torch.from_numpy(np.array(np.load(cache/'flat.npy',mmap_mode='r'))).cuda()
 if c['mode']=='mean_pre':vector=torch.tensor((means[ri]-b['embedding_mean'])/b['embedding_std'],device='cuda')
 batches=[]
 for group in np.array_split(np.arange(len(ids)),max(1,int(np.ceil(len(ids)/128)))):
  part=[];mx=0
  for i in group:
   nxt=max(mx,int(length[i]))
   if part and nxt*(len(part)+1)>16384:batches.append(part);part=[];mx=0
   part.append(int(i));mx=max(mx,int(length[i]))
  if part:batches.append(part)
 pred=[]
 with torch.inference_mode():
  for z in batches:
   if c['mode']=='charge_deep':out=m(None,extra[z])
   elif c['mode']=='mean_pre':out=m(vector[z],extra[z])
   else:
    L=int(max(length[z]));positions=torch.arange(L,device='cuda');mask=positions[None]<torch.tensor(length[z],device='cuda')[:,None];index=torch.tensor(offset[z],device='cuda')[:,None]+positions[None];index=torch.where(mask,index,torch.zeros_like(index));out=m((flat[index].float()-em)/es,extra[z],mask)
   pred.append(out.softmax(1).cpu().numpy())
 pred=np.concatenate(pred);saved=np.load(f.with_name('evaluation_prediction.npy'));err=float(abs(pred-saved).max());assert np.allclose(pred,saved,atol=1e-6,rtol=1e-5),(str(f),err);checks.append({'run_id':f.parent.name,'family':c['mode'],'policy':c['policy'],'fold':c['fold'],'seed':c['seed'],'max_absolute_prediction_error':err,'passed':True});del m
out={'status':'passed','checkpoints':checks,'elapsed_seconds':time.monotonic()-start,'scope':'fresh process, saved model and scaler, immutable frozen cache; same evaluation batching; meaningful FP32 tolerance atol1e-6 rtol1e-5'};(r/'results/fresh_replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
