"""Fresh-process GPU replay of selected exploratory and confirmation checkpoints."""
import os

import json,os,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
r=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'overnight_20261106/explore'));sys.path.insert(0,str(r/'code'));from overnight_explore_model import PoolingModel
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ["GPU_UUID_0"]
p=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'atlas_scale_20261105'));cache=Path(str(Path(os.environ["PROJECT_ROOT"]) / 'pooling_20261105/cache'));d=pd.read_parquet(p/'data/samples.parquet').set_index('sample_id');charge=np.load(p/'data/composition_charge.npy');ix=pd.read_parquet(cache/'index.parquet').set_index('sequence_sha256');flat=torch.from_numpy(np.array(np.load(cache/'flat.npy',mmap_mode='r'))).cuda();position=pd.Series(np.arange(len(d)),index=d.index)
summ=json.loads((r/'results/summary.json').read_text());paths=[Path(v['checkpoint']) for v in summ['selected_by_stop'].values()];paths+=list((r/'confirmation/runs').glob('*/best.pt'));assert len(paths)==6
results=[];start=time.monotonic()
for checkpoint in paths:
 b=torch.load(checkpoint,map_location='cpu',weights_only=False);m=PoolingModel(b['config']['mode']).cuda().eval();m.load_state_dict(b['state_dict']);ids=json.loads(checkpoint.with_name('roles.json').read_text())['evaluation_ids'];sub=d.loc[ids];rows=ix.loc[sub.sequence_sha256];length=rows.length.to_numpy();offset=rows.offset.to_numpy();raw=charge[position.loc[ids].to_numpy()];flags=np.isnan(raw).astype('float32');v=np.where(np.isnan(raw),b['charge_median'],raw);extra=torch.tensor(np.concatenate([(v-b['charge_mean'])/b['charge_std'],flags],1),device='cuda');em=torch.tensor(b['embedding_mean'],device='cuda');es=torch.tensor(b['embedding_std'],device='cuda');pred=[]
 segments=[]
 for group in np.array_split(np.arange(len(ids)),max(1,int(np.ceil(len(ids)/128)))):
  part=[];mx=0
  for i in group:
   nxt=max(mx,int(length[i]))
   if part and nxt*(len(part)+1)>16384:segments.append(part);part=[];mx=0
   part.append(int(i));mx=max(mx,int(length[i]))
  if part:segments.append(part)
 with torch.inference_mode():
  for batch in segments:
   j=batch[0];k=batch[-1]+1
   L=int(max(length[j:k]));pos=torch.arange(L,device='cuda');mask=pos[None]<torch.tensor(length[j:k],device='cuda')[:,None];index=torch.tensor(offset[j:k],device='cuda')[:,None]+pos[None];index=torch.where(mask,index,torch.zeros_like(index));x=(flat[index].float()-em)/es
   pred.append(m(x,extra[j:k],mask).softmax(1).cpu().numpy())
 pred=np.concatenate(pred);saved=np.load(checkpoint.with_name('evaluation_prediction.npy'));error=float(np.max(abs(pred-saved)));assert np.allclose(pred,saved,atol=1e-7,rtol=1e-6),(str(checkpoint),error)
 results.append({'checkpoint':str(checkpoint),'mode':b['config']['mode'],'fold':b['config']['fold'],'seed':b['config']['seed'],'rows':len(ids),'max_absolute_prediction_error':error,'fresh_process_replay_passed':True});del m
result={'status':'passed','checkpoints':results,'elapsed_seconds':time.monotonic()-start,'scope':'fresh model+checkpoint+saved transforms; identical immutable FP16 cache and original evaluation batch partition; no training','replay_adjustment':'First replay used different batch partitions; first checkpoint max absolute difference 3.129243850708008e-7 exceeded the strict tolerance. Restored original batching for this exact replay; no model/training/prediction artifact changed.'}
(r/'results/fresh_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
