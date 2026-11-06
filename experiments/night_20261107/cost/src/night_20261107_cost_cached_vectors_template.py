"""Small cached mean-vector route: do not recompute means from residue cache."""
import time
START=time.perf_counter()
import hashlib,json,os,resource,sys
from pathlib import Path
import numpy as np,torch
R=Path('/path/to/seq2scp/night_20261107/cost');p=json.loads((R/'records/plan.json').read_text())
assert set(os.sched_getaffinity(0))=={174,175} and os.environ['CUDA_VISIBLE_DEVICES']==p['GPU']
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path.insert(0,str(Path(p['head_code']).parent));from pooling_model import PoolingModel
sys.path.insert(0,'/path/to/seq2scp/bridge_20261106/roles/code');from bridge_roles_guard import check
rows=json.loads((R/'private/panel.json').read_text());guard=check('/path/to/seq2scp/bridge_20261106/roles/registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':'predict','stage':'development','sequence_sha256':[x['sequence_sha256'] for x in rows],'aliases':[]})
h=next(x for x in p['heads'] if x['mode']=='mean_pre');assert hashlib.sha256(Path(h['checkpoint']).read_bytes()).hexdigest()==h['sha256']
c=torch.load(h['checkpoint'],map_location='cpu',weights_only=False);model=PoolingModel('mean_pre').cuda().eval();model.load_state_dict(c['state_dict'])
norm={k:torch.tensor(c[k],device='cuda') for k in ['embedding_mean','embedding_std','charge_mean','charge_std']}
t=time.perf_counter();mean=np.load(R/'private/mean.npy');ch=np.load(R/'private/charge.npy');io=time.perf_counter()-t
def run(batch):
 out=[]
 for i in range(0,len(mean),batch):
  x=(torch.tensor(mean[i:i+batch],device='cuda')-norm['embedding_mean'])/norm['embedding_std'];q=torch.tensor(ch[i:i+batch],device='cuda');e=torch.cat([(q-norm['charge_mean'])/norm['charge_std'],torch.zeros((len(q),39),device='cuda')],1);out.append(model(x,e).softmax(1).cpu().numpy())
 return np.concatenate(out)
results=[]
with torch.inference_mode():
 ref=run(1)
 for batch in [1,32,95]:
  samples=[]
  for rep in range(20):
   torch.cuda.synchronize();t=time.perf_counter();v=run(batch);torch.cuda.synchronize();samples.append(time.perf_counter()-t)
  assert np.isfinite(v).all() and (v>=0).all() and abs(v.sum(1)-1).max()<1e-6 and abs(v-ref).max()<1e-5
  results.append({'batch':batch,'repeats':20,'wall_seconds_per_95':samples,'prediction_max_difference_from_singleton':float(abs(v-ref).max())})
u=resource.getrusage(resource.RUSAGE_SELF)
out={'schema':'cached_mean_vector_cost_v1','status':'complete','rows':len(mean),'results':results,'input_load_seconds':io,'input_bytes':mean.nbytes+ch.nbytes,'scope':'Already available mean and charge vectors on CPU; includes normalization, host-device copies, model and output CPU copies; excludes encoder and residue reduction. OS file cache not forced cold.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((R/'records/plan.json').read_bytes()).hexdigest(),'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':u.ru_maxrss*1024,'process_CPU_seconds':u.ru_utime+u.ru_stime,'process_wall_seconds':time.perf_counter()-START,'new_fits':0,'targets_read':False}
with (R/'results/cached_mean_cost_public.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='results'}))
