"""Bounded input-only frozen-encoder and cached-readout timing; no accuracy claims."""
import time
START=time.perf_counter()
import argparse,hashlib,json,os,resource,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
plan=json.loads((R/'records/plan.json').read_text());assert set(os.sched_getaffinity(0))==set(plan['CPU'])
assert os.environ['CUDA_VISIBLE_DEVICES']==plan['GPU']
put(R/'records/started.json',{'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'epoch':time.time(),'code_sha256':sha(__file__),'plan_sha256':sha(R/'records/plan.json')})
import numpy as np,torch
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==plan['GPU'].lower().removeprefix('gpu-')
sys.path.insert(0,plan['backend_src']);sys.path.insert(0,str(Path(plan['head_code']).parent));sys.path.insert(0,'/path/to/seq2scp/package/src')
from seq2scp_training.hf_backend import LocalHFBackend
from seq2scp_training.charge_features import extract_features
from seq2scp_training.features import VIEWS
from pooling_model import PoolingModel
rows=json.loads((R/'private/panel.json').read_text());assert sha(R/'private/panel.json')==plan['panel_sha256']
assert sha(Path(plan['backend_src'])/'seq2scp_training/hf_backend.py')==plan['backend_sha256']
assert sha(plan['head_code'])==plan['head_code_sha256'] and sha(plan['checkpoint_manifest'])==plan['checkpoint_manifest_sha256']
sys.path.insert(0,'/path/to/seq2scp/bridge_20261106/roles/code')
from bridge_roles_guard import check
guard=check('/path/to/seq2scp/bridge_20261106/roles/registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':'predict','stage':'development','sequence_sha256':[x['sequence_sha256'] for x in rows],'aliases':[]});put(R/'records/runtime_guard.json',guard)
models={};norm={}
for h in plan['heads']:
 assert sha(h['checkpoint'])==h['sha256'];c=torch.load(h['checkpoint'],map_location='cpu',weights_only=False)
 m=PoolingModel(h['mode']).cuda().eval();m.load_state_dict(c['state_dict']);models[h['mode']]=m
 norm[h['mode']]={k:torch.tensor(c[k],device='cuda') for k in ['embedding_mean','embedding_std','charge_mean','charge_std']}
backend=LocalHFBackend(json.loads(Path(plan['checkpoint_manifest']).read_text()),device='cuda:0')
torch.cuda.synchronize();load_seconds=time.perf_counter()-START
def budget():
 assert time.perf_counter()-START<plan['max_GPU_process_seconds']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<plan['max_RSS_bytes']
def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();z=fn();torch.cuda.synchronize();return z,time.perf_counter()-t
def head(mode,mats,charges):
 n=norm[mode];ch=torch.tensor(np.stack(charges),device='cuda');extra=torch.cat([(ch-n['charge_mean'])/n['charge_std'],torch.zeros((len(mats),39),device='cuda')],1)
 if mode=='mean_pre':x=(torch.tensor(np.stack([v.mean(0,dtype=np.float32) for v in mats]),device='cuda')-n['embedding_mean'])/n['embedding_std'];out=models[mode](x,extra)
 else:
  size=max(map(len,mats));v=np.zeros((len(mats),size,960),np.float32);mask=np.zeros((len(mats),size),bool)
  for i,z in enumerate(mats):v[i,:len(z)]=z;mask[i,:len(z)]=True
  x=(torch.tensor(v,device='cuda')-n['embedding_mean'])/n['embedding_std'];out=models[mode](x,extra,torch.tensor(mask,device='cuda'))
 return out.softmax(1).cpu().numpy()
cache=[];charges=[];pred0=[];times=[];passes=[]
with torch.inference_mode():
 for rep in range(plan['repetitions']):
  ts=time.perf_counter();records=[];pred=[]
  for i,row in enumerate(rows):
   budget();whole=time.perf_counter()
   t=time.perf_counter();feat=extract_features(row['sequence'],sequence_view='historical_exposed_cost_input',full_chain=True);ch=np.asarray([feat[k] for k in VIEWS['composition_charge']],dtype=np.float32);desc=time.perf_counter()-t
   zz,enc=timed(lambda:backend.extract_batch([row['sequence']],[30],precision='fp32')[0]);assert zz['sequence_sha256']==row['sequence_sha256'] and zz['positions']==list(range(row['length']))
   mat=zz['layers'][30];assert mat.shape==(row['length'],960) and np.isfinite(mat).all();mat=mat.astype(np.float16).astype(np.float32)
   pr,rd=timed(lambda:head('residue_mean',[mat],[ch]));assert np.isfinite(pr).all() and (pr>=0).all() and abs(pr.sum()-1)<1e-6
   records.append({'index':i,'stratum':row['stratum'],'length':row['length'],'descriptor_seconds':desc,'encoder_seconds':enc,'readout_seconds':rd,'pipeline_seconds':time.perf_counter()-whole});pred.append(pr)
   if rep==0:cache.append(mat.astype(np.float16));charges.append(ch)
  current=np.concatenate(pred);delta=0 if rep==0 else float(abs(current-pred0).max())
  if rep==0:pred0=current
  assert delta<1e-6;times.extend(dict(x,repeat=rep) for x in records);passes.append({'repeat':rep,'wall_seconds':time.perf_counter()-ts,'prediction_repeat_maxdiff':delta})
 cold_to_prob_seconds=load_seconds+passes[0]['wall_seconds']
 # Retain small private cache, never the full training cache. File I/O is measured as warm/unknown-cache-state.
 flat=np.concatenate(cache);offset=np.r_[0,np.cumsum([len(x) for x in cache])];means=np.stack([v.astype(np.float32).mean(0) for v in cache]);ch=np.stack(charges)
 assert flat.nbytes+means.nbytes+ch.nbytes<plan['max_cache_bytes']
 t=time.perf_counter()
 for name,v in [('residues',flat),('mean',means),('charge',ch),('offset',offset)]:
  with (R/'private'/(name+'.npy')).open('xb') as f:np.save(f,v);f.flush();os.fsync(f.fileno())
 write_s=time.perf_counter()-t
 t=time.perf_counter();reloaded=np.load(R/'private/residues.npy');read_s=time.perf_counter()-t;assert np.array_equal(reloaded,flat)
 cached=[]
 for mode in ['residue_mean','mean_pre']:
  for rep in range(3):
   for i,row in enumerate(rows):
    z,dt=timed(lambda:head(mode,[cache[i].astype(np.float32)],[charges[i]]));cached.append({'mode':mode,'repeat':rep,'stratum':row['stratum'],'index':i,'seconds':dt})
  ordered=sorted(range(len(rows)),key=lambda i:rows[i]['length']);batches=[];b=[]
  for i in ordered:
   if b and (len(b)>=8 or (len(b)+1)*rows[i]['length']>16384):batches.append(b);b=[]
   b.append(i)
  if b:batches.append(b)
  batchtimes=[]
  for rep in range(3):
   ts=time.perf_counter();outputs={}
   for ids in batches:
    pr,dt=timed(lambda:head(mode,[cache[i].astype(np.float32) for i in ids],[charges[i] for i in ids]))
    for i,v in zip(ids,pr):outputs[i]=v
   batchtimes.append(time.perf_counter()-ts)
   if mode=='residue_mean':assert np.max(np.abs(np.stack([outputs[i] for i in range(len(rows))])-pred0))<1e-5
  cached.append({'mode':mode,'batched_total_seconds':batchtimes,'batches':len(batches),'max_batch':8,'max_padded_residues':16384})
usage=resource.getrusage(resource.RUSAGE_SELF)
out={'schema':'length_stratified_cost_benchmark_v1','status':'complete','code_sha256':sha(__file__),'plan_sha256':sha(R/'records/plan.json'),'rows':len(rows),'residues':sum(x['length'] for x in rows),'load_seconds':load_seconds,'first_process_to_primary_predictions_seconds':cold_to_prob_seconds,'end_to_end_passes':passes,'timings':times,'cached_readout':cached,'cache_IO':{'write_fsync_seconds':write_s,'read_copy_seconds':read_s,'bytes':int(flat.nbytes),'page_cache_state':'OS page cache possibly warm; no cache dropping; storage bandwidth not certified'},'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':usage.ru_maxrss*1024,'process_CPU_seconds':usage.ru_utime+usage.ru_stime,'total_process_wall_seconds':time.perf_counter()-START,'GPU_busy_seconds_measured':False,'new_fits':0,'targets_read':False,'limits':'Historical length-stratified input engineering test, not accuracy or AFDB representativeness. FP32 singletons; external database I/O and sequence acquisition excluded.'}
put(R/'private/benchmark.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['timings','cached_readout']},indent=2))
