"""Bounded follow-up: exact historical charge-only OOF heads, no ESMC or fits."""
import time
START=time.perf_counter()
import argparse,csv,hashlib,json,os,resource,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--plan-sha256',required=True);a=ap.parse_args();R=Path(a.root);P=Path('/path/to/seq2scp');out=R/'charge_followup'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
plan=load(out/'plan.json');assert sha(out/'plan.json')==a.plan_sha256
assert set(os.sched_getaffinity(0))=={174,175} and os.getpriority(os.PRIO_PROCESS,0)>=15
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-REPLACE_WITH_ALLOCATED_DEVICE'
put(out/'started.json',{'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'epoch':time.time(),'code_sha256':sha(__file__),'plan_sha256':a.plan_sha256})
import numpy as np,pandas as pd,torch
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path.insert(0,str(P/'package/src'));from seq2scp_training.models import SequenceModel
from seq2scp_training.charge_features import extract_features
from seq2scp_training.features import VIEWS
sys.path.insert(0,str(P/'bridge_20261106/roles/code'));from bridge_roles_guard import check
rows=load(R/'private/panel.json');assert sha(R/'private/panel.json')==plan['panel_sha256']
guard=check(P/'bridge_20261106/roles/registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':'predict','stage':'development','sequence_sha256':[x['sequence_sha256'] for x in rows],'aliases':[]});put(out/'guard.json',guard)
for ref in plan['sources']:assert sha(ref['path'])==ref['sha256']
meta=pd.read_parquet(P/'atlas_scale_20261105/data/samples.parquet');canonical=meta.reset_index().sort_values('sample_id').drop_duplicates('sequence_sha256').set_index('sequence_sha256');assert canonical.index.is_unique
chosen=canonical.loc[[x['sequence_sha256'] for x in rows]];folds=chosen.fold.to_numpy();features=np.load(P/'atlas_scale_20261105/data/composition_charge.npy',mmap_mode='r')[chosen['index'].to_numpy()]
models={};norms={};saved={};headrefs=[]
for ref in plan['heads']:
 fold=ref['fold'];run=Path(ref['run']);assert sha(run/'best.pt')==ref['checkpoint_sha256']
 ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config'];assert cfg['input_dim']==78 and cfg['seed']==1701 and cfg['fold']==fold
 m=SequenceModel(input_dim=cfg['input_dim'],bins=63,kind='vector',pool='mean',hidden=cfg['hidden'],dropout=cfg['dropout']).cuda().eval();m.load_state_dict(ck['state_dict']);models[fold]=m;norms[fold]=ck
 roles=load(run/'roles.json');pos={x:i for i,x in enumerate(roles['evaluation_ids'])};pred=np.load(run/'evaluation_prediction.npy',mmap_mode='r')
 for i,row in enumerate(chosen.itertuples()):
  if row.fold==fold:saved[i]=np.array(pred[pos[row.sample_id]])
 headrefs.append({'fold':fold,'checkpoint_sha256':ref['checkpoint_sha256'],'model_source_sha256':cfg['model_sha256'],'hidden':cfg['hidden'],'parameters':sum(x.numel() for x in m.parameters())})
assert len(saved)==len(rows)
def features_for(row):
 x=extract_features(row['sequence'],sequence_view='historical_exposed_cost_input',full_chain=True);return np.asarray([x[k] for k in VIEWS['composition_charge']],np.float32)
def head(fold,raw):
 ck=norms[int(fold)];flag=np.isnan(raw).astype(np.float32);x=np.concatenate([(np.where(np.isnan(raw),ck['median'],raw)-ck['mean'])/ck['std'],flag],axis=1)
 return models[int(fold)](torch.tensor(x,device='cuda'))['logits'].softmax(1).cpu().numpy()
def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();z=fn();torch.cuda.synchronize();return z,time.perf_counter()-t
def budget():
 assert time.perf_counter()-START<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
torch.cuda.synchronize();load_seconds=time.perf_counter()-START;timings=[];passes=[];prediction0=None
with torch.inference_mode():
 for rep in range(3):
  begin=time.perf_counter();pred=[]
  for i,row in enumerate(rows):
   budget();t=time.perf_counter();raw,feat=timed(lambda:features_for(row));assert np.allclose(raw,features[i],atol=1e-7,rtol=1e-7,equal_nan=True)
   z,rd=timed(lambda:head(folds[i],raw[None]));elapsed=time.perf_counter()-t
   timings.append({'repeat':rep,'stratum':row['stratum'],'index':i,'length':row['length'],'feature_seconds':feat,'head_seconds':rd,'pipeline_seconds':elapsed});pred.append(z[0])
  pred=np.stack(pred);diff=float(np.max(abs(pred-np.stack([saved[i] for i in range(len(rows))]))));assert diff<2e-6 and np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max()<1e-6
  if prediction0 is None:prediction0=pred
  assert np.max(abs(pred-prediction0))<1e-6
  passes.append({'repeat':rep,'wall_seconds':time.perf_counter()-begin,'saved_OOF_prediction_maxdiff':diff})
 cached=[]
 for rep in range(3):
  begin=time.perf_counter()
  for i,row in enumerate(rows):
   z,dt=timed(lambda:head(folds[i],features[i:i+1]));cached.append({'repeat':rep,'index':i,'stratum':row['stratum'],'seconds':dt})
 batch=[]
 for rep in range(3):
  begin=time.perf_counter();pred=np.zeros((len(rows),63),np.float32)
  for f in sorted(models):
   ids=np.flatnonzero(folds==f);z,dt=timed(lambda:head(f,features[ids]));pred[ids]=z
  batch.append(time.perf_counter()-begin);assert np.max(abs(pred-prediction0))<2e-6
u=resource.getrusage(resource.RUSAGE_SELF)
result={'schema':'matched_OOF_charge_cost_v1','status':'complete','code_sha256':sha(__file__),'plan_sha256':a.plan_sha256,'rows':len(rows),'repeats':3,'load_seconds':load_seconds,'cold_process_to_first_predictions_seconds':load_seconds+passes[0]['wall_seconds'],'end_to_end_passes':passes,'timings':timings,'cached_singleton':cached,'cached_five_fold_batches_seconds':batch,'heads':headrefs,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':u.ru_maxrss*1024,'CPU_seconds':u.ru_utime+u.ru_stime,'process_wall_seconds':time.perf_counter()-START,'new_fits':0,'encoder_loaded':False,'targets_read':False,'scope':'Same 95 inputs; each sequence uses exactly its original evaluation-fold charge model, not an ensemble. Saved OOF probability replay establishes identity, not new accuracy. Fresh descriptors match old composition_charge features. Includes singleton H2/CE-free charge feature and head path; repeated full-library IO unmeasured.'}
put(out/'private_result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['timings','cached_singleton']},indent=2))
