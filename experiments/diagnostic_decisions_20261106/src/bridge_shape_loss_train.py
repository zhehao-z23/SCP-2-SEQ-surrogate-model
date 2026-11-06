"""Fixed-path H2 versus CE/4 probe, derived from the accepted optimization trainer.
Only stop H selects the checkpoint; both arms use the previously observed H2 LR path.
"""
import argparse,hashlib,json,os,random,resource,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def cpu_copy(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_copy(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
 return x
def snapshot(opt):
 return {'optimizer_state':cpu_copy(opt.state_dict()),'rng_state':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}}
def digest(x):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.detach().cpu().numpy().tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
  elif isinstance(v,(tuple,list)):
   for k in v:add(k)
  else:h.update(repr(v).encode())
 add(x);return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--parent',required=True);ap.add_argument('--mode',choices=['mean_pre','residue_mean','charge_deep'],required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--seed',type=int,default=1701);ap.add_argument('--max-seconds',type=int,default=900);ap.add_argument('--max-epochs',type=int,default=300);ap.add_argument('--cache-root',required=True);ap.add_argument('--policy',choices=['fresh','plateau','reset','retain'],required=True);ap.add_argument('--warmstart');ap.add_argument('--loss',choices=['H2','CE_quarter'],required=True);a=ap.parse_args()
 r=Path(a.root);p=Path(a.parent)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 plan=json.loads((r/'plan.json').read_text());assert sha(__file__)==plan['trainer_sha256']
 assert a.mode=='residue_mean' and a.fold==plan['fold'] and a.seed==plan['seed'] and a.policy=='plateau' and not a.warmstart
 assert a.max_epochs==plan['epochs']==134 and a.max_seconds==900 and a.lr==.0003
 assert sorted(os.sched_getaffinity(0))==plan['CPU_affinity'] and os.environ['CUDA_VISIBLE_DEVICES']==plan['GPU']
 assert sha(Path(__file__).with_name('bridge_optimization_model.py'))==plan['model_sha256']
 from bridge_optimization_model import PoolingModel
 assert sha(p/'code/train_atlas_scale.py')==plan['metrics_code_sha256']
 sys.path.insert(0,str(p/'code'));from train_atlas_scale import select,metrics
 reference=Path(plan['reference_directory']);assert sha(reference/'receipt.json')==plan['reference_receipt_sha256'] and sha(reference/'curve.json')==plan['reference_curve_sha256']
 reference_receipt=json.loads((reference/'receipt.json').read_text());reference_curve=json.loads((reference/'curve.json').read_text())
 assert len(reference_curve)==plan['epochs']
 for item in reference_curve:
  stage=[z for z in plan['LR_path'] if z['first_epoch']<=item['epoch']<=z['last_epoch']];assert len(stage)==1 and item['lr']==stage[0]['lr'] and item['LR_stage']==stage[0]['stage']
 data=p/'data';rel=json.loads((data/'release.json').read_text());assert sha(data/'release.json')==plan['release_sha256']
 assert sha(data/'samples.parquet')==rel['artifacts']['samples.parquet'];d=pd.read_parquet(data/'samples.parquet')
 guard_path=Path(plan['roles_root'])/'code/bridge_roles_guard.py';assert sha(guard_path)==plan['guard_code_sha256']
 sys.path.insert(0,str(guard_path.parent));from bridge_roles_guard import check
 request={'stage':'development','sequence_sha256':sorted(d.sequence_sha256.unique().tolist()),'aliases':[]}
 guards=[check(Path(plan['roles_root'])/'registry_v1/guard_contract.json',plan['guard_contract_sha256'],{'operation':op,**request}) for op in ['read_target','train','stop','model_select','model_eval']]
 assert sha(data/'p.npy')==rel['artifacts']['p.npy'];y=np.load(data/'p.npy')
 tr,va,te=select(d,a.fold,'all')
 expected_roles=json.loads(Path(plan['reference_directory']).joinpath('roles.json').read_text())
 assert sha(Path(plan['reference_directory'])/'roles.json')==plan['reference_roles_sha256']
 assert expected_roles=={'train_ids':d.iloc[tr].sample_id.tolist(),'stop_ids':d.iloc[va].sample_id.tolist(),'evaluation_ids':d.iloc[te].sample_id.tolist()}
 for left,right in [(tr,va),(tr,te),(va,te)]:
  for key in ['sequence_sha256','group_id']:assert not set(d.iloc[left][key])&set(d.iloc[right][key])
 cache_source=Path(a.cache_root);cache=cache_source/'cache';cr=json.loads((cache/'receipt.json').read_text());assert cr['status']=='complete'
 assert sha(cache/'receipt.json')==plan['cache_receipt_sha256']==reference_receipt['config']['cache_receipt_sha256']
 assert sha(cache/'index.parquet')==plan['cache_index_sha256']==reference_receipt['config']['cache_index_sha256']
 acceptance=json.loads((cache_source/'records/cache_acceptance.json').read_text());assert acceptance['status']=='accepted' and acceptance['receipt_sha256']==sha(cache/'receipt.json')
 for name,check in acceptance['files'].items():
  fp=cache/name;assert fp.stat().st_size==check['size'] and fp.stat().st_mtime_ns==check['mtime_ns'],'immutable cache modified after acceptance'
  if name!='flat.npy':assert sha(fp)==check['sha256']
 ix=pd.read_parquet(cache/'index.parquet');assert ix.sequence_sha256.is_unique
 lookup=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256);ri=lookup.loc[d.sequence_sha256].to_numpy();length=ix.length.to_numpy()[ri];offset=ix.offset.to_numpy()[ri]
 means=np.load(cache/'mean.npy')[ri];assert means.shape==(len(d),960) and np.isfinite(means).all()
 em=means[tr].mean(0);es=means[tr].std(0);es=np.where(es<1e-6,1,es)
 assert sha(data/'composition_charge.npy')==rel['artifacts']['composition_charge.npy']
 charge=np.load(data/'composition_charge.npy');med=np.nan_to_num(np.nanmedian(charge[tr],0));flags=np.isnan(charge).astype('float32');charge=np.where(np.isnan(charge),med,charge);cm=charge[tr].mean(0);cs=charge[tr].std(0);cs=np.where(cs<1e-6,1,cs);extra=np.concatenate([(charge-cm)/cs,flags],1)
 cfg={'mode':a.mode,'fold':a.fold,'lr':a.lr,'seed':a.seed,'hidden':[1024,512,256],'dropout':.3,'input_dim':78 if a.mode=='charge_deep' else 1038,'loss':a.loss,'batch':128,'max_padded_residues':16384,'epochs':a.max_epochs,'max_seconds':a.max_seconds,'min_epochs':40,'patience':30,'min_delta':1e-5,'optimizer':'AdamW','weight_decay':1e-4,'clip':1.,'LR_schedule':'frozen historical H2 path, before-epoch assignment; not adaptive plateau','release_sha256':sha(data/'release.json'),'cache_receipt_sha256':sha(cache/'receipt.json'),'cache_index_sha256':sha(cache/'index.parquet'),'script_sha256':sha(__file__),'model_sha256':sha(Path(__file__).with_name('bridge_optimization_model.py')),'parent_metrics_and_split_sha256':sha(p/'code/train_atlas_scale.py'),'normalization':'fit-only mean-embedding mean/std shared across all residue aggregation; fit-only charge median/mean/std','microbatch_order':'permutation chunks1024 sort by length into128; shuffle batches; identical for paired modes','precision':'FP32 math; FP16 frozen cache; no AMP or TF32','not_blind':True,'new_C_transfer_permitted':False,'policy':a.policy,'warmstart_sha256':sha(a.warmstart) if a.warmstart else None,'warmstart_path':a.warmstart,'plan_sha256':sha(r/'plan.json'),'best_epoch_optimizer_and_rng_saved':True,'epoch0_eligible':True,'fixed_epochs':True,'source_trainer_sha256':plan['source_trainer_sha256'],'frozen_LR_path':plan['LR_path'],'plateau_lrs':[.0003,.0001,.00003],'plateau_patience':15,'final_stage_min_epochs':20}
 ident=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();out=r/'runs'/ident;out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.json').exists():print('exists',ident);return
 with (out/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time()},f)
 put(out/'role_preflight.json',guards);put(out/'config.json',cfg);put(out/'roles.json',{'train_ids':d.iloc[tr].sample_id.tolist(),'stop_ids':d.iloc[va].sample_id.tolist(),'evaluation_ids':d.iloc[te].sample_id.tolist()})
 model=PoolingModel(a.mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
 initial_model_digest=digest(model.state_dict());initial_rng_digest=digest(snapshot(opt)['rng_state'])
 assert bool(a.warmstart)==(a.policy in ['reset','retain'])
 source_state_digest=None;restored_rng=None
 if a.warmstart:
  base=torch.load(a.warmstart,map_location='cpu',weights_only=False);bc=base['config']
  assert bc['policy']=='fresh' and bc['mode']==a.mode and bc['fold']==a.fold and bc['seed']==a.seed and bc['release_sha256']==cfg['release_sha256'] and bc['cache_receipt_sha256']==cfg['cache_receipt_sha256']
  assert json.loads(Path(a.warmstart).with_name('roles.json').read_text())==json.loads((out/'roles.json').read_text())
  for k,v in [('embedding_mean',em),('embedding_std',es),('charge_median',med),('charge_mean',cm),('charge_std',cs)]:assert np.array_equal(base[k],v),k
  model.load_state_dict(base['state_dict']);source_state_digest=digest(base['optimizer_state']['state']);restored_rng=base['rng_state']
  if a.policy=='retain':opt.load_state_dict(base['optimizer_state'])
  for group in opt.param_groups:group['lr']=a.lr
  assert (digest(opt.state_dict()['state'])==source_state_digest) if a.policy=='retain' else not opt.state
 initial_optimizer_digest=digest(opt.state_dict()['state'])
 initial_optimizer_steps=[float(v['step']) for v in opt.state.values() if 'step' in v]

 ex=torch.tensor(extra,dtype=torch.float32,device='cuda');yt=torch.tensor(y,dtype=torch.float32,device='cuda');emg=torch.tensor(em,device='cuda');esg=torch.tensor(es,device='cuda')
 isvector=a.mode in ['mean_pre','charge_deep'];flat=None;on_gpu=False
 if isvector:
  vector=means
  vector=torch.tensor((vector-em)/es,device='cuda',dtype=torch.float32)
 else:
  flat=np.load(cache/'flat.npy',mmap_mode='r');assert flat.dtype==np.float16 and flat.shape[1]==960
  assert (offset+length<=len(flat)).all()
  if flat.nbytes<16*1024**3:
   flat=torch.from_numpy(np.array(flat)).to('cuda');on_gpu=True
  offsets=torch.tensor(offset,device='cuda');lens=torch.tensor(length,device='cuda')
 weights=d.iloc[tr].protein_id.map(d.iloc[tr].protein_id.value_counts()).to_numpy();weights=1/weights;weights/=weights.mean();wt=torch.ones(len(d),device='cuda');wt[torch.tensor(tr,device='cuda')]=torch.tensor(weights,dtype=torch.float32,device='cuda')
 def micro(ids):
  result=[];part=[];mx=0
  for i in ids:
   nxt=max(mx,int(length[i]))
   if part and nxt*(len(part)+1)>16384:result.append(np.asarray(part));part=[];mx=0
   part.append(int(i));mx=max(mx,int(length[i]))
  if part:result.append(np.asarray(part))
  return result
 def forward(ids):
  if isvector:return model(vector[ids] if a.mode=='mean_pre' else None,ex[ids])
  L=int(length[ids].max());pos=torch.arange(L,device='cuda');mask=pos[None]<lens[ids,None]
  if on_gpu:
   ind=offsets[ids,None]+pos[None];ind=torch.where(mask,ind,torch.zeros_like(ind));x=flat[ind].float()
  else:
   x=np.zeros((len(ids),L,960),dtype=np.float16)
   for j,i in enumerate(ids):x[j,:length[i]]=flat[offset[i]:offset[i]+length[i]]
   x=torch.tensor(x,device='cuda',dtype=torch.float32)
  x=(x-emg)/esg;return model(x,ex[ids],mask)
 def predict(ids):
  model.eval();result=[]
  with torch.inference_mode():
   for group in np.array_split(ids,max(1,int(np.ceil(len(ids)/128)))):
    for small in micro(group):result.append(forward(small).softmax(1).cpu().numpy())
  return np.concatenate(result)
 def evaluation_with_CE(ids,pred):
  result,per=metrics(y[ids],pred,d.iloc[ids])
  ce=d.iloc[ids][['protein_id','group_id']].copy()
  ce['CE']=-(y[ids].astype('float64')*np.log(np.maximum(pred.astype('float64'),1e-30))).sum(1)
  result['group_macro_CE']=float(ce.groupby(['group_id','protein_id']).CE.mean().groupby('group_id').mean().mean())
  result['CE_probability_floor_count']=int((pred<1e-30).sum())
  return result,per
 def batches():
  order=np.random.permutation(tr);blocks=[]
  for pos in range(0,len(order),1024):
   b=order[pos:pos+1024];b=b[np.argsort(length[b],kind='stable')];blocks.extend([b[j:j+128] for j in range(0,len(b),128)])
  np.random.shuffle(blocks);return blocks
 if restored_rng:
  random.setstate(restored_rng['python']);np.random.set_state(restored_rng['numpy']);torch.set_rng_state(restored_rng['torch']);torch.cuda.set_rng_state_all(restored_rng['cuda'])
 start=time.monotonic();initial_stop=metrics(y[va],predict(va),d.iloc[va])[0]['group_macro_H'];best=initial_stop;best_ep=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bundle=snapshot(opt);curve=[];reason='epoch_budget';updates=0;presentations=0;residues=0
 stage=0;stage_start=0;stage_best=initial_stop;stage_best_ep=0
 put(out/'initialization.json',{'stop_H':initial_stop,'source_optimizer_digest':source_state_digest,'initial_optimizer_digest':initial_optimizer_digest,'initial_optimizer_steps':initial_optimizer_steps,'rng_restored_from_source_best':bool(restored_rng),'policy':a.policy,'initial_model_digest':initial_model_digest,'initial_rng_digest':initial_rng_digest})
 for epoch in range(1,a.max_epochs+1):
  applicable=[z for z in plan['LR_path'] if z['first_epoch']<=epoch<=z['last_epoch']];assert len(applicable)==1
  stage=applicable[0]['stage']
  for group in opt.param_groups:group['lr']=applicable[0]['lr']
  model.train();tot=0.;norms=[];batch_hash=hashlib.sha256()
  for ids in batches():
   batch_hash.update(ids.astype('<i8').tobytes())
   opt.zero_grad(set_to_none=True)
   for small in micro(ids):
    logp=forward(small).log_softmax(1)
    loss=.5*((torch.exp(.5*logp)-yt[small].sqrt())**2).sum(1) if a.loss=='H2' else -.25*(yt[small]*logp).sum(1)
    loss=(loss*wt[small]).sum()/len(ids);assert torch.isfinite(loss);loss.backward();tot+=float(loss.detach())*len(ids)
   norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)));opt.step();updates+=1;presentations+=len(ids);residues+=int(length[ids].sum())
  stop_metrics=evaluation_with_CE(va,predict(va))[0];score=stop_metrics['group_macro_H'];elapsed=time.monotonic()-start
  curve.append({'lr':opt.param_groups[0]['lr'],'LR_stage':stage,'epoch':epoch,'online_train_loss':tot/len(tr),'loss':a.loss,'stop_H':score,'stop_CE':stop_metrics['group_macro_CE'],'stop_CE_floor_count':stop_metrics['CE_probability_floor_count'],'stop_CDF_MSE':stop_metrics['group_macro_cutpoint_CDF_MSE'],'stop_edge_MAE':stop_metrics['group_macro_edge_bin_mass_MAE'],'gradient_norm_median':float(np.median(norms)),'clip_fraction':float(np.mean(np.asarray(norms)>1.)),'batch_order_sha256':batch_hash.hexdigest(),'end_epoch_RNG_digest':digest(snapshot(opt)['rng_state']),'elapsed_seconds':elapsed,'updates':updates,'residue_presentations':residues})
  if score<best-1e-5:best=score;best_ep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bundle=snapshot(opt)
  put(out/'progress.json',curve[-1]);put(out/'curve.json',curve)
  if elapsed>=a.max_seconds:raise TimeoutError('Incomplete fixed schedule: wall budget reached')
 reason='fixed_schedule_complete'
 assert state is not None;model.load_state_dict(state)
 torch.save({**bundle,'state_dict':state,'config':cfg,'embedding_mean':em,'embedding_std':es,'charge_median':med,'charge_mean':cm,'charge_std':cs},out/'best.pt')
 pred=predict(te);res,per=metrics(y[te],pred,d.iloc[te]);np.save(out/'evaluation_prediction.npy',pred);per.to_parquet(out/'evaluation_metrics.parquet',index=False)
 partition_results={}
 for name,ids in [('train',tr),('stop',va),('evaluation',te)]:
  pp=pred if name=='evaluation' else predict(ids)
  sm,rows=evaluation_with_CE(ids,pp);np.save(out/(name+'_prediction.npy'),pp);np.save(out/(name+'_source_indices.npy'),ids)
  partition_results[name]={'metrics':sm,'prediction_sha256':sha(out/(name+'_prediction.npy')),'indices_sha256':sha(out/(name+'_source_indices.npy')),'rows':len(ids)}
 put(out/'fixed_best_partitions.json',partition_results)
 torch.cuda.synchronize();t=time.monotonic();again=predict(te);torch.cuda.synchronize();inference=time.monotonic()-t;assert np.allclose(pred,again,atol=1e-7)
 receipt={'initial_model_digest':initial_model_digest,'initial_rng_digest':initial_rng_digest,'fixed_best_partitions_sha256':sha(out/'fixed_best_partitions.json'),'initial_stop_H':initial_stop,'source_optimizer_digest':source_state_digest,'initial_optimizer_digest':initial_optimizer_digest,'initial_optimizer_steps':initial_optimizer_steps,'rng_restored':bool(restored_rng),'final_LR_stage':stage,'status':'complete','run_id':ident,'config':cfg,'result':res,'best_stop_H':best,'best_epoch':best_ep,'epochs':epoch,'stop_reason':reason,'training_samples':len(tr),'training_proteins':int(d.iloc[tr].protein_id.nunique()),'training_groups':int(d.iloc[tr].group_id.nunique()),'updates':updates,'sample_presentations':presentations,'residue_presentations':residues,'fit_wall_seconds':time.monotonic()-start,'cached_readout_sequences_per_second':len(te)/inference,'cache_on_GPU':on_gpu,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(out/'best.pt'),'prediction_sha256':sha(out/'evaluation_prediction.npy'),'environment':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'gpu':torch.cuda.get_device_name()}}
 put(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['run_id','result','best_epoch','epochs','stop_reason','fit_wall_seconds']}),flush=True)
if __name__=='__main__':main()
