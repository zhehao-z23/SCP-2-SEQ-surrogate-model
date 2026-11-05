"""Matched, bounded readout trials; immutable parent data and explicit cache identity."""
import argparse,hashlib,json,os,random,resource,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from pooling_model import PoolingModel

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--parent',required=True);ap.add_argument('--mode',choices=['mean_pre','max_pre','residue_mean','residue_attention'],required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--seed',type=int,default=1701);ap.add_argument('--max-seconds',type=int,default=900);ap.add_argument('--max-epochs',type=int,default=300);a=ap.parse_args()
 r=Path(a.root);p=Path(a.parent);sys.path.insert(0,str(p/'code'));from train_atlas_scale import select,metrics
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 data=p/'data';d=pd.read_parquet(data/'samples.parquet');y=np.load(data/'p.npy');rel=json.loads((data/'release.json').read_text());assert sha(data/'samples.parquet')==rel['artifacts']['samples.parquet'] and sha(data/'p.npy')==rel['artifacts']['p.npy']
 tr,va,te=select(d,a.fold,'all');cache=r/'cache';cr=json.loads((cache/'receipt.json').read_text());assert cr['status']=='complete'
 acceptance=json.loads((r/'records/cache_acceptance.json').read_text());assert acceptance['status']=='accepted' and acceptance['receipt_sha256']==sha(cache/'receipt.json')
 for name,check in acceptance['files'].items():
  fp=cache/name;assert fp.stat().st_size==check['size'] and fp.stat().st_mtime_ns==check['mtime_ns'],'immutable cache modified after acceptance'
  if name!='flat.npy':assert sha(fp)==check['sha256']
 ix=pd.read_parquet(cache/'index.parquet');assert ix.sequence_sha256.is_unique
 lookup=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256);ri=lookup.loc[d.sequence_sha256].to_numpy();length=ix.length.to_numpy()[ri];offset=ix.offset.to_numpy()[ri]
 means=np.load(cache/'mean.npy')[ri];assert means.shape==(len(d),960) and np.isfinite(means).all()
 em=means[tr].mean(0);es=means[tr].std(0);es=np.where(es<1e-6,1,es)
 assert sha(data/'composition_charge.npy')==rel['artifacts']['composition_charge.npy']
 charge=np.load(data/'composition_charge.npy');med=np.nan_to_num(np.nanmedian(charge[tr],0));flags=np.isnan(charge).astype('float32');charge=np.where(np.isnan(charge),med,charge);cm=charge[tr].mean(0);cs=charge[tr].std(0);cs=np.where(cs<1e-6,1,cs);extra=np.concatenate([(charge-cm)/cs,flags],1)
 cfg={'mode':a.mode,'fold':a.fold,'lr':a.lr,'seed':a.seed,'hidden':[1024,512,256],'dropout':.3,'input_dim':1038,'loss':'H2','batch':128,'max_padded_residues':16384,'epochs':a.max_epochs,'max_seconds':a.max_seconds,'min_epochs':40,'patience':30,'min_delta':1e-5,'optimizer':'AdamW','weight_decay':1e-4,'clip':1.,'LR_schedule':'constant','release_sha256':sha(data/'release.json'),'cache_receipt_sha256':sha(cache/'receipt.json'),'cache_index_sha256':sha(cache/'index.parquet'),'script_sha256':sha(__file__),'model_sha256':sha(Path(__file__).with_name('pooling_model.py')),'parent_metrics_and_split_sha256':sha(p/'code/train_atlas_scale.py'),'normalization':'fit-only mean-embedding mean/std shared across all residue aggregation; fit-only charge median/mean/std','microbatch_order':'permutation chunks1024 sort by length into128; shuffle batches; identical for paired modes','precision':'FP32 math; FP16 frozen cache; no AMP or TF32','not_blind':True,'new_C_transfer_permitted':False}
 ident=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();out=r/'runs'/ident;out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.json').exists():print('exists',ident);return
 with (out/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time()},f)
 put(out/'config.json',cfg);put(out/'roles.json',{'train_ids':d.iloc[tr].sample_id.tolist(),'stop_ids':d.iloc[va].sample_id.tolist(),'evaluation_ids':d.iloc[te].sample_id.tolist()})
 model=PoolingModel(a.mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
 ex=torch.tensor(extra,dtype=torch.float32,device='cuda');yt=torch.tensor(y,dtype=torch.float32,device='cuda');emg=torch.tensor(em,device='cuda');esg=torch.tensor(es,device='cuda')
 isvector=a.mode.endswith('_pre');flat=None;on_gpu=False
 if isvector:
  vector=means if a.mode=='mean_pre' else np.load(cache/'max.npy')[ri]
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
  if isvector:return model(vector[ids],ex[ids])
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
 def batches():
  order=np.random.permutation(tr);blocks=[]
  for pos in range(0,len(order),1024):
   b=order[pos:pos+1024];b=b[np.argsort(length[b],kind='stable')];blocks.extend([b[j:j+128] for j in range(0,len(b),128)])
  np.random.shuffle(blocks);return blocks
 start=time.monotonic();best=float('inf');best_ep=0;state=None;curve=[];reason='epoch_budget';updates=0;presentations=0;residues=0
 for epoch in range(1,a.max_epochs+1):
  model.train();tot=0.
  for ids in batches():
   opt.zero_grad(set_to_none=True)
   for small in micro(ids):
    logp=forward(small).log_softmax(1);loss=.5*((torch.exp(.5*logp)-yt[small].sqrt())**2).sum(1)
    loss=(loss*wt[small]).sum()/len(ids);assert torch.isfinite(loss);loss.backward();tot+=float(loss)*len(ids)
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();updates+=1;presentations+=len(ids);residues+=int(length[ids].sum())
  score=metrics(y[va],predict(va),d.iloc[va])[0]['group_macro_H'];elapsed=time.monotonic()-start
  curve.append({'epoch':epoch,'train_H2':tot/len(tr),'stop_H':score,'elapsed_seconds':elapsed,'updates':updates,'residue_presentations':residues})
  if score<best-1e-5:best=score;best_ep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  put(out/'progress.json',curve[-1]);put(out/'curve.json',curve)
  if epoch>=40 and epoch-best_ep>=30:reason='early_stop';break
  if elapsed>=a.max_seconds:reason='wall_budget';break
 assert state is not None;model.load_state_dict(state)
 torch.save({'state_dict':state,'config':cfg,'embedding_mean':em,'embedding_std':es,'charge_median':med,'charge_mean':cm,'charge_std':cs},out/'best.pt')
 pred=predict(te);res,per=metrics(y[te],pred,d.iloc[te]);np.save(out/'evaluation_prediction.npy',pred);per.to_parquet(out/'evaluation_metrics.parquet',index=False)
 torch.cuda.synchronize();t=time.monotonic();again=predict(te);torch.cuda.synchronize();inference=time.monotonic()-t;assert np.allclose(pred,again,atol=1e-7)
 receipt={'status':'complete','run_id':ident,'config':cfg,'result':res,'best_stop_H':best,'best_epoch':best_ep,'epochs':epoch,'stop_reason':reason,'training_samples':len(tr),'training_proteins':int(d.iloc[tr].protein_id.nunique()),'training_groups':int(d.iloc[tr].group_id.nunique()),'updates':updates,'sample_presentations':presentations,'residue_presentations':residues,'fit_wall_seconds':time.monotonic()-start,'cached_readout_sequences_per_second':len(te)/inference,'cache_on_GPU':on_gpu,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(out/'best.pt'),'prediction_sha256':sha(out/'evaluation_prediction.npy'),'environment':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'gpu':torch.cuda.get_device_name()}}
 put(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['run_id','result','best_epoch','epochs','stop_reason','fit_wall_seconds']}),flush=True)
if __name__=='__main__':main()
