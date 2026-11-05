"""Bounded vector learning curves reusing the validated Seq2SCP model/loss.

No source data is modified. This is retrospective development, not blind testing.
"""
import argparse,copy,hashlib,json,os,random,sys,time,resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def metrics(y,p,meta):
 h=np.sqrt(.5*((np.sqrt(y)-np.sqrt(p))**2).sum(1))
 c=((np.cumsum(y,1)[:,:-1]-np.cumsum(p,1)[:,:-1])**2).mean(1)
 t=np.abs(y[:,[0,-1]]-p[:,[0,-1]]).mean(1)
 d=meta[['sample_id','protein_id','group_id','length','coverage','source']].copy().reset_index(drop=True)
 d['H']=h;d['cutpoint_CDF_MSE']=c;d['edge_bin_mass_MAE']=t
 per=d.groupby(['group_id','protein_id'])[['H','cutpoint_CDF_MSE','edge_bin_mass_MAE']].mean()
 group=per.groupby('group_id').mean()
 out={'group_macro_'+k:float(v) for k,v in group.mean().items()}
 out.update({'protein_macro_'+k:float(v) for k,v in per.mean().items()})
 out.update(samples=len(d),proteins=len(per),groups=len(group))
 return out,d
def select(meta,fold,scale):
 fit=meta[~meta.fold.isin([fold,(fold+1)%5])]
 groups=sorted(fit.group_id.unique(),key=lambda x:hashlib.sha256(('atlas-scale-1701:'+x).encode()).hexdigest())
 if scale!='all':groups=groups[:int(scale)]
 train=fit.index[fit.group_id.isin(groups)].to_numpy()
 stop=meta.index[meta.fold==(fold+1)%5].to_numpy();test=meta.index[meta.fold==fold].to_numpy()
 assert not(set(meta.loc[train].group_id)&set(meta.loc[stop].group_id))
 assert not(set(meta.loc[train].group_id)&set(meta.loc[test].group_id))
 return train,stop,test
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--source',required=True);ap.add_argument('--family',required=True,choices=['composition_charge_deep']);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--scale',required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--seed',type=int,default=1701);ap.add_argument('--max-epochs',type=int,default=300);ap.add_argument('--max-seconds',type=int,default=600);a=ap.parse_args()
 root=Path(a.root);data=root/'data';source=Path(a.source)
 sys.path.insert(0,str(source/'package/src'))
 from seq2scp_training.models import SequenceModel
 from seq2scp_training.losses import distribution_loss
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 meta=pd.read_parquet(data/'samples.parquet');y=np.load(data/'p.npy',allow_pickle=False)
 release=json.loads((data/'release.json').read_text())
 for name in ['samples.parquet','p.npy']:assert sha(data/name)==release['artifacts'][name]
 tr,va,te=select(meta,a.fold,a.scale)
 feature_paths=[]
 if a.family.startswith('composition'):
  feature_paths=[data/'composition_charge.npy'];x=np.load(feature_paths[0]);hidden=[1024,512,256];drop=.3
 else:
  cache=root/'data/esmc_300m_mean'
  receipt=json.loads((cache/'receipt.json').read_text())
  assert receipt['status']=='complete'
  idx=pd.read_parquet(cache/'index.parquet');emb=np.load(cache/'mean.npy',allow_pickle=False)
  assert idx.sequence_sha256.is_unique and meta.sequence_sha256.isin(idx.sequence_sha256).all()
  # The cache writer supplies its own receipt; bind exact final payload independently here.
  x=emb[pd.Series(np.arange(len(idx)),index=idx.sequence_sha256).loc[meta.sequence_sha256].to_numpy()]
  feature_paths=[cache/'receipt.json',cache/'index.parquet',cache/'mean.npy']
  hidden=[1024,512,256];drop=.3
  if a.family=='esmc_charge':
   feature_paths.append(data/'composition_charge.npy');x=np.concatenate([x,np.load(feature_paths[-1])],1)
 assert len(x)==len(meta) and np.isfinite(y).all() and not np.isinf(x).any()
 med=np.nanmedian(x[tr],axis=0);med=np.nan_to_num(med)
 # Always retain explicit missingness for handcrafted descriptors; encoder values must be finite.
 flag=np.isnan(x).astype(np.float32);filled=np.where(np.isnan(x),med,x)
 mean=filled[tr].mean(0);std=filled[tr].std(0);std=np.where(std<1e-6,1,std)
 x=(filled-mean)/std
 if a.family.startswith('composition'):x=np.concatenate([x,flag],1)
 elif a.family=='esmc_charge':x=np.concatenate([x,flag[:,-39:]],1)
 assert np.isfinite(x).all()
 cfg={'family':a.family,'fold':a.fold,'train_groups_requested':a.scale,'lr':a.lr,'seed':a.seed,'hidden':hidden,'dropout':drop,'epochs':a.max_epochs,'max_seconds':a.max_seconds,'min_epochs':40,'patience':30,'min_delta':1e-5,'batch':128,'optimizer':'AdamW','weight_decay':1e-4,'schedule':'fixed constant LR, not scanned','loss':'H2','selection':'stop group-macro H','precision':'FP32','input_dim':x.shape[1],'data_release_sha256':sha(data/'release.json'),'script_sha256':sha(__file__),'model_sha256':sha(source/'package/src/seq2scp_training/models.py'),'loss_sha256':sha(source/'package/src/seq2scp_training/losses.py'),'feature_hashes':{str(p):sha(p) for p in feature_paths},'split':'train other3 folds; stop next cyclic fold; evaluation fold i; retrospective exposed development'}
 identity=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();out=root/'runs'/identity
 out.mkdir(parents=True,exist_ok=True)
 if (out/'receipt.json').exists():print('existing',identity);return
 lock=out/'started.json'
 with lock.open('x') as f:json.dump({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'epoch':time.time()},f)
 put(out/'config.json',cfg)
 put(out/'roles.json',{'train_ids':meta.loc[tr].sample_id.tolist(),'stop_ids':meta.loc[va].sample_id.tolist(),'evaluation_ids':meta.loc[te].sample_id.tolist()})
 device='cuda';xt=torch.tensor(x,dtype=torch.float32,device=device);yt=torch.tensor(y,device=device)
 model=SequenceModel(input_dim=x.shape[1],bins=63,kind='vector',pool='mean',hidden=hidden,dropout=drop).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
 count=meta.loc[tr].protein_id.value_counts();weights=torch.tensor(1/meta.loc[tr].protein_id.map(count).to_numpy(),dtype=torch.float32,device=device)
 weights=weights/weights.mean()
 def predict(ids):
  model.eval()
  with torch.inference_mode():return np.concatenate([model(xt[ix])['logits'].softmax(1).cpu().numpy() for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/1024))))])
 started=time.monotonic();best=float('inf');best_epoch=0;best_state=None;curve=[];reason='epoch_budget';steps=0
 for epoch in range(1,a.max_epochs+1):
  model.train();order=np.random.permutation(len(tr));total=0.
  for offset in range(0,len(tr),128):
   local=order[offset:offset+128];ids=tr[local];opt.zero_grad(set_to_none=True)
   loss=(distribution_loss(model(xt[ids])['logits'],yt[ids])*weights[local]).mean()
   assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();steps+=1;total+=float(loss)*len(ids)
  valid,_=metrics(y[va],predict(va),meta.loc[va]);score=valid['group_macro_H']
  curve.append({'epoch':epoch,'train_H2':total/len(tr),'stop_H':score,'elapsed_seconds':time.monotonic()-started,'updates':steps})
  if score<best-cfg['min_delta']:
   best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  if epoch%10==0:put(out/'progress.json',curve[-1])
  if epoch>=40 and epoch-best_epoch>=30:reason='early_stop';break
  if time.monotonic()-started>=a.max_seconds:reason='wall_budget';break
 assert best_state is not None
 model.load_state_dict(best_state);torch.save({'state_dict':best_state,'config':cfg,'median':med,'mean':mean,'std':std,'training_release':release},out/'best.pt')
 pred=predict(te);torch.cuda.synchronize();t0=time.monotonic();pred2=predict(te);torch.cuda.synchronize();infer=time.monotonic()-t0
 assert np.allclose(pred,pred2,atol=1e-7)
 result,per=metrics(y[te],pred,meta.loc[te]);np.save(out/'evaluation_prediction.npy',pred,allow_pickle=False);per.to_parquet(out/'evaluation_metrics.parquet',index=False)
 strata={}
 per['length_bin']=pd.cut(per.length,[0,150,300,600,np.inf]).astype(str)
 per['coverage_bin']=pd.cut(per.coverage,[-np.inf,.7,.9,.98,np.inf]).astype(str)
 for c in ['length_bin','coverage_bin','source']:strata[c]={str(k):metrics(y[te][v.index.to_numpy()],pred[v.index.to_numpy()],meta.loc[te].iloc[v.index.to_numpy()])[0] for k,v in per.groupby(c)}
 put(out/'strata.json',strata);put(out/'curve.json',curve)
 baseline=np.repeat(y[tr].mean(0,keepdims=True),len(te),axis=0);center,_=metrics(y[te],baseline,meta.loc[te])
 receipt={'status':'complete','run_id':identity,'config':cfg,'result':result,'training_mean_baseline':center,'best_epoch':best_epoch,'epochs':epoch,'stop_reason':reason,'best_stop_H':best,'training_samples':len(tr),'training_proteins':meta.loc[tr].protein_id.nunique(),'training_groups':meta.loc[tr].group_id.nunique(),'stop_samples':len(va),'evaluation_samples':len(te),'updates':steps,'sample_presentations':epoch*len(tr),'wall_seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'head_inference_sequences_per_second':len(te)/infer,'encoder_extraction_cost_included':False,'checkpoint_sha256':sha(out/'best.pt'),'prediction_sha256':sha(out/'evaluation_prediction.npy'),'not_blind':True,'new_C_transfer_permitted':False,'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name()}}
 put(out/'receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ['run_id','result','training_samples','training_groups','best_epoch','stop_reason','wall_seconds']}),flush=True)
if __name__=='__main__':main()
