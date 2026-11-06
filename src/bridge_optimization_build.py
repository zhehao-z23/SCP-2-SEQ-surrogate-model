import os
"""Create private experiment trainer from the accepted pooling trainer, without altering source."""
from pathlib import Path
r=Path(str(Path(os.environ['PROJECT_ROOT']) / 'bridge_20261106/optimization'));s=Path(str(Path(os.environ['PROJECT_ROOT']) / 'pooling_20261105/code/train_pooling.py')).read_text()
def change(a,b):
 global s
 assert a in s,a
 s=s.replace(a,b)
change('from pooling_model import PoolingModel','from bridge_optimization_model import PoolingModel')
change("choices=['mean_pre','max_pre','residue_mean','residue_attention']","choices=['mean_pre','residue_mean','charge_deep']")
change('a=ap.parse_args()',"ap.add_argument('--cache-root',required=True);ap.add_argument('--policy',choices=['fresh','plateau','reset','retain'],required=True);ap.add_argument('--warmstart');a=ap.parse_args()")
change("tr,va,te=select(d,a.fold,'all');cache=r/'cache'","tr,va,te=select(d,a.fold,'all');cache_source=Path(a.cache_root);cache=cache_source/'cache'")
change("(r/'records/cache_acceptance.json')","(cache_source/'records/cache_acceptance.json')")
change("sha(Path(__file__).with_name('pooling_model.py'))","sha(Path(__file__).with_name('bridge_optimization_model.py'))")
change("'not_blind':True,'new_C_transfer_permitted':False}","'not_blind':True,'new_C_transfer_permitted':False,'policy':a.policy,'warmstart_sha256':sha(a.warmstart) if a.warmstart else None,'warmstart_path':a.warmstart,'plan_sha256':sha(r/'plan.json'),'best_epoch_optimizer_and_rng_saved':True,'epoch0_eligible':True,'plateau_lrs':[.0003,.0001,.00003],'plateau_patience':15,'final_stage_min_epochs':20}")
change("'input_dim':1038","'input_dim':78 if a.mode=='charge_deep' else 1038")
change("'LR_schedule':'constant'","'LR_schedule':'continuous plateau' if a.policy=='plateau' else 'constant within fit'")
change("model=PoolingModel(a.mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)","""model=PoolingModel(a.mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
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
""")
change("isvector=a.mode.endswith('_pre')","isvector=a.mode in ['mean_pre','charge_deep']")
change("vector=means if a.mode=='mean_pre' else np.load(cache/'max.npy')[ri]","vector=means")
change("if isvector:return model(vector[ids],ex[ids])","if isvector:return model(vector[ids] if a.mode=='mean_pre' else None,ex[ids])")
change("start=time.monotonic();best=float('inf');best_ep=0;state=None;curve=[];reason='epoch_budget';updates=0;presentations=0;residues=0","""if restored_rng:
  random.setstate(restored_rng['python']);np.random.set_state(restored_rng['numpy']);torch.set_rng_state(restored_rng['torch']);torch.cuda.set_rng_state_all(restored_rng['cuda'])
 start=time.monotonic();initial_stop=metrics(y[va],predict(va),d.iloc[va])[0]['group_macro_H'];best=initial_stop;best_ep=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bundle=snapshot(opt);curve=[];reason='epoch_budget';updates=0;presentations=0;residues=0
 stage=0;stage_start=0;stage_best=initial_stop;stage_best_ep=0
 put(out/'initialization.json',{'stop_H':initial_stop,'source_optimizer_digest':source_state_digest,'initial_optimizer_digest':initial_optimizer_digest,'initial_optimizer_steps':initial_optimizer_steps,'rng_restored_from_source_best':bool(restored_rng),'policy':a.policy})""")
change("tot+=float(loss)*len(ids)","tot+=float(loss.detach())*len(ids)")
change("curve.append({'epoch':epoch,","curve.append({'lr':opt.param_groups[0]['lr'],'LR_stage':stage,'epoch':epoch,")
change("if score<best-1e-5:best=score;best_ep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}","if score<best-1e-5:best=score;best_ep=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bundle=snapshot(opt)")
change("if epoch>=40 and epoch-best_ep>=30:reason='early_stop';break", """if a.policy=='plateau':
   if score<stage_best-1e-5:stage_best=score;stage_best_ep=epoch
   if stage<2 and epoch-stage_best_ep>=15:
    stage+=1;stage_start=epoch;stage_best=score;stage_best_ep=epoch
    for group in opt.param_groups:group['lr']=cfg['plateau_lrs'][stage]
   if stage==2 and epoch-stage_start>=20 and epoch>=40 and epoch-best_ep>=30:reason='early_stop';break
  elif epoch>=40 and epoch-best_ep>=30:reason='early_stop';break""")
change("torch.save({'state_dict':state,'config':cfg,","torch.save({**bundle,'state_dict':state,'config':cfg,")
change("receipt={'status':'complete'","receipt={'initial_stop_H':initial_stop,'source_optimizer_digest':source_state_digest,'initial_optimizer_digest':initial_optimizer_digest,'initial_optimizer_steps':initial_optimizer_steps,'rng_restored':bool(restored_rng),'final_LR_stage':stage,'status':'complete'")
helper='''
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
'''
s=s.replace('def main():',helper+'\ndef main():');compile(s,'bridge_optimization_train.py','exec');p=r/'code/bridge_optimization_train.py';assert not p.exists();p.write_text(s);print(len(s))
