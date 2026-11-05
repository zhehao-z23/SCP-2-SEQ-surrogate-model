"""Resumable disjoint-worker FP16 residue cache from frozen singleton FP32 ESMC."""
import argparse,fcntl,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--backend-src',required=True);p.add_argument('--checkpoint-manifest',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--gpu-uuid',required=True);p.add_argument('--max-items',type=int,default=0);p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--attempt',required=True);a=p.parse_args()
 out=Path(a.output)/('worker'+str(a.rank));out.mkdir(parents=True,exist_ok=True)
 lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import numpy as np,torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu_uuid
 assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==a.gpu_uuid.lower().removeprefix('gpu-')
 rows=json.loads(Path(a.source).read_text());assert len({x['sequence_sha256'] for x in rows})==len(rows)
 for x in rows:assert len(x['sequence'])==x['length'] and hashlib.sha256(x['sequence'].encode()).hexdigest()==x['sequence_sha256']
 ids=list(range(a.rank,len(rows),a.workers));ordered=sorted(ids,key=lambda i:rows[i]['length']);pilot=[ordered[round(i*(len(ordered)-1)/7)] for i in range(8)];order=pilot+[i for i in ids if i not in pilot]
 lengths=[rows[i]['length'] for i in order];offsets=np.r_[0,np.cumsum(lengths)].tolist();assert offsets[-1]*960*2<20*1024**3
 identity={'source_sha256':sha(a.source),'script_sha256':sha(__file__),'backend_sha256':sha(Path(a.backend_src)/'seq2scp_training/hf_backend.py'),'checkpoint_manifest_sha256':sha(a.checkpoint_manifest),'rank':a.rank,'workers':a.workers,'row_order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),'dtype':'float16','compute':'fp32','encoder':'esmc_300m','layer':30,'width':960,'truncation':False,'special_tokens':False,'singleton_unpadded':True}
 ip=out/'identity.json'
 if ip.exists():assert json.loads(ip.read_text())==identity
 else:put(ip,identity)
 initial=not (out/'flat.npy').exists()
 if initial:
  flat=np.lib.format.open_memmap(out/'flat.npy',mode='w+',dtype='float16',shape=(offsets[-1],960))
  means=np.lib.format.open_memmap(out/'mean.npy',mode='w+',dtype='float32',shape=(len(order),960));maxima=np.lib.format.open_memmap(out/'max.npy',mode='w+',dtype='float32',shape=(len(order),960));original=np.lib.format.open_memmap(out/'original_mean.npy',mode='w+',dtype='float32',shape=(len(order),960))
  put(out/'index.json',[dict(row_index=ri,sequence_sha256=rows[ri]['sequence_sha256'],offset=offsets[j],length=lengths[j]) for j,ri in enumerate(order)])
  completed=0
 else:
  flat=np.load(out/'flat.npy',mmap_mode='r+');means=np.load(out/'mean.npy',mmap_mode='r+');maxima=np.load(out/'max.npy',mmap_mode='r+');original=np.load(out/'original_mean.npy',mmap_mode='r+');completed=json.loads((out/'progress.json').read_text())['completed_rows']
 assert flat.shape==(offsets[-1],960) and flat.dtype==np.float16
 start=time.monotonic();count0=completed;stop=False;state={'status':'loading','attempt':a.attempt,'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time(),'completed_rows':completed,'total_rows':len(order),'residues':offsets[-1],'gpu_uuid':a.gpu_uuid,'max_seconds':a.max_seconds,'max_RSS_bytes':16*1024**3,'source_read_only':True,'labels_read':False}
 def flush(status='running'):
  for ar in [flat,means,maxima,original]:ar.flush()
  state.update(status=status,completed_rows=completed,elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated());put(out/'progress.json',state)
 def halt(*_):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);flush('loading')
 sys.path.insert(0,a.backend_src)
 from seq2scp_training.hf_backend import LocalHFBackend
 try:
  manifest=json.loads(Path(a.checkpoint_manifest).read_text());assert manifest['encoder']=='esmc_300m' and manifest['compute_precision']=='fp32'
  backend=LocalHFBackend(manifest,device='cuda:0');assert backend.layers==30 and not backend.model.training and not any(p.requires_grad for p in backend.model.parameters())
  flush()
  for j in range(completed,len(order)):
   if stop or time.monotonic()-start>=a.max_seconds or (a.max_items and completed-count0>=a.max_items):break
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3,'RSS budget'
   ri=order[j];x=rows[ri];result=backend.extract_batch([x['sequence']],[30],precision='fp32')[0];mat=result['layers'][30]
   assert result['sequence_sha256']==x['sequence_sha256'] and result['positions']==list(range(x['length'])) and mat.shape==(x['length'],960) and np.isfinite(mat).all()
   quant=mat.astype('float16');assert np.isfinite(quant).all();flat[offsets[j]:offsets[j+1]]=quant;restored=quant.astype('float32');means[j]=restored.mean(0,dtype=np.float32);maxima[j]=restored.max(0);original[j]=mat.mean(0,dtype=np.float32);completed=j+1
   if completed%50==0:flush()
  status='complete' if completed==len(order) else ('pilot_complete' if a.max_items and completed-count0>=a.max_items else 'budget_stopped');flush(status)
  delta=np.abs(means[:completed]-original[:completed]);state['quantized_mean_absolute_error']={'max':float(delta.max()),'mean':float(delta.mean()),'per_sequence_max_quantiles':np.quantile(delta.max(1),[0,.5,.95,.99,1]).tolist()}
  artifacts={f:sha(out/f) for f in ['flat.npy','mean.npy','max.npy','original_mean.npy','index.json']} if status=='complete' else {'index.json':sha(out/'index.json')}
  put(out/('receipt_'+a.attempt+'.json'),dict(state,identity=identity,artifacts=artifacts));print(json.dumps(state),flush=True)
 except BaseException as e:
  state['error']=repr(e);flush('failed');put(out/('receipt_'+a.attempt+'.json'),dict(state,identity=identity));raise
if __name__=='__main__':main()
