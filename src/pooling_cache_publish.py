"""Publish complete FP16 ragged cache; validate every residue and quantized pooling."""
import argparse,json,hashlib,resource,time
from pathlib import Path
import numpy as np,pandas as pd
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--legacy-cache',required=True);a=p.parse_args();r=Path(a.root);out=r/'cache';out.mkdir(exist_ok=True);assert not (out/'receipt.json').exists()
started=time.monotonic();rows=json.loads(Path(a.source).read_text());worker=[];lookup={};receipts=[]
for rank in [0,1]:
 d=r/'cache_work'/f'worker{rank}';receipt=json.loads((d/'receipt_full.json').read_text());assert receipt['status']=='complete'
 for name,h in receipt['artifacts'].items():assert sha(d/name)==h
 idx=json.loads((d/'index.json').read_text());arr=np.load(d/'flat.npy',mmap_mode='r');mu=np.load(d/'mean.npy',mmap_mode='r');ma=np.load(d/'max.npy',mmap_mode='r');orig=np.load(d/'original_mean.npy',mmap_mode='r')
 worker.append((arr,mu,ma,orig));receipts.append(receipt)
 for j,x in enumerate(idx):
  assert x['row_index'] not in lookup and x['sequence_sha256']==rows[x['row_index']]['sequence_sha256'] and x['length']==rows[x['row_index']]['length'];lookup[x['row_index']]=(rank,j,x)
assert set(lookup)==set(range(len(rows)))
offset=np.r_[0,np.cumsum([x['length'] for x in rows])];flat=np.lib.format.open_memmap(out/'flat.npy',mode='w+',dtype='float16',shape=(int(offset[-1]),960));means=np.zeros((len(rows),960),dtype='float32');maxima=np.zeros_like(means);origmeans=np.zeros_like(means);index=[]
for i,row in enumerate(rows):
 rank,j,x=lookup[i];ar,mu,ma,org=worker[rank];part=ar[x['offset']:x['offset']+x['length']];assert part.shape==(row['length'],960) and np.isfinite(part).all()
 restored=part.astype('float32');m=restored.mean(0,dtype=np.float32);mx=restored.max(0);assert np.array_equal(m,mu[j]) and np.array_equal(mx,ma[j])
 flat[offset[i]:offset[i+1]]=part;means[i]=m;maxima[i]=mx;origmeans[i]=org[j];index.append({'sequence_sha256':row['sequence_sha256'],'offset':int(offset[i]),'length':row['length']})
flat.flush();np.save(out/'mean.npy',means,allow_pickle=False);np.save(out/'max.npy',maxima,allow_pickle=False);pd.DataFrame(index).to_parquet(out/'index.parquet',index=False)
legacy=Path(a.legacy_cache);li=pd.read_parquet(legacy/'index.parquet');lm=np.load(legacy/'mean.npy');lidx=pd.Series(np.arange(len(li)),index=li.sequence_sha256);old=lm[lidx.loc[[x['sequence_sha256'] for x in rows]].to_numpy()]
fp32delta=np.abs(old-origmeans);assert np.allclose(old,origmeans,rtol=1e-4,atol=1e-5),'encoder source parity failed'
delta=np.abs(means-old);artifacts={f:sha(out/f) for f in ['flat.npy','index.parquet','mean.npy','max.npy']}
runtime_to_receipt=sum((r/'cache_work'/f'worker{k}'/f'receipt_{attempt}.json').stat().st_mtime-json.loads((r/'records'/f'cache_launch_{attempt}.json').read_text())['created_epoch'] for k in [0,1] for attempt in ['pilot','full'])
all_receipts=receipts+[json.loads((r/'cache_work'/f'worker{k}'/'receipt_pilot.json').read_text()) for k in [0,1]]
receipt={'schema':'pooling.residue.cache.v1','status':'complete','rows':len(rows),'residues':int(offset[-1]),'width':960,'dtype':'float16','encoder_compute':'FP32 singleton','no_truncation':True,'special_tokens_excluded':True,'derived_mean_max':'FP32 reductions of quantized FP16 residue cache','source_sha256':sha(a.source),'script_sha256':sha(__file__),'artifacts':artifacts,'worker_identities':[x['identity'] for x in receipts],'original_FP32_mean_vs_legacy_max_abs_error':float(fp32delta.max()),'quantized_mean_vs_legacy':{'max_absolute_error':float(delta.max()),'mean_absolute_error':float(delta.mean()),'per_sequence_max_quantiles':np.quantile(delta.max(1),[0,.5,.95,.99,1]).tolist()},'encoder_worker_reported_seconds':sum(x['elapsed_seconds']+json.loads((r/'cache_work'/f'worker{k}'/'receipt_pilot.json').read_text())['elapsed_seconds'] for k,x in enumerate(receipts)),'encoder_launch_to_receipt_seconds_sum':runtime_to_receipt,'time_definition':'reported worker time ends before final artifact hashing; launch-to-receipt includes startup and hashing, not measured GPU busy time','encoder_peak_GPU_bytes':max(x['peak_GPU_bytes'] for x in receipts),'encoder_peak_RSS_bytes':max(x['peak_RSS_bytes'] for x in receipts),'publisher_seconds':time.monotonic()-started,'publisher_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'source_sequence_order_preserved':True,'labels_read':False}
receipt['encoder_peak_GPU_bytes']=max(x['peak_GPU_bytes'] for x in all_receipts);receipt['encoder_peak_RSS_bytes']=max(x['peak_RSS_bytes'] for x in all_receipts)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,indent=2))
