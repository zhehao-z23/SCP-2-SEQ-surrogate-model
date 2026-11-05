"""Check all 16 pilot residues/means against the previously qualified encoder output."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--legacy-cache',required=True);a=p.parse_args();r=Path(a.root);legacy=Path(a.legacy_cache)
li=pd.read_parquet(legacy/'index.parquet');lm=np.load(legacy/'mean.npy');lookup=pd.Series(np.arange(len(li)),index=li.sequence_sha256);checks=[]
for rank in [0,1]:
 w=r/'cache_work'/f'worker{rank}';receipt=json.loads((w/'receipt_pilot.json').read_text());assert receipt['status']=='pilot_complete' and receipt['completed_rows']==8
 idx=json.loads((w/'index.json').read_text())[:8];flat=np.load(w/'flat.npy',mmap_mode='r');means=np.load(w/'mean.npy',mmap_mode='r');maxima=np.load(w/'max.npy',mmap_mode='r');orig=np.load(w/'original_mean.npy',mmap_mode='r')
 for j,x in enumerate(idx):
  part=flat[x['offset']:x['offset']+x['length']].astype('float32');assert len(part)==x['length'] and np.isfinite(part).all() and np.array_equal(part.mean(0,dtype='float32'),means[j]) and np.array_equal(part.max(0),maxima[j])
  old=lm[lookup.loc[x['sequence_sha256']]];assert np.allclose(old,orig[j],rtol=1e-4,atol=1e-5)
  checks.append({'rank':rank,'length':x['length'],'fp32_legacy_max_abs_error':float(abs(old-orig[j]).max()),'quantized_mean_max_abs_error':float(abs(old-means[j]).max())})
report={'status':'passed','rows':len(checks),'checks':checks,'maximum_fp32_parity_error':max(x['fp32_legacy_max_abs_error'] for x in checks),'maximum_quantized_mean_error':max(x['quantized_mean_max_abs_error'] for x in checks),'full_extraction_permitted':True}
(r/'records/cache_pilot_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
