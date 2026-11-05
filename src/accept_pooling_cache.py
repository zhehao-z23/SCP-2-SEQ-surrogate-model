"""Independent complete cache hash acceptance before any head fit."""
import argparse,json,hashlib
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root);c=r/'cache';receipt=json.loads((c/'receipt.json').read_text());assert receipt['status']=='complete';files={}
for name,h in receipt['artifacts'].items():
 f=c/name;assert sha(f)==h;files[name]={'sha256':h,'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns}
out={'status':'accepted','receipt_sha256':sha(c/'receipt.json'),'files':files,'scope':'all published cache bytes independently hashed; no qualification of downstream scientific generalization'}
(r/'records/cache_acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
