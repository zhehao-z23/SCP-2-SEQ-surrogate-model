"""Append explicit historical bin/protocol binding; never relabel or rebuild arrays."""
import argparse,json,hashlib
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--root',required=True);args=a.parse_args()
s=Path(args.source);r=Path(args.root);p=s/'data/labels/release.pending.json';d=json.loads(p.read_text())
assert d['release_id']=='historical_canonical_p_byte_20261012' and d['target']=='p' and len(d['edges'])==62
assert sha(s/'data/labels/p.float32.npy')==json.loads((r/'data/release.json').read_text())['sources'][str(s/'data/labels/p.float32.npy')]
record={'status':'bound','release_sha256':sha(r/'data/release.json'),'label_contract_path':str(p),'label_contract_sha256':sha(p),'label_contract':d,'correction':'Original development release sources included source/shared_bins.json as an extra reference. It is a different, new physical bin contract and was NOT used to transform targets or calculate any reported metric. This addendum binds the actual historical label edges and protocol. Original receipt and all arrays remain unchanged.','numerical_recomputation_required':False,'metrics':'H independent of bin numeric positions; CDF is unweighted 62-cutpoint MSE; tails are first/last bins under this historical boundary rule','qualification':'Canonical probabilities and historical structure pairing are inherited; new protocol physical accuracy is not claimed; historical full-U eligibility is not used to assert T=U.'}
(r/'data/label_contract_addendum.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':'bound','edges_min':d['edges'][0],'edges_max':d['edges'][-1],'units':d['units'],'physical_protocol':d['physical_protocol']}))
