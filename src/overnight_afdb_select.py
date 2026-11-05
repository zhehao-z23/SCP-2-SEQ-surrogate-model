"""Select a small deterministic length-stratified engineering panel from a frozen FASTA."""
import argparse,hashlib,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--fasta',required=True);ap.add_argument('--receipt',required=True);a=ap.parse_args()
p=Path(a.fasta);rows=[];sid=None;seq=[]
for line in p.read_text().splitlines()+['>END']:
 if line.startswith('>'):
  if sid is not None:
   s=''.join(seq);rows.append({'source_id':sid,'sequence':s,'length':len(s),'sequence_sha256':hashlib.sha256(s.encode()).hexdigest()})
  sid=line[1:].split()[0];seq=[]
 else:seq.append(line.strip())
assert len({r['source_id'] for r in rows})==len(rows)
valid=[r for r in rows if 40<=r['length']<=2400 and set(r['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY')]
seen=set();unique=[]
for r in sorted(valid,key=lambda r:r['source_id']):
 if r['sequence_sha256'] not in seen:unique.append(r);seen.add(r['sequence_sha256'])
selected=[];counts={}
for lo,hi in [(40,150),(151,300),(301,600),(601,2400)]:
 pool=[r for r in unique if lo<=r['length']<=hi];pool.sort(key=lambda r:hashlib.sha256(('afdb-engineering-1701:'+r['sequence_sha256']).encode()).hexdigest());assert len(pool)>=16
 for r in pool[:16]:r['selection_bin']=f'{lo}-{hi}'
 selected.extend(pool[:16]);counts[f'{lo}-{hi}']={'eligible':len(pool),'selected':16}
print(json.dumps({'rows':selected,'source':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'receipt_sha256':hashlib.sha256(Path(a.receipt).read_bytes()).hexdigest(),'all_rows':len(rows),'canonical_length_eligible':len(valid),'eligible_unique_sequences':len(unique),'strata':counts},'qualification':'Engineering distribution-shift and inference diagnosis only. Biased legacy annotation/candidate subset; no verified paired current 63-bin physical target; no AFDB-wide accuracy claim.'},indent=2))
