import os
"""G3 in-place screening: emit public IDs/hashes/lengths only, never FASTA."""
import hashlib,json,sys,base64
from pathlib import Path
spec=json.loads(base64.b64decode(sys.argv[1]));excluded=set(spec['excluded_accessions'])
p=Path(str(Path(os.environ['AFDB_PROJECT_ROOT']) / 'data/retrieval/legacy_sequence_search/legacy_sequences.fasta'))
rows=[];sid=None;parts=[]
def save():
 if sid:
  s=''.join(parts).upper()
  if set(s)<=set('ACDEFGHIKLMNPQRSTVWY'):
   rows.append(dict(accession=sid,length=len(s),sequence_sha256=hashlib.sha256(s.encode()).hexdigest()))
for line in p.read_text().splitlines():
 if line.startswith('>'):save();sid=line[1:].split()[0];parts=[]
 else:parts.append(line.strip())
save();unique={}
for row in sorted(rows,key=lambda x:x['accession']):unique.setdefault(row['sequence_sha256'],row)
old64=[]
for lo,hi in [(40,150),(151,300),(301,600),(601,2400)]:
 old64+=sorted([r for r in unique.values() if lo<=r['length']<=hi],key=lambda r:hashlib.sha256(('afdb-engineering-1701:'+r['sequence_sha256']).encode()).hexdigest())[:16]
excluded.update(r['accession'] for r in old64)
excluded_sha={r['sequence_sha256'] for r in rows if r['accession'] in excluded}
chosen=[]
for lo,hi in [(100,200),(201,400),(401,600),(601,1000)]:
 eligible=[r for r in unique.values() if lo<=r['length']<=hi and r['accession'] not in excluded and r['sequence_sha256'] not in excluded_sha]
 for r in sorted(eligible,key=lambda r:(hashlib.sha256(('bridge_protected_inputs_v2:'+r['sequence_sha256']).encode()).hexdigest(),r['accession']))[:24]:chosen.append(dict(r,length_stratum=f'{lo}-{hi}'))
assert len(chosen)==spec['count']==96 and len({r['sequence_sha256'] for r in chosen})==96
print(json.dumps(dict(status='input_ID_screen_only',source_fasta_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_records=len(rows),excluded_ID_union=len(excluded),excluded_sequence_hashes=len(excluded_sha),exposure_inventory_sha256=spec['exposure_inventory_sha256'],initial24_sha256=spec['initial24_sha256'],selection='24 per length stratum, salted sequence-hash ordering; complete recorded exposure ID union and exact prior64 salted panel excluded',selected=chosen,limitation='Historical biased candidate pool; neither absence of remote homology nor pLM pretraining novelty implied; no target or model prediction read'),indent=2))
