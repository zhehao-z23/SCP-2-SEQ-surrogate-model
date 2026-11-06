import os
"""G3 read-only historical exposure identity inventory. Emit hashes, never sequences."""
import base64,hashlib,json,sys
from pathlib import Path
import pandas as pd

spec=json.loads(base64.b64decode(sys.argv[1]));wanted=set(spec['public_accessions'])
r=Path(str(Path(os.environ['AFDB_PROJECT_ROOT']) / ''));rows=[];sources={}
for rel in ['data/retrieval/review_pool.parquet','data/metadata/review_metadata.parquet']:
    p=r/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    d=pd.read_parquet(p,columns=['sample_id','sequence','sequence_sha256'])
    for x in d.itertuples(index=False):
        if x.sample_id not in wanted or not isinstance(x.sequence,str):continue
        h=hashlib.sha256(x.sequence.encode()).hexdigest()
        assert h==x.sequence_sha256
        rows.append({'accession':x.sample_id,'sequence_sha256':h,'length':len(x.sequence),'source':rel})
p=r/'data/retrieval/legacy_sequence_search/legacy_sequences.fasta';sources[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
sid=None;parts=[]
def add():
    if sid in wanted:
        s=''.join(parts).upper()
        rows.append({'accession':sid,'sequence_sha256':hashlib.sha256(s.encode()).hexdigest(),'length':len(s),'source':'legacy_sequence_search'})
for l in p.open():
    if l.startswith('>'):add();sid=l[1:].split()[0];parts=[]
    else:parts.append(l.strip())
add()
print(json.dumps({'status':'historical_exposure_identity_hashes_only','requested_accessions':len(wanted),'sources':sources,'rows':rows,'unresolved_accessions':sorted(wanted-{x['accession'] for x in rows}),'no_sequences_transferred':True},indent=2))
