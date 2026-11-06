import os
"""Bounded AF-input versus historical/new-C relation audit. No targets/predictions."""
import hashlib,json,os,resource,subprocess,time,signal
from pathlib import Path
import pandas as pd
ROOT=Path(str(Path(os.environ['PROJECT_ROOT']) / ''));r=ROOT/'bridge_20261106/roles';out=r/'search_v2'
assert not out.exists();out.mkdir();(out/'tmp').mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def fasta(p):
 key=None;seq=[]
 for line in Path(p).open():
  if line.startswith('>'):
   if key:yield key,''.join(seq)
   key=line[1:].split()[0];seq=[]
  else:seq.append(line.strip())
 if key:yield key,''.join(seq)
paths={'old_search_manifest':ROOT/'data/homology/sequence_search_manifest.parquet','normalization_source':ROOT/'tools/cohort_prepare.py','old_views':ROOT/'data/homology/all_views.fasta','newC':r/'source/newC_homology.fasta','aliases':r/'source/newC_aliases.tsv','old_groups':ROOT/'data/qualification/release_v1/all_sequence_groups.parquet','old_manifest':ROOT/'data/qualification/release_v1/sample_manifest.parquet','initial':r/'source/public_AF_inventory_private.json','screen_v2':r/'screen_v2/source/public_AF_inventory_private.json','old33':ROOT/'bridge_20261106/physical/inputs/existing_inventory.json','old64':ROOT/'overnight_20261106/afdb/data/overnight_afdb_sequences.json','exposure':r/'source/exposure_ids_private.json'}
v2=json.loads(paths['screen_v2'].read_text());assert v2['status']=='complete_input_screen' and len(v2['rows'])==96 and v2['CIF_requests']<=8
records=[]
for kind,key in [('initial_input','initial'),('screen_v2','screen_v2'),('old33_exposed','old33'),('old64_model_evaluated','old64')]:
 z=json.loads(paths[key].read_text());rows=z['rows'] if isinstance(z,dict) else z
 for row in rows:
  s=row.get('sequence');h=row.get('sequence_sha256');sid=row.get('accession',row.get('sample_id',row.get('source_id',row.get('id',h))))
  if s:
   assert hashlib.sha256(s.encode()).hexdigest()==h
   assert set(s)<=set('ACDEFGHIKLMNPQRSTVWY')
  records.append(dict(accession=sid,sequence_sha256=h,sequence=s,length=len(s) if s else row.get('length'),source=kind,input_status=row.get('status'),cif_sha256=row.get('cif_sha256',row.get('source_cif_sha256')),input_eligible=row.get('status')=='input_eligible_role_pending',exposed=kind!='screen_v2'))
unique={x['sequence_sha256']:x['sequence'] for x in records if x['sequence']}
with (out/'panel.fasta').open('w') as f:
 for h,s in sorted(unique.items()):f.write('>p_'+h+'\n'+s+'\n')
# Sequence strings stay private; role metadata need only identities/hashes.
put(out/'panel_manifest_private.json',dict(records=[{k:v for k,v in x.items() if k!='sequence'} for x in records],query_sequences=len(unique)))
counts={};keys=set()
search_map=pd.read_parquet(paths['old_search_manifest']).set_index('sequence_sha256')
group_ids=set(pd.read_parquet(paths['old_groups']).sequence_sha256)
assert len(search_map)==187939 and search_map.index.is_unique
assert set(search_map.index)==group_ids
reference_identities=[]
with (out/'reference.fasta').open('w') as f:
 for prefix,key in [('o_','old_views'),('c_','newC')]:
  n=0
  for k,s in fasta(paths[key]):
   h=hashlib.sha256(s.encode()).hexdigest();original=k.removeprefix('c_')
   if key=='old_views':
    m=search_map.loc[original];assert m.search_sequence_sha256==h and int(m.length)==len(s)
   else:assert original==h
   reference_identities.append(dict(node=prefix+original,identity_sha256=original,search_sha256=h))
   q=prefix+original;assert q not in keys;keys.add(q);f.write('>'+q+'\n'+s+'\n');n+=1
  counts[key]=n
put(out/'reference_identity_private.json',reference_identities)
assert counts=={'old_views':187939,'newC':133315},counts
plan=dict(status='registered',script_sha256=sha(__file__),sources={k:dict(relative=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in paths.items()},reference_counts=counts,panel_unique_sequences=len(unique),identity_min=.3,shorter_sequence_coverage_min=.8,evalue_max=.001,sensitivity=7.5,threads=4,max_address_space_bytes=16*1024**3,split_memory_limit='10G',disk_cap_bytes=20*1024**3,wall_seconds_cap=7200,directions=['panel_to_reference','reference_to_panel','panel_to_panel'],normalization='Inherited old FASTA BZJUO->X verified against immutable search manifest; original view identity preserved, actual search hash separately recorded.',scope='Sequence search only; all scored hits retained before explicit shorter-sequence coverage filter. Not a full newC-to-old search, profile or remote-family closure.')
put(out/'plan.json',plan);start=time.monotonic();done=[]
def size():return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
def limit():resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
mm=os.environ['MMSEQS']
try:
 for name,q,t,n in [('panel_to_reference','panel.fasta','reference.fasta',sum(counts.values())),('reference_to_panel','reference.fasta','panel.fasta',len(unique)),('panel_to_panel','panel.fasta','panel.fasta',len(unique))]:
  command=[mm,'easy-search',str(out/q),str(out/t),str(out/(name+'.tsv')),str(out/'tmp'/name),'--threads','4','--split-memory-limit','10G','--max-seqs',str(n+1),'--min-seq-id','0.3','-e','0.001','-s','7.5','-c','0','--alignment-mode','3','--search-type','1','--format-output','query,target,fident,qcov,tcov,qlen,tlen,evalue','--remove-tmp-files','1']
  with (out/(name+'.log')).open('x') as log:
   p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,preexec_fn=limit,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   put(out/'progress.json',dict(status='running',direction=name,pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],elapsed=time.monotonic()-start,completed=done))
   while p.poll() is None:
    if time.monotonic()-start>7200 or size()>20*1024**3:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     raise RuntimeError('search budget exceeded; incomplete relations cannot authorize protection')
    time.sleep(2)
   assert p.returncode==0,(name,p.returncode)
  done.append(dict(direction=name,command=command,sha256=sha(out/(name+'.tsv')),bytes=(out/(name+'.tsv')).stat().st_size))
 put(out/'receipt.json',dict(status='complete',script_sha256=sha(__file__),plan_sha256=sha(out/'plan.json'),directions=done,elapsed_seconds=time.monotonic()-start,disk_bytes=size(),scope=plan['scope']))
except BaseException as e:
 put(out/'receipt.json',dict(status='failed',error=repr(e),completed=done,elapsed_seconds=time.monotonic()-start));raise
