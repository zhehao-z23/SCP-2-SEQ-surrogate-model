import os
"""G3-local historical exposure audit; transfer only identities/relations, never FASTA."""
import base64,hashlib,json,resource,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
import pandas as pd

spec=json.loads(base64.b64decode(sys.argv[1]));r=Path(str(Path(os.environ['AFDB_PROJECT_ROOT']) / ''))
wanted=set(spec['public_accessions']);start=time.time();seqs={};records=[];sources={}
def h(b):return hashlib.sha256(b).hexdigest()
def add(s,identity,source):
    key=h(s.encode());seqs[key]=s;records.append({'accession':identity,'sequence_sha256':key,'length':len(s),'source':source})
for rel in ['data/retrieval/review_pool.parquet','data/metadata/review_metadata.parquet']:
    p=r/rel;sources[rel]=h(p.read_bytes())
    for x in pd.read_parquet(p,columns=['sample_id','sequence','sequence_sha256']).itertuples(index=False):
        if x.sample_id in wanted and isinstance(x.sequence,str):
            assert h(x.sequence.encode())==x.sequence_sha256;add(x.sequence,x.sample_id,rel)
missing=wanted-{x['accession'] for x in records}
for identity in sorted(missing):
    if not identity.startswith('REF_'):continue
    p=r/'bridge/chemistry'/identity/'PARSE/structure_0/manifest.json'
    if not p.is_file():continue
    z=json.loads(p.read_text());sources[str(p.relative_to(r))]=h(p.read_bytes())
    for key in ['entity_sequence','prepared_observed_sequence']:
        s=z[key];assert h(s.encode())==z[key+'_sha256'];add(s,identity,key)
query={};api=[]
for x in spec['candidates']:
    url='https://alphafold.ebi.ac.uk/api/prediction/'+x['accession']
    with urllib.request.urlopen(url,timeout=20) as f:
        assert f.url.startswith('https://alphafold.ebi.ac.uk/');body=f.read(20_000_001)
    assert len(body)<=20_000_000
    z=[v for v in json.loads(body) if v.get('uniprotAccession')==x['accession'] and h(v.get('uniprotSequence','').encode())==x['sequence_sha256']]
    assert len(z)==1
    query[x['sequence_sha256']]=z[0]['uniprotSequence'];api.append({'accession':x['accession'],'API_sha256':h(body),'sequence_sha256':x['sequence_sha256']})
def limit():resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
hits=[];raw={};logs={};version=subprocess.check_output([os.environ['MMSEQS'],'version'],text=True).strip()
with tempfile.TemporaryDirectory(prefix='bridge_roles_exposure_') as tmp:
    t=Path(tmp)
    for name,vals,prefix in [('query',query,'p_'),('exposed',seqs,'e_')]:
        with (t/(name+'.fa')).open('w') as f:
            for key,s in sorted(vals.items()):f.write('>'+prefix+key+'\n'+s.translate(str.maketrans({c:'X' for c in 'BZJUO'}))+'\n')
    for direction,q,target,n in [('candidate_to_exposed','query','exposed',len(seqs)),('exposed_to_candidate','exposed','query',len(query))]:
        cmd=[os.environ['MMSEQS'],'easy-search',str(t/(q+'.fa')),str(t/(target+'.fa')),str(t/(direction+'.tsv')),str(t/('tmp_'+direction)),
             '--threads','2','--split-memory-limit','2G','--max-seqs',str(n+1),'--min-seq-id','0.3','-e','0.001','-s','7.5','-c','0',
             '--alignment-mode','3','--search-type','1','--format-output','query,target,fident,qcov,tcov,qlen,tlen,evalue','--remove-tmp-files','1']
        done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180,preexec_fn=limit)
        logs[direction]={'returncode':done.returncode,'log_sha256':h(done.stdout.encode())};assert done.returncode==0,done.stdout[-2000:]
        output=t/(direction+'.tsv');logs[direction]['output_sha256']=h(output.read_bytes());raw[direction]=0
        for line in output.open():
            qn,tn,ident,qcov,tcov,qlen,tlen,evalue=line.split();raw[direction]+=1
            cov=float(qcov) if int(qlen)<=int(tlen) else float(tcov)
            if float(ident)>=.3 and cov>=.8 and float(evalue)<=.001:
                hits.append({'direction':direction,'query':qn,'target':tn,'fident':float(ident),'shorter_coverage':cov,'evalue':float(evalue)})
print(json.dumps({'status':'complete','scope':'G3 local historical exposure search; two candidate queries, no historical sequence transfer or label/prediction use','script_sha256':spec['script_sha256'],
 'sources':sources,'exposure_records':records,'unique_exposed_sequences':len(seqs),'resolved_accessions':len({x['accession'] for x in records}),
 'unresolved_accessions':sorted(wanted-{x['accession'] for x in records}),'queries':api,'mmseqs_version':version,'raw_counts':raw,'accepted_hits':hits,'receipts':logs,'elapsed_seconds':time.time()-start,
 'limits':{'threads':2,'address_space_gib':4,'each_direction_seconds':180,'API_requests':len(query),'CIF_requests':0},
 'limitations':'Small panel direct relationships only; profile closure absent. Inputs remain provisional even if no accepted relationships.'},indent=2))
