"""Bounded official AFDB input-only retrieval; never loads physical targets/predictions."""
import argparse,hashlib,json,time,urllib.request,urllib.error,importlib.util
from pathlib import Path
def sha(x):return hashlib.sha256(x).hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise urllib.error.HTTPError(args[1],302,'redirect rejected for request accounting',{},None)
OPENER=urllib.request.build_opener(NoRedirect)
def get(url):
    assert url.startswith('https://alphafold.ebi.ac.uk/')
    with OPENER.open(urllib.request.Request(url,headers={'User-Agent':'Seq2SCP-input-audit/1.0'}),timeout=20) as f:
        assert f.url.startswith('https://alphafold.ebi.ac.uk/');b=f.read(20_000_001)
        assert len(b)<=20_000_000;return b
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--selection',required=True);a=ap.parse_args();r=Path(a.root)
    assert not r.exists(),'immutable new screening directory required';(r/'source/public_AF').mkdir(parents=True);(r/'records').mkdir();out=r/'source/public_AF'
    selected=json.loads(Path(a.selection).read_text());rows=[];beg=time.time()
    assert len(selected['selected'])==96
    plan=dict(status='registered',max_API_requests=96,max_CIF_requests=8,timeout_seconds=20,minimum_API_mean_pLDDT=85,maximum_fraction_pLDDT_below70=.10,max_structures_per_length_stratum=2,role='unassigned_input_screen; no target/prediction access',structural_requirements='single complete F1 fragment, exact source sequence SHA, full contiguous CA mapping to API sequence, mean pLDDT>=85 and <=10% residues below70',novelty='Will be checked against full historical views and newC; not inferred from accession or exact-sequence nonoverlap.')
    plan.update(script_sha256=sha(Path(__file__).read_bytes()),max_wall_seconds=1200,CIF_cap_counts='attempted HTTP requests, including failed/unqualified responses; automatic redirects disabled');put(r/'records/public_AF_retrieval_plan.json',plan)
    api_attempts=0;cif_attempts=0;counts={};have_bio=bool(importlib.util.find_spec('Bio'))
    if have_bio:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        from Bio.SeqUtils import seq1
    for row in selected['selected']:
        if time.time()-beg>1200:break
        z=dict(row);sid=z['accession'];apiurl='https://alphafold.ebi.ac.uk/api/prediction/'+sid
        try:
            p=out/(sid+'_api.json')
            if not p.exists():
                assert api_attempts<96;api_attempts+=1;p.write_bytes(get(apiurl))
            b=p.read_bytes();arr=json.loads(b);assert isinstance(arr,list) and arr
            matches=[x for x in arr if x.get('uniprotAccession')==sid and int(x.get('uniprotStart',0))==1 and int(x.get('uniprotEnd',0))==len(x.get('uniprotSequence',''))]
            assert len(matches)==1,'absent/ambiguous complete structure'
            x=matches[0];sequence=x['uniprotSequence'];assert sha(sequence.encode())==z['sequence_sha256'],'current source sequence differs'
            z.update(sequence=sequence,api_sha256=sha(b),api_url=apiurl,entry_id=x['entryId'],version=x.get('latestVersion'),cif_url=x['cifUrl'],mean_plddt=x.get('globalMetricValue'),status='api_bound')
            good=float(z['mean_plddt'])>=85
            if good and counts.get(z['length_stratum'],0)<2 and cif_attempts<8:
                assert have_bio,'Bio parser unavailable; do not assert structure qualification'
                cp=out/(sid+'.cif')
                if not cp.exists():
                    assert cif_attempts<8;cif_attempts+=1;put(r/'records/request_counts.json',dict(API=api_attempts,CIF=cif_attempts));cp.write_bytes(get(x['cifUrl']))
                d=MMCIF2Dict(str(cp));names=d['_atom_site.label_atom_id'];take=[i for i,n in enumerate(names) if n=='CA' and d['_atom_site.group_PDB'][i]=='ATOM']
                chains={d['_atom_site.label_asym_id'][i] for i in take};assert len(chains)==1
                keys=[int(d['_atom_site.label_seq_id'][i]) for i in take]
                assert keys==list(range(1,len(sequence)+1)),'non-contiguous/duplicate/missing CA'
                observed=''.join(seq1(d['_atom_site.label_comp_id'][i]) for i in take);assert observed==sequence,'CIF/API sequence mismatch'
                confidence=[float(d['_atom_site.B_iso_or_equiv'][i]) for i in take];low=sum(q<70 for q in confidence)/len(confidence)
                z.update(cif_path=str(cp),cif_sha256=sha(cp.read_bytes()),auth_chain=d['_atom_site.auth_asym_id'][take[0]],observed_sequence_sha256=sha(observed.encode()),CA_count=len(take),CA_mean_pLDDT=sum(confidence)/len(confidence),CA_fraction_below70=low,structure_identity_verified=True)
                if low<=.10 and z['CA_mean_pLDDT']>=85:
                    counts[z['length_stratum']]=counts.get(z['length_stratum'],0)+1;z['status']='input_eligible_role_pending'
                else:z['status']='confidence_gate_not_met'
            elif not good:z['status']='API_confidence_gate_not_met'
            else:z['status']='API_bound_CIF_budget_or_stratum_cap'
        except Exception as ex:z.update(status='unavailable_or_not_qualified',error=str(ex))
        rows.append(z);put(r/'records/public_AF_retrieval_progress.json',dict(rows=rows,API_requests=api_attempts,CIF_requests=cif_attempts,elapsed_seconds=time.time()-beg,eligible_by_stratum=counts))
    put(r/'source/public_AF_inventory_private.json',dict(status='complete_input_screen' if len(rows)==96 else 'wall_budget',API_requests=api_attempts,CIF_requests=cif_attempts,script_sha256=sha(Path(__file__).read_bytes()),rows=rows,elapsed_seconds=time.time()-beg,Bio_available=have_bio,eligible_by_stratum=counts,selection_sha256=sha(Path(a.selection).read_bytes()),physical_accuracy_evaluated=False))
    print(json.dumps(dict(rows=len(rows),API_requests=api_attempts,CIF_requests=cif_attempts,status_counts={k:sum(z['status']==k for z in rows) for k in sorted({z['status'] for z in rows})},eligible_by_stratum=counts,elapsed_seconds=time.time()-beg)))
if __name__=='__main__':main()
