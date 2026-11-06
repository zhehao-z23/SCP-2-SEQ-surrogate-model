"""Verify saved charge bytes and cache addresses for all prior duplicate T pairs.

No embedding array is loaded or rehashed. Acceptance metadata plus fresh index
hash bind addresses; detailed identities stay in the existing private audit root.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--audit-root',type=Path,required=True);ap.add_argument('--code-sha256',required=True);a=ap.parse_args()
    import resource,signal
    os.sched_setaffinity(0,{174});resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30));signal.alarm(600)
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    start=time.monotonic();assert __debug__ and sha(__file__)==a.code_sha256
    d=a.source_root/'atlas_scale_20261105/data';c=a.source_root/'pooling_20261105/cache';roles=a.source_root/'bridge_20261106/roles'
    assert sha(a.audit_root/'summary.json')=='40d97475b6dbdce650add4d10e1c39162eec3fd6f273cd3a9c80a14a4ff6595a'
    old=load(a.audit_root/'summary.json');assert sha(d/'release.json')==old['release_sha256'];rel=load(d/'release.json')
    assert sha(d/'samples.parquet')==rel['artifacts']['samples.parquet']
    m=pq.read_table(d/'samples.parquet',columns=['sample_id','sequence_sha256','length','fold','group_id']).to_pandas().reset_index(drop=True)
    v=m.loc[m.sequence_sha256.duplicated(False)];assert len(v)==12 and v.sequence_sha256.nunique()==6
    assert sha(roles/'code/bridge_roles_guard.py')=='5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'
    sys.path.insert(0,str(roles/'code'));from bridge_roles_guard import check
    guard=check(roles/'registry_v1/guard_contract.json',old['guard_contract_sha256'],{'operation':'read_target','stage':'development','sequence_sha256':sorted(v.sequence_sha256.unique().tolist()),'aliases':[]})
    assert sha(d/'composition_charge.npy')==rel['artifacts']['composition_charge.npy']
    charge=np.load(d/'composition_charge.npy',mmap_mode='r',allow_pickle=False);assert charge.shape[0]==len(m) and charge.ndim==2
    receipt_pin='c278ca5f93b52070d9b10a86c5934d369b7d371bdc7013f17a4f56399e6f0c23'
    index_pin='83c8f9f753fa27e38c86156dce2fd2deb826c3dceaa6210d20dd1674a73d2c0a'
    assert sha(c/'receipt.json')==receipt_pin;cr=load(c/'receipt.json');assert cr['status']=='complete'
    acceptance_path=c.parent/'records/cache_acceptance.json';ac=load(acceptance_path)
    assert ac['status']=='accepted' and ac['receipt_sha256']==receipt_pin
    for name,info in ac['files'].items():
        assert cr['artifacts'][name]==info['sha256']
        st=(c/name).stat();assert st.st_size==info['size'] and st.st_mtime_ns==info['mtime_ns']
    assert sha(c/'index.parquet')==index_pin==ac['files']['index.parquet']['sha256']
    ix=pq.read_table(c/'index.parquet',columns=['sequence_sha256','offset','length']).to_pandas().reset_index(drop=True)
    assert ix.sequence_sha256.is_unique
    lookup=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256);out=[]
    for t,z in v.groupby('sequence_sha256',sort=True):
        assert len(z)==2 and z.fold.nunique()==z.group_id.nunique()==1
        i,j=z.index;ri=lookup.loc[z.sequence_sha256].to_numpy();offset=ix.offset.to_numpy()[ri];length=ix.length.to_numpy()[ri]
        assert np.array_equal(length,z.length.to_numpy()) and (length>0).all() and (offset>=0).all()
        x,y=np.asarray(charge[i]),np.asarray(charge[j]);mask_equal=np.array_equal(np.isnan(x),np.isnan(y))
        equal=np.array_equal(x,y,equal_nan=True);bits=x.tobytes()==y.tobytes()
        sameaddress=bool(ri[0]==ri[1] and offset[0]==offset[1] and length[0]==length[1])
        assert mask_equal and equal and bits and sameaddress
        out.append({'T_sha256':t,'sample_ids':z.sample_id.tolist(),'charge_bits_equal':bits,'NaN_masks_equal':mask_equal,'charge_values_equal_nan':equal,'same_cache_index_offset_length':sameaddress,'cache_index_rows':ri.tolist(),'cache_offsets':offset.tolist(),'cache_lengths':length.tolist(),'charge_row_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    private={'pairs':out,'guard_receipt':guard,'code_sha256':sha(__file__),'prior_summary_sha256':sha(a.audit_root/'summary.json')}
    put(a.audit_root/'input_closure_private.json',private)
    summary={'schema':'duplicate_T_actual_input_closure_v1','status':'passed','rows':12,'duplicate_T_pairs':6,'charge_shape_per_row':[int(charge.shape[1])],'charge_dtype':str(charge.dtype),'charge_byte_identical_pairs':sum(p['charge_bits_equal'] for p in out),'NaN_mask_identical_pairs':sum(p['NaN_masks_equal'] for p in out),'same_cache_index_offset_length_pairs':sum(p['same_cache_index_offset_length'] for p in out),'current_model_input_scope':'charge descriptor bytes and frozen residue-cache addresses; same per-checkpoint normalization then gives the same deterministic eval-mode input for these exact T pairs','embedding_arrays_loaded':0,'embedding_bytes_rehashed':False,'cache_check':'receipt and index freshly hashed; all accepted cache file size/mtime identities checked; prior full-byte acceptance retained','new_fits':0,'model_reads':0,'raw_phi_reads':0,'sequence_text_reads':0,'gallery_modified':False,'release_sha256':old['release_sha256'],'charge_artifact_sha256':rel['artifacts']['composition_charge.npy'],'cache_receipt_sha256':receipt_pin,'cache_index_sha256':index_pin,'cache_acceptance_sha256':sha(acceptance_path),'source_code_sha256':sha(__file__),'prior_summary_sha256':sha(a.audit_root/'summary.json'),'private_input_closure_sha256':sha(a.audit_root/'input_closure_private.json'),'resource':{'affinity':sorted(os.sched_getaffinity(0)),'BLAS_threads':1,'GPU':False,'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    put(a.audit_root/'input_closure_summary.json',summary);print(json.dumps({'status':'passed','pairs':6,'charge_bits_equal_pairs':6,'same_cache_address_pairs':6,'summary_sha256':sha(a.audit_root/'input_closure_summary.json'),'resource':summary['resource']}))
if __name__=='__main__':main()
