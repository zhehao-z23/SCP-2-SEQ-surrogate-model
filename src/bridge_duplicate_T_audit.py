"""All exact-T duplicates in one pinned development release; no fitting/phi IO.

The cohort selection is made from identity metadata before reading targets.
Raw statistics are previously saved QC summaries, not re-evaluated phi arrays.
Per-case identities and metadata are written only to the private output root.
"""
import argparse, hashlib, itertools, json, os, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
sys.dont_write_bytecode=True

PINS={'release':'67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b',
      'contract':'05932133251f6c20956d9dbf77af4d857a78137bf125b3485c1e97098ba61c9c',
      'guard':'71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',
      'guard_code':'5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def hellinger(p,q):
    import numpy as np
    return float(np.sqrt(.5*((np.sqrt(p)-np.sqrt(q))**2).sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--code-sha256',required=True);a=ap.parse_args()
    import resource,signal
    os.sched_setaffinity(0,{174});resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s bound')));signal.alarm(600)
    import numpy as np,pandas as pd,pyarrow.parquet as pq
    start=time.monotonic();assert __debug__ and sha(__file__)==a.code_sha256
    assert os.sched_getaffinity(0)=={174} and not a.output.exists();a.output.mkdir(parents=True)
    assert hellinger(np.array([1.,0.]),np.array([1.,0.]))==0
    assert hellinger(np.array([1.,0.]),np.array([0.,1.]))==1
    plan={'selection':'all sequence_sha256 duplicated across rows of pinned 21921-row release, no target/error selection',
          'comparison':'all unordered row pairs in each duplicate T; same-label and same-source strata retained',
          'input_identity':'exact stored T SHA256, no sequence text loaded',
          'metrics':'current FP32 p bit equality and float64 H; exact raw count equality and QC quantile/mean differences',
          'source_identity':'p artifact+row, saved matrix path, structure SHA and pairing/target source metadata; no original structure bytes replay',
          'no_causal_attribution':True,'CPU':174,'BLAS_threads':1,'GPU':False,'RAM_GiB':8,'wall_seconds':600,
          'new_fits':0,'raw_phi_reads':0,'model_reads':0,'code_sha256':sha(__file__)}
    put(a.output/'plan.json',plan)
    refs=[]
    def verify(p,expected):
        actual=sha(p);assert actual==expected,str(p)
        refs.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size});return actual
    root=a.source_root;data=root/'atlas_scale_20261105/data';roles=root/'bridge_20261106/roles'
    verify(data/'release.json',PINS['release']);release=load(data/'release.json')
    verify(data/'samples.parquet',release['artifacts']['samples.parquet'])
    cols=['sample_id','protein_id','sequence_sha256','length','group_id','fold','target_row','source_structure_sha256','source_origin','source','coverage']
    meta=pq.read_table(data/'samples.parquet',columns=cols).to_pandas().reset_index(drop=True)
    assert len(meta)==21921 and meta.sample_id.is_unique and meta.sequence_sha256.nunique()==21915 and meta.group_id.nunique()==7999
    assert (meta.protein_id=='T_'+meta.sequence_sha256).all()
    selected=meta.loc[meta.sequence_sha256.duplicated(False)].copy();indices=selected.index.to_numpy();ids=selected.sample_id.tolist()
    assert len(selected)==12 and selected.sequence_sha256.nunique()==6
    assert selected.groupby('sequence_sha256').size().eq(2).all()
    verify(roles/'code/bridge_roles_guard.py',PINS['guard_code']);sys.path.insert(0,str(roles/'code'))
    from bridge_roles_guard import check
    request={'operation':'read_target','stage':'development','sequence_sha256':sorted(selected.sequence_sha256.unique().tolist()),'aliases':[]}
    guard=check(roles/'registry_v1/guard_contract.json',PINS['guard'],request)
    put(a.output/'role_preflight.json',{'request':request,'receipt':guard})
    verify(root/'data/labels/release.pending.json',PINS['contract']);contract=load(root/'data/labels/release.pending.json')
    qcpath=root/'source/labels/source_snapshots/matrix_phi_qc.parquet'
    countpath=root/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet'
    manifestpath=root/'data/qualification/release_v1/sample_manifest.parquet';indexpath=root/'data/labels/index.parquet'
    for p in [qcpath,countpath]:verify(p,contract['source_hashes']['source_snapshots/'+p.name])
    for p in [manifestpath,indexpath]:verify(p,release['sources'][str(p)])
    verify(data/'p.npy',release['artifacts']['p.npy'])
    def aligned(path,columns):
        d=pq.read_table(path,columns=['sample_id']+columns,filters=[('sample_id','in',ids)]).to_pandas()
        assert d.sample_id.is_unique and set(d.sample_id)==set(ids)
        return d.set_index('sample_id').loc[ids].reset_index()
    qcols=['q0p01','q0p05','q0p1','q0p5','q1','q5','q25','q50','q75','q95','q99','q99p5','q99p9','q99p95','q99p99']
    qc=aligned(qcpath,['matrix_path','readable','phi_column','matrix_columns','n_rows','n_vertices','n_nan_phi','n_inf_phi','phi_mean','phi_min','phi_max']+qcols)
    mc=['teacher_sequence_sha256','teacher_length','sequence_sha256','source_structure_sha256','source_origin','source_release','source_sequence_id','target_row','target_file_sha256','target_source_sha256','pairing_receipt_sha256','role','group_id','outer_fold','label_join_valid','static_eligible','sequence_view','strict_sequence_equal','T_U_coverage']
    manifest=aligned(manifestpath,mc);ix=aligned(indexpath,['row','label_integrity_valid','n_vertices'])
    bins=[f'bin_{i:03d}' for i in range(63)];count=aligned(countpath,bins)[bins].to_numpy();n=count.sum(1,dtype=np.int64)
    assert np.issubdtype(count.dtype,np.integer) and (count>=0).all() and (n>0).all()
    assert np.array_equal(n,qc.n_vertices) and np.array_equal(n,qc.n_rows) and np.array_equal(n,ix.n_vertices)
    assert qc.readable.all() and qc.phi_column.eq('phi').all() and qc.matrix_columns.eq('phi,x,y,z').all()
    assert qc.n_nan_phi.eq(0).all() and qc.n_inf_phi.eq(0).all() and ix.label_integrity_valid.all()
    assert manifest.role.eq('development').all() and manifest.label_join_valid.all() and manifest.static_eligible.all()
    assert np.array_equal(manifest.teacher_sequence_sha256,selected.sequence_sha256)
    assert np.array_equal(manifest.teacher_length,selected.length)
    assert np.array_equal(manifest.source_structure_sha256,selected.source_structure_sha256)
    assert np.array_equal(manifest.target_row,selected.target_row) and np.array_equal(ix.row,selected.target_row)
    assert np.array_equal(manifest.group_id,selected.group_id) and np.array_equal(manifest.outer_fold,selected.fold)
    for c in ['source_structure_sha256','target_file_sha256','target_source_sha256','pairing_receipt_sha256']:
        assert manifest[c].dropna().map(lambda v:isinstance(v,str) and len(v)==64 and all(x in '0123456789abcdef' for x in v)).all(),c
    pmem=np.load(data/'p.npy',mmap_mode='r',allow_pickle=False);assert pmem.shape==(len(meta),63) and pmem.dtype==np.float32
    p=np.asarray(pmem[indices]);assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
    assert np.array_equal(p,(count/n[:,None]).astype(np.float32))
    qvalues=qc[qcols+['phi_mean']].to_numpy(float);assert np.isfinite(qvalues).all()
    m=selected.reset_index(drop=True);pairs=[]
    identity_cols=['source_structure_sha256','source_origin','source_release','source_sequence_id','target_row','target_file_sha256','target_source_sha256','pairing_receipt_sha256','sequence_sha256','strict_sequence_equal','T_U_coverage']
    for t,rows in m.groupby('sequence_sha256',sort=True):
        assert rows.group_id.nunique()==rows.fold.nunique()==1
        for i,j in itertools.combinations(rows.index,2):
            same={c:None if pd.isna(manifest.loc[i,c]) or pd.isna(manifest.loc[j,c]) else bool(manifest.loc[i,c]==manifest.loc[j,c]) for c in identity_cols}
            same['matrix_path']=bool(qc.loc[i,'matrix_path']==qc.loc[j,'matrix_path'])
            same['current_p_artifact_and_row']=bool(indices[i]==indices[j])
            same['original_label_artifact_and_row']=same['target_file_sha256'] and same['target_row']
            same['n_vertices']=bool(n[i]==n[j]);same['count_vector']=bool(np.array_equal(count[i],count[j]));same['current_p']=bool(np.array_equal(p[i],p[j]))
            delta={c:float(qc.loc[j,c]-qc.loc[i,c]) for c in qcols+['phi_mean']}
            pairs.append({'T_sha256':t,'sample_ids':[m.loc[i,'sample_id'],m.loc[j,'sample_id']],
                'group_id':m.loc[i,'group_id'],'fold':int(m.loc[i,'fold']),
                'H':hellinger(p[i].astype(float),p[j].astype(float)),
                'max_abs_p_difference':float(abs(p[i].astype(float)-p[j].astype(float)).max()),
                'identity_equal':same,'QC_signed_row2_minus_row1':delta,
                'n_vertices':[int(n[i]),int(n[j])],
                'metadata':[manifest.iloc[k].to_dict() for k in [i,j]],
                'QC_metadata':[qc.iloc[k].to_dict() for k in [i,j]]})
    assert len(pairs)==6
    private={'status':'complete','selection':'all duplicate T, no error selection','pairs':pairs,'source_refs':refs,
             'role_preflight_sha256':sha(a.output/'role_preflight.json'),'code_sha256':sha(__file__)}
    put(a.output/'pairs_private.json',private)
    same_counts={k:sum(x['identity_equal'][k] is True for x in pairs) for k in pairs[0]['identity_equal']}
    missing_counts={k:sum(x['identity_equal'][k] is None for x in pairs) for k in pairs[0]['identity_equal']}
    hs=np.array([x['H'] for x in pairs]);affected=m.loc[m.sequence_sha256.isin([x['T_sha256'] for x in pairs if not x['identity_equal']['current_p']])]
    group_sizes=meta.groupby('group_id').sequence_sha256.nunique()
    group_support=sum(1/float(group_sizes[g]) for g in affected.drop_duplicates('sequence_sha256').group_id)/meta.group_id.nunique()
    # Triangle inequality gives a lower bound for the two rows receiving one common deterministic prediction.
    floor_row=float(hs.sum()/len(meta));floor_T=float((hs/2).sum()/meta.sequence_sha256.nunique())
    floor_group=float(sum(x['H']/(2*group_sizes[x['group_id']]) for x in pairs)/meta.group_id.nunique())
    summary={'schema':'duplicate_T_audit_summary_v1','status':'complete','selection':plan['selection'],
       'cohort':{'rows':len(meta),'unique_T':int(meta.sequence_sha256.nunique()),'groups':int(meta.group_id.nunique())},
       'duplicate_support':{'rows':len(m),'unique_T':int(m.sequence_sha256.nunique()),'groups':int(m.group_id.nunique()),'unordered_pairs':len(pairs),'rows_per_duplicate_T':2},
       'nonidentical_p_support':{'rows':len(affected),'unique_T':int(affected.sequence_sha256.nunique()),'groups':int(affected.group_id.nunique())},
       'same_identity_pair_counts':same_counts,'unavailable_identity_comparison_pair_counts':missing_counts,'pair_H':{'min':float(hs.min()),'median':float(np.median(hs)),'max':float(hs.max()),'nonzero':int((hs>0).sum())},
       'max_abs_p_difference':max(x['max_abs_p_difference'] for x in pairs),
       'max_abs_saved_raw_QC_differences':{c:max(abs(x['QC_signed_row2_minus_row1'][c]) for x in pairs) for c in qcols+['phi_mean']},
       'affected_max_metric_contribution_support_bounds':{'row_mean_H':len(affected)/len(meta),'T_macro_H':affected.sequence_sha256.nunique()/meta.sequence_sha256.nunique(),'group_macro_H_exact_T_weight':float(group_support),'assumption':'H<=1, and row->T->group aggregation; possible contribution ceiling, not observed error or causal explanation'},
       'triangle_inequality_lower_bounds_if_one_deterministic_T_prediction':{'row_mean_H':floor_row,'T_macro_H':floor_T,'group_macro_H':floor_group,'scope':'empirical duplicate rows only; not global conditional noise estimate or source of model residuals'},
       'controls':{'all_12_current_p_bit_equal_exact_count_normalization_FP32':True,'all_12_saved_QC_count_index_and_manifest_bindings_match':True,'identity_self_H_zero':True,'onehot_disjoint_H_one':True,'same_input_and_same_target_pair_count':same_counts['current_p'],'same_input_and_same_source_structure_pair_count':same_counts['source_structure_sha256'],'controls_do_not_create_an_independent_physical_replicate':True},
       'not_identified':['causal decomposition into structure, preparation, protocol, grid or numerical effects','per-object PQR/mesh/grid equality from raw solver receipts','original matrix bytes or continuous phi replay','sequence-exact unseen input conditional variability'],
       'source_code_sha256':sha(__file__),'plan_sha256':sha(a.output/'plan.json'),'release_sha256':PINS['release'],'label_contract_sha256':PINS['contract'],'guard_contract_sha256':PINS['guard'],
       'role_preflight_sha256':sha(a.output/'role_preflight.json'),'private_pairs_sha256':sha(a.output/'pairs_private.json'),'p_sha256':release['artifacts']['p.npy'],
       'resource':{'affinity':sorted(os.sched_getaffinity(0)),'BLAS_threads':1,'GPU':False,'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
       'data_access':{'metadata_rows_for_selection':len(meta),'actual_target_rows':len(m),'saved_QC_rows':len(qc),'count_rows':len(count),'sequence_text_rows':0,'raw_phi_reads':0,'models_loaded':0,'new_fits':0,'protected_targets':0,'gallery_modified':False}}
    put(a.output/'summary.json',summary)
    print(json.dumps({'status':'complete','summary_sha256':sha(a.output/'summary.json'),'duplicate_support':summary['duplicate_support'],'pair_H':summary['pair_H'],'same_identity_pair_counts':same_counts,'resource':summary['resource']}))

if __name__=='__main__':main()
