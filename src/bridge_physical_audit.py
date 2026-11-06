"""Independent read-only audit of a completed AF surface label chain.

This does not grant numerical qualification: it separates source/export/ledger
integrity from required numerical comparisons and any explicitly approved
method-evidence inheritance.
"""
from pathlib import Path
import argparse,json,hashlib,sqlite3,sys,time
import numpy as np
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def checked(r):
    assert sha(r['path'])==r['sha256'],r['path'];return Path(r['path'])

def reference(path):return {'path':str(path),'sha256':sha(path)}

def sampling_audit(root,sid,variant,out,inputs):
    """Every query retains eight corners and one owner, including fallbacks."""
    d=root/'cases'/sid/variant;phi=np.load(inputs['corrected_phi']['path']);raw=np.load(inputs['raw_trilinear_phi']['path']);weights=np.load(inputs['area_weights']['path']);vertices=np.load(inputs['vertices']['path'])
    assert phi.shape==raw.shape==weights.shape==(len(vertices),)and np.isfinite(phi).all()and np.isfinite(raw).all()
    assert np.isfinite(vertices).all()and np.isfinite(weights).all()and(weights>=0).all()and weights.sum()>0
    if variant=='further':
        rows=out['field_sampling_receipts']
    else:
        plan=load(d/'geometry_plan.json');rows=[]
        for part in plan['parts']:
            rank=part['rank'];pd=d/('part_'+str(rank))
            rows.append({'rank':rank,'query_indices':reference(pd/'query_indices.npy'),'phi':reference(pd/'phi_jump.npy'),'raw':reference(pd/'phi_raw.npy'),'accepted':reference(pd/'query_accepted.npy'),'field':reference(root/'records/SES_map_fields'/(sid+'_'+variant+'_fine_'+str(rank)+'.json'))})
    assert rows and len({r['rank']for r in rows})==len(rows)
    ownership=np.zeros(len(phi),np.int32);accepted=np.zeros(len(phi),bool);parts=[]
    for row in rows:
        for key in ['query_indices','phi','raw','accepted','field']:checked(row[key])
        idx=np.load(row['query_indices']['path']);px=np.load(row['phi']['path']);rx=np.load(row['raw']['path']);ok=np.load(row['accepted']['path'])
        assert idx.ndim==1 and len(idx)>0 and np.issubdtype(idx.dtype,np.integer)and idx.min()>=0 and idx.max()<len(phi)and len(np.unique(idx))==len(idx)
        assert px.shape==rx.shape==ok.shape==idx.shape and ok.dtype==np.bool_
        assert np.isfinite(px).all()and np.isfinite(rx).all()and np.array_equal(px,phi[idx])and np.array_equal(rx,raw[idx])
        assert np.array_equal(px[~ok],rx[~ok]),'Rejected correction must retain ordinary trilinear value.'
        np.add.at(ownership,idx,1);accepted[idx]=ok
        field=load(row['field']['path']);header={};deltas=[]
        with open(field['dx_path'])as f:
            for _ in range(100):
                line=f.readline();words=line.split()
                if 'gridpositions'in words:header['dims']=list(map(int,words[-3:]))
                elif words and words[0]=='origin':header['origin']=list(map(float,words[1:]))
                elif words and words[0]=='delta':deltas.append(list(map(float,words[1:])))
                if 'data follows'in line:break
        assert header['dims']==field['dims']and np.allclose(header['origin'],field['origin'],rtol=0,atol=5e-5)and np.allclose(deltas,np.diag(field['h']),rtol=0,atol=5e-6)
        h=np.diag(np.asarray(deltas));assert np.all(h>0)
        lower=np.floor((vertices[idx]-np.asarray(header['origin']))/h).astype(np.int64)
        assert(lower>=0).all()and(lower+1<np.asarray(header['dims'])).all(),'All eight interpolation corners must exist.'
        parts.append({'rank':row['rank'],'points':len(idx),'all_eight_corners_present':True,'part_global_arrays_exact':True,'accepted_boolean':True,'false_correction_retains_raw_exactly':True,'actual_grid':{'origin':header['origin'],'h':h.tolist(),'dims':header['dims']},'references':{key:row[key]for key in ['query_indices','phi','raw','accepted','field']}})
    assert np.all(ownership==1)
    fallback=float(weights[~accepted].sum()/weights.sum())
    assert abs(fallback-out['fallback_area_fraction'])<=1e-13
    code=Path(inputs['protocol_configuration']['path']).parent.parent/'code'
    return {'variant':variant,'pass':True,'source_outcome':reference(d/'outcome.json'),'representation':out['representation'],'source_inputs':inputs,'all_queries_retained':True,'queries':len(phi),'ownership_min':int(ownership.min()),'ownership_max':int(ownership.max()),'all_eight_corners_present':True,'raw_and_corrected_finite':True,'part_global_arrays_exact':True,'accepted_boolean':True,'weights_nonnegative_positive_total':True,'false_correction_retains_raw_exactly':True,'fallback_area_fraction':fallback,'recorded_fallback_area_fraction':out['fallback_area_fraction'],'fallback_fraction_difference':fallback-out['fallback_area_fraction'],'accepted_false_queries':int((~accepted).sum()),'no_invented_fallback_fraction_threshold':True,'parts':parts,'sampling_code':[reference(code/name)for name in ['surface_field_sampling.py','planar_jump_interpolation.py','vendor_revision003/solver_surface.py']]}
def audit(root,sid,rehash_dx=False):
    d=root/'cases'/sid;pre=load(d/'prepared_case.json');obj=pre['source'];checked(pre['source_cif']);context=pre['structure'];chem=pre['PQR']
    assert obj['source_sha256']==pre['source_cif']['sha256']
    assert context['status']=='ready_for_parameterization'and chem['status']=='ok'and chem['chemical_integrity_status']=='passed'
    assert sha(pre['pqr_path'])==chem['pqr_sha256']
    assert hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()==chem['source_preparation_sha256']
    if obj.get('sequence_sha256'):assert obj['sequence_sha256']==context['prepared_observed_sequence_sha256']
    variants=[];seen=set();field_audits=[];hashes={};sampling=[]
    def verify(ref):
        path=ref['path']
        if path not in hashes:hashes[path]=sha(path)
        assert hashes[path]==ref['sha256'],path
        return Path(path)
    with sqlite3.connect('file:'+str(root/'records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)as db:
        db.row_factory=sqlite3.Row
        for op in sorted(d.glob('*/outcome.json')):
            out=load(op);assert out['status']=='completed';mp=checked(out['representation']);manifest=load(mp)
            assert manifest['primary_array']=='corrected_area_p'and manifest['potential_clipping_or_rescaling']is False
            for item in manifest['products']:assert sha(mp.parent/item['name'])==item['sha256']
            v=np.load(mp.parent/'vectors.npz');p=v['corrected_area_p'];assert p.shape==(63,)and np.isfinite(p).all()and (p>=0).all()and abs(p.sum()-1)<2e-12
            assert np.array_equal(v['corrected_area_sqrt_p'],np.sqrt(p));assert v['internal_edges_kBT_per_e'].shape==(62,)
            spec=manifest['spec'];ins=spec['inputs']
            for ref in ins.values():checked(ref)
            assert np.array_equal(v['internal_edges_kBT_per_e'],np.array(load(ins['shared_bins']['path'])['internal_edges']))
            q=load(ins['source_case_record']['path']);pqrsha=sha(q['pqr_path']);phi=np.load(ins['corrected_phi']['path']);w=np.load(ins['area_weights']['path']);assert phi.shape==w.shape and np.isfinite(phi).all()and np.isfinite(w).all()and(w>=0).all()and w.sum()>0
            edges=np.r_[-np.inf,v['internal_edges_kBT_per_e'],np.inf];rp=np.histogram(phi,edges,weights=w/w.sum())[0];assert np.max(abs(rp-p))<1e-12
            ids=[]
            for rr in spec['PB_field_receipts']:
                receipt=checked(rr);field=load(receipt);pid=field['PB_attempt_id'];entry=db.execute('select * from attempts where id=?',(pid,)).fetchone();assert entry['status']=='completed'and Path(entry['result_path'])==receipt
                assert field['source']['source_PQR_sha256']==pqrsha and field['status']=='completed'and field['logs']['logs_acceptable']
                assert sha(Path(field['result']['cwd'])/'apbs.in')==field['source']['input_sha256']
                assert field['source']['physical']==load(ins['protocol_configuration']['path'])['PB_physical_parameters']
                if rehash_dx and pid not in seen:
                    verify({'path':field['dx_path'],'sha256':field['dx_sha256']})
                    source=field['source'];maps=source['coefficient_maps'];cwd=Path(field['result']['cwd'])
                    verify({'path':str(cwd/'molecule.pqr'),'sha256':source['normalized_PQR_sha256']})
                    assert load(cwd/'bindings.json')==source
                    assert entry['input_hash']==hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
                    raw_logs=[cwd/name for name in ['apbs.stdout.log','apbs.stderr.log','apbs.internal.log','io.mc']]
                    assert all(path.is_file()for path in raw_logs)
                    for ref in maps:verify(ref)
                    if maps:
                        map_manifest=Path(maps[0]['path']).parent/'map_manifest.json';mr=load(map_manifest)
                        assert mr['maps']==maps and mr['source']['PQR_sha256']==pqrsha
                        assert mr['source']['vertices_sha256']==ins['vertices']['sha256']and mr['source']['faces_sha256']==ins['faces']['sha256']
                        assert mr['source']['dims']==field['dims']and np.allclose(mr['source']['origin_A'],field['origin'],rtol=0,atol=5e-5)and np.allclose(mr['source']['h_A'],field['h'],rtol=0,atol=5e-6)
                    else:map_manifest=None
                    if source.get('boundary_map'):verify({'path':source['boundary_map'],'sha256':source['boundary_map_sha256']})
                    archive=field['archive'];assert archive['lossless_roundtrip']and archive['original_retained']and archive['decompressed_sha256']==field['dx_sha256']
                    assert archive['decompressed_bytes']==Path(field['dx_path']).stat().st_size
                    verify(archive['compressed'])
                    runtime=source['native_runtime'];verify({'path':runtime['manifest'],'sha256':runtime['manifest_sha256']})
                    apbs=next(x for x in field['result']['command']if x.endswith('/bin/apbs'));verify({'path':apbs,'sha256':source['APBS_sha256']})
                    assert field['result']['returncode']==0 and not field['result']['timed_out']and field['result']['threads']==1
                    field_audits.append({'PB_attempt_id':pid,'receipt':rr,'DX_sha256':field['dx_sha256'],'coefficient_maps_rehashed':len(maps),'map_manifest':{'path':str(map_manifest),'sha256':sha(map_manifest)}if map_manifest else None,'boundary_hash_verified':bool(source.get('boundary_map')),'archive_compressed_hash_verified':True,'native_roundtrip_receipt_bound':True,'fresh_decompression_in_this_audit':False,'actual_APBS_threads':1,'normalized_PQR_hash_verified':True,'bindings_json_matches_source':True,'ledger_canonical_source_hash_matches':True,'all_four_raw_logs_present':True,'execution_source_references':[reference(cwd/'bindings.json'),reference(cwd/'molecule.pqr')],'raw_log_references':[reference(path)for path in raw_logs]})
                ids.append(pid);seen.add(pid)
            sampling.append(sampling_audit(root,sid,out['variant'],out,ins))
            variants.append({'variant':out['variant'],'surface_points':len(phi),'PB_attempt_ids':ids,'source_export_ledger_integrity':True,'recomputed_raw_area_p_max_abs_difference':float(np.max(abs(rp-p))),'bin_sha256':ins['shared_bins']['sha256'],'protocol_sha256':ins['protocol_configuration']['sha256']})
    qual=load(root/'results'/(sid+'_qualification.json'))
    return {'schema':'AF_surface_label_integrity_audit_v2','sample_id':sid,'source_export_ledger_integrity_pass':True,'variants':variants,'sampling_integrity':{'pass':all(x['pass']for x in sampling),'variants':sampling},'unique_completed_attempts':len(seen),'DX_rehashed_in_this_audit':rehash_dx,'all_native_field_maps_and_archive_hashes_verified':rehash_dx and len(field_audits)==len(seen),'field_audits':field_audits,'heavy_unique_files_actually_hashed':len(hashes),'source_qualification':{'path':str(root/'results'/(sid+'_qualification.json')),'sha256':sha(root/'results'/(sid+'_qualification.json'))},'prepared_case':{'path':str(d/'prepared_case.json'),'sha256':sha(d/'prepared_case.json')},'strict_per_instance_numerical_qualified':qual['numerical_qualified'],'observed_comparison_axes':[r['axis']for r in qual['comparisons']],'pending_comparison_axes':sorted(set(qual['expected_axes'])-{r['axis']for r in qual['comparisons']}),'method_inheritance_not_granted_by_this_audit':True,'auditor_sha256':sha(__file__),'epoch':time.time()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--rehash-dx',action='store_true');a=p.parse_args();r=audit(a.root,a.case,a.rehash_dx);r['audit_code']={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)};out=a.root/'results'/(a.case+'_integrity_audit.json');out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
