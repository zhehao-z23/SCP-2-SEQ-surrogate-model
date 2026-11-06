"""Create an immutable development-label receipt only after two complete audits.

Historical outcomes retain their original qualification flags. This successor
receipt is the sole acceptance entry; file existence never substitutes for gates.
No physical calculation or training is performed by this program.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,sqlite3
import numpy as np

AXES={'spacing','further_refinement','phase','rotation','surface_resolution','expanded_domain','coarse_parent','overlap'}
LIMITS={'Hellinger':.02,'W1_kBT_per_e':.1,'CDF_sup':.01}
VARIANTS={'base','refined','further','shifted','rotation','surface16','expanded','parent_refined','overlap12'}
PAIRS={'spacing':('base','refined'),'further_refinement':('refined','further'),'phase':('base','shifted'),'rotation':('base','rotation'),'surface_resolution':('base','surface16'),'expanded_domain':('base','expanded'),'coarse_parent':('base','parent_refined'),'overlap':('refined','overlap12')}

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def verify(item):
    assert sha(item['path'])==item['sha256'],item['path']
    return Path(item['path'])
def same_ref(a,b):return Path(a['path'])==Path(b['path'])and a['sha256']==b['sha256']
def ref_key(item):return(str(Path(item['path'])),item['sha256'])
def check_axes(rows):
    assert len(rows)==8 and {x['axis']for x in rows}==AXES
    for row in rows:
        assert row.get('pass')is True
        assert(row['test'],row['reference'])==PAIRS[row['axis']]
        for key,limit in LIMITS.items():
            value=row['metrics'][key];assert np.isfinite(value)and 0<=value<=limit
def immutable(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as f:json.dump(data,f,indent=2,ensure_ascii=False);f.write('\n')

def accept(root,sid,independent_path):
    if not __debug__:raise RuntimeError('Acceptance must run with validation assertions enabled.')
    root=root.resolve();case=root/'cases'/sid
    destination=root/'accepted_development_labels'/sid/'qualification_receipt.json'
    assert not destination.exists(),'An accepted receipt is immutable; do not silently replace it.'
    pilots=load(root/'inputs/pilots.json');source=next(x for x in pilots['cases']if x['sample_id']==sid)
    assert 'development'in source['role']and set(pilots['required_axes'])==AXES
    qualification=root/'results'/(sid+'_qualification.json');qual=load(qualification)
    assert qual['numerical_qualified']is True and set(qual['expected_axes'])==AXES
    check_axes(qual['comparisons']);qref=ref(qualification)

    integrity_path=root/'results'/(sid+'_integrity_audit.json');integrity=load(integrity_path)
    assert integrity['schema']=='AF_surface_label_integrity_audit_v2'
    assert integrity['source_export_ledger_integrity_pass']is True
    assert integrity['all_native_field_maps_and_archive_hashes_verified']is True and integrity['DX_rehashed_in_this_audit']is True
    assert integrity['strict_per_instance_numerical_qualified']is True and not integrity['pending_comparison_axes']
    assert set(integrity['observed_comparison_axes'])==AXES
    assert len(integrity['variants'])==9 and {x['variant']for x in integrity['variants']}==VARIANTS
    sampling=integrity['sampling_integrity'];assert sampling['pass']is True and len(sampling['variants'])==9 and {x['variant']for x in sampling['variants']}==VARIANTS
    sampling_by_variant={x['variant']:x for x in sampling['variants']};integrity_by_variant={x['variant']:x for x in integrity['variants']}
    for row in sampling['variants']:
        assert row['pass']is True and row['all_queries_retained']is True and row['all_eight_corners_present']is True and row['raw_and_corrected_finite']is True and row['false_correction_retains_raw_exactly']is True
        assert row['ownership_min']==row['ownership_max']==1
        assert all(row[key]is True for key in ['weights_nonnegative_positive_total','accepted_boolean','part_global_arrays_exact'])
        assert row['queries']>0 and row['parts']and sum(part['points']for part in row['parts'])==row['queries']
        assert row['queries']==integrity_by_variant[row['variant']]['surface_points']and len({p['rank']for p in row['parts']})==len(row['parts'])
        for part in row['parts']:
            assert part['points']>0 and part['all_eight_corners_present']is True and set(part['references'])=={'query_indices','phi','raw','accepted','field'}
            for item in part['references'].values():verify(item)
        assert row['sampling_code']
        for item in row['sampling_code']:verify(item)
    assert same_ref(integrity['source_qualification'],qref);verify(integrity['audit_code']);verify(integrity['prepared_case'])
    assert integrity['prepared_case']['path']==str(case/'prepared_case.json')

    independent=load(independent_path)
    assert independent['schema']=='independent_AF_qualification_audit_v2'
    assert independent['status']=='complete'and independent['pass']is True
    assert set(independent['required_axes'])==AXES and same_ref(independent['source_qualification'],qref)
    assert independent['case']==qual['sample_id']==integrity['sample_id']==sid
    check_axes(independent['axes'])
    registered={row['axis']:row for row in qual['comparisons']}
    for row in independent['axes']:
        assert all(abs(row['metrics'][key]-registered[row['axis']]['metrics'][key])<=1e-11 for key in LIMITS)
    assert independent['audit_code']and independent['required_references']
    for item in independent['audit_code']:verify(item)
    for item in independent['required_references']:
        assert item['verified']is True;verify(item)
    required_refs={ref_key(ref(path))for path in [case/'prepared_case.json',qualification,case/'further/nested_plan.json',root/'records/geometry_v3'/sid/'all_variant_admissions.json']}
    expected_fields={};variant_inputs={}
    for variant in VARIANTS:
        op=case/variant/'outcome.json';out=load(op);required_refs.add(ref_key(ref(op)))
        assert out['status']=='completed'and out['sample_id']==sid and out['variant']==variant
        mp=verify(out['representation']);required_refs.add(ref_key(out['representation']));m=load(mp)
        assert m['primary_array']=='corrected_area_p';variant_inputs[variant]=m['spec']['inputs']
        sampling_row=sampling_by_variant[variant]
        assert same_ref(sampling_row['source_outcome'],ref(op))and same_ref(sampling_row['representation'],out['representation'])
        assert set(sampling_row['source_inputs'])==set(m['spec']['inputs'])
        for key,item in m['spec']['inputs'].items():assert same_ref(sampling_row['source_inputs'][key],item)
        field_keys={ref_key(item)for item in m['spec']['PB_field_receipts']}
        for part in sampling_row['parts']:
            item=part['references']['field'];assert ref_key(item)in field_keys
            expected_key=sid+('_further_nested_fine_'if variant=='further'else '_'+variant+'_fine_')+str(part['rank'])
            assert Path(item['path']).stem==expected_key
        assert sampling_row['recorded_fallback_area_fraction']==out['fallback_area_fraction']and abs(sampling_row['fallback_area_fraction']-out['fallback_area_fraction'])<=1e-13 and abs(sampling_row['fallback_fraction_difference'])<=1e-13
        required_refs.update(ref_key(item)for item in m['spec']['inputs'].values())
        for field in m['spec']['PB_field_receipts']:
            if field['path']in expected_fields:assert same_ref(field,expected_fields[field['path']])
            expected_fields[field['path']]=field
    required_refs.update(ref_key(item)for item in expected_fields.values())
    assert required_refs<={ref_key(item)for item in independent['required_references']},'Independent audit does not cover the actual nine-variant evidence closure.'
    for row in independent['axes']:
        for side in ['test','reference']:
            evidence=row['inputs'][side];variant=row[side]
            assert evidence['variant']==variant
            for key,item in variant_inputs[variant].items():assert same_ref(evidence[key],item)
    nested=independent['nested_geometry']
    assert nested['pass']is True and nested['child_fields']==68
    assert nested['query_ownership_min']==1 and nested['query_ownership_max']==1

    role_binding_path=root/'records/final_acceptance_role_guard_binding.json';role_binding=load(role_binding_path)
    for key in ['request','receipt','guard_code','roles_contract','prepared_case']:verify(role_binding[key])
    assert role_binding['development_only']is True and role_binding['existing_release_role_gate_still_required']is True
    assert same_ref(role_binding['prepared_case'],integrity['prepared_case'])
    role_request=load(role_binding['request']['path']);role_receipt=load(role_binding['receipt']['path'])
    pre=load(case/'prepared_case.json');actual_sha=pre['structure']['prepared_observed_sequence_sha256']
    assert role_request=={'operation':'read_target','stage':'development','sequence_sha256':[actual_sha],'aliases':[sid]}
    assert role_binding['actual_prepared_sequence_sha256']==actual_sha
    assert role_receipt['status']=='passed_bridge_protection_preflight'and role_receipt['development_only']is True
    assert role_receipt['existing_release_role_gate_still_required']is True and role_receipt['final_closure_ready']is False
    assert role_receipt['contract_sha256']==role_binding['registered_contract_pin']==role_binding['roles_contract']['sha256']
    assert role_receipt['request_sha256']==hashlib.sha256(json.dumps(role_request,sort_keys=True).encode()).hexdigest()
    roles_contract=load(role_binding['roles_contract']['path'])
    assert roles_contract['schema']=='bridge_guard_contract.v1'and roles_contract['search_complete']is True and roles_contract['locks']
    for path,expected in roles_contract['locks'].items():verify({'path':path,'sha256':expected})

    geometry_path=root/'records/geometry_v3'/sid/'all_variant_admissions.json';geometry=load(geometry_path)
    assert geometry['eight_variants_bound']is True and geometry['operator_changed']is False
    assert {r['variant']for r in geometry['rows']}==VARIANTS-{'further'}
    for row in geometry['rows']:
        admission=load(verify(row['admission']));assert admission['pass']is True
        for key in ['certificate','revision_contract','coherent_normals','original_orientation','gate_code']:verify(admission[key])
        for item in [*admission['mesh_identity'].values(),*admission['original_qualification_code_bindings'].values()]:verify(item)
    pools=[root/'records/budget/resources.json',root/'records/budget/further_resources.json']
    execution=load(root/'records/further_execution_preflight.json')
    pools.append(Path(execution['heavy_root'])/'records/disk_reservations.json')
    for path in pools:
        if path.exists():assert load(path)=={},'Outstanding resource reservation must be reconciled.'

    with sqlite3.connect('file:'+str(root/'records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)as db:
        db.row_factory=sqlite3.Row;attempts=[dict(row)for row in db.execute('select * from attempts order by id')]
    assert len(attempts)<=120 and all(x['status']not in {'running','reserved'}for x in attempts)
    by_id={x['id']:x for x in attempts}
    assert len({x['PB_attempt_id']for x in integrity['field_audits']})==len(integrity['field_audits'])
    assert len({x['receipt']['path']for x in integrity['field_audits']})==len(integrity['field_audits'])
    assert {ref_key(x['receipt'])for x in integrity['field_audits']}=={ref_key(x)for x in expected_fields.values()}
    for field in integrity['field_audits']:
        native=load(verify(field['receipt']));row=by_id[field['PB_attempt_id']]
        assert row['status']=='completed'and row['result_path']==field['receipt']['path']
        assert all(field[key]is True for key in ['normalized_PQR_hash_verified','bindings_json_matches_source','ledger_canonical_source_hash_matches','all_four_raw_logs_present'])
        cwd=Path(native['result']['cwd'])
        assert {Path(item['path'])for item in field['execution_source_references']}=={cwd/'bindings.json',cwd/'molecule.pqr'}
        assert {Path(item['path'])for item in field['raw_log_references']}=={cwd/name for name in ['apbs.stdout.log','apbs.stderr.log','apbs.internal.log','io.mc']}
        for item in field['execution_source_references']+field['raw_log_references']:verify(item)
    assert len(integrity['field_audits'])==integrity['unique_completed_attempts']

    # Accept the production base-resolution operator that the eight axes check.
    # A finer result is a numerical reference, not a silent change of the target.
    outcome=load(case/'base/outcome.json');manifest_path=verify(outcome['representation']);manifest=load(manifest_path)
    assert outcome['numerical_qualified']is False
    assert manifest['primary_array']=='corrected_area_p'
    inputs=manifest['spec']['inputs']
    for item in inputs.values():verify(item)
    products=[]
    for item in manifest['products']:
        rr={'path':str(manifest_path.parent/item['name']),'sha256':item['sha256']};verify(rr);products.append(rr)
    vertices=np.load(inputs['vertices']['path'],mmap_mode='r')
    assert nested['total_queries']==len(vertices)
    history_path=root/'records/further_resume_history.json';history_ref=None
    if history_path.exists():
        history=load(history_path);verify(history['original_interrupted_stage']);assert history['resumed_stages']
        for item in history.get('interrupted_stages',[]):verify(item)
        for item in history['resumed_stages']:
            stage=load(verify(item));assert stage['status']=='validated_for_resume'
            for bound in stage['snapshots'].values():verify(bound)
            for evidence in stage['cached_sampling_evidence']:verify(evidence['sampling_receipt'])
        history_ref=ref(history_path)
    state={'schema':'accepted_development_ledger_snapshot_v1','attempts':attempts,'cap_including_failed_attempts':120}
    ledger_path=destination.parent/'ledger_snapshot.json';immutable(ledger_path,state)
    receipt={
        'schema':'qualified_AF_development_label_receipt_v1','status':'accepted','numerical_qualified':True,
        'accepted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sample_id':sid,'role':'development_only','source_identity':source,
        'development_role_guard':ref(role_binding_path),'existing_release_role_gate_still_required':True,
        'role_guard_timing':role_binding['check_timing'],
        'target':'Area-weighted electrostatic-potential distribution on this exact AF coordinate object under the bound chemical/SES/NPBE protocol.',
        'not_experimental_absolute_accuracy':True,'not_global_AFDB_or_long_sequence_qualification':True,
        'not_protected_test_or_benchmark':True,'old_outcome_flags_retained':True,
        'accepted_operator_variant':'base','primary_array':'corrected_area_p',
        'sequence_and_structure':ref(case/'prepared_case.json'),
        'label_representation':ref(manifest_path),'products':products,'source_inputs':inputs,
        'required_axes':sorted(AXES),'self_executed_axes':sorted(AXES),'inherited_instead_of_self_executed_axes':[],
        'thresholds':LIMITS,'qualification':qref,
        'integrity_audit':ref(integrity_path),'independent_audit':ref(independent_path),
        'geometry_validation_revision':ref(root/'contracts/geometry_check_revision_v3.json'),
        'geometry_variant_admissions':ref(geometry_path),
        'nested_plan':ref(case/'further/nested_plan.json'),'execution_preflight':ref(root/'records/further_execution_preflight.json'),
        'execution_stage_history':history_ref,
        'ledger_snapshot':ref(ledger_path),'acceptance_code':ref(__file__),
        'consumption_rule':'Verify this receipt and bound hashes; require this accepted successor instead of reading a legacy numerical_qualified flag or relying on an existing vector file.',
    }
    immutable(destination,receipt);return ref(destination)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--independent-audit',type=Path,required=True);a=p.parse_args()
    print(json.dumps(accept(a.root,a.case,a.independent_audit)))
if __name__=='__main__':main()
