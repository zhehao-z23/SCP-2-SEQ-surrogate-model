"""Finalize an independent eight-axis AF audit only after all solves stop.

Reuses seven hash-bound independent comparisons and triangle-area checks;
calculates only the new further-refinement comparison. Independently checks
all sampling joins/corners/fallbacks and the actual nested parent-child grids.
Does not import project scientific functions or recompute PB fields.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import sqlite3
import sys

sys.dont_write_bytecode = True
import numpy as np

PAIRS = {'spacing': ('base', 'refined'), 'further_refinement': ('refined', 'further'),
         'phase': ('base', 'shifted'), 'rotation': ('base', 'rotation'),
         'surface_resolution': ('base', 'surface16'), 'expanded_domain': ('base', 'expanded'),
         'coarse_parent': ('base', 'parent_refined'), 'overlap': ('refined', 'overlap12')}
LIMITS = {'Hellinger': .02, 'W1_kBT_per_e': .1, 'CDF_sup': .01}
VARIANTS = ['base', 'refined', 'shifted', 'rotation', 'surface16', 'expanded', 'parent_refined', 'overlap12', 'further']


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def header(path):
    result = {}; deltas = []
    with Path(path).open() as f:
        for _ in range(100):
            line = f.readline(); words = line.split()
            if 'gridpositions' in words:
                result['dims'] = list(map(int, words[-3:]))
            elif words and words[0] == 'origin':
                result['origin'] = list(map(float, words[1:]))
            elif words and words[0] == 'delta':
                deltas.append(list(map(float, words[1:])))
            if 'data follows' in line:
                break
    delta = np.asarray(deltas, dtype=float)
    assert delta.shape == (3, 3) and np.isfinite(delta).all()
    assert np.array_equal(delta, np.diag(np.diag(delta))) and np.all(np.diag(delta) > 0)
    assert len(result['dims']) == len(result['origin']) == 3 and min(result['dims']) >= 2
    assert np.isfinite(result['origin']).all()
    result['h'] = np.diag(delta).tolist()
    return result


def corners_present(q, grid):
    lower = np.floor((q - np.asarray(grid['origin'])) / np.asarray(grid['h'])).astype(np.int64)
    assert (lower >= 0).all() and (lower + 1 < np.asarray(grid['dims'])).all()


def full_metrics(x, wx, y, wy, edges):
    def bins(phi, w):
        p = np.bincount(np.searchsorted(edges, phi, side='right'), weights=w, minlength=63)
        return p / w.sum()
    px, py = bins(x, wx), bins(y, wy)
    support = np.concatenate([x, y]); mass = np.concatenate([wx / wx.sum(), -wy / wy.sum()])
    order = np.argsort(support, kind='mergesort'); values = support[order]
    cumulative = np.cumsum(mass[order], dtype=np.float64)
    last = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    unique, delta = values[last], cumulative[last]
    metrics = {'Hellinger': float(np.sqrt(.5 * ((np.sqrt(px) - np.sqrt(py)) ** 2).sum())),
               'W1_kBT_per_e': float((np.abs(delta[:-1]) * np.diff(unique)).sum()),
               'CDF_sup': float(np.abs(delta).max())}
    return metrics, px, py


def validate_metrics(metrics):
    # Qualification also retains diagnostic vectors/areas; only these three
    # predeclared metrics are acceptance gates.
    assert set(LIMITS) <= set(metrics)
    for k, limit in LIMITS.items():
        assert np.isfinite(metrics[k]) and 0 <= metrics[k] <= limit


def compare_interface(path):
    values = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'AXES', 'PAIRS', 'LIMITS', 'VARIANTS'}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    assert values == {'AXES': set(PAIRS), 'PAIRS': PAIRS, 'LIMITS': LIMITS, 'VARIANTS': set(VARIANTS)}
    assert "independent_AF_qualification_audit_v2" in Path(path).read_text(encoding='utf-8')
    return {'path': str(Path(path).resolve()), 'sha256': digest(path), 'constants_and_schema_match': True}


def run(root, sid, prior_dir, prior_receipt_sha, acceptor):
    assert __debug__, 'Assertions must remain enabled.'
    root = root.resolve(); case = root / 'cases' / sid
    references = {}; hash_cache = {}

    def reference(path, expected=None):
        path = Path(path).resolve(); key = str(path)
        if key not in hash_cache:
            hash_cache[key] = digest(path)
        actual = hash_cache[key]
        assert expected is None or actual == expected, key
        value = {'path': key, 'sha256': actual, 'verified': True}
        references[key] = value
        return value

    def verify(item):
        value = reference(item['path'], item['sha256'])
        if 'bytes' in item:
            assert Path(item['path']).stat().st_size == item['bytes']
        return value

    # Fail before array work if any physical child/claim is still live.
    with sqlite3.connect('file:' + str(root / 'records/budget/PB_attempts.sqlite') + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        attempts = [dict(x) for x in db.execute('select * from attempts order by id')]
    assert len(attempts) <= 120 and not any(x['status'] in {'running', 'reserved'} for x in attempts)
    ledger = {x['id']: x for x in attempts}
    execution = load(root / 'records/further_execution_preflight.json')
    heavy = Path(execution['heavy_root'])
    for path in [root / 'records/budget/resources.json', root / 'records/budget/further_resources.json', heavy / 'records/disk_reservations.json']:
        assert load(path) == {}, str(path)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes()
            assert not (b'bridge_physical_further_parallel.py' in cmd and str(root).encode() in cmd), 'Further worker/controller is still live.'
        except (OSError, ProcessLookupError):
            pass
    qual_path = root / 'results' / (sid + '_qualification.json')
    qref = reference(qual_path); qual = load(qual_path)
    assert qual['sample_id'] == sid and qual['numerical_qualified'] is True
    assert set(qual['expected_axes']) == set(PAIRS) and len(qual['comparisons']) == 8
    recorded = {x['axis']: x for x in qual['comparisons']}
    assert set(recorded) == set(PAIRS)
    for axis, row in recorded.items():
        assert (row['test'], row['reference']) == PAIRS[axis] and row['pass'] is True
        validate_metrics(row['metrics'])
    interface = compare_interface(acceptor); verify(interface)
    integrity_path = root / 'results' / (sid + '_integrity_audit.json')
    integrity = load(integrity_path); integrity_ref = reference(integrity_path)
    assert integrity['sample_id'] == sid and integrity['source_qualification']['sha256'] == qref['sha256']
    assert Path(integrity['source_qualification']['path']) == qual_path
    assert integrity['source_export_ledger_integrity_pass'] is True
    assert integrity['all_native_field_maps_and_archive_hashes_verified'] is True and integrity['DX_rehashed_in_this_audit'] is True
    assert integrity['strict_per_instance_numerical_qualified'] is True and not integrity['pending_comparison_axes']
    assert set(integrity['observed_comparison_axes']) == set(PAIRS)
    verify(integrity['audit_code'])
    assert integrity['sampling_integrity']['pass'] is True
    assert {x['variant'] for x in integrity['sampling_integrity']['variants']} == set(VARIANTS)

    # A pinned immutable snapshot is the authority for seven already-recomputed axes.
    pr = prior_dir / 'archive_receipt.json'; reference(pr, prior_receipt_sha)
    archive = load(pr); assert archive['all_hashes_equal'] is True and archive['immutable_snapshot'] is True
    for item in archive['artifacts']:
        assert item['local_sha256'] == item['remote_sha256'] and item['hashes_equal'] is True
        reference(prior_dir / item['name'], item['remote_sha256'])
    prior = load(prior_dir / 'bridge_physical_independent_audit.json')
    assert prior['case'] == sid and prior['verified_axis_count'] == 7 and prior['full_eight_axis_qualified'] is False
    axes = {row['axis']: dict(row) for row in prior['axis_results']}
    assert set(axes) == set(PAIRS) - {'further_refinement'}
    previous = {}; area_checks = {}
    for n in ['01', '02', '03']:
        batch = load(prior_dir / ('bridge_physical_independent_audit_batch_' + n + '.json'))
        assert batch['case'] == sid and batch['whole_surface_no_query_subsampling'] is True
        for row in batch['axes']:
            assert row['metrics'] == axes[row['axis']]['metrics']
        for name, value in batch['variants'].items():
            if name in previous:
                for key in ['phi_sha256', 'weights_sha256', 'mesh_identity']:
                    assert previous[name][key] == value[key]
            previous[name] = value
        area_checks.update(batch.get('mesh_audits', {}))
    assert set(previous) == set(VARIANTS) - {'further'} and area_checks
    for name in ['base', 'refined']:
        for key in ['phi_sha256', 'weights_sha256', 'mesh_identity']:
            assert previous[name][key] == prior['prior_inline_spacing_bindings'][name][key]

    manifests = {}; outcomes = {}; field_refs = {}; sampling_rows = []
    source_prepared = reference(case / 'prepared_case.json')
    reference(root / 'records/geometry_v3' / sid / 'all_variant_admissions.json')
    for variant in VARIANTS:
        d = case / variant; out_ref = reference(d / 'outcome.json'); out = load(d / 'outcome.json')
        assert out['status'] == 'completed' and out['sample_id'] == sid and out['variant'] == variant
        verify(out['representation']); manifest = load(out['representation']['path']); ins = manifest['spec']['inputs']
        assert manifest['primary_array'] == 'corrected_area_p' and manifest['potential_clipping_or_rescaling'] is False
        assert manifest['potential_unit'] == 'kBT/e' and manifest['area_unit'] == 'angstrom^2'
        for item in ins.values():
            verify(item)
        for item in manifest['products']:
            reference(Path(out['representation']['path']).parent / item['name'], item['sha256'])
        own_fields = {x['path']: x for x in manifest['spec']['PB_field_receipts']}
        assert len(own_fields) == len(manifest['spec']['PB_field_receipts'])
        assert {(x['path'], x['sha256']) for x in out['fields']} == {(x['path'], x['sha256']) for x in own_fields.values()}
        pqr_sha = reference(load(ins['source_case_record']['path'])['pqr_path'])['sha256']
        physical = load(ins['protocol_configuration']['path'])['PB_physical_parameters']
        for item in own_fields.values():
            verify(item); field = load(item['path']); entry = ledger[field['PB_attempt_id']]
            assert field['status'] == entry['status'] == 'completed' and Path(entry['result_path']) == Path(item['path'])
            assert field['source']['source_PQR_sha256'] == pqr_sha and field['source']['physical'] == physical
            assert field['logs']['logs_acceptable'] is True
            reference(Path(field['result']['cwd']) / 'apbs.in', field['source']['input_sha256'])
            if item['path'] in field_refs:
                assert field_refs[item['path']]['sha256'] == item['sha256']
            field_refs[item['path']] = item
        if variant != 'further':
            old = previous[variant]
            assert out_ref['sha256'] == old['outcome_sha256']
            assert out['representation']['sha256'] == old['representation_sha256']
            assert ins['corrected_phi']['sha256'] == old['phi_sha256'] and ins['area_weights']['sha256'] == old['weights_sha256']
            assert ins['vertices']['sha256'] + ':' + ins['faces']['sha256'] == old['mesh_identity']
        manifests[variant] = manifest; outcomes[variant] = out

        # Independent sampling joins for every variant. No project sampler import.
        phi = np.load(ins['corrected_phi']['path'], mmap_mode='r'); raw = np.load(ins['raw_trilinear_phi']['path'], mmap_mode='r')
        w = np.load(ins['area_weights']['path'], mmap_mode='r'); vertices = np.load(ins['vertices']['path'], mmap_mode='r')
        assert phi.shape == raw.shape == w.shape == (len(vertices),)
        assert vertices.shape == (len(phi), 3) and np.isfinite(vertices).all()
        assert np.isfinite(phi).all() and np.isfinite(raw).all() and np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0
        if variant == 'further':
            rows = out['field_sampling_receipts']
        else:
            gp = d / 'geometry_plan.json'; reference(gp); rows = []
            for part in load(gp)['parts']:
                rank = part['rank']; pd = d / ('part_' + str(rank))
                rows.append({'rank': rank, 'query_indices': reference(pd / 'query_indices.npy'),
                             'phi': reference(pd / 'phi_jump.npy'), 'raw': reference(pd / 'phi_raw.npy'),
                             'accepted': reference(pd / 'query_accepted.npy'),
                             'field': reference(root / 'records/SES_map_fields' / (sid + '_' + variant + '_fine_' + str(rank) + '.json'))})
        assert rows and len({x['rank'] for x in rows}) == len(rows)
        ownership = np.zeros(len(phi), dtype=np.int32); accepted = np.zeros(len(phi), dtype=bool); parts = []
        for row in rows:
            for key in ['query_indices', 'phi', 'raw', 'accepted', 'field']:
                verify(row[key])
            expected_key = sid + ('_further_nested_fine_' if variant == 'further' else '_' + variant + '_fine_') + str(row['rank'])
            assert Path(row['field']['path']).stem == expected_key
            assert row['field']['path'] in own_fields and row['field']['sha256'] == own_fields[row['field']['path']]['sha256']
            idx = np.load(row['query_indices']['path']); px = np.load(row['phi']['path']); rx = np.load(row['raw']['path']); ok = np.load(row['accepted']['path'])
            assert idx.ndim == 1 and len(idx) and np.issubdtype(idx.dtype, np.integer)
            assert idx.min() >= 0 and idx.max() < len(phi) and len(np.unique(idx)) == len(idx)
            assert px.shape == rx.shape == ok.shape == idx.shape and ok.dtype == np.bool_
            assert np.isfinite(px).all() and np.isfinite(rx).all()
            assert np.array_equal(px, phi[idx]) and np.array_equal(rx, raw[idx]) and np.array_equal(px[~ok], rx[~ok])
            np.add.at(ownership, idx, 1); accepted[idx] = ok
            field = load(row['field']['path']); actual = header(field['dx_path'])
            assert actual['dims'] == field['dims']
            assert np.allclose(actual['origin'], field['origin'], rtol=0, atol=5e-5)
            assert np.allclose(actual['h'], field['h'], rtol=0, atol=5e-6)
            corners_present(vertices[idx], actual)
            parts.append({'rank': row['rank'], 'points': len(idx), 'actual_grid': actual,
                          'all_eight_corners_present': True,
                          'references': {key: row[key] for key in ['query_indices', 'phi', 'raw', 'accepted', 'field']}})
        assert np.all(ownership == 1)
        fallback = float(w[~accepted].sum() / w.sum())
        assert abs(fallback - out['fallback_area_fraction']) <= 1e-13
        sampling_rows.append({'variant': variant, 'pass': True, 'queries': len(phi), 'ownership_min': int(ownership.min()),
                              'ownership_max': int(ownership.max()), 'all_queries_retained': True, 'all_eight_corners_present': True,
                              'raw_and_corrected_finite': True, 'weights_nonnegative_positive_total': True,
                              'accepted_boolean': True, 'part_global_arrays_exact': True, 'false_correction_retains_raw_exactly': True,
                              'fallback_area_fraction': fallback, 'recorded_fallback_area_fraction': out['fallback_area_fraction'],
                              'no_invented_fallback_fraction_threshold': True, 'parts': parts})

    expected_fields = {(x['path'], x['sha256']) for x in field_refs.values()}
    assert expected_fields == {(x['receipt']['path'], x['receipt']['sha256']) for x in integrity['field_audits']}
    assert len(field_refs) == integrity['unique_completed_attempts'] == len(integrity['field_audits'])
    assert len({load(x['path'])['PB_attempt_id'] for x in field_refs.values()}) == len(field_refs)
    # Prior mesh/area checks are reused by exact identity; no triangle recomputation.
    ins = manifests['further']['spec']['inputs']; base = manifests['base']['spec']['inputs']
    for key in ['vertices', 'faces', 'area_weights', 'shared_bins', 'protocol_configuration']:
        assert ins[key]['sha256'] == base[key]['sha256']
    for variant in VARIANTS:
        vi = manifests[variant]['spec']['inputs']
        mesh_key = vi['vertices']['sha256'] + ':' + vi['faces']['sha256']
        assert mesh_key in area_checks and area_checks[mesh_key]['weights_sha256'] == vi['area_weights']['sha256']

    # Actual nested h.125 -> h.0625 containment and original parent ownership.
    plan_path = case / 'further/nested_plan.json'; plan_ref = reference(plan_path); plan = load(plan_path)
    assert plan['case'] == sid and plan['parent_variant'] == 'overlap12' and plan['parent_h_A'] == .125 and plan['fine_h_A'] == .0625
    assert plan['fine_halo_A'] == 8 and len(plan['rows']) == 68 and plan['new_PB_attempts'] == 68
    verify(plan['same_PQR']); verify(plan['mesh_manifest'])
    assert {row['rank'] for row in plan['rows']} == set(range(68))
    verify(outcomes['further']['nested_reference_plan']); assert outcomes['further']['nested_reference_plan']['sha256'] == plan_ref['sha256']
    verify(outcomes['further']['execution_preflight'])
    assert Path(outcomes['further']['execution_preflight']['path']) == root / 'records/further_execution_preflight.json'
    assert execution['geometry_plan']['sha256'] == plan_ref['sha256']
    records = outcomes['further']['field_sampling_receipts']; assert len(records) == 68
    by_rank = {x['rank']: x for x in records}; assert set(by_rank) == set(range(68))
    actual_parts = {x['rank']: x for x in sampling_rows[-1]['parts']}
    vertices = np.load(ins['vertices']['path'], mmap_mode='r'); parent_owner = np.full(len(vertices), -1, dtype=np.int32)
    for part in next(x for x in sampling_rows if x['variant'] == 'overlap12')['parts']:
        idx = np.load(part['references']['query_indices']['path']); assert np.all(parent_owner[idx] == -1); parent_owner[idx] = part['rank']
    assert (parent_owner >= 0).all() and len(vertices) == plan['full_query_count']
    nested_rows = []; parents = set()
    for row in plan['rows']:
        rank = row['rank']; sr = by_rank[rank]; child = load(sr['field']['path']); parent = load(sr['parent_field']['path']); verify(sr['parent_field'])
        assert Path(sr['parent_field']['path']).stem == row['parent_source_key']
        assert row['parent_source_key'] == sid + '_overlap12_fine_' + str(row['parent_rank'])
        p_entry = ledger[parent['PB_attempt_id']]
        assert p_entry['status'] == parent['status'] == 'completed' and Path(p_entry['result_path']) == Path(sr['parent_field']['path'])
        assert child['source']['source_PQR_sha256'] == parent['source']['source_PQR_sha256'] == plan['same_PQR']['sha256']
        assert child['source']['physical'] == parent['source']['physical']
        assert child['source']['boundary_map'] == parent['dx_path'] and child['source']['boundary_map_sha256'] == parent['dx_sha256']
        pg = header(parent['dx_path']); cg = actual_parts[rank]['actual_grid']
        assert pg['dims'] == parent['dims'] and cg['dims'] == row['local_dime']
        assert np.allclose(pg['h'], .125, rtol=0, atol=5e-6) and np.allclose(cg['h'], .0625, rtol=0, atol=5e-6)
        pp = row['parent_query_plan']; expected_origin = np.asarray(pp['global_origin']) + np.asarray(pp['compute_start']) * .125
        assert pg['dims'] == pp['local_dime'] and np.allclose(pg['origin'], expected_origin, rtol=0, atol=5e-5)
        assert np.allclose(pg['origin'], parent['origin'], rtol=0, atol=5e-5)
        assert np.allclose(cg['origin'], row['origin'], rtol=0, atol=5e-5)
        po, co = np.asarray(pg['origin']), np.asarray(cg['origin'])
        pu = po + (np.asarray(pg['dims']) - 1) * np.asarray(pg['h'])
        cu = co + (np.asarray(cg['dims']) - 1) * np.asarray(cg['h'])
        assert np.allclose(cu - co, row['length'], rtol=0, atol=5e-5)
        margin = np.minimum(co - po, pu - cu); assert (margin >= .125 - 1e-5).all()
        assert np.allclose(margin, row['parent_minimum_margin_A'], rtol=0, atol=1e-4)
        assert np.allclose(margin, sr['actual_parent_margin_A'], rtol=0, atol=1e-4)
        idx = np.load(sr['query_indices']['path']); assert np.all(parent_owner[idx] == row['parent_rank'])
        query_margin = float(np.minimum(vertices[idx] - co, cu - vertices[idx]).min()); assert query_margin >= 8 - 5e-5
        sampling_path = Path(sr['phi']['path']).parent / 'sampling_receipt.json'; reference(sampling_path)
        assert load(sampling_path) == sr
        parents.add(sr['parent_field']['path'])
        nested_rows.append({'rank': rank, 'parent_rank': row['parent_rank'], 'child_field': sr['field'], 'parent_field': sr['parent_field'],
                            'actual_parent_grid': pg, 'actual_child_grid': cg, 'minimum_parent_margin_A': float(margin.min()),
                            'minimum_query_margin_A': query_margin, 'parent_query_ownership_verified': True, 'sampling_receipt': reference(sampling_path)})
    assert len(parents) == 7

    # Only this final axis is newly computed; the previous seven remain unchanged.
    a = manifests['refined']['spec']['inputs']; b = manifests['further']['spec']['inputs']
    edges = np.asarray(load(b['shared_bins']['path'])['internal_edges']); assert edges.shape == (62,) and (np.diff(edges) > 0).all()
    metrics, px, py = full_metrics(np.load(a['corrected_phi']['path']), np.load(a['area_weights']['path']),
                                   np.load(b['corrected_phi']['path']), np.load(b['area_weights']['path']), edges)
    validate_metrics(metrics)
    for variant, p in [('refined', px), ('further', py)]:
        vectors = Path(outcomes[variant]['representation']['path']).parent / 'vectors.npz'
        assert np.allclose(p, np.load(vectors)['corrected_area_p'], atol=1e-13, rtol=0)
    axes['further_refinement'] = {'axis': 'further_refinement', 'test': 'refined', 'reference': 'further', 'metrics': metrics, 'pass': True}
    for axis, row in axes.items():
        assert (row['test'], row['reference']) == PAIRS[axis] and row['pass'] is True
        validate_metrics(row['metrics'])
        assert all(abs(row['metrics'][k] - recorded[axis]['metrics'][k]) <= 1e-11 for k in LIMITS)
        row['inputs'] = {side: dict(variant=row[side], **manifests[row[side]]['spec']['inputs']) for side in ['test', 'reference']}
        row['calculation_reused_without_recomputing'] = axis != 'further_refinement'

    history_path = root / 'records/further_resume_history.json'; history = load(history_path); reference(history_path)
    verify(history['original_interrupted_stage']); assert history['resumed_stages']
    interrupted = history.get('interrupted_stages', [history['original_interrupted_stage']])
    assert interrupted and len({(x['path'], x['sha256']) for x in interrupted}) == len(interrupted)
    assert (history['original_interrupted_stage']['path'], history['original_interrupted_stage']['sha256']) in {
        (x['path'], x['sha256']) for x in interrupted}
    interrupted_checks = []
    for item in interrupted:
        verify(item); archived = load(item['path'])
        assert archived['verified_no_branch_cwd_processes'] is True
        assert archived['resource_and_disk_pools_empty'] is True
        assert archived['failure_before_new_PB_reservation'] is True
        sources = archived['source_refs']; assert isinstance(sources, list) and sources
        for source in sources:
            # Preserve both immutable historical manifest formats. Original
            # source paths may legitimately change; verify the archived copy.
            if 'snapshot' in source:
                verify(source['snapshot'])
            else:
                assert {'source_path', 'snapshot_path', 'sha256'} <= set(source)
                verify({'path': source['snapshot_path'], 'sha256': source['sha256']})
        verify(archived['ledger_snapshot'])
        originals = archived.get('original_sampling_receipts', [])
        for sample in originals:
            verify(sample); saved = load(sample['path']); assert saved == by_rank[saved['rank']]
        interrupted_checks.append({'manifest': item, 'archived_source_references_verified': len(sources),
                                   'ledger_snapshot': archived['ledger_snapshot'],
                                   'original_sampling_receipts_verified': len(originals),
                                   'completed_child_fields_at_interruption': archived['completed_child_fields']})
    cached_receipts = {}; stage_counts = []
    for item in history['resumed_stages']:
        verify(item); stage = load(item['path']); assert stage['status'] == 'validated_for_resume'
        assert stage['numerical_one_field_function_unchanged'] is True
        for snapshot in stage['snapshots'].values():
            verify(snapshot)
        assert stage['completed_sampling_reused_without_rewrite'] + stage['remaining_new_fields'] == 68
        assert len(stage['cached_sampling_evidence']) == stage['completed_sampling_reused_without_rewrite']
        stage_counts.append({'receipt': item, 'reused_sampling_fields': stage['completed_sampling_reused_without_rewrite'],
                             'remaining_fields_at_resume': stage['remaining_new_fields']})
        for evidence in stage['cached_sampling_evidence']:
            verify(evidence['sampling_receipt']); sr = load(evidence['sampling_receipt']['path'])
            assert sr == by_rank[evidence['rank']]
            verify(evidence['native_field']); verify(evidence['parent_field'])
            key = evidence['sampling_receipt']['path']
            if key in cached_receipts:
                assert cached_receipts[key]['sha256'] == evidence['sampling_receipt']['sha256']
            cached_receipts[key] = evidence['sampling_receipt']
    # Rehash mutable final gate inputs at the end; a concurrent rewrite fails.
    assert digest(qual_path) == qref['sha256'] and digest(integrity_path) == integrity_ref['sha256']
    audit_code = [reference(__file__), reference(prior_dir / 'bridge_physical_independent_audit.py'),
                  reference(prior_dir / 'bridge_physical_independent_local_overlap.py')]
    return {'schema': 'independent_AF_qualification_audit_v2', 'case': sid, 'status': 'complete', 'pass': True,
            'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_qualification': qref,
            'required_axes': list(PAIRS), 'axes': [axes[x] for x in PAIRS], 'audit_code': audit_code,
            'required_references': list(references.values()), 'prepared_case': source_prepared,
            'prior_seven_axis_archive': reference(pr), 'seven_comparisons_and_triangle_areas_recomputed': False,
            'prior_mesh_area_audits_reused_by_exact_hash': area_checks,
            'sampling_integrity': {'pass': True, 'variants': sampling_rows},
            'nested_geometry': {'pass': True, 'child_fields': len(nested_rows), 'parent_fields': len(parents),
                                'total_queries': len(vertices), 'query_ownership_min': 1, 'query_ownership_max': 1,
                                'minimum_actual_parent_containment_A': min(x['minimum_parent_margin_A'] for x in nested_rows),
                                'minimum_actual_query_halo_A': min(x['minimum_query_margin_A'] for x in nested_rows),
                                'all_child_queries_have_original_parent_owner': True, 'rows': nested_rows},
            'execution_stage_history': reference(history_path), 'execution_stage_count': 1 + len(history['resumed_stages']),
            'resumed_stages': stage_counts, 'interrupted_stages': interrupted_checks,
            'preserved_cached_sampling_receipts': list(cached_receipts.values()),
            'main_integrity_audit': integrity_ref, 'acceptance_interface': interface,
            'full_DX_maps_rehashed_by_this_independent_audit': False,
            'full_DX_maps_rehash_required_and_bound_main_audit': True,
            'archive_native_roundtrip_receipts_bound_not_freshly_decompressed_here': True,
            'PB_calls': 0, 'new_model_metrics': 0, 'potential_clipping_or_rescaling': False,
            'scope': 'Numerical qualification of this exact AF coordinate object under the bound protocol; not experimental structural accuracy, all AFDB, full sequence space, or a protected benchmark.'}


def self_test(acceptor):
    edges = np.linspace(-30, 31, 62)
    m, _, _ = full_metrics(np.array([0., 1.]), np.array([1., 1.]), np.array([0., 1.]), np.array([1., 1.]), edges)
    assert max(m.values()) == 0
    m, _, _ = full_metrics(np.array([0., 0.]), np.array([1., 1.]), np.array([2.]), np.array([1.]), edges)
    assert abs(m['W1_kBT_per_e'] - 2) < 1e-15 and abs(m['CDF_sup'] - 1) < 1e-15
    grid = {'origin': [0, 0, 0], 'h': [1, 1, 1], 'dims': [3, 3, 3]}
    corners_present(np.array([[.5, .5, .5], [1.999, 1., 1.]]), grid)
    refused = 0
    for q in [np.array([[2., 1., 1.]]), np.array([[-.001, 1., 1.]])]:
        try:
            corners_present(q, grid)
        except AssertionError:
            refused += 1
    for bad in [float('nan'), float('inf'), -.1, .0200001]:
        try:
            validate_metrics({'Hellinger': bad, 'W1_kBT_per_e': 0, 'CDF_sup': 0})
        except AssertionError:
            refused += 1
    assert refused == 6
    return {'status': 'local_synthetic_checks_passed', 'negative_checks_rejected': refused,
            'no_real_case_arrays_read': True, 'no_final_audit_created': True,
            'interface': compare_interface(acceptor), 'code_sha256': digest(__file__)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--self-test', action='store_true')
    p.add_argument('--acceptor', type=Path, required=True); p.add_argument('--root', type=Path)
    p.add_argument('--case'); p.add_argument('--prior-dir', type=Path); p.add_argument('--prior-receipt-sha')
    p.add_argument('--output', type=Path); a = p.parse_args()
    if a.self_test:
        print(json.dumps(self_test(a.acceptor), indent=2)); return
    assert all([a.root, a.case, a.prior_dir, a.prior_receipt_sha, a.output])
    assert not a.output.exists(), 'Final receipt is immutable; never overwrite.'
    result = run(a.root, a.case, a.prior_dir, a.prior_receipt_sha, a.acceptor)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'pass': result['pass'], 'path': str(a.output), 'sha256': digest(a.output),
                      'axes': len(result['axes']), 'nested_children': result['nested_geometry']['child_fields']}))


if __name__ == '__main__':
    main()
