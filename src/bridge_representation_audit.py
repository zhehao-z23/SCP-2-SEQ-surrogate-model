"""Development-only audit of historical 63-count quantile representation.

No model, phi matrix, new fit, mode classifier or source-percentile screening.
Private row-level outputs stay beside the Byte receipt.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

RELEASE_PIN = '67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b'
CONTRACT_PIN = '05932133251f6c20956d9dbf77af4d857a78137bf125b3485c1e97098ba61c9c'
GUARD_PIN = '71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c'
GUARD_CODE_PIN = '5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'
BOUNDS_CODE_PIN = '28c30fc2cbb34cb388bf2495b58a8d567521b5ed323f42845dc327dabedbd04e'
ABS_TOL = 1e-10
EDGE_SCALE_REL_TOL = 1e-12
BOWLEY_TOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def put(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def sizes(meta, mask=None):
    z = meta if mask is None else meta.loc[np.asarray(mask, dtype=bool)]
    return {'rows': len(z), 'unique_T': int(z.sequence_sha256.nunique()), 'groups': int(z.group_id.nunique())}


def describe(values):
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    return {'finite_n': len(z), 'min': float(z.min()) if len(z) else None,
            'median': float(np.median(z)) if len(z) else None,
            'mean': float(z.mean()) if len(z) else None,
            'p90': float(np.quantile(z, .90)) if len(z) else None,
            'p95': float(np.quantile(z, .95)) if len(z) else None,
            'max': float(z.max()) if len(z) else None}


def bowley_box_bounds(lower, upper, padding):
    """Conservative bounds, with certification only for finite IQR >= 1 boxes.

    a=Q25,c=Q50,b=Q75. For ordered a<=c<=b, S increases in a,b
    and decreases in c. If bL-aU>=1, all denominators are positive.
    Extreme feasible c is min(cU,bL) for the minimum, max(cL,aU)
    for the maximum. Clipping the corresponding corner value to [-1,1]
    implements this constraint. Other quantile dependence is relaxed.
    """
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    assert lower.shape == upper.shape and lower.ndim == 2 and lower.shape[1] == 3
    assert (lower <= upper).all()
    assert (lower[:, 0] <= lower[:, 1]).all() and (lower[:, 1] <= lower[:, 2]).all()
    assert (upper[:, 0] <= upper[:, 1]).all() and (upper[:, 1] <= upper[:, 2]).all()
    lo = np.nextafter(lower-padding, -np.inf)
    hi = np.nextafter(upper+padding, np.inf)
    finite = np.isfinite(lo).all(1) & np.isfinite(hi).all(1)
    iqr_lo = np.zeros(len(lo))
    iqr_hi = np.full(len(lo), np.inf)
    iqr_lo[finite] = np.maximum(0., np.nextafter(lo[finite, 2]-hi[finite, 0], -np.inf))
    iqr_hi[finite] = np.nextafter(hi[finite, 2]-lo[finite, 0], np.inf)
    eligible = finite & (iqr_lo >= 1.)
    sl = np.full(len(lo), np.nan)
    su = np.full(len(lo), np.nan)
    if eligible.any():
        l, h = lo[eligible], hi[eligible]
        sl[eligible] = np.maximum(-1., (l[:, 0]+l[:, 2]-2*h[:, 1])/(l[:, 2]-l[:, 0])-BOWLEY_TOL)
        su[eligible] = np.minimum(1., (h[:, 0]+h[:, 2]-2*l[:, 1])/(h[:, 2]-h[:, 0])+BOWLEY_TOL)
        assert (sl[eligible] <= su[eligible]).all()
    return {'finite': finite, 'IQR_lower': iqr_lo, 'IQR_upper': iqr_hi,
            'certificate_eligible': eligible, 'S_lower': sl, 'S_upper': su,
            'certified_negative': eligible & (su <= -.30),
            'certified_positive': eligible & (sl >= .30)}


def math_self_test():
    """Synthetic sorted endpoint checks only; no actual target access."""
    rng = np.random.default_rng(1701)
    boxes = triples = 0
    for _ in range(240):
        ends = np.sort(rng.uniform(-8., 8., 6))
        lower = ends[[0, 1, 4]]
        upper = ends[[2, 3, 5]]
        # Include c touching/crossing the a/b intervals as well as separated c.
        if boxes % 2:
            lower[1] = lower[0]
            upper[1] = upper[2]
        z = bowley_box_bounds(lower[None], upper[None], 1e-10)
        if not z['certificate_eligible'][0]:
            continue
        for a in np.linspace(lower[0], upper[0], 4):
            for b in np.linspace(lower[2], upper[2], 4):
                for c in np.linspace(max(a, lower[1]), min(b, upper[1]), 4):
                    if not a <= c <= b:
                        continue
                    s = (a+b-2*c)/(b-a)
                    # Direct floating evaluation can give -1-1ulp at c=b.
                    assert z['S_lower'][0]-BOWLEY_TOL <= s <= z['S_upper'][0]+BOWLEY_TOL
                    triples += 1
        boxes += 1
    assert boxes > 30 and triples > 1000
    return {'status': 'passed', 'finite_ordered_boxes': boxes, 'sorted_triples': triples,
            'actual_proteins_read': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(math_self_test()))
        return
    import resource
    import pandas as pd
    import pyarrow
    import pyarrow.parquet as pq
    assert __debug__ and a.source_root is not None and a.output is not None
    assert os.sched_getaffinity(0) == {172}
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    start = time.monotonic()
    put(a.output/'started.json', {'pid': os.getpid(), 'unix': time.time(), 'code_sha256': sha(__file__),
                                'CPU_affinity': sorted(os.sched_getaffinity(0)), 'max_RSS_budget_GiB': 8,
                                'timeout_seconds': 600, 'new_fits': 0, 'raw_phi_reads': 0})
    refs = []

    def verified(path, expected):
        actual = sha(path)
        item = {'path': str(path), 'sha256': actual, 'expected_sha256': expected,
                'bytes': path.stat().st_size, 'match': actual == expected}
        refs.append(item)
        if actual != expected:
            raise RuntimeError('source SHA mismatch: '+str(path))
        return item

    try:
        root = a.source_root
        data = root/'atlas_scale_20261105/data'
        roles = root/'bridge_20261106/roles'
        verified(data/'release.json', RELEASE_PIN)
        release = load(data/'release.json')
        verified(data/'samples.parquet', release['artifacts']['samples.parquet'])
        m = pq.read_table(data/'samples.parquet', columns=['sample_id', 'sequence_sha256', 'group_id',
                            'fold', 'length', 'target_row', 'source_structure_sha256']).to_pandas()
        assert m.sample_id.is_unique and sizes(m) == {'rows': 21921, 'unique_T': 21915, 'groups': 7999}
        verified(roles/'code/bridge_roles_guard.py', GUARD_CODE_PIN)
        sys.path.insert(0, str(roles/'code'))
        from bridge_roles_guard import check
        request = {'operation': 'read_target', 'stage': 'development',
                   'sequence_sha256': sorted(m.sequence_sha256.unique().tolist()), 'aliases': []}
        guard = check(roles/'registry_v1/guard_contract.json', GUARD_PIN, request)
        put(a.output/'role_preflight.json', {'request': request, 'receipt': guard})
        # The policy preflight precedes all target/QC/count bytes and numerical arrays.
        verified(root/'data/labels/release.pending.json', CONTRACT_PIN)
        contract = load(root/'data/labels/release.pending.json')
        qc_path = root/'source/labels/source_snapshots/matrix_phi_qc.parquet'
        counts_path = root/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet'
        edges_path = root/'source/labels/source_snapshots/q05_995_bins63_edges.tsv'
        for path in [qc_path, counts_path, edges_path]:
            verified(path, contract['source_hashes']['source_snapshots/'+path.name])
        verified(data/'p.npy', release['artifacts']['p.npy'])
        index_path = root/'data/labels/index.parquet'
        verified(index_path, release['sources'][str(index_path)])
        bounds_file = Path(__file__).with_name('bridge_binned_quantile_bounds.py')
        verified(bounds_file, BOUNDS_CODE_PIN)
        from bridge_binned_quantile_bounds import linear_sample_quantile_bounds
        ids = m.sample_id.tolist()
        filters = [('sample_id', 'in', ids)]

        def aligned(path, columns):
            t = pq.read_table(path, columns=['sample_id']+columns, filters=filters).to_pandas()
            assert t.sample_id.is_unique and set(t.sample_id) == set(ids)
            return t.set_index('sample_id').loc[ids].reset_index()

        qcols = ['q25', 'q50', 'q75']
        qc = aligned(qc_path, qcols+['n_vertices', 'n_rows', 'n_nan_phi', 'n_inf_phi', 'phi_column',
                                     'matrix_columns', 'matrix_path', 'readable'])
        bincols = [f'bin_{i:03d}' for i in range(63)]
        cnt_df = aligned(counts_path, bincols)
        counts = cnt_df[bincols].to_numpy()
        assert np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
        totals = counts.sum(1, dtype=np.int64)
        assert (totals > 0).all() and (totals < 2**53).all()
        assert np.array_equal(totals, qc.n_vertices.to_numpy())
        assert np.array_equal(totals, qc.n_rows.to_numpy())
        assert qc.readable.all() and qc.phi_column.eq('phi').all() and qc.matrix_columns.eq('phi,x,y,z').all()
        assert qc.n_nan_phi.eq(0).all() and qc.n_inf_phi.eq(0).all()
        ix = aligned(index_path, ['row', 'label_integrity_valid', 'n_vertices'])
        assert np.array_equal(ix.row.to_numpy(), m.target_row.to_numpy())
        assert ix.label_integrity_valid.all() and np.array_equal(ix.n_vertices.to_numpy(), totals)
        y = np.load(data/'p.npy', allow_pickle=False)
        assert y.dtype == np.float32 and y.shape == counts.shape and np.isfinite(y).all() and (y >= 0).all()
        expected_p = counts / totals[:, None]
        exact_fp32 = np.array_equal(y, expected_p.astype(np.float32))
        max_p_diff = float(abs(y.astype(float)-expected_p).max())
        assert exact_fp32, 'current p differs from exact count normalization rounded to FP32'
        edges = pd.read_csv(edges_path, sep='\t', float_precision='round_trip').edge.to_numpy()
        assert np.array_equal(edges, np.asarray(contract['edges'])) and len(edges) == 62
        actual = qc[qcols].to_numpy(float)
        assert np.isfinite(actual).all() and (np.diff(actual, axis=1) >= 0).all()
        identity = {'status': 'passed', 'cohort': sizes(m), 'all_target_hashes_freshly_verified': True,
                    'all_metadata_and_count_rows_aligned_by_sample_id': True,
                    'counts_sum_equals_QC_n_vertices_and_full_n_rows': True, 'all_QC_NaN_Inf_counts_zero': True,
                    'all_phi_columns_named_phi': True, 'target_index_exact_match': True,
                    'current_p_bit_equal_count_normalization_rounded_FP32': exact_fp32,
                    'max_abs_current_p_minus_exact_float64_probability': max_p_diff,
                    'original_matrix_bytes_replayed': False, 'refs': refs}
        put(a.output/'identity.json', identity)
        math_check = math_self_test()
        put(a.output/'mathematical_preflight.json', math_check)
        tolerance = ABS_TOL + EDGE_SCALE_REL_TOL*float(abs(edges).max())
        lo = np.empty_like(actual); hi = np.empty_like(actual)
        left_rank = np.empty_like(actual, dtype=np.int64); right_rank = left_rank.copy()
        left_bin = left_rank.copy(); right_bin = left_rank.copy(); fraction = np.empty_like(actual)
        for i, count in enumerate(counts):
            z = linear_sample_quantile_bounds(count, edges, np.array([.25, .5, .75]))
            lo[i], hi[i] = z['lower'], z['upper']
            left_rank[i], right_rank[i] = z['left_rank_zero_based'], z['right_rank_zero_based']
            left_bin[i], right_bin[i], fraction[i] = z['left_bin'], z['right_bin'], z['fraction']
        violation = np.maximum(np.maximum(lo-actual, actual-hi), 0.)
        mismatch = violation > tolerance
        b = bowley_box_bounds(lo, hi, tolerance)
        iqr = actual[:, 2]-actual[:, 0]
        raw_s = np.full(len(m), np.nan)
        valid = iqr > 0
        raw_s[valid] = (actual[valid, 0]+actual[valid, 2]-2*actual[valid, 1])/iqr[valid]
        negative = (iqr >= 1.) & (raw_s <= -.30)
        positive = (iqr >= 1.) & (raw_s >= .30)
        bneg, bpos = b['certified_negative'], b['certified_positive']
        eligible = b['certificate_eligible']
        certificate_checks = {
            'QC_within_Bowley_bounds_on_eligible_rows': bool(((raw_s[eligible] >= b['S_lower'][eligible]-BOWLEY_TOL) & (raw_s[eligible] <= b['S_upper'][eligible]+BOWLEY_TOL)).all()),
            'every_negative_certificate_satisfies_raw_absolute_condition': bool((~bneg | negative).all()),
            'every_positive_certificate_satisfies_raw_absolute_condition': bool((~bpos | positive).all())}
        out = m.copy(); out['n_vertices'] = totals
        out['matrix_path'] = qc.matrix_path
        for j, name in enumerate(qcols):
            for suffix, values in [('raw_QC', actual), ('lower', lo), ('upper', hi), ('excess_outside_bounds', violation),
                                   ('left_rank', left_rank), ('right_rank', right_rank), ('left_bin', left_bin), ('right_bin', right_bin), ('fraction', fraction)]:
                out[name+'_'+suffix] = values[:, j]
            out[name+'_outside_tolerance'] = mismatch[:, j]
            out[name+'_finite_support'] = np.isfinite(lo[:, j]) & np.isfinite(hi[:, j])
        out['raw_IQR'] = iqr; out['raw_Bowley'] = raw_s
        out['raw_negative_absolute_shape'] = negative; out['raw_positive_absolute_shape'] = positive
        for name, values in b.items():
            out['box_'+name] = values
        out.to_parquet(a.output/'per_sample.parquet', index=False)
        quantile_summary = {}
        for j, name in enumerate(qcols):
            finite = np.isfinite(lo[:, j]) & np.isfinite(hi[:, j])
            quantile_summary[name] = {'mismatch_after_tolerance': sizes(m, mismatch[:, j]),
                'strict_positive_roundoff_excess': sizes(m, violation[:, j] > 0),
                'maximum_excess_before_tolerance': float(violation[:, j].max()),
                'finite_support': sizes(m, finite), 'unbounded_support': sizes(m, ~finite),
                'finite_width': describe((hi-lo)[finite, j])}
        shape = {'definition': 'Historical absolute conditions only: raw IQR>=1 and Bowley<=-0.30 or >=+0.30; source-percentile requirement NOT applied.',
                 'denominator': sizes(m), 'raw_IQR_at_least_1': sizes(m, iqr >= 1.),
                 'raw_negative': sizes(m, negative), 'raw_positive': sizes(m, positive),
                 'raw_either': sizes(m, negative | positive), 'certificate_eligible': sizes(m, eligible),
                 'certified_negative': sizes(m, bneg), 'certified_positive': sizes(m, bpos),
                 'raw_negative_not_certified': sizes(m, negative & ~bneg),
                 'raw_positive_not_certified': sizes(m, positive & ~bpos),
                 'not_certificate_eligible': sizes(m, ~eligible), 'checks': certificate_checks}
        passed = not mismatch.any() and all(certificate_checks.values())
        summary = {'schema': 'historical_representation_audit_v1', 'status': 'complete',
                   'outcome': 'all_quantile_and_certificate_consistency_checks_passed' if passed else 'inconsistency_retained_for_review',
                   'identity_status': identity['status'], 'identity_sha256': sha(a.output/'identity.json'),
                   'code_sha256': sha(__file__), 'bounds_code_sha256': BOUNDS_CODE_PIN,
                   'cohort': sizes(m), 'quantiles': quantile_summary, 'any_quantile_mismatch': sizes(m, mismatch.any(1)),
                   'quantile_tolerance': {'formula': '1e-10 + 1e-12*max(abs(internal_edges))', 'value_kBT_per_e': tolerance,
                                          'formula_fixed_before_execution': True},
                   'shape': shape, 'raw_IQR_distribution': describe(iqr),
                   'finite_IQR_upper_distribution': describe(b['IQR_upper']),
                   'finite_certifiable_Bowley_interval_width': describe(b['S_upper']-b['S_lower']),
                   'method': {'quantiles': 'NumPy linear/type7 two-order-statistic interval from exact integer counts; no uniform-bin assumption',
                              'certificate': 'Quantile support expanded by fixed tolerance; finite ordered box with IQR lower>=1; monotonic corner extrema clipped to [-1,1], then outward S tolerance 1e-12',
                              'unit': 'equal retained historical vertices, kBT/e, original U-inclusive last central bin',
                              'unique_T_and_groups': 'Counts of distinct identifiers among matching rows, no averaging of per-sample distributions; one group/T may appear in multiple shape categories'},
                   'limits': ['Retrospective exposed development only; no source-population percentile ranking, hence not full historical extreme screening.',
                              'No model scoring, new fit, mode classification, raw phi read or new physical qualification.',
                              'Uncertified by these sufficient conservative bounds does not prove irrecoverable information loss; quantile dependence can narrow the true feasible set.',
                              'Archived producer uses np.quantile without explicit method; original execution NumPy version/exact producer execution SHA remain unbound.',
                              'Fresh hashes bind current snapshots and p. This is not original-matrix byte replay.'],
                   'private_table': {'path': str(a.output/'per_sample.parquet'), 'sha256': sha(a.output/'per_sample.parquet'), 'rows': len(out)},
                   'mathematical_preflight': math_check, 'elapsed_seconds': time.monotonic()-start,
                   'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                                   'pyarrow': pyarrow.__version__, 'CPU_affinity': sorted(os.sched_getaffinity(0))}}
        put(a.output/'summary.json', summary)
        print(json.dumps({'status': summary['status'], 'outcome': summary['outcome'], 'cohort': summary['cohort'],
                          'any_quantile_mismatch': summary['any_quantile_mismatch'], 'shape': shape,
                          'elapsed_seconds': summary['elapsed_seconds'], 'summary_sha256': sha(a.output/'summary.json')}))
    except Exception as exc:
        put(a.output/'failure.json', {'status': 'failed_preserved', 'type': type(exc).__name__, 'error': str(exc),
                                    'traceback': traceback.format_exc(), 'verified_so_far': refs,
                                    'elapsed_seconds': time.monotonic()-start, 'code_sha256': sha(__file__)})
        raise


if __name__ == '__main__':
    main()
