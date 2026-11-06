"""Conservative bounds for linear sample quantiles from exact bin counts.

Historical equal-vertex counts only; not weighted-area quantiles. Closed bounds
are used conservatively even where a histogram interval is half open. Open tail
positions remain unbounded. No bin centers or uniform-within-bin assumption.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def linear_sample_quantile_bounds(counts, internal_edges, quantiles):
    counts = np.asarray(counts)
    edges = np.asarray(internal_edges, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    assert counts.ndim == 1 and np.issubdtype(counts.dtype, np.integer)
    assert len(counts) == len(edges)+1 and (counts >= 0).all()
    assert edges.ndim == 1 and len(edges) >= 2 and np.isfinite(edges).all() and (np.diff(edges) > 0).all()
    assert q.ndim == 1 and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    n = int(sum(int(x) for x in counts))
    assert 0 < n < 2**53
    cumulative = counts.cumsum(dtype=np.int64)
    index = (n-1)*q
    left_rank = np.floor(index).astype(np.int64)
    right_rank = np.ceil(index).astype(np.int64)
    fraction = index-left_rank
    left_bin = np.searchsorted(cumulative, left_rank+1, side='left')
    right_bin = np.searchsorted(cumulative, right_rank+1, side='left')
    bounds = np.r_[-np.inf, edges, np.inf]
    lower = bounds[left_bin].copy()
    upper = bounds[left_bin+1].copy()
    mixed = fraction > 0
    # Avoid the undefined 0*infinity on exact-rank quantiles.
    lower[mixed] = (1-fraction[mixed])*bounds[left_bin[mixed]] + fraction[mixed]*bounds[right_bin[mixed]]
    upper[mixed] = (1-fraction[mixed])*bounds[left_bin[mixed]+1] + fraction[mixed]*bounds[right_bin[mixed]+1]
    assert not np.isnan(lower).any() and not np.isnan(upper).any() and (lower <= upper).all()
    return {'n': n, 'quantiles': q, 'lower': lower, 'upper': upper,
            'left_rank_zero_based': left_rank, 'right_rank_zero_based': right_rank,
            'fraction': fraction, 'left_bin': left_bin, 'right_bin': right_bin}


def historical_counts(values, internal_edges):
    """Historical central last bin includes U; right overflow is strictly > U."""
    values = np.asarray(values, dtype=float); edges = np.asarray(internal_edges, dtype=float)
    assert values.ndim == 1 and np.isfinite(values).all()
    index = np.searchsorted(edges, values, side='right')
    index[values == edges[-1]] = len(edges)-1
    return np.bincount(index, minlength=len(edges)+1)


def self_test():
    rng = np.random.default_rng(1701)
    edges = np.array([-3., -1., 0., 2., 5.])
    qs = np.array([0., .01, .25, .5, .75, .99, 1.])
    tested = 0
    # Empty middle bins, edge-equality, ties, one element, both open tails and
    # an interpolated quantile crossing multiple empty bins are all included.
    examples = [np.array([0.]), np.array([-10., 10.]), np.array([-2., 4.]),
                np.repeat(edges, 3), np.array([-1., -1., 0., 5., 5.])]
    for _ in range(500):
        values = rng.normal(0, 5, size=int(rng.integers(1, 150)))
        if rng.random() < .3: values = np.round(values)
        examples.append(values)
    for values in examples:
        result = linear_sample_quantile_bounds(historical_counts(values, edges), edges, qs)
        actual = np.quantile(values, qs, method='linear')
        assert (actual >= result['lower']-1e-12).all() and (actual <= result['upper']+1e-12).all()
        tested += len(qs)
    example = np.array([-2., 4.])
    result = linear_sample_quantile_bounds(historical_counts(example, edges), edges, np.array([.5]))
    actual = float(np.quantile(example, .5, method='linear'))
    assert actual == 1. and result['lower'][0] <= actual <= result['upper'][0]
    naive_crossing_bin = int(np.searchsorted(historical_counts(example, edges).cumsum()/2, .5, side='left'))
    assert not edges[naive_crossing_bin-1] <= actual <= edges[naive_crossing_bin]
    return {'status': 'synthetic_checks_passed', 'datasets': len(examples), 'quantiles_checked': tested,
            'interpolation_counterexample': {'values': example.tolist(), 'q50_linear': actual,
                 'two_rank_interval': [float(result['lower'][0]), float(result['upper'][0])],
                 'single_crossing_bin_interval': [-3., -1.], 'single_bin_interval_contains_q50': False},
            'actual_protein_arrays_read': False, 'actual_model_inferences': 0,
            'numpy_version_used_for_synthetic_checks': np.__version__,
            'historical_execution_numpy_version': None,
            'scope': 'Equal-count linear sample quantile bounds; not continuous-density, inverse-CDF, or area-weighted quantiles.'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--self-test', action='store_true'); ap.add_argument('--output', type=Path)
    args = ap.parse_args(); assert args.self_test
    result = self_test(); result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output:
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
