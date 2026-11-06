"""Diagnose rare-shape fitting versus generalization using six saved arrays.

No model loads, new epochs, historical reconstruction, or protected labels.
Two fit-only constant distribution references do not use evaluation targets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text())


def put(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    assert __debug__ and os.sched_getaffinity(0) == {174}
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    import numpy as np
    import pandas as pd
    root = a.root; supp = root/'bridge_20261106/diagnostic_supplements'
    assert not a.output.exists(); a.output.mkdir(parents=True)
    start = time.monotonic(); refs = []

    def bind(p, expected=None):
        actual = sha(p)
        if expected: assert actual == expected, str(p)
        refs.append({'path': str(p), 'sha256': actual})
        return actual

    put(a.output/'started.json', {'code_sha256': sha(__file__), 'PID': os.getpid(),
        'new_fits': 0, 'GPU': False, 'CPU': [174], 'max_seconds': 600})
    helper = supp/'code/bridge_shape_model_comparison.py'
    bind(helper, '4be007ff1b869a4a71b800713f3e833f79b04770f7f27dd66aeb22507da16ce6')
    sys.path.insert(0, str(helper.parent))
    from bridge_shape_model_comparison import uniform_quantiles, probability_errors, macro, confusion, rank_one_percent
    data = root/'atlas_scale_20261105/data'
    bind(data/'release.json', '67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b')
    release = load(data/'release.json'); bind(data/'samples.parquet', release['artifacts']['samples.parquet'])
    meta = pd.read_parquet(data/'samples.parquet')
    roles = root/'bridge_20261106/roles'
    bind(roles/'code/bridge_roles_guard.py', '5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3')
    sys.path.insert(0, str(roles/'code')); from bridge_roles_guard import check
    gp = roles/'registry_v1/guard_contract.json'; gh = '71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c'
    request = {'stage': 'development', 'sequence_sha256': sorted(meta.sequence_sha256.unique().tolist()), 'aliases': []}
    put(a.output/'role_preflight.json', [check(gp, gh, {'operation': op, **request}) for op in ['read_target', 'model_eval']])
    bind(data/'p.npy', release['artifacts']['p.npy']); y = np.load(data/'p.npy')
    binding = load(data/'label_contract_addendum.json'); bind(binding['label_contract_path'], binding['label_contract_sha256'])
    edges = np.asarray(binding['label_contract']['edges'])
    rep = supp/'representation_audit_v1'; bind(rep/'summary.json', '995a2e7b6502226ea39540a752b0574af0cf498d2b90919732447cf6f87ad41f')
    rs = load(rep/'summary.json'); bind(rep/'per_sample.parquet', rs['private_table']['sha256'])
    raw = pd.read_parquet(rep/'per_sample.parquet').set_index('sample_id').loc[meta.sample_id].reset_index()
    assert np.array_equal(raw.sequence_sha256, meta.sequence_sha256)
    best = supp/'best_states_v1'; bind(best/'summary.json', '292f35d956933b8ff0706a37b2ecf8ad12ff693b07c62c10f5e39801d8083742')
    bs = load(best/'summary.json'); arrays = {}; partitions = {}
    for r in bs['results']:
        policy = r['policy']
        for part, item in r['partitions'].items():
            pp = best/policy/(part+'_prediction.npy'); ip = best/policy/(part+'_source_indices.npy')
            bind(pp, item['prediction_sha256']); bind(ip, item['indices_sha256'])
            ids = np.load(ip); pred = np.load(pp)
            assert pred.shape == (len(ids), 63)
            assert np.isfinite(pred).all() and (pred >= 0).all() and abs(pred.sum(1)-1).max() < 1e-6
            if part in partitions: assert np.array_equal(partitions[part], ids)
            partitions[part] = ids; arrays[(policy, part)] = pred
    assert np.array_equal(np.sort(np.concatenate(list(partitions.values()))), np.arange(len(meta)))
    for left, right in [('train','stop'), ('train','evaluation'), ('stop','evaluation')]:
        for k in ['group_id', 'sequence_sha256']:
            assert not set(meta.iloc[partitions[left]][k]) & set(meta.iloc[partitions[right]][k])
    tr = partitions['train']; counts = meta.iloc[tr].sequence_sha256.value_counts()
    weights = 1/meta.iloc[tr].sequence_sha256.map(counts).to_numpy(); weights /= weights.sum()
    yy = y[tr].astype(float); yy /= yy.sum(1)[:, None]
    mean = (weights[:, None]*yy).sum(0)
    hc = (weights[:, None]*np.sqrt(yy)).sum(0)**2; hc /= hc.sum()
    centers = {'fit_only_mean_PMF': mean, 'fit_only_Hellinger_center': hc}
    results = []; frames = []
    for part, ids in partitions.items():
        m = meta.iloc[ids].reset_index(drop=True); rw = raw.iloc[ids].reset_index(drop=True)
        nq = rw.raw_negative_absolute_shape.to_numpy(bool); pq = rw.raw_positive_absolute_shape.to_numpy(bool)
        items = [(policy, arrays[(policy, part)]) for policy in ['fresh', 'plateau']]
        items += [(name, np.broadcast_to(p, (len(ids), 63))) for name, p in centers.items()]
        for name, pred in items:
            q, bins = uniform_quantiles(pred, edges); iqr = q[:,2]-q[:,0]
            eligible = np.isfinite(q).all(1) & (iqr >= 1)
            s = np.full(len(ids), np.nan); s[eligible] = (q[eligible,0]+q[eligible,2]-2*q[eligible,1])/iqr[eligible]
            f = m[['sample_id','sequence_sha256','group_id']].copy()
            for k,v in probability_errors(y[ids], pred).items(): f[k] = v
            f['Q50_AE'] = abs(q[:,1]-rw.q50_raw_QC.to_numpy())
            f['IQR_AE'] = abs(iqr-rw.raw_IQR.to_numpy())
            f['Bowley_AE'] = abs(s-rw.raw_Bowley.to_numpy())
            f['predicted_Bowley'] = s; f['raw_Bowley'] = rw.raw_Bowley.to_numpy()
            f['predicted_IQR'] = iqr; f['raw_IQR'] = rw.raw_IQR.to_numpy()
            f['Bowley_eligible'] = eligible
            f['negative_edge_bias'] = pred[:,0]-y[ids,0]
            f['positive_edge_bias'] = pred[:,-1]-y[ids,-1]
            r = {'predictor': name, 'partition': part, 'rows': len(ids),
                 'unique_T': int(m.sequence_sha256.nunique()), 'groups': int(m.group_id.nunique()),
                 'metrics': macro(f, ['H','CDF_MSE','edge_MAE','Q50_AE','IQR_AE','Bowley_AE','negative_edge_bias','positive_edge_bias']),
                 'negative': confusion(m, nq, eligible & (s <= -.30), eligible),
                 'positive': confusion(m, pq, eligible & (s >= .30), eligible)}
            if name in ['fresh','plateau']:
                r['fixed_one_percent_ranking'], _ = rank_one_percent(f, nq, pq)
            else:
                r['fixed_one_percent_ranking'] = {'status':'uninformative_constant_score_ties',
                    'expected_random_recall': min(int(np.ceil(.01*m.sequence_sha256.nunique()))/m.sequence_sha256.nunique(),1.)}
            r['truth_strata'] = {}
            for side, mask in [('strong_negative', nq), ('strong_positive', pq), ('central_abs_Bowley_lt_point1', abs(rw.raw_Bowley.to_numpy()) < .1)]:
                z = f.loc[mask]
                r['truth_strata'][side] = {'rows': len(z), 'unique_T': int(z.sequence_sha256.nunique()), 'groups': int(z.group_id.nunique()),
                    'raw_Bowley_range': [float(z.raw_Bowley.min()),float(z.raw_Bowley.max())] if len(z) else None,
                    'predicted_Bowley_range': [float(z.predicted_Bowley.min()),float(z.predicted_Bowley.max())] if len(z) else None,
                    'metrics': macro(z, ['H','Bowley_AE','negative_edge_bias','positive_edge_bias'])}
            results.append(r); f['predictor'] = name; f['partition'] = part; frames.append(f)
    pd.concat(frames).to_parquet(a.output/'private_per_sample.parquet', index=False)
    put(a.output/'identity.json', {'refs': refs, 'status':'passed', 'group_disjoint':True})
    put(a.output/'private_fit_only_centers.json', {k: v.tolist() for k,v in centers.items()})
    summary = {'status':'complete', 'schema':'shape_partition_diagnostics_v1', 'results':results,
        'new_fits':0, 'model_loads':0, 'raw_phi_reads':0, 'code_sha256':sha(__file__),
        'identity_sha256':sha(a.output/'identity.json'), 'private_table_sha256':sha(a.output/'private_per_sample.parquet'),
        'constant_reference_weighting':'Equal training T, then equal views within each T; no stop/evaluation targets fitted.',
        'scope':'Two selected states; exploratory development data. Rare-threshold comparisons use absolute raw Bowley conditions, not complete historical source-percentile rule. Fit scores reflect in-sample capability, not generalization.',
        'elapsed_seconds':time.monotonic()-start, 'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    put(a.output/'summary.json', summary)
    print(json.dumps({'status':'complete', 'elapsed_seconds':summary['elapsed_seconds'], 'summary_sha256':sha(a.output/'summary.json')}))


if __name__ == '__main__': main()
