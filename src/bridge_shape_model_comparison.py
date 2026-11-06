"""Compare saved OOF distributions with historical raw-QC shape summaries.

Within-bin uniform inverse-CDF quantiles are explicit proxies, not sample type-7
quantiles and not an optimal/irreducible error bound. No model or phi is loaded.
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
GUARD_PIN = '71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c'
GUARD_CODE_PIN = '5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'
DIAGNOSTICS_PIN = 'a39427ca30cc7fdc652f5355bc3854d2d10288cfd47c1b4937d7a6ca453cfa15'
FIT_INDEX_PIN = '224251b6320031487be1514222d8cea3a9d0201f6dfea4ddfbdc1d9e1eeff9ef'
REPRESENTATION_PIN = '995a2e7b6502226ea39540a752b0574af0cf498d2b90919732447cf6f87ad41f'
FAMILIES = ['composition_charge_deep', 'mean_pre', 'residue_mean']
SEEDS = [1701, 3407]
METRICS = ['H', 'CDF_MSE', 'edge_MAE', 'Q50_AE', 'IQR_AE', 'Bowley_AE']
DUPLICATE_P_ATOL = 1e-6


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def put(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def sizes(meta, mask=None):
    z = meta if mask is None else meta.loc[np.asarray(mask, bool)]
    return {'rows': len(z), 'unique_T': int(z.sequence_sha256.nunique()), 'groups': int(z.group_id.nunique())}


def uniform_quantiles(probability, edges):
    p = np.asarray(probability, float)
    edges = np.asarray(edges, float)
    assert p.ndim == 2 and p.shape[1] == len(edges)+1
    assert np.isfinite(p).all() and (p >= 0).all() and (p.sum(1) > 0).all()
    assert np.isfinite(edges).all() and (np.diff(edges) > 0).all()
    p = p/p.sum(1)[:, None]  # Only quantile proxies, never original probability metrics.
    cumulative = p.cumsum(1)
    quantiles = np.full((len(p), 3), np.nan)
    bins = np.empty((len(p), 3), np.int64)
    for j, q in enumerate([.25, .5, .75]):
        k = (cumulative < q).sum(1)
        assert (k < p.shape[1]).all()
        bins[:, j] = k
        central = (k > 0) & (k < p.shape[1]-1)
        rows = np.flatnonzero(central); b = k[rows]
        mass = p[rows, b]
        assert (mass > 0).all()
        fraction = (q-cumulative[rows, b-1])/mass
        assert ((fraction >= -1e-12) & (fraction <= 1+1e-12)).all()
        quantiles[rows, j] = edges[b-1] + fraction*(edges[b]-edges[b-1])
    return quantiles, bins


def probability_errors(truth, pred):
    y, p = np.asarray(truth, float), np.asarray(pred, float)
    return {'H': np.sqrt(.5*((np.sqrt(y)-np.sqrt(p))**2).sum(1)),
            'CDF_MSE': ((y.cumsum(1)[:, :-1]-p.cumsum(1)[:, :-1])**2).mean(1),
            'edge_MAE': abs(y[:, [0, -1]]-p[:, [0, -1]]).mean(1)}


def macro(frame, columns):
    proteins = frame.groupby(['group_id', 'sequence_sha256'])[columns].mean()
    groups = proteins.groupby('group_id').mean()
    return {c: {'group_macro': float(groups[c].mean()) if groups[c].notna().any() else None,
                'protein_macro': float(proteins[c].mean()) if proteins[c].notna().any() else None,
                'support': {'rows': int(frame[c].notna().sum()), 'unique_T': int(proteins[c].notna().sum()),
                            'groups': int(groups[c].notna().sum())}} for c in columns}


def confusion(meta, truth, predicted, eligible):
    truth, predicted, eligible = np.asarray(truth, bool), np.asarray(predicted, bool), np.asarray(eligible, bool)
    events = {'TP': truth & predicted, 'FP': ~truth & predicted, 'FN': truth & ~predicted,
              'TN': ~truth & ~predicted, 'abstained': ~eligible, 'positive_truth_abstained': truth & ~eligible}
    row = {key: sizes(meta, val) for key, val in events.items()}
    tp, fp, fn = [row[k]['rows'] for k in ['TP', 'FP', 'FN']]
    frame = meta[['group_id', 'sequence_sha256']].copy()
    for k in ['TP', 'FP', 'FN']:
        frame[k] = events[k].astype(float)
    rates = {k: v['group_macro'] for k, v in macro(frame, ['TP', 'FP', 'FN']).items()}
    return {'row_events': row, 'truth_positive': sizes(meta, truth), 'predicted_positive': sizes(meta, predicted),
            'row_precision': tp/(tp+fp) if tp+fp else None, 'row_recall': tp/(tp+fn) if tp+fn else None,
            'group_balanced_event_masses': rates,
            'group_balanced_precision': rates['TP']/(rates['TP']+rates['FP']) if rates['TP']+rates['FP'] else None,
            'group_balanced_recall': rates['TP']/(rates['TP']+rates['FN']) if rates['TP']+rates['FN'] else None,
            'abstention_rule': 'No positive call; therefore positive truth rejected remains FN. TN includes negative-truth abstentions and is not called supported specificity.'}


def rank_one_percent(frame, raw_negative, raw_positive):
    # Scores from one fixed row per T; never average probabilities or shapes.
    canonical = frame.sort_values('sample_id').drop_duplicates('sequence_sha256').copy()
    truth = frame[['sequence_sha256', 'group_id']].copy()
    truth['negative'] = raw_negative; truth['positive'] = raw_positive
    truth = truth.groupby('sequence_sha256')[['negative', 'positive']].any()
    canonical = canonical.join(truth, on='sequence_sha256')
    budget = math.ceil(.01*len(canonical))
    valid = canonical[canonical.Bowley_eligible].copy()
    result = {'budget_fraction': .01, 'budget_T': budget, 'denominator_T': len(canonical),
              'rankable_T': len(valid), 'rejected_T': len(canonical)-len(valid),
              'score_selection': 'One lexicographically first sample_id row per T, fixed before inspecting scores; no view averaging',
              'truth_selection': 'Any raw view of the T satisfies the same-direction absolute condition; raw views remain separate elsewhere',
              'population': 'Exposed development T pool, not historical source population percentiles'}
    selected_refs = []
    for direction, ascending in [('negative', True), ('positive', False)]:
        selected = valid.sort_values(['predicted_Bowley', 'sequence_sha256'], ascending=[ascending, True]).head(budget)
        positive = selected[direction].to_numpy(bool)
        tp = int(positive.sum()); total_truth = int(canonical[direction].sum())
        result[direction] = {'selected_T': len(selected), 'selected_groups': int(selected.group_id.nunique()),
            'true_positive_T': tp, 'true_positive_groups': int(selected.loc[positive, 'group_id'].nunique()),
            'false_positive_T': len(selected)-tp, 'false_negative_T': total_truth-tp, 'truth_positive_T': total_truth,
            'precision': tp/len(selected) if len(selected) else None, 'recall': tp/total_truth if total_truth else None}
        for r in selected[['sample_id', 'sequence_sha256', 'group_id', 'predicted_Bowley']].to_dict('records'):
            selected_refs.append({'direction': direction, **r})
    return result, selected_refs


def self_test():
    e = np.array([-3., -1., 0., 2., 5.])
    a, bins = uniform_quantiles(np.array([[0., 1., 0., 0., 0., 0.], [0., .5, 0., 0., .5, 0.], [.3, 0., 0., 0., .7, 0.]]), e)
    assert np.allclose(a[0], [-2.5, -2., -1.5])
    assert np.allclose(a[1], [-2., -1., 3.5])
    assert bins[1, 1] == 1 and a[1, 1] == -1.  # First CDF >= q.
    assert np.isnan(a[2, 0]) and np.isfinite(a[2, 1:]).all()
    assert float(np.quantile([-2., 4.], .5, method='linear')) == 1. != a[1, 1]
    return {'status': 'passed', 'empty_bin_and_internal_boundary_checked': True,
            'open_tail_refused': True, 'uniform_inverse_CDF_not_sample_type7_demonstrated': True,
            'no_actual_proteins_read': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    import resource
    import pandas as pd
    import pyarrow.parquet as pq
    assert __debug__ and args.root is not None and args.output is not None
    assert os.sched_getaffinity(0) == {172}
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    assert not args.output.exists(); args.output.mkdir(parents=True)
    output, root = args.output, args.root
    start = time.monotonic(); refs = []
    put(output/'started.json', {'pid': os.getpid(), 'unix': time.time(), 'code_sha256': sha(__file__),
                               'CPU_affinity': [172], 'max_address_space_GiB': 8, 'timeout_seconds': 600,
                               'fits_loaded': 0, 'raw_phi_reads': 0})

    def bind(path, expected=None):
        h = sha(path)
        refs.append({'path': str(path), 'sha256': h, 'bytes': path.stat().st_size, 'expected_sha256': expected})
        if expected is not None:
            assert h == expected, 'changed bound input: '+str(path)
        return h

    try:
        data = root/'atlas_scale_20261105/data'
        bind(data/'release.json', RELEASE_PIN); release = load(data/'release.json')
        bind(data/'samples.parquet', release['artifacts']['samples.parquet'])
        m = pq.read_table(data/'samples.parquet', columns=['sample_id', 'sequence_sha256', 'group_id', 'fold', 'target_row']).to_pandas()
        assert m.sample_id.is_unique and sizes(m) == {'rows': 21921, 'unique_T': 21915, 'groups': 7999}
        assert m.groupby('sequence_sha256').fold.nunique().max() == 1
        assert m.groupby('sequence_sha256').group_id.nunique().max() == 1
        assert m.groupby('group_id').fold.nunique().max() == 1
        roles_root = root/'bridge_20261106/roles'
        bind(roles_root/'code/bridge_roles_guard.py', GUARD_CODE_PIN)
        sys.path.insert(0, str(roles_root/'code'))
        from bridge_roles_guard import check
        request = {'stage': 'development', 'sequence_sha256': sorted(m.sequence_sha256.unique().tolist()), 'aliases': []}
        guards = [check(roles_root/'registry_v1/guard_contract.json', GUARD_PIN, {'operation': op, **request}) for op in ['read_target', 'model_eval']]
        put(output/'role_preflight.json', {'request': request, 'receipts': guards})
        # No target or prediction payload precedes protection preflight.
        rep = root/'bridge_20261106/diagnostic_supplements/representation_audit_v1'
        bind(rep/'summary.json', REPRESENTATION_PIN); rep_summary = load(rep/'summary.json')
        assert rep_summary['outcome'] == 'all_quantile_and_certificate_consistency_checks_passed'
        bind(rep/'identity.json', rep_summary['identity_sha256']); identity = load(rep/'identity.json')
        for old in identity['refs']:
            bind(Path(old['path']), old['sha256'])
        bind(rep/'per_sample.parquet', rep_summary['private_table']['sha256'])
        raw = pq.read_table(rep/'per_sample.parquet').to_pandas().set_index('sample_id').loc[m.sample_id].reset_index()
        assert np.array_equal(raw.sequence_sha256, m.sequence_sha256) and np.array_equal(raw.group_id, m.group_id)
        raw_q = raw[['q25_raw_QC', 'q50_raw_QC', 'q75_raw_QC']].to_numpy(float)
        raw_iqr = raw.raw_IQR.to_numpy(float); raw_s = raw.raw_Bowley.to_numpy(float)
        negative = raw.raw_negative_absolute_shape.to_numpy(bool); positive = raw.raw_positive_absolute_shape.to_numpy(bool)
        assert negative.sum() == 61 and positive.sum() == 17
        y = np.load(data/'p.npy', allow_pickle=False)
        counts_path = root/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet'
        c = pq.read_table(counts_path, filters=[('sample_id', 'in', m.sample_id.tolist())]).to_pandas().set_index('sample_id').loc[m.sample_id]
        counts = c[[f'bin_{i:03d}' for i in range(63)]].to_numpy()
        assert np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
        total = counts.sum(1)
        assert np.array_equal(total, raw.n_vertices) and np.array_equal((counts/total[:, None]).astype(np.float32), y)
        contract = load(root/'data/labels/release.pending.json'); edges = np.asarray(contract['edges'])
        diagnostics = root/'bridge_20261106/diagnostics/public/receipt.json'
        bind(diagnostics, DIAGNOSTICS_PIN); d = load(diagnostics)
        index = root/'overnight_20261106/confirmation/results/all_fits.csv'
        bind(index, FIT_INDEX_PIN); assert d['fit_index_sha256'] == FIT_INDEX_PIN
        fits = pd.read_csv(index)
        assert len(fits) == 30 and set(map(tuple, fits[['family', 'seed', 'fold']].to_numpy())) == {(f,s,k) for f in FAMILIES for s in SEEDS for k in range(5)}
        pins = {(r['family'], r['seed'], r['fold']): r for r in d['source_fits']}
        ix = pd.Series(np.arange(len(m)), index=m.sample_id)
        predictions = {(f,s): np.full(y.shape, np.nan, np.float32) for f in FAMILIES for s in SEEDS}
        coverage = {key: np.zeros(len(m), np.int64) for key in predictions}
        fit_checks = []
        for fit in fits.to_dict('records'):
            family, seed, fold = fit['family'], int(fit['seed']), int(fit['fold'])
            run = Path(fit['artifact_path']); run.relative_to(root)
            bind(run/'receipt.json'); rec = load(run/'receipt.json'); cfg = rec['config']
            assert rec['status'] == 'complete' and rec['run_id'] == fit['run_id']
            assert cfg.get('family', cfg.get('mode')) == family and cfg['fold'] == fold and cfg['seed'] == seed
            assert cfg.get('data_release_sha256', cfg.get('release_sha256')) == RELEASE_PIN
            assert cfg['script_sha256'] == fit['trainer_sha256'] and cfg['model_sha256'] == fit['model_sha256']
            pin = pins[(family, seed, fold)]
            bind(run/'roles.json', pin['roles_sha256']); roles = load(run/'roles.json')
            sets = {k: set(v) for k,v in roles.items()}
            assert set(sets) == {'train_ids', 'stop_ids', 'evaluation_ids'}
            assert all(len(roles[k]) == len(sets[k]) for k in sets)
            assert set.union(*sets.values()) == set(m.sample_id) and sum(map(len, sets.values())) == len(m)
            ids = ix.loc[roles['evaluation_ids']].to_numpy()
            assert len(ids) == len(set(ids)) and set(ids) == set(np.flatnonzero(m.fold.to_numpy() == fold))
            for key, wanted in [('stop_ids', {(fold+1)%5}), ('train_ids', set(range(5))-{fold,(fold+1)%5})]:
                assert set(m.iloc[ix.loc[roles[key]].to_numpy()].fold) == wanted
            assert rec['prediction_sha256'] == pin['prediction_sha256']
            bind(run/'evaluation_prediction.npy', pin['prediction_sha256'])
            p = np.load(run/'evaluation_prediction.npy', allow_pickle=False)
            assert p.shape == (len(ids),63) and np.isfinite(p).all() and (p >= 0).all() and abs(p.sum(1)-1).max() < 1e-6
            f = m.iloc[ids][['group_id', 'sequence_sha256']].reset_index(drop=True).copy()
            for name, val in probability_errors(y[ids], p).items(): f[name] = val
            calculated = macro(f, ['H', 'CDF_MSE', 'edge_MAE'])
            differences = {}
            for name, old, tolerance in [('H','group_macro_H',1e-7),('CDF_MSE','group_macro_cutpoint_CDF_MSE',1e-8),('edge_MAE','group_macro_edge_bin_mass_MAE',1e-8)]:
                differences[name] = calculated[name]['group_macro']-rec['result'][old]
                assert abs(differences[name]) <= tolerance
            assert abs(calculated['H']['group_macro']-fit['H']) <= 1e-7
            predictions[(family, seed)][ids] = p; coverage[(family, seed)][ids] += 1
            fit_checks.append({'family':family,'seed':seed,'fold':fold,'run_id':fit['run_id'],
                               'receipt_sha256':sha(run/'receipt.json'),'prediction_sha256':pin['prediction_sha256'],
                               'roles_sha256':pin['roles_sha256'],'metric_replay_differences':differences})
        assert all(np.all(v == 1) for v in coverage.values())
        duplicate_groups = [np.asarray(v) for v in m.groupby('sequence_sha256').indices.values() if len(v)>1]
        duplicate_checks = []
        for (family,seed), p in predictions.items():
            maximum = max((float(abs(p[ids]-p[ids[0]]).max()) for ids in duplicate_groups), default=0.)
            assert maximum <= DUPLICATE_P_ATOL
            duplicate_checks.append({'family':family,'seed':seed,'repeated_T_groups':len(duplicate_groups),
                                     'max_abs_prediction_difference':maximum,'absolute_tolerance':DUPLICATE_P_ATOL,'same_fold':True})
        put(output/'identity_and_prediction_replay.json', {'status':'passed','refs':refs,'fit_checks':fit_checks,
                                                          'duplicate_T_checks':duplicate_checks,'all_30_fits_complete_OOF':True})
        put(output/'mathematical_preflight.json', self_test())
        frames = {}; summaries = []
        datasets = [('oracle_true63_uniform', None, y, counts/total[:, None])]
        datasets += [(family, seed, predictions[(family,seed)], predictions[(family,seed)]) for seed in SEEDS for family in FAMILIES]
        for family, seed, probability, quantile_probability in datasets:
            q, bins = uniform_quantiles(quantile_probability, edges)
            q50_ok = np.isfinite(q[:,1]); iqr_ok = np.isfinite(q[:,[0,2]]).all(1)
            pred_iqr = q[:,2]-q[:,0]
            bowley_ok = np.isfinite(q).all(1) & (pred_iqr >= 1.)
            pred_s = np.full(len(m), np.nan)
            pred_s[bowley_ok] = (q[bowley_ok,0]+q[bowley_ok,2]-2*q[bowley_ok,1])/pred_iqr[bowley_ok]
            frame = m.copy(); frame['family'] = family; frame['seed'] = seed
            for name, values in probability_errors(y, probability).items(): frame[name] = values
            frame['Q50_AE'] = np.where(q50_ok, abs(q[:,1]-raw_q[:,1]), np.nan)
            frame['IQR_AE'] = np.where(iqr_ok, abs(pred_iqr-raw_iqr), np.nan)
            frame['Bowley_AE'] = np.where(bowley_ok & (raw_iqr >= 1.), abs(pred_s-raw_s), np.nan)
            frame['predicted_Bowley'] = pred_s; frame['predicted_IQR'] = pred_iqr; frame['Bowley_eligible'] = bowley_ok
            frame['raw_Bowley'] = raw_s; frame['raw_IQR'] = raw_iqr
            for j,n in enumerate(['q25','q50','q75']): frame[n+'_proxy'] = q[:,j]; frame[n+'_bin'] = bins[:,j]; frame[n+'_raw'] = raw_q[:,j]
            frame['raw_negative'] = negative; frame['raw_positive'] = positive
            frame['predicted_negative'] = bowley_ok & (pred_s <= -.30)
            frame['predicted_positive'] = bowley_ok & (pred_s >= .30)
            support = {'Q50_estimable':sizes(m,q50_ok),'Q50_open_tail_rejected':sizes(m,~q50_ok),
                       'IQR_estimable':sizes(m,iqr_ok),'IQR_open_tail_rejected':sizes(m,~iqr_ok),
                       'Bowley_eligible':sizes(m,bowley_ok),'any_quantile_open_tail':sizes(m,~np.isfinite(q).all(1)),
                       'finite_predicted_IQR_below_1':sizes(m,iqr_ok & (pred_iqr < 1.))}
            item = {'family':family,'seed':seed,'metrics':macro(frame,METRICS),'support':support,
                    'negative':confusion(m,negative,frame.predicted_negative,bowley_ok),
                    'positive':confusion(m,positive,frame.predicted_positive,bowley_ok)}
            summaries.append(item); frames[(family,seed)] = frame
        # A fixed 1% candidate budget is reported only when threshold calls are sparse.
        k = math.ceil(.01*m.sequence_sha256.nunique())
        sparse = any(item[side]['predicted_positive']['unique_T'] < k for item in summaries if item['seed'] is not None for side in ['negative','positive'])
        selected_rows = []
        if sparse:
            for item in summaries:
                frame = frames[(item['family'],item['seed'])]
                item['fixed_one_percent_ranking'], selected = rank_one_percent(frame,negative,positive)
                selected_rows += [{'family':item['family'],'seed':item['seed'],**r} for r in selected]
        common = []
        oracle = frames[('oracle_true63_uniform',None)]
        for seed in SEEDS:
            for metric in ['Q50_AE','IQR_AE','Bowley_AE']:
                mask = np.isfinite(oracle[metric].to_numpy())
                for family in FAMILIES: mask &= np.isfinite(frames[(family,seed)][metric].to_numpy())
                for family in ['oracle_true63_uniform']+FAMILIES:
                    z = oracle if family=='oracle_true63_uniform' else frames[(family,seed)]
                    common.append({'family':family,'seed_block':seed,'metric':metric,
                                   'summary':macro(z.loc[mask],[metric])[metric]})
        table = pd.concat(frames.values(), ignore_index=True)
        table.to_parquet(output/'per_sample_metrics.parquet',index=False)
        if sparse: pd.DataFrame(selected_rows).to_parquet(output/'one_percent_selected_T_private.parquet',index=False)
        per_group = table.groupby(['family','seed','group_id','sequence_sha256'],dropna=False)[METRICS].mean().groupby(['family','seed','group_id'],dropna=False).mean().reset_index()
        per_group.to_parquet(output/'per_group_metrics.parquet',index=False)
        summary = {'schema':'shape_model_comparison_v1','status':'complete','cohort':sizes(m),'OOF_fits':30,
                   'seeds':SEEDS,'seeds_are_not_independent_sample_replicates':True,'results':summaries,
                   'common_support_comparison':common,'duplicate_T_checks':duplicate_checks,
                   'ranking_trigger':{'fixed_budget_fraction':.01,'K_distinct_T':k,'trigger':'At least one model-direction has fewer threshold-positive T than the fixed 1% budget','triggered':sparse},
                   'probability_replay':{'fits':len(fit_checks),'all_passed':True,
                       'max_absolute_H_difference':max(abs(r['metric_replay_differences']['H']) for r in fit_checks)},
                   'definitions':{'proxy':'First cumulative p>=q bin; finite central interval uses uniform-density interpolation. Open-tail bins abstain even at boundary. Model p is row-renormalized only for proxy quantiles.',
                       'truth':'Previously verified raw QC type7-compatible q25/q50/q75; IQR and Bowley computed per original sample.',
                       'absolute_shape':'Raw IQR>=1 and Bowley<=-.30 or >=+.30; no original source-percentile condition.',
                       'aggregation':'Each row metric -> same T within group mean -> equal groups. Missing proxy metrics excluded only from that metric and support reported; common support supplied separately.',
                       'confusion':'Full row events with distinct T/groups plus ratios of group-balanced TP/FP/FN masses; rejected positives remain FN.',
                       'oracle':'True63 under the same within-bin-uniform inverse-CDF rule. Assumption-based reference; not best attainable or irreducible error lower bound. Includes quantile-estimator convention differences.'},
                   'limits':['All observations are previously exposed development; neither final-test claims nor source-population extreme screening.',
                             'Two seeds are reported separately on the same samples; no seed-as-independent-n pooling or best-seed selection.',
                             'Multiple target views remain separate. No probability or raw shape averaging; fixed canonical-row scores used only for T-budget ranking and any-view raw truth is explicit.',
                             'No mode classification, AP/PR sweep, model loading, new fit or raw phi matrix access.'],
                   'code_sha256':sha(__file__),'identity_sha256':sha(output/'identity_and_prediction_replay.json'),
                   'private_outputs':{'per_sample_metrics':{'path':str(output/'per_sample_metrics.parquet'),'sha256':sha(output/'per_sample_metrics.parquet'),'rows':len(table)},
                                      'per_group_metrics':{'path':str(output/'per_group_metrics.parquet'),'sha256':sha(output/'per_group_metrics.parquet'),'rows':len(per_group)}},
                   'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'CPU_affinity':[172]}}
        put(output/'summary.json',summary)
        print(json.dumps({'status':'complete','OOF_fits':30,'new_fits':0,'summary_sha256':sha(output/'summary.json'),
                          'elapsed_seconds':summary['elapsed_seconds'],'peak_RSS_KiB':summary['peak_RSS_KiB']}))
    except Exception as exc:
        put(output/'failure.json', {'status':'failed_preserved','type':type(exc).__name__,'error':str(exc),
                                   'traceback':traceback.format_exc(),'source_refs_so_far':refs,'code_sha256':sha(__file__)})
        raise


if __name__ == '__main__':
    main()
