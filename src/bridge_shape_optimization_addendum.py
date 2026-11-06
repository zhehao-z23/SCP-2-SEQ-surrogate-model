"""Saved-prediction shape comparison of eight matched optimization pairs.

No fitting, inference, checkpoint load, phi access, or complete-OOF claim.
The frozen original shape tool supplies all numerical definitions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HELPER_SHA = '4be007ff1b869a4a71b800713f3e833f79b04770f7f27dd66aeb22507da16ce6'
OOF_SUMMARY_SHA = '675c87743ed787135fa99b1d25c98ebfa0fa1bb427a06681c96f5b54b7e292cd'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def put(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    a = ap.parse_args()
    import resource
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    assert __debug__ and os.sched_getaffinity(0) == {172}
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    plan = load(a.plan); root = Path(plan['root']); out = Path(plan['output'])
    assert sha(__file__) == plan['code_sha256']
    assert not out.exists(); out.mkdir(parents=True)
    start = time.monotonic(); refs = []
    put(out/'started.json', {'pid': os.getpid(), 'unix': time.time(), 'CPU_affinity': [172],
        'code_sha256': sha(__file__), 'plan_sha256': sha(a.plan), 'new_fits': 0,
        'model_loads': 0, 'raw_phi_reads': 0, 'address_space_GiB': 8, 'timeout_seconds': 600})

    def bind(p, expected=None):
        p = Path(p); actual = sha(p)
        refs.append({'path': str(p), 'sha256': actual, 'bytes': p.stat().st_size, 'expected_sha256': expected})
        if expected is not None:
            assert actual == expected, str(p)
        return actual

    try:
        helper = Path(__file__).with_name('bridge_shape_model_comparison.py')
        bind(helper, HELPER_SHA)
        sys.path.insert(0, str(helper.parent))
        import bridge_shape_model_comparison as h
        data = root/'atlas_scale_20261105/data'
        bind(data/'release.json', h.RELEASE_PIN); rel = load(data/'release.json')
        bind(data/'samples.parquet', rel['artifacts']['samples.parquet'])
        meta = pq.read_table(data/'samples.parquet', columns=['sample_id','sequence_sha256','group_id','fold','target_row']).to_pandas()
        assert meta.sample_id.is_unique and h.sizes(meta) == {'rows':21921,'unique_T':21915,'groups':7999}
        assert meta.groupby('sequence_sha256').fold.nunique().max() == 1
        assert meta.groupby('sequence_sha256').group_id.nunique().max() == 1
        assert meta.groupby('group_id').fold.nunique().max() == 1
        role_root = root/'bridge_20261106/roles'
        bind(role_root/'code/bridge_roles_guard.py', h.GUARD_CODE_PIN)
        sys.path.insert(0,str(role_root/'code'))
        from bridge_roles_guard import check
        req = {'stage':'development','sequence_sha256':sorted(meta.sequence_sha256.unique().tolist()),'aliases':[]}
        guards = [check(role_root/'registry_v1/guard_contract.json',h.GUARD_PIN,{'operation':op,**req}) for op in ['read_target','model_eval']]
        put(out/'role_preflight.json',{'request':req,'receipts':guards})
        # All target and prediction payloads are below this successful guard.
        rep = root/'bridge_20261106/diagnostic_supplements/representation_audit_v1'
        bind(rep/'summary.json',h.REPRESENTATION_PIN); rs = load(rep/'summary.json')
        bind(rep/'identity.json',rs['identity_sha256']); ri = load(rep/'identity.json')
        for r in ri['refs']: bind(r['path'],r['sha256'])
        bind(rep/'per_sample.parquet',rs['private_table']['sha256'])
        raw = pq.read_table(rep/'per_sample.parquet').to_pandas().set_index('sample_id').loc[meta.sample_id].reset_index()
        assert np.array_equal(raw.sequence_sha256,meta.sequence_sha256) and np.array_equal(raw.group_id,meta.group_id)
        bind(data/'p.npy',rel['artifacts']['p.npy']); y = np.load(data/'p.npy',allow_pickle=False)
        edges = np.asarray(load(root/'data/labels/release.pending.json')['edges'])
        base = root/'bridge_20261106/diagnostic_supplements/shape_model_comparison_v1'
        bind(base/'summary.json',OOF_SUMMARY_SHA); bs = load(base/'summary.json')
        bind(base/'identity_and_prediction_replay.json',bs['identity_sha256'])
        bind(base/'per_sample_metrics.parquet',bs['private_outputs']['per_sample_metrics']['sha256'])
        oracle = pq.read_table(base/'per_sample_metrics.parquet',filters=[('family','=','oracle_true63_uniform')]).to_pandas().set_index('sample_id').loc[meta.sample_id].reset_index()
        assert np.array_equal(oracle.sequence_sha256,meta.sequence_sha256)
        assert np.array_equal(oracle.q25_raw,raw.q25_raw_QC) and np.array_equal(oracle.q50_raw,raw.q50_raw_QC) and np.array_equal(oracle.q75_raw,raw.q75_raw_QC)
        for src in plan['evidence']: bind(src['path'],src['sha256'])
        audit_rows = {}
        for src in plan['evidence'][:2]:
            audit = load(src['path']); assert audit['status']=='complete'
            audit_rows.update({z['run_id']:z for z in audit['all_fits']})
        reference_evidence = load(plan['evidence'][2]['path'])['comparisons']
        ix = pd.Series(np.arange(len(meta)),index=meta.sample_id)
        records = {}; frames = {}; roles_all = {}; replays = []
        selected = []; results = []

        def describe(frame, identity):
            m = frame[['sample_id','sequence_sha256','group_id','fold']]
            neg = frame.raw_negative.to_numpy(bool); pos = frame.raw_positive.to_numpy(bool)
            ok = frame.Bowley_eligible.to_numpy(bool)
            rank, chosen = h.rank_one_percent(frame,neg,pos)
            for r in chosen: selected.append({**identity,**r})
            return {**identity,'cohort':h.sizes(m),'metrics':h.macro(frame,h.METRICS),
                'support':{'Q50_estimable':h.sizes(m,np.isfinite(frame.Q50_AE)),
                           'IQR_estimable':h.sizes(m,np.isfinite(frame.IQR_AE)),
                           'Bowley_eligible':h.sizes(m,ok),'Bowley_rejected':h.sizes(m,~ok),
                           'any_quantile_open_tail':h.sizes(m,~np.isfinite(frame[['q25_proxy','q50_proxy','q75_proxy']]).all(1)),
                           'finite_predicted_IQR_below_1':h.sizes(m,np.isfinite(frame.predicted_IQR)&(frame.predicted_IQR<1))},
                'negative':h.confusion(m,neg,frame.predicted_negative,ok),
                'positive':h.confusion(m,pos,frame.predicted_positive,ok),'fixed_one_percent_ranking':rank}

        for item in plan['runs']:
            run = Path(item['directory']); run.relative_to(root)
            bind(run/'receipt.json',item['receipt_sha256']); rec = load(run/'receipt.json'); cfg = rec['config']
            assert rec['status']=='complete' and rec['run_id']==item['run_id']
            assert cfg['mode']==item['family'] and cfg['fold']==item['fold'] and cfg['seed']==item['seed']
            assert cfg['release_sha256']==h.RELEASE_PIN
            if item['cohort']=='optimization12':
                assert cfg['policy']==item['policy']
                audit = audit_rows[item['run_id']]
                assert audit['prediction_sha256']==rec['prediction_sha256']==item['prediction_sha256']
                assert audit['checkpoint_sha256']==rec['checkpoint_sha256']
                assert audit['data_and_scaler_identity_validated'] and audit['prediction_and_checkpoint_validated']
                assert audit['script_sha256']==cfg['script_sha256'] and audit['model_sha256']==cfg['model_sha256']
            bind(run/'roles.json',item.get('roles_sha256')); roles = load(run/'roles.json')
            sets = {k:set(v) for k,v in roles.items()}
            assert set(sets)=={'train_ids','stop_ids','evaluation_ids'}
            assert set.union(*sets.values())==set(meta.sample_id) and sum(map(len,sets.values()))==len(meta)
            assert all(len(roles[k])==len(sets[k]) for k in sets)
            ids = ix.loc[roles['evaluation_ids']].to_numpy(); fold = item['fold']
            assert set(ids)==set(np.flatnonzero(meta.fold.to_numpy()==fold))
            assert np.array_equal(ids,np.flatnonzero(meta.fold.to_numpy()==fold)), 'prediction rows must use the recorded original evaluation order'
            assert set(meta.iloc[ix.loc[roles['stop_ids']]].fold)=={(fold+1)%5}
            assert set(meta.iloc[ix.loc[roles['train_ids']]].fold)==set(range(5))-{fold,(fold+1)%5}
            bind(run/'evaluation_prediction.npy',item['prediction_sha256'])
            assert rec['prediction_sha256']==item['prediction_sha256']
            p = np.load(run/'evaluation_prediction.npy',allow_pickle=False)
            assert p.shape==(len(ids),63) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
            z = meta.iloc[ids].reset_index(drop=True).copy(); r = raw.iloc[ids].reset_index(drop=True)
            for name,v in h.probability_errors(y[ids],p).items(): z[name]=v
            mm=h.macro(z,['H','CDF_MSE','edge_MAE']); differences={}
            for name,old,tol in [('H','group_macro_H',1e-7),('CDF_MSE','group_macro_cutpoint_CDF_MSE',1e-8),('edge_MAE','group_macro_edge_bin_mass_MAE',1e-8)]:
                differences[name]=mm[name]['group_macro']-rec['result'][old]; assert abs(differences[name])<=tol
            q,bins=h.uniform_quantiles(p,edges); iqr=q[:,2]-q[:,0]; ok=np.isfinite(q).all(1)&(iqr>=1)
            s=np.full(len(z),np.nan); s[ok]=(q[ok,0]+q[ok,2]-2*q[ok,1])/iqr[ok]
            z['Q50_AE']=abs(q[:,1]-r.q50_raw_QC); z['IQR_AE']=abs(iqr-r.raw_IQR)
            z['Bowley_AE']=np.where(ok&(r.raw_IQR>=1),abs(s-r.raw_Bowley),np.nan)
            z['predicted_Bowley']=s; z['predicted_IQR']=iqr; z['Bowley_eligible']=ok
            z['raw_Bowley']=r.raw_Bowley; z['raw_IQR']=r.raw_IQR
            z['raw_negative']=r.raw_negative_absolute_shape; z['raw_positive']=r.raw_positive_absolute_shape
            z['predicted_negative']=ok&(s<=-.3); z['predicted_positive']=ok&(s>=.3)
            for j,n in enumerate(['q25','q50','q75']): z[n+'_proxy']=q[:,j]; z[n+'_bin']=bins[:,j]; z[n+'_raw']=r[n+'_raw_QC']
            dup=[np.asarray(v) for v in z.groupby('sequence_sha256').indices.values() if len(v)>1]
            maxdiff=max((float(abs(p[v]-p[v[0]]).max()) for v in dup),default=0.)
            assert maxdiff<=h.DUPLICATE_P_ATOL
            identity={k:item[k] for k in ['run_id','family','policy','fold','seed']}
            identity['source_cohort']=item['cohort']
            for k,v in identity.items(): z[k]=v
            result=describe(z,identity); result['best_epoch']=rec['best_epoch']; result['best_stop_H']=rec['best_stop_H']
            result['duplicate_T_check']={'groups':len(dup),'max_abs_p_difference':maxdiff,'atol':h.DUPLICATE_P_ATOL}
            results.append(result); records[item['run_id']]=rec; frames[item['run_id']]=z; roles_all[item['run_id']]=roles
            replays.append({**identity,'metric_replay_differences':differences,'receipt_sha256':item['receipt_sha256'],'prediction_sha256':item['prediction_sha256'],'roles_sha256':sha(run/'roles.json')})
        assert len(results)==16 and sum(r['source_cohort']=='optimization12' for r in results)==12
        pairs=[]; by_id={z['run_id']:z for z in results}
        for pair in plan['pairs']:
            a_id,b_id=pair['plateau'],pair['reference']; za,zb=frames[a_id],frames[b_id]
            assert roles_all[a_id]==roles_all[b_id]
            assert za.sample_id.tolist()==zb.sample_id.tolist()
            ca,cb=records[a_id]['config'],records[b_id]['config']
            for k in ['mode','fold','seed','release_sha256','cache_receipt_sha256']:
                assert ca[k]==cb[k],k
            if by_id[b_id]['source_cohort']=='old_fixed_LR_reference':
                proof=[r for r in reference_evidence if r['new_run_id']==a_id and r['reference_run_id']==b_id]
                assert len(proof)==1 and proof[0]['roles_and_scalers_identical']
            differing={k:{'plateau':ca.get(k),'reference':cb.get(k)} for k in sorted(set(ca)|set(cb)) if ca.get(k)!=cb.get(k)}
            equal_keys=[k for k in sorted(set(ca)&set(cb)) if ca[k]==cb[k]]
            common={}
            for metric in h.METRICS:
                mask=np.isfinite(za[metric])&np.isfinite(zb[metric])
                delta=za.loc[mask,['group_id','sequence_sha256']].copy(); delta['delta']=za.loc[mask,metric]-zb.loc[mask,metric]
                common[metric]={'plateau':h.macro(za.loc[mask],[metric])[metric],'reference':h.macro(zb.loc[mask],[metric])[metric],
                    'paired_delta_plateau_minus_reference':h.macro(delta,['delta'])['delta']}
            pairs.append({**pair,'family':ca['mode'],'fold':ca['fold'],'seed':ca['seed'],'cohort':h.sizes(za),
                'roles_and_order_identical':True,'same_cache_receipt':True,'equal_config_keys':equal_keys,'different_config_values':differing,
                'scaler_evidence':pair['scaler_evidence'],'scalers_loaded_in_this_analysis':False,
                'common_support_paired_metrics':common,
                'plateau_summary':by_id[a_id],'reference_summary':by_id[b_id],
                'interpretation':'Matched evaluation-object engineering comparison; cross-runner reference is not proof of identical full training trajectory or an isolated causal LR effect.'})
        oracle_summaries=[]
        for fold in sorted({r['fold'] for r in results}):
            z=oracle.loc[oracle.fold==fold].reset_index(drop=True)
            oracle_summaries.append(describe(z,{'run_id':'oracle_fold_'+str(fold),'family':'oracle_true63_uniform','policy':'assumption_reference','fold':fold,'seed':None,'source_cohort':'true63_reference'}))
        tables=pd.concat(frames.values(),ignore_index=True); tables.to_parquet(out/'per_sample_metrics.parquet',index=False)
        pd.DataFrame(selected).to_parquet(out/'one_percent_selected_T_private.parquet',index=False)
        tables.groupby(['run_id','group_id','sequence_sha256'])[h.METRICS].mean().groupby(['run_id','group_id']).mean().reset_index().to_parquet(out/'per_group_metrics.parquet',index=False)
        put(out/'identity_and_replay.json',{'status':'passed','refs':refs,'run_replays':replays,'all_16_saved_predictions_verified':True})
        summary={'schema':'shape_optimization_addendum_v2','status':'complete','optimization_observations':12,'old_fixed_LR_references':4,
            'saved_prediction_files':16,'paired_comparisons':8,'complete_fivefold_OOF':False,'new_fits':0,'model_loads':0,
            'results':results,'pairs':pairs,'oracle_fold_references':oracle_summaries,
            'max_abs_H_replay_difference':max(abs(x['metric_replay_differences']['H']) for x in replays),
            'definitions_from_frozen_OOF_summary':{'path':str(base/'summary.json'),'sha256':OOF_SUMMARY_SHA},
            'limits':['Each pair has its own matching evaluation fold; never pooled as complete OOF.',
                      'Seeds repeat samples and are not independent n; both available seeds kept without choosing the better.',
                      'Fixed 1% distinct-T budget is computed within each evaluation fold, not the full development pool or original source population.',
                      'Absolute shape conditions omit historical source-percentile condition; raw positives are rare.',
                      'Oracle uses uniform-bin inverse CDF, not raw type7 or an irreducible lower bound.',
                      'Cross-runner references preserve documented config differences; no new scaler/checkpoint load or isolated LR-causality claim.'],
            'identity_sha256':sha(out/'identity_and_replay.json'),'code_sha256':sha(__file__),'plan_sha256':sha(a.plan),
            'private_outputs':[{'path':str(out/n),'sha256':sha(out/n)} for n in ['per_sample_metrics.parquet','per_group_metrics.parquet','one_percent_selected_T_private.parquet']],
            'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'CPU_affinity':[172]}}
        put(out/'summary.json',summary)
        print(json.dumps({'status':'complete','pairs':8,'new_fits':0,'summary_sha256':sha(out/'summary.json'),'elapsed_seconds':summary['elapsed_seconds'],'peak_RSS_KiB':summary['peak_RSS_KiB']}))
    except Exception as exc:
        put(out/'failure.json',{'status':'failed_preserved','error':str(exc),'traceback':traceback.format_exc(),'refs_so_far':refs})
        raise


if __name__=='__main__':
    main()
