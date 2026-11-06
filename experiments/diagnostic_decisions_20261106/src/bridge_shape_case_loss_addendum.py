"""Posthoc fixed-case description of two already-selected loss-probe checkpoints.

No fitting, model loading, raw phi reads, case reselection, or promotion changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text())


def put(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False); f.write('\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True); a = ap.parse_args()
    assert __debug__ and os.sched_getaffinity(0) == {174} and not os.environ.get('CUDA_VISIBLE_DEVICES')
    assert all(os.environ.get(k) == '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'])
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    import numpy as np
    import pandas as pd
    plan = load(a.plan); assert sha(__file__) == plan['code_sha256']
    out = Path(plan['output']); assert not out.exists(); out.mkdir()
    start = time.monotonic(); refs = []
    put(out/'started.json', {'pid': os.getpid(), 'unix': time.time(), 'CPU': [174], 'threads': 1, 'max_AS_GiB': 8, 'timeout_seconds': 600, 'code_sha256': sha(__file__), 'plan_sha256': sha(a.plan)})

    def bind(p, expected):
        p = Path(p); actual = sha(p); assert actual == expected, str(p)
        refs.append({'path': str(p), 'sha256': actual, 'bytes': p.stat().st_size}); return p

    try:
        supp = Path(plan['supplement_root']); probe = supp/'loss_probe_v1'
        bind(probe/'plan.json', plan['probe_plan_sha256']); pp = load(probe/'plan.json')
        data = Path(pp['parent'])/'data'; bind(data/'release.json', pp['release_sha256']); release = load(data/'release.json')
        bind(data/'samples.parquet', release['artifacts']['samples.parquet'])
        meta = pd.read_parquet(data/'samples.parquet', columns=['sample_id','sequence_sha256','group_id','fold','length','source','coverage'])
        guard = Path(pp['roles_root'])/'code/bridge_roles_guard.py'; bind(guard, pp['guard_code_sha256'])
        sys.path.insert(0, str(guard.parent)); from bridge_roles_guard import check
        request = {'stage': 'development', 'sequence_sha256': sorted(meta.sequence_sha256.unique().tolist()), 'aliases': []}
        put(out/'role_preflight.json', [check(Path(pp['roles_root'])/'registry_v1/guard_contract.json', pp['guard_contract_sha256'], {'operation': op, **request}) for op in ['read_target','model_eval']])
        # Target-bearing packets and saved predictions are read only after the guard.
        source = supp/'shape_case_facts_v2'
        bind(source/'private_facts.json', plan['case_facts_sha256']); oldfacts = load(source/'private_facts.json')
        bind(source/'identity.json', oldfacts['identity_sha256'])
        bind(source/'private_case_packet.json', oldfacts['private_packet_sha256']); packet = load(source/'private_case_packet.json')
        chosen = [x for x in packet['cases'] if x['case_type'] == 'raw_absolute_positive']
        assert len(chosen) == 2 and {x['role'] for x in chosen} == {'frozen_badcase','matched_normal_control'}
        assert next(x for x in packet['matching'] if x['case_type'] == 'raw_absolute_positive')['control_sample_id'] == next(x['sample_id'] for x in chosen if x['role'] == 'matched_normal_control')
        rep = supp/'representation_audit_v1'; bind(rep/'summary.json', '995a2e7b6502226ea39540a752b0574af0cf498d2b90919732447cf6f87ad41f')
        rep_summary = load(rep/'summary.json'); bind(rep/'per_sample.parquet', rep_summary['private_table']['sha256'])
        raw_full = pd.read_parquet(rep/'per_sample.parquet', columns=['sample_id','sequence_sha256','q25_raw_QC','q50_raw_QC','q75_raw_QC','raw_IQR','raw_Bowley'], filters=[('sample_id','in',[x['sample_id'] for x in chosen])]).set_index('sample_id')
        assert raw_full.index.is_unique and len(raw_full)==2
        bind(data/'p.npy', release['artifacts']['p.npy']); truth = np.load(data/'p.npy', allow_pickle=False)
        contract = Path(plan['label_contract']['path']); bind(contract, plan['label_contract']['sha256']); edges = np.asarray(load(contract)['edges'])
        assert edges.shape == (62,) and np.all(np.diff(edges) > 0) and edges[0] < 20 < 40 < edges[-1]
        helper = Path(plan['shape_helper']['path']); bind(helper, plan['shape_helper']['sha256'])
        sys.path.insert(0, str(helper.parent)); from bridge_shape_model_comparison import uniform_quantiles, probability_errors, macro
        coarse = Path(plan['coarse_helper']['path']); bind(coarse, plan['coarse_helper']['sha256'])
        sys.path.insert(0, str(coarse.parent)); from bridge_coarse_shape_support import peaks
        bind(probe/'analysis/summary.json', plan['probe_summary_sha256']); summary = load(probe/'analysis/summary.json')
        assert summary['status'] == 'complete' and summary['promote_to_one_paired_confirmation'] is False
        bind(probe/'paired_path_checks.json', summary['validity_receipt_sha256']); valid = load(probe/'paired_path_checks.json')
        assert valid['status'] == 'passed' and valid['plan_sha256'] == sha(probe/'plan.json')
        bind(Path(pp['reference_directory'])/'roles.json', pp['reference_roles_sha256']); expected_roles = load(Path(pp['reference_directory'])/'roles.json')
        canonical = {part+'_ids': meta.loc[mask,'sample_id'].tolist() for part,mask in [('evaluation',meta.fold==0),('stop',meta.fold==1),('train',meta.fold.isin([2,3,4]))]}
        assert expected_roles == canonical
        for part in ['train','stop']:
            assert not set(meta.loc[meta.sample_id.isin(canonical[part+'_ids']),'group_id']) & set(meta.loc[meta.fold==0,'group_id'])
        positions = pd.Series(np.arange(len(meta)), index=meta.sample_id); assert positions.index.is_unique
        ids = np.flatnonzero(meta.fold.to_numpy() == 0); preds = {}; runs = []; replays = []
        for rp in sorted((probe/'runs').glob('*/receipt.json')):
            rec = load(rp); loss = rec['config']['loss']; assert loss in ['H2','CE_quarter'] and loss not in preds
            bind(rp, valid['H2_receipt_sha256' if loss == 'H2' else 'CE_receipt_sha256'])
            assert rec['status'] == 'complete' and rec['config']['fold'] == 0 and rec['config']['seed'] == 1701
            run = rp.parent; bind(run/'roles.json', pp['reference_roles_sha256']); assert load(run/'roles.json') == expected_roles
            bind(run/'fixed_best_partitions.json', rec['fixed_best_partitions_sha256']); part = load(run/'fixed_best_partitions.json')['evaluation']
            predp = run/'evaluation_prediction.npy'; indp = run/'evaluation_source_indices.npy'
            bind(predp, part['prediction_sha256']); bind(indp, part['indices_sha256'])
            assert part['prediction_sha256'] == rec['prediction_sha256']
            oldinput = next(x for x in summary['input_predictions'] if x['loss'] == loss and x['partition'] == 'evaluation')
            assert all(oldinput[k] == part[k] for k in ['prediction_sha256','indices_sha256'])
            pred = np.load(predp, allow_pickle=False); savedids = np.load(indp, allow_pickle=False)
            assert np.array_equal(savedids, ids) and pred.shape == (len(ids),63)
            assert np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max() < 1e-6
            frame = meta.iloc[ids][['sample_id','sequence_sha256','group_id']].copy()
            for k,v in probability_errors(truth[ids],pred).items(): frame[k] = v
            mm = macro(frame,['H','CDF_MSE','edge_MAE'])
            previous = next(x for x in summary['results'] if x['loss']==loss and x['partition']=='evaluation')
            diffs = {k: abs(mm[k]['group_macro']-previous['metrics'][k]['group_macro']) for k in mm}
            assert max(diffs.values()) <= 1e-12
            replays.append({'loss':loss,'full_evaluation_probability_metric_maxdiff':max(diffs.values())})
            preds[loss] = pred
            runs.append({'loss':loss,'selected_epoch':rec['best_epoch'],'receipt':str(rp),'receipt_sha256':sha(rp),'prediction_sha256':sha(predp),'indices_sha256':sha(indp),'checkpoint_sha256_from_receipt':rec['checkpoint_sha256'],'checkpoint_loaded':False})
        assert set(preds) == {'H2','CE_quarter'}

        def describe(p, y, raw):
            p = np.asarray(p,float); y = np.asarray(y,float)
            density = p[1:-1]/np.diff(edges); ii = int(np.argmax(density)); left = right = ii; eps = float(density.max())*1e-12
            while left>0 and abs(density[left-1]-density[ii])<=eps: left-=1
            while right+1<len(density) and abs(density[right+1]-density[ii])<=eps: right+=1
            loc = peaks(density,0.)
            for v in loc:
                v['interval'] = [float(edges[v['left_bin']]),float(edges[v['right_bin']+1])]
                v['prominence_fraction_of_global_max'] = v['prominence']/float(density.max())
                v['passes_existing_5percent_prominence'] = bool(v['prominence']+1e-12*density.max()>=.05*density.max())
            inside=(edges[:-1]>=20)&(edges[1:]<=40); intersect=(edges[1:]>20)&(edges[:-1]<40)
            lengths=np.maximum(0,np.minimum(edges[1:],40)-np.maximum(edges[:-1],20))
            q,_ = uniform_quantiles(p[None,:],edges); q=q[0]; iqr=float(q[2]-q[0]); eligible=bool(np.isfinite(q).all() and iqr>=1)
            bow = float((q[0]+q[2]-2*q[1])/iqr) if eligible else None
            delta=p.cumsum()[:-1]-y.cumsum()[:-1]
            return {'window20_40_mass':{'lower':float(p[1:-1][inside].sum()),'upper':float(p[1:-1][intersect].sum()),'uniform_proxy':float(np.sum(p[1:-1]*lengths/np.diff(edges)))},
                'maximum_native_density_interval':[float(edges[left]),float(edges[right+1])],'maximum_density':float(density[ii]),
                'unthresholded_finite_interior_local_peaks':loc,'local_peaks_with_center_inside20_40':int(sum(20<(edges[v['left_bin']]+edges[v['right_bin']+1])/2<40 for v in loc)),
                'cutpoint_CDF_delta':{'minimum':float(delta.min()),'maximum':float(delta.max()),'positive_over_1e8':int((delta>1e-8).sum()),'negative_below_minus1e8':int((delta< -1e-8).sum()),'near_zero_1e8':int((abs(delta)<=1e-8).sum())},
                'H':float(np.sqrt(.5*((np.sqrt(p)-np.sqrt(y))**2).sum())), 'cutpoint_CDF_MSE':float((delta**2).mean()),
                'uniform_proxy_quantiles':[float(x) if np.isfinite(x) else None for x in q], 'uniform_proxy_IQR':iqr if np.isfinite(iqr) else None,
                'uniform_proxy_Bowley':bow,'Bowley_eligible':eligible,'Bowley_AE_vs_raw_QC':abs(bow-raw['Bowley']) if eligible else None}

        records=[]; light=[]
        for c in chosen:
            sid=c['sample_id']; index=int(positions[sid]); assert int(meta.iloc[index].fold)==0
            assert c['metadata']['sequence_sha256']==meta.iloc[index].sequence_sha256
            y=truth[index]; assert np.array_equal(y,np.asarray(c['true_p'],np.float32))
            counts=np.asarray(c['counts']); assert np.issubdtype(counts.dtype,np.integer) and np.array_equal((counts/counts.sum()).astype(np.float32),y)
            assert int(counts.sum())==int(c['raw_QC']['n_vertices'])==int(c['raw_QC']['n_rows'])
            assert raw_full.loc[sid,'sequence_sha256']==meta.iloc[index].sequence_sha256
            raw={k:float(raw_full.loc[sid,k+'_raw_QC']) for k in ['q25','q50','q75']}
            assert all(abs(raw[k]-c['raw_QC'][k])<=1e-9 for k in raw)  # Earlier compact packet rounded metadata to 10 decimal places.
            raw['IQR']=raw['q75']-raw['q25']; raw['Bowley']=(raw['q25']+raw['q75']-2*raw['q50'])/raw['IQR']
            assert raw['IQR']==raw_full.loc[sid,'raw_IQR'] and raw['Bowley']==raw_full.loc[sid,'raw_Bowley']
            rowpos=int(np.flatnonzero(ids==index)[0]); dy=describe(y,y,raw); modeled={k:describe(v[rowpos],y,raw) for k,v in preds.items()}
            # Recheck the earlier exactly declared window values, without reselecting or tuning it.
            for k in ['lower','upper','uniform_proxy']: assert abs(dy['window20_40_mass'][k]-c['truth63']['window20_40_mass'][k])<=1e-15
            for v in modeled.values():
                t=dy['window20_40_mass']; p=v['window20_40_mass']
                v['truth_minus_prediction_mass_interval']=[t['lower']-p['upper'],t['upper']-p['lower']]
                v['guaranteed_positive_mass_deficit']=max(0.,t['lower']-p['upper'])
                v['uniform_proxy_mass_deficit']=t['uniform_proxy']-p['uniform_proxy']
            # The true peak coordinates are fixed once from truth; report model density there too.
            for v in dy['unthresholded_finite_interior_local_peaks']:
                jj=int(v['left_bin']); v['model_density_at_truth_peak_leftmost_bin']={k:float(pred[rowpos,1+jj]/np.diff(edges)[jj]) for k,pred in preds.items()}
            item={'case_alias':'case06' if c['role']=='frozen_badcase' else 'normal06','role':c['role'],'evaluation_fold':0,'raw_QC':raw,'truth63':dy,'models':modeled,
                'CE_minus_H2_guaranteed_deficit':modeled['CE_quarter']['guaranteed_positive_mass_deficit']-modeled['H2']['guaranteed_positive_mass_deficit'],
                'CE_minus_H2_uniform_proxy_deficit':modeled['CE_quarter']['uniform_proxy_mass_deficit']-modeled['H2']['uniform_proxy_mass_deficit']}
            light.append(item); records.append({'sample_id':sid,'metadata':c['metadata'],'true_p':y.tolist(),'predicted_p':{k:v[rowpos].tolist() for k,v in preds.items()},**item})
        put(out/'private_case_packet.json',{'cases':records,'selected_runs':runs,'original_case_packet_sha256':oldfacts['private_packet_sha256']})
        put(out/'identity.json',{'status':'passed','refs':refs,'full_evaluation_probability_replay':replays,'no_checkpoint_load':True,'no_raw_phi_read':True,'roles_exact_and_fold0_only':True})
        result={'schema':'fixed_case_loss_addendum_v1','status':'complete','privacy':'Private diagnostic case facts despite anonymous aliases; not a public aggregate or Git artifact.','cases':light,'selected_epochs':{x['loss']:x['selected_epoch'] for x in runs},'promotion_unchanged':True,'promote_to_one_paired_confirmation':False,
            'scope':'Posthoc descriptive comparison of frozen case06 and its already-selected normal06; not independent testing, no reselection or promotion-rule change. case03 fold2 is outside this probe and omitted.',
            'definitions':{'window':'Fixed postvisual [20,40]: fully-contained-bin lower and intersecting-bin upper; uniform proxy separately labeled. Stored FP32 PMF masses are used without probability renormalization for window/CDF/H; quantile helper normalizes as in the original shape comparison.',
                'peaks':'Same finite-density interior peak and plateau handling as prior facts; zero relative-prominence descriptive listing, not major-mode reclassification. Main peak is leftmost argmax plateau at relative1e-12 tolerance.',
                'quantiles':'Uniform within finite bins inverse-CDF proxy, not raw type7; open-tail hits cannot be point estimated. Bowley requires all finite quantiles and IQR>=1.',
                'interpretation':'A changed interval lower bound is not an exact continuous-phi change. Overlapping predicted window intervals cannot establish the exact change without within-bin information.'},
            'identity_sha256':sha(out/'identity.json'),'private_packet_sha256':sha(out/'private_case_packet.json'),'code_sha256':sha(__file__),'plan_sha256':sha(a.plan),
            'resources':{'CPU':[174],'threads':1,'AS_GiB':8,'wall_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},'new_fits':0,'models_loaded':0,'raw_phi_reads':0}
        put(out/'anonymous_summary.json',result)
        print(json.dumps({'status':'complete','summary_sha256':sha(out/'anonymous_summary.json'),'wall_seconds':result['resources']['wall_seconds']}))
    except Exception as exc:
        put(out/'failure.json',{'status':'failed_preserved','error':str(exc),'traceback':traceback.format_exc(),'refs_so_far':refs}); raise


if __name__=='__main__':main()
