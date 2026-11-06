"""Apply the frozen probe decision rules to two valid paired fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text())


def put(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    assert __debug__ and os.sched_getaffinity(0)=={172} and not os.environ.get('CUDA_VISIBLE_DEVICES')
    resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
    import numpy as np
    import pandas as pd
    start=time.monotonic();r=a.root;plan=load(r/'plan.json');out=r/'analysis'
    assert not out.exists();out.mkdir()
    valid=load(r/'paired_path_checks.json');assert valid['status']=='passed' and valid['plan_sha256']==sha(r/'plan.json')
    supp=r.parent;helper=supp/'code/bridge_shape_model_comparison.py'
    assert sha(helper)=='4be007ff1b869a4a71b800713f3e833f79b04770f7f27dd66aeb22507da16ce6'
    sys.path.insert(0,str(helper.parent))
    from bridge_shape_model_comparison import probability_errors,uniform_quantiles,macro,confusion,rank_one_percent
    assert sha(supp/'code/bridge_tail_fraction_bounds.py')=='c947afb5920ec89cfd964cb03e1fa63676dfcb622887e1b9cdf0b544415aab00'
    from bridge_tail_fraction_bounds import fraction_bounds
    data=Path(plan['parent'])/'data';release=load(data/'release.json');assert sha(data/'release.json')==plan['release_sha256']
    assert sha(data/'samples.parquet')==release['artifacts']['samples.parquet'];m=pd.read_parquet(data/'samples.parquet')
    guard=Path(plan['roles_root'])/'code/bridge_roles_guard.py';assert sha(guard)==plan['guard_code_sha256']
    sys.path.insert(0,str(guard.parent));from bridge_roles_guard import check
    request={'stage':'development','sequence_sha256':sorted(m.sequence_sha256.unique().tolist()),'aliases':[]}
    put(out/'role_preflight.json',[check(Path(plan['roles_root'])/'registry_v1/guard_contract.json',plan['guard_contract_sha256'],{'operation':op,**request}) for op in ['read_target','model_eval']])
    assert sha(data/'p.npy')==release['artifacts']['p.npy'];y=np.load(data/'p.npy')
    counts_path=Path(plan['parent']).parent/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet'
    assert sha(counts_path)=='71f3cb95b348af64019ca10deab39f9380585fd43b150f0598cfbbedf3ceab1e'
    count_frame=pd.read_parquet(counts_path).set_index('sample_id').loc[m.sample_id]
    counts=count_frame[[f'bin_{i:03d}' for i in range(63)]].to_numpy()
    assert np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and np.array_equal((counts/counts.sum(1)[:,None]).astype(np.float32),y)
    binding=load(data/'label_contract_addendum.json');assert sha(binding['label_contract_path'])==binding['label_contract_sha256'];edges=np.asarray(binding['label_contract']['edges'])
    rep=supp/'representation_audit_v1';assert sha(rep/'summary.json')=='995a2e7b6502226ea39540a752b0574af0cf498d2b90919732447cf6f87ad41f'
    rs=load(rep/'summary.json');assert sha(rep/'per_sample.parquet')==rs['private_table']['sha256']
    raw=pd.read_parquet(rep/'per_sample.parquet').set_index('sample_id').loc[m.sample_id].reset_index()
    assert np.array_equal(raw.sequence_sha256,m.sequence_sha256)
    runs={}
    for p in (r/'runs').glob('*/receipt.json'):
        rec=load(p);assert rec['status']=='complete';loss=rec['config']['loss'];assert loss not in runs
        assert sha(p)==valid['H2_receipt_sha256' if loss=='H2' else 'CE_receipt_sha256'];runs[loss]=(p.parent,rec)
    assert set(runs)=={'H2','CE_quarter'}
    results=[];frames={};inputs=[];partition_ids={};observed_trajectories=[]
    columns=['H','CDF_MSE','edge_MAE','Q50_AE','IQR_AE','Bowley_AE','negative_edge_bias','positive_edge_bias','negative_edge_AE','positive_edge_AE']
    for loss,(run,rec) in runs.items():
        assert sha(run/'fixed_best_partitions.json')==rec['fixed_best_partitions_sha256'];partitions=load(run/'fixed_best_partitions.json')
        curve=load(run/'curve.json');assert len(curve)==134
        observed_trajectories.append({'loss':loss,'selected_epoch':rec['best_epoch'],'selected_stop_H':rec['best_stop_H'],
            'best_at_final_epoch':rec['best_epoch']==134,
            'last10_minus_previous10_stop_CE':float(np.mean([x['stop_CE'] for x in curve[-10:]])-np.mean([x['stop_CE'] for x in curve[-20:-10]])),
            'last10_minus_previous10_stop_H':float(np.mean([x['stop_H'] for x in curve[-10:]])-np.mean([x['stop_H'] for x in curve[-20:-10]])),
            'mean_clip_fraction':float(np.mean([x['clip_fraction'] for x in curve])),
            'CE_floor_count_max':max(x['stop_CE_floor_count'] for x in curve),
            'gradient_norm_median_by_LR_stage':{str(k):float(np.median([x['gradient_norm_median'] for x in curve if x['LR_stage']==k])) for k in [0,1,2]}})
        for part,pins in partitions.items():
            pp=run/(part+'_prediction.npy');ip=run/(part+'_source_indices.npy')
            assert sha(pp)==pins['prediction_sha256'] and sha(ip)==pins['indices_sha256']
            pred=np.load(pp);ids=np.load(ip)
            if part in partition_ids:assert np.array_equal(ids,partition_ids[part])
            partition_ids[part]=ids
            assert pred.shape==(len(ids),63) and np.isfinite(pred).all() and (pred>=0).all() and abs(pred.sum(1)-1).max()<1e-6
            rw=raw.iloc[ids].reset_index(drop=True);meta=m.iloc[ids].reset_index(drop=True)
            f=meta[['sample_id','sequence_sha256','group_id']].copy()
            for k,v in probability_errors(y[ids],pred).items():f[k]=v
            q,b=uniform_quantiles(pred,edges);iqr=q[:,2]-q[:,0];eligible=np.isfinite(q).all(1)&(iqr>=1)
            s=np.full(len(ids),np.nan);s[eligible]=(q[eligible,0]+q[eligible,2]-2*q[eligible,1])/iqr[eligible]
            negative=rw.raw_negative_absolute_shape.to_numpy(bool);positive=rw.raw_positive_absolute_shape.to_numpy(bool)
            f['predicted_Bowley']=s;f['Bowley_eligible']=eligible;f['raw_Bowley']=rw.raw_Bowley.to_numpy()
            f['Q50_AE']=abs(q[:,1]-rw.q50_raw_QC.to_numpy());f['IQR_AE']=abs(iqr-rw.raw_IQR.to_numpy());f['Bowley_AE']=abs(s-rw.raw_Bowley.to_numpy())
            f['negative_edge_bias']=pred[:,0]-y[ids,0];f['positive_edge_bias']=pred[:,-1]-y[ids,-1]
            f['negative_edge_AE']=abs(f.negative_edge_bias);f['positive_edge_AE']=abs(f.positive_edge_bias)
            metric=macro(f,columns)
            for key,old,tol in [('H','group_macro_H',1e-7),('CDF_MSE','group_macro_cutpoint_CDF_MSE',1e-8),('edge_MAE','group_macro_edge_bin_mass_MAE',1e-8)]:assert abs(metric[key]['group_macro']-pins['metrics'][old])<=tol
            item={'loss':loss,'partition':part,'rows':len(meta),'unique_T':int(meta.sequence_sha256.nunique()),'groups':int(meta.group_id.nunique()),
                'metrics':metric,'common_CE':pins['metrics']['group_macro_CE'],
                'negative':confusion(meta,negative,eligible&(s<=-.30),eligible),
                'positive':confusion(meta,positive,eligible&(s>=.30),eligible),
                'strong_shape':{name:macro(f.loc[mask],['H','Bowley_AE']) for name,mask in [('union',negative|positive),('negative',negative),('positive',positive)]}}
            item['fixed_one_percent_ranking'],_=rank_one_percent(f,negative,positive)
            item['directional_mass_concentration']={}
            def count_support(mask):
                z=meta.loc[mask];return {'rows':len(z),'unique_T':int(z.sequence_sha256.nunique()),'groups':int(z.group_id.nunique())}
            for side,interval in [('negative',(-np.inf,-5.)),('positive',(5.,np.inf))]:
                tl,tu=fraction_bounds(counts[ids],edges,interval);pl,pu=fraction_bounds(pred,edges,interval)
                tpos=tl>=.8;tneg=tu<.8;unknown=~(tpos|tneg);ppos=pl>=.8;pneg=pu<.8
                item['directional_mass_concentration'][side]={
                    'definition':'Whole-support mass beyond +/-5 >=0.8, original independent concentration flag; not complete historical source-percentile extreme rule.',
                    'certified_truth_positive':count_support(tpos),'unresolved_truth':count_support(unknown),
                    'certified_prediction_positive':count_support(ppos),'prediction_deferred':count_support(~(ppos|pneg)),
                    'TP_on_certified_truth':count_support(tpos&ppos),'FP_on_certified_truth':count_support(tneg&ppos),
                    'FN_on_certified_truth_including_defer':count_support(tpos&~ppos),'unknown_truth_called':count_support(unknown&ppos)}
            results.append(item);frames[(loss,part)]=f
            inputs.append({'loss':loss,'partition':part,'prediction_sha256':sha(pp),'indices_sha256':sha(ip)})
    by={(x['loss'],x['partition']):x for x in results};decisions=[]
    def value(item,key):return item['metrics'][key]['group_macro']
    for part in ['stop','evaluation']:
        h=by[('H2',part)];c=by[('CE_quarter',part)]
        bh=h['strong_shape']['union']['Bowley_AE']['group_macro'];bc=c['strong_shape']['union']['Bowley_AE']['group_macro']
        shape_ok=bc is not None and bh is not None and bc<=.9*bh
        tail_ok=abs(value(c,'negative_edge_bias'))<=.8*abs(value(h,'negative_edge_bias')) and value(c,'negative_edge_AE')<value(h,'negative_edge_AE')
        distribution_ok=value(c,'H')<=1.01*value(h,'H') and value(c,'CDF_MSE')<=1.01*value(h,'CDF_MSE')
        hr=sum(h['fixed_one_percent_ranking'][side]['true_positive_T'] for side in ['negative','positive'])
        cr=sum(c['fixed_one_percent_ranking'][side]['true_positive_T'] for side in ['negative','positive'])
        decisions.append({'partition':part,'rare_shape_10pct_improvement':shape_ok,'tail_bias20pct_and_tail_AE_improvement':tail_ok,
            'H_CDF_within1pct_guardrail':distribution_ok,'combined_fixed_budget_TP_not_lower':cr>=hr,
            'H2_combined_TP':hr,'CE_combined_TP':cr,'pass':bool((shape_ok or tail_ok) and distribution_ok and cr>=hr)})
    pd.concat([f.assign(loss=k[0],partition=k[1]) for k,f in frames.items()]).to_parquet(out/'private_per_sample.parquet',index=False)
    summary={'status':'complete','schema':'fixed_path_shape_loss_probe_analysis_v1','results':results,'trajectory_diagnostics':observed_trajectories,
        'frozen_promotion_decisions':decisions,'promote_to_one_paired_confirmation':all(d['pass'] for d in decisions),
        'validity_receipt_sha256':sha(r/'paired_path_checks.json'),'control_replay_sha256':sha(r/'H2_control_replay.json'),
        'plan_sha256':sha(r/'plan.json'),'analysis_code_sha256':sha(__file__),'input_predictions':inputs,
        'private_table_sha256':sha(out/'private_per_sample.parquet'),'elapsed_seconds':time.monotonic()-start,
        'scope':'Exposed fold0 development probe; same realized H2-derived LR path and stop-H selection. Loss intervention includes resulting gradients/AdamW trajectory. It does not isolate an asymptotic Bayes-center cause or establish CE optimality. No evaluation selection of epoch.',
        'thresholds_are_engineering_decisions_not_significance_tests':True}
    put(out/'summary.json',summary)
    print(json.dumps({'status':'complete','promote':summary['promote_to_one_paired_confirmation'],'summary_sha256':sha(out/'summary.json')}))


if __name__=='__main__':main()
