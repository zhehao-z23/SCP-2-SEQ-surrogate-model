"""Matched continuous-shape probes of fixed model features; all tuning uses stop only."""
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


def load(p):return json.loads(Path(p).read_bytes())


def put(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True)
    ap.add_argument('--code-sha256',required=True);a=ap.parse_args()
    assert __debug__ and sha(__file__)==a.code_sha256 and sha(a.plan)==a.plan_sha256
    assert sorted(os.sched_getaffinity(0))==[172,173] and not os.environ.get('CUDA_VISIBLE_DEVICES')
    resource.setrlimit(resource.RLIMIT_AS,(12<<30,12<<30))
    import numpy as np
    import pandas as pd
    start=time.monotonic();out=a.root/'linear_probes_v1';assert not out.exists();out.mkdir()
    put(out/'started.json',{'PID':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'source_sha256':sha(__file__),'plan_sha256':sha(a.plan),'new_backbone_training':False})
    froot=a.root/'features_v1';fs=load(froot/'summary.json');assert fs['status']=='complete' and fs['plan_sha256']==a.plan_sha256
    assert not fs['model_weights_changed'] and all(x['prediction_max_abs_difference']<=1e-6 for x in fs['checks'].values())
    assert sha(froot/'private_meta.parquet')==fs['private_meta_sha256'];meta=pd.read_parquet(froot/'private_meta.parquet')
    base=Path(os.environ['SCP_DATA_ROOT']);bridge=base/'bridge_20261106';supp=bridge/'diagnostic_supplements';data=base/'atlas_scale_20261105/data'
    guard=bridge/'roles';assert sha(guard/'code/bridge_roles_guard.py')=='5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'
    sys.path.insert(0,str(guard/'code'));from bridge_roles_guard import check
    req={'stage':'development','sequence_sha256':sorted(meta.sequence_sha256.unique()),'aliases':[]}
    put(out/'role_preflight.json',[check(guard/'registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':op,**req}) for op in ['read_target','train','stop','model_select','model_eval']])
    helper=supp/'code/bridge_shape_model_comparison.py';assert sha(helper)=='4be007ff1b869a4a71b800713f3e833f79b04770f7f27dd66aeb22507da16ce6'
    sys.path.insert(0,str(helper.parent));from bridge_shape_model_comparison import uniform_quantiles,macro,confusion,rank_one_percent
    assert sha(data/'release.json')=='67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b';release=load(data/'release.json')
    assert sha(data/'p.npy')==release['artifacts']['p.npy'];p=np.load(data/'p.npy')
    assert sha(data/'samples.parquet')==release['artifacts']['samples.parquet'];original=pd.read_parquet(data/'samples.parquet')
    assert np.array_equal(meta.sample_id,original.sample_id) and np.array_equal(meta.sequence_sha256,original.sequence_sha256)
    binding=load(data/'label_contract_addendum.json');assert sha(binding['label_contract_path'])==binding['label_contract_sha256'];edges=np.array(binding['label_contract']['edges'])
    rep=supp/'representation_audit_v1';assert sha(rep/'summary.json')=='995a2e7b6502226ea39540a752b0574af0cf498d2b90919732447cf6f87ad41f'
    rs=load(rep/'summary.json');assert sha(rep/'per_sample.parquet')==rs['private_table']['sha256']
    raw=pd.read_parquet(rep/'per_sample.parquet').set_index('sample_id').loc[meta.sample_id].reset_index()
    assert np.array_equal(raw.sequence_sha256,meta.sequence_sha256)
    def array(name):
        path=froot/(name+'.npy');assert sha(path)==fs['outputs'][name]['sha256'];return np.load(path)
    parts={name:array(name+'_indices') for name in ['train','stop','evaluation']};tr=parts['train'];va=parts['stop']
    assert np.array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(len(meta)))
    for left,right in [('train','stop'),('train','evaluation'),('stop','evaluation')]:
        assert not set(meta.iloc[parts[left]].group_id)&set(meta.iloc[parts[right]].group_id)
        assert not set(meta.iloc[parts[left]].sequence_sha256)&set(meta.iloc[parts[right]].sequence_sha256)
    raw_target=raw.raw_Bowley.to_numpy(float);q,_=uniform_quantiles(p,edges)
    binned_target=(q[:,0]+q[:,2]-2*q[:,1])/(q[:,2]-q[:,0]);assert np.isfinite(raw_target).all() and np.isfinite(binned_target).all()
    probability=array('predicted_PMF');pq,_=uniform_quantiles(probability,edges)
    baseline=(pq[:,0]+pq[:,2]-2*pq[:,1])/(pq[:,2]-pq[:,0]);eligible=np.isfinite(pq).all(1)&((pq[:,2]-pq[:,0])>=1)
    assert eligible.all()  # Recorded cohort fact, not use of raw eligibility at prediction time.
    negative=raw.raw_negative_absolute_shape.to_numpy(bool);positive=raw.raw_positive_absolute_shape.to_numpy(bool);strong=negative|positive
    weights=1/meta.iloc[tr].sequence_sha256.map(meta.iloc[tr].sequence_sha256.value_counts()).to_numpy(float)
    weights/=weights.mean();sumw=weights.sum()
    def wmean(v):return np.sum(v*weights[:,None],axis=0)/sumw if v.ndim==2 else np.sum(v*weights)/sumw
    def gm(v,ids,mask=None):
        z=meta.iloc[ids][['group_id','sequence_sha256']].copy();z['v']=np.asarray(v)
        if mask is not None:z=z.loc[np.asarray(mask)]
        return None if len(z)==0 else float(z.groupby(['group_id','sequence_sha256']).v.mean().groupby('group_id').mean().mean())
    targets=np.stack([raw_target,binned_target],1);ym=wmean(targets[tr]);yc=targets[tr]-ym
    predictions={'PMF_uniform_Bowley':baseline.copy(),'constant_train_mean':np.full(len(meta),ym[0])}
    x=baseline[tr];xm=wmean(x);cov=wmean((x-xm)*(raw_target[tr]-ym[0]));var=wmean((x-xm)**2)
    slope=max(0.,float(cov/var)) if var>0 else 0.;intercept=float(ym[0]-slope*xm)
    predictions['affine_scalar_raw']=intercept+slope*baseline
    controls={'affine_slope':slope,'affine_intercept':intercept,'affine_nonnegative_slope':True,
        'rank_invariance_expected_if_positive_slope':slope>0,'constant_has_no_ranking_information':True}
    chosen=[];coefficients={};lambdas=[.0001,.01,1.]
    for feature in ['charge','input_mean_charge','hidden256','predicted_PMF']:
        feature_start=time.monotonic();X=array(feature).astype('float64');xm=wmean(X[tr]);center=X[tr]-xm
        xs=np.sqrt(wmean(center**2));active=xs>1e-8;xs=np.where(active,xs,1.)
        X=(X-xm)/xs;X[:,~active]=0.;Xt=X[tr]
        C=(Xt.T*weights)@Xt/sumw;D=(Xt.T*weights)@yc/sumw
        eigen,U=np.linalg.eigh(C);assert eigen.min()>-1e-8;eigen=np.maximum(eigen,0)
        rotated=U.T@D;candidates=[]
        for lam in lambdas:
            coef=U@(rotated/(eigen[:,None]+lam));stop_score=X[va]@coef[:,0]+ym[0]
            selection=gm((np.clip(stop_score,-1,1)-raw_target[va])**2,va)
            residual=float(np.max(abs((C+lam*np.eye(C.shape[0]))@coef-D)))
            assert residual<1e-8
            candidates.append({'lambda':lam,'stop_raw_group_MSE':selection,'normal_equation_residual':residual,'coef':coef})
        best=min(candidates,key=lambda z:(z['stop_raw_group_MSE'],-z['lambda']))
        scores=X@best['coef']+ym
        for i,target in enumerate(['raw','binned']):predictions[feature+'__'+target]=scores[:,i]
        np.savez(out/(feature+'_private_fit.npz'),coefficient=best['coef'],feature_mean=xm,feature_std=xs,active=active,target_mean=ym,lambda_value=best['lambda'])
        chosen.append({'feature':feature,'dimension':X.shape[1],'active_dimensions':int(active.sum()),'lambda':best['lambda'],
            'stop_candidates':[{k:v for k,v in c.items() if k!='coef'} for c in candidates],
            'selection_target':'raw Bowley stop group-MSE; same chosen lambda reused for binned-target intervention',
            'fits_coefficient_vectors':6,'wall_seconds':time.monotonic()-feature_start,'coefficient_archive_sha256':sha(out/(feature+'_private_fit.npz'))})
        assert time.monotonic()-start<600
    result=[];private=[]
    for name,scores in predictions.items():
        clipped=np.clip(scores,-1,1)
        for part,ids in parts.items():
            m=meta.iloc[ids].reset_index(drop=True);rawp=raw_target[ids]
            f=m[['sample_id','sequence_sha256','group_id']].copy();f['Bowley_AE']=abs(clipped[ids]-rawp);f['Bowley_MSE']=(clipped[ids]-rawp)**2
            f['predicted_Bowley']=clipped[ids];f['score_before_clip']=scores[ids];f['Bowley_eligible']=eligible[ids]
            f['raw_Bowley']=rawp;f['true63_uniform_Bowley']=binned_target[ids]
            item={'method':name,'partition':part,'rows':len(ids),'unique_T':int(m.sequence_sha256.nunique()),'groups':int(m.group_id.nunique()),
                'overall':macro(f,['Bowley_AE','Bowley_MSE']),
                'strong':{label:macro(f.loc[mask[ids]],['Bowley_AE','Bowley_MSE']) for label,mask in [('union',strong),('negative',negative),('positive',positive)]},
                'negative':confusion(m,negative[ids],eligible[ids]&(clipped[ids]<=-.30),eligible[ids]),
                'positive':confusion(m,positive[ids],eligible[ids]&(clipped[ids]>=.30),eligible[ids]),
                'clipped_rows':int((abs(scores[ids])>1).sum()),'distinct_scores':int(np.unique(scores[ids]).size)}
            if np.unique(scores[ids]).size==1:item['ranking']=None;item['ranking_reason']='constant score: no informative ordering'
            else:
                item['ranking'],_=rank_one_percent(f.assign(predicted_Bowley=scores[ids]),negative[ids],positive[ids])
                item['ranking']['score']='preclip continuous score; threshold/value metrics use clipped [-1,1] estimate'
            result.append(item);private.append(f.assign(method=name,partition=part))
    by={(x['method'],x['partition']):x for x in result};decisions=[]
    controls['affine_exact_rank_order_by_partition']={}
    if slope>0:
        for part,ids in parts.items():
            canonical=meta.iloc[ids].sort_values('sample_id').drop_duplicates('sequence_sha256')
            same=True
            for ascending in [True,False]:
                before=canonical.assign(score=baseline[canonical.index]).sort_values(['score','sample_id'],ascending=[ascending,True]).index.to_numpy()
                after=canonical.assign(score=predictions['affine_scalar_raw'][canonical.index]).sort_values(['score','sample_id'],ascending=[ascending,True]).index.to_numpy()
                same=same and np.array_equal(before,after)
            assert same;controls['affine_exact_rank_order_by_partition'][part]=bool(same)
    for name in predictions:
        if name in ['PMF_uniform_Bowley','constant_train_mean']:continue
        checks=[]
        for part in ['stop','evaluation']:
            b=by[('PMF_uniform_Bowley',part)];c=by[(name,part)]
            counts=lambda it:None if it['ranking'] is None else sum(it['ranking'][side]['true_positive_T'] for side in ['negative','positive'])
            bc=counts(b);cc=counts(c)
            checks.append({'partition':part,'rare_AE_reduction10pct':c['strong']['union']['Bowley_AE']['group_macro']<=.9*b['strong']['union']['Bowley_AE']['group_macro'],
                'overall_MSE_within5pct':c['overall']['Bowley_MSE']['group_macro']<=1.05*b['overall']['Bowley_MSE']['group_macro'],
                'ranking_not_lower':cc is not None and cc>=bc,'TP_delta':None if cc is None else cc-bc})
        passes=all(c['rare_AE_reduction10pct'] and c['overall_MSE_within5pct'] and c['ranking_not_lower'] for c in checks) and any(c['TP_delta'] is not None and c['TP_delta']>=2 for c in checks)
        eligible_candidate=name.endswith('__raw') or name=='affine_scalar_raw'
        decisions.append({'method':name,'checks':checks,'candidate_eligible':eligible_candidate,
            'promising_for_strong_control_comparison':bool(passes and eligible_candidate)})
    pd.concat(private,ignore_index=True).to_parquet(out/'private_predictions.parquet',index=False)
    summary={'status':'complete','schema':'frozen_feature_shape_probe_results.v1','plan_sha256':sha(a.plan),'code_sha256':sha(__file__),
        'feature_summary_sha256':sha(froot/'summary.json'),'controls':controls,'ridge_selection':chosen,'results':result,'decisions':decisions,
        'private_prediction_sha256':sha(out/'private_predictions.parquet'),'process_seconds':time.monotonic()-start,
        'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_backbone_fits':0,'ridge_coefficient_vectors_fitted':24,
        'PMF_unchanged':True,'eligibility':'Inherited finite baseline-PMF quartiles and predicted IQR>=1; all current rows eligible; raw eligibility used only for truth.',
        'scope':'Fold0/seed1701 exposed development model. Existing stop-H checkpoint selection precedes stop-only ridge tuning. No final independent test or probability-calibration guarantee. Failed linear probes do not imply absence of information.',
        'target_intervention':'Within each feature source both target columns use identical X/scaler/lambda/weights; raw target chose lambda on stop, binned target reuses it. This asymmetric selection favors the intended raw-target task and is explicit.'}
    put(out/'summary.json',summary)
    print(json.dumps({'status':'complete','summary_sha256':sha(out/'summary.json'),'process_seconds':summary['process_seconds'],'decisions':decisions}))


if __name__=='__main__':main()
