"""Compare the historical 0.8 concentration flag using interval-valid masses.

This is not the historical source-percentile extreme rule. Conditional-window
fractions are counterfactual representations of this Atlas field, not AF labels.
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
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text())


def put(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert __debug__ and os.sched_getaffinity(0)=={172} and not os.environ.get('CUDA_VISIBLE_DEVICES')
    resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
    import numpy as np
    import pandas as pd
    helper=Path(__file__).with_name('bridge_tail_fraction_bounds.py')
    assert sha(helper)=='c947afb5920ec89cfD964cb03e1fa63676dfcb622887e1b9cdf0b544415aab00'.lower()
    from bridge_tail_fraction_bounds import fraction_bounds,selftest
    start=time.monotonic();root=a.root;supp=root/'bridge_20261106/diagnostic_supplements'
    assert not a.output.exists();a.output.mkdir(parents=True);refs=[]
    def bind(p,pin=None):
        h=sha(p)
        if pin:assert h==pin,str(p)
        refs.append({'path':str(p),'sha256':h})
    put(a.output/'selftest.json',selftest())
    data=root/'atlas_scale_20261105/data'
    bind(data/'release.json','67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b')
    release=load(data/'release.json');bind(data/'samples.parquet',release['artifacts']['samples.parquet'])
    m=pd.read_parquet(data/'samples.parquet',columns=['sample_id','sequence_sha256','group_id','fold'])
    roles=root/'bridge_20261106/roles';bind(roles/'code/bridge_roles_guard.py','5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3')
    sys.path.insert(0,str(roles/'code'));from bridge_roles_guard import check
    request={'stage':'development','sequence_sha256':sorted(m.sequence_sha256.unique().tolist()),'aliases':[]}
    put(a.output/'role_preflight.json',[check(roles/'registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':op,**request}) for op in ['read_target','model_eval']])
    bind(data/'p.npy',release['artifacts']['p.npy']);y=np.load(data/'p.npy')
    counts_path=root/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet'
    bind(counts_path,'71f3cb95b348af64019ca10deab39f9380585fd43b150f0598cfbbedf3ceab1e')
    count_frame=pd.read_parquet(counts_path).set_index('sample_id').loc[m.sample_id]
    counts=count_frame[[f'bin_{i:03d}' for i in range(63)]].to_numpy()
    assert np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and (counts.sum(1)>0).all() and (counts.sum(1)<2**53).all()
    assert np.array_equal((counts/counts.sum(1)[:,None]).astype(np.float32),y)
    binding=load(data/'label_contract_addendum.json');bind(binding['label_contract_path'],binding['label_contract_sha256']);edges=np.asarray(binding['label_contract']['edges'])
    old=supp/'shape_model_comparison_v1';bind(old/'summary.json','675c87743ed787135fa99b1d25c98ebfa0fa1bb427a06681c96f5b54b7e292cd')
    summary=load(old/'summary.json');bind(old/'identity_and_prediction_replay.json',summary['identity_sha256'])
    identity=load(old/'identity_and_prediction_replay.json')
    fitpins={(x['family'],x['seed'],x['fold']):x for x in identity['fit_checks']}
    fitpath=root/'overnight_20261106/confirmation/results/all_fits.csv';bind(fitpath,'224251b6320031487be1514222d8cea3a9d0201f6dfea4ddfbdc1d9e1eeff9ef');fits=pd.read_csv(fitpath)
    index=pd.Series(np.arange(len(m)),index=m.sample_id); predictions={}; seen={}
    for f in fits.to_dict('records'):
        key=(f['family'],int(f['seed']));pin=fitpins[(*key,int(f['fold']))];run=Path(f['artifact_path']);run.relative_to(root)
        bind(run/'roles.json',pin['roles_sha256']);r=load(run/'roles.json');ids=index.loc[r['evaluation_ids']].to_numpy()
        assert np.array_equal(np.sort(ids),np.flatnonzero(m.fold.to_numpy()==f['fold']))
        bind(run/'evaluation_prediction.npy',pin['prediction_sha256']);p=np.load(run/'evaluation_prediction.npy')
        predictions.setdefault(key,np.empty_like(y))[ids]=p;seen.setdefault(key,np.zeros(len(m),int))[ids]+=1
    assert len(predictions)==6 and all((v==1).all() for v in seen.values())
    def support(mask):
        z=m.loc[mask];return {'rows':len(z),'unique_T':int(z.sequence_sha256.nunique()),'groups':int(z.group_id.nunique())}
    def categories(lo,hi):return np.where(lo>=.8,1,np.where(hi<.8,-1,0))
    def compare(truth,pred):
        called=pred==1;knownpositive=truth==1;knownnegative=truth==-1;unknown=truth==0
        aa=int((called&knownpositive).sum());bb=int((~called&knownpositive).sum());cc=int((called&unknown).sum());dd=int((~called&unknown).sum());fp=int((called&knownnegative).sum());calls=int(called.sum())
        return {'called_positive':support(called),'deferred_prediction':support(pred==0),
            'known_truth_TP':support(called&knownpositive),'known_truth_FP':support(called&knownnegative),
            'known_truth_FN_including_defer':support(~called&knownpositive),
            'positive_truth_prediction_deferred':support((pred==0)&knownpositive),
            'unknown_truth_called':support(called&unknown),
            'row_precision_interval':[aa/calls,(aa+cc)/calls] if calls else None,
            'row_recall_interval':[aa/(aa+bb+dd) if aa+bb+dd else None,(aa+cc)/(aa+bb+cc) if aa+bb+cc else None],
            'interval_scope':'All allocations within the observed bins; unknown truths retained; undefined ratios remain null.'}
    outputs=[];tables=[]
    for window,den in [('whole_support',(-np.inf,np.inf)),('conditional_minus60_plus60',(-60.,60.))]:
        for side,num in [('negative',(den[0],-5.)),('positive',(5.,den[1]))]:
            lo,hi=fraction_bounds(counts,edges,num,den);truth=categories(lo,hi)
            oldlo,oldhi=fraction_bounds(y,edges,num,den)
            item={'window':window,'side':side,'threshold_mass':.8,
                'truth_support':{name:support(truth==value) for name,value in [('certified_positive',1),('certified_negative',-1),('unresolved',0)]},
                'truth_mass_interval_width_quantiles':dict(zip(['min','p25','median','p75','max'],map(float,np.quantile(hi-lo,[0,.25,.5,.75,1])))),
                'fp32_vs_exact_count_sensitivity':{'max_interval_endpoint_difference':float(max(abs(lo-oldlo).max(),abs(hi-oldhi).max())),
                    'truth_concentration_category_changes':int((truth!=categories(oldlo,oldhi)).sum())},
                'models':[]}
            for (family,seed),p in predictions.items():
                pl,pu=fraction_bounds(p,edges,num,den);pred=categories(pl,pu)
                z={'family':family,'seed':seed,**compare(truth,pred)}
                # Interval separation is a lower bound on mass error, not an estimate of its exact value.
                gap=np.maximum.reduce([lo-pu,pl-hi,np.zeros(len(m))])
                f=m.copy();f['minimum_mass_error']=gap
                z['group_macro_minimum_mass_error']=float(f.groupby(['group_id','sequence_sha256']).minimum_mass_error.mean().groupby('group_id').mean().mean())
                z['provably_nonoverlapping_mass_intervals']=support(gap>1e-7)
                item['models'].append(z)
                f['window']=window;f['side']=side;f['family']=family;f['seed']=seed
                f['target_lower']=lo;f['target_upper']=hi;f['prediction_lower']=pl;f['prediction_upper']=pu
                f['truth_category']=truth;f['prediction_category']=pred;tables.append(f)
            outputs.append(item)
    pd.concat(tables).to_parquet(a.output/'private_per_sample.parquet',index=False)
    put(a.output/'identity.json',{'refs':refs,'source_fits':30,'source_prediction_sets':6})
    result={'status':'complete','schema':'interval_tail5_concentration_audit_v2_exact_counts','cohort':support(np.ones(len(m),bool)),
        'results':outputs,'code_sha256':sha(__file__),'bounds_helper_sha256':sha(helper),
        'identity_sha256':sha(a.output/'identity.json'),'private_table_sha256':sha(a.output/'private_per_sample.parquet'),
        'definition':'Concentration flag >=0.8; target intervals from original integer counts, with bitwise current-label normalization check and FP32 sensitivity comparison. Intervals for phi<-5 or phi>5 use no open-tail centers. Query endpoints are strictly inside bins, avoiding producer-boundary atom ambiguity.',
        'not_the_historical_extreme_rule':'Source-population percentile and old AF physical field are unavailable here. Conditional[-60,60) changes only the measure on this Atlas distribution; it does not recreate old AF computations. No claim of full extreme-screening validation.',
        'no_false_precision':'Intervals are sharp over arbitrary within-bin mass allocations, not confidence intervals or proofs of physically realizable alternatives; row/T/group presence counts may overlap for categories.',
        'new_fits':0,'model_loads':0,'raw_phi_reads':0,'elapsed_seconds':time.monotonic()-start,
        'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    put(a.output/'summary.json',result)
    print(json.dumps({'status':'complete','elapsed_seconds':result['elapsed_seconds'],'summary_sha256':sha(a.output/'summary.json')}))


if __name__=='__main__':main()
