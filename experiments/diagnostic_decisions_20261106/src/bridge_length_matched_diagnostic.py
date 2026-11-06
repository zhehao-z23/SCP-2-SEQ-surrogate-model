"""Target-matched long/short comparison; an observational mechanism check.

Tests whether long-sequence error differences persist when the target PMF,
tail mass, source, evaluation fold and near-complete coverage are comparable.
It does not identify a causal effect of length or a region's physical role.
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
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
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
    root=a.root;supp=root/'bridge_20261106/diagnostic_supplements';start=time.monotonic()
    assert not a.output.exists();a.output.mkdir(parents=True);refs=[]
    method={'long_length':'>600','short_length':'151..300','coverage':'>=0.98 and finite',
        'exact_match':['evaluation_fold','source_origin'],'target_H_max':.03,
        'negative_tail_mass_difference_max':.005,'positive_tail_mass_difference_max':.002,
        'canonical_T':'lexicographically first sample_id within T, before role/pool selection',
        'pair_selection':'Greedy long sequence_sha256 order, nearest target H then short sequence_sha256; every homology group used in at most one pair across both sides.',
        'minimum_interpretable_pairs':20,'bootstrap':'2000 pair resamples, fixed seed1701; exploratory 2.5/97.5 percentile interval, no causal claim',
        'primary':'Long minus short H for each family/seed; interaction in residue-minus-charge H gap',
        'secondary':'CDF and open-edge MAE; raw group/membership and complete pair table private only',
        'new_fits':0,'raw_phi_reads':0,'CPU':[172],'RAM_GiB':8,'timeout_seconds':600}
    put(a.output/'method_before_data.json',method)
    def bind(p,pin=None):
        h=sha(p)
        if pin:assert h==pin,str(p)
        refs.append({'path':str(p),'sha256':h})
    data=root/'atlas_scale_20261105/data';bind(data/'release.json','67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b')
    release=load(data/'release.json');bind(data/'samples.parquet',release['artifacts']['samples.parquet'])
    m=pd.read_parquet(data/'samples.parquet',columns=['sample_id','sequence_sha256','group_id','fold','length','coverage','source_origin'])
    roles=root/'bridge_20261106/roles';bind(roles/'code/bridge_roles_guard.py','5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3')
    sys.path.insert(0,str(roles/'code'));from bridge_roles_guard import check
    request={'stage':'development','sequence_sha256':sorted(m.sequence_sha256.unique().tolist()),'aliases':[]}
    put(a.output/'role_preflight.json',[check(roles/'registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',{'operation':op,**request}) for op in ['read_target','model_eval']])
    bind(data/'p.npy',release['artifacts']['p.npy']);y=np.load(data/'p.npy').astype(float);sqrt=np.sqrt(y)
    source=supp/'shape_model_comparison_v1';bind(source/'summary.json','675c87743ed787135fa99b1d25c98ebfa0fa1bb427a06681c96f5b54b7e292cd')
    old=load(source/'summary.json');table=source/'per_sample_metrics.parquet';bind(table,old['private_outputs']['per_sample_metrics']['sha256'])
    table=pd.read_parquet(table,columns=['sample_id','group_id','sequence_sha256','family','seed','H','CDF_MSE','edge_MAE'])
    canonical=m.sort_values('sample_id').drop_duplicates('sequence_sha256').copy()
    eligible=canonical[(canonical.coverage>=.98)&np.isfinite(canonical.coverage)].copy()
    eligible['origin']=eligible.source_origin.fillna('unavailable')
    longer=eligible[eligible.length>600].sort_values('sequence_sha256')
    shorter=eligible[(eligible.length>=151)&(eligible.length<=300)].sort_values('sequence_sha256')
    def support(z):return {'rows':len(z),'unique_T':int(z.sequence_sha256.nunique()),'groups':int(z.group_id.nunique())}
    used=set();pairs=[];reasons={'long_group_already_used':0,'no_available_same_source_fold_group':0,'no_target_caliper_match':0}
    for i,l in longer.iterrows():
        if l.group_id in used:reasons['long_group_already_used']+=1;continue
        candidates=shorter[(shorter.fold==l.fold)&(shorter.origin==l.origin)&~shorter.group_id.isin(used|{l.group_id})]
        if not len(candidates):reasons['no_available_same_source_fold_group']+=1;continue
        ix=candidates.index.to_numpy();dist=np.sqrt(np.maximum(0,.5*(y[i].sum()+y[ix].sum(1)-2*(sqrt[ix]@sqrt[i]))))
        mask=(dist<=.03)&(abs(y[ix,0]-y[i,0])<=.005)&(abs(y[ix,-1]-y[i,-1])<=.002)
        if not mask.any():reasons['no_target_caliper_match']+=1;continue
        z=candidates.loc[ix[mask]].copy();z['distance']=dist[mask];s=z.sort_values(['distance','sequence_sha256'],kind='stable').iloc[0];j=int(s.name)
        used|={l.group_id,s.group_id}
        pairs.append({'long_sample_id':l.sample_id,'short_sample_id':s.sample_id,'long_group':l.group_id,'short_group':s.group_id,
            'long_T':l.sequence_sha256,'short_T':s.sequence_sha256,'fold':int(l.fold),'origin':l.origin,
            'long_length':int(l.length),'short_length':int(s.length),'long_coverage':float(l.coverage),'short_coverage':float(s.coverage),
            'target_H':float(s.distance),'negative_tail_mass_difference':float(y[i,0]-y[j,0]),'positive_tail_mass_difference':float(y[i,-1]-y[j,-1])})
    pair=pd.DataFrame(pairs);assert len(used)==2*len(pair)
    if not pairs:
        put(a.output/'identity.json',{'refs':refs})
        result={'status':'complete_support_insufficient','schema':'label_matched_length_diagnostic_v1',
            'long_pool':support(longer),'short_pool':support(shorter),'pairs':0,
            'unmatched_long_reasons':reasons,'code_sha256':sha(__file__),
            'method_sha256':sha(a.output/'method_before_data.json'),'identity_sha256':sha(a.output/'identity.json'),
            'new_fits':0,'new_inference':0,'elapsed_seconds':time.monotonic()-start}
        put(a.output/'summary.json',result);print(json.dumps(result));return
    results=[];interactions=[];private=[]
    rng=np.random.default_rng(1701);indices=rng.integers(0,len(pair),(2000,len(pair))) if len(pair) else None
    def estimate(x):
        if not len(x):return {'mean':None,'interval':None}
        means=x[indices].mean(1)
        return {'mean':float(x.mean()),'median':float(np.median(x)),
            'exploratory_pair_bootstrap_95pct_interval':[float(v) for v in np.quantile(means,[.025,.975])] if len(x)>=20 else None}
    metric_arrays={}
    for seed in [1701,3407]:
        for family in ['composition_charge_deep','mean_pre','residue_mean']:
            t=table[(table.family==family)&(table.seed==seed)].set_index('sample_id')
            assert len(t)==len(m) and t.index.is_unique
            a1=t.loc[pair.long_sample_id].reset_index();b1=t.loc[pair.short_sample_id].reset_index()
            item={'family':family,'seed':seed,'pairs':len(pair),'metrics':{}}
            for k in ['H','CDF_MSE','edge_MAE']:
                x=a1[k].to_numpy();z=b1[k].to_numpy();metric_arrays[(family,seed,k)]=x-z
                item['metrics'][k]={'long_mean':float(x.mean()),'short_mean':float(z.mean()),'long_minus_short':estimate(x-z)}
            results.append(item)
            for row,x,z in zip(pairs,a1.H,b1.H):private.append({**row,'family':family,'seed':seed,'long_H':x,'short_H':z,'H_difference':x-z})
        x=metric_arrays[('residue_mean',seed,'H')]-metric_arrays[('composition_charge_deep',seed,'H')]
        interactions.append({'seed':seed,'meaning':'(residue H - charge H) long minus short','estimate':estimate(x)})
    pd.DataFrame(private).to_parquet(a.output/'private_pairs.parquet',index=False)
    put(a.output/'identity.json',{'refs':refs})
    balancing={k:dict(zip(['min','median','max'],[float(v) for v in pair[k].quantile([0,.5,1])])) for k in ['long_length','short_length','target_H','long_coverage','short_coverage','negative_tail_mass_difference','positive_tail_mass_difference']}
    result={'status':'complete','schema':'label_matched_length_diagnostic_v1','cohort':support(m),'canonical_T':support(canonical),
        'near_complete_coverage':support(eligible),'long_pool':support(longer),'short_pool':support(shorter),'pairs':len(pair),'distinct_groups_used':len(used),
        'unmatched_long_reasons':reasons,'balance':balancing,'results':results,'residue_vs_charge_interaction':interactions,
        'code_sha256':sha(__file__),'method_sha256':sha(a.output/'method_before_data.json'),'identity_sha256':sha(a.output/'identity.json'),
        'private_pair_sha256':sha(a.output/'private_pairs.parquet'),'new_fits':0,'new_inference':0,
        'limitation':'Greedy caliper matching and near-complete coverage select a restricted population. Matching on target distributions is a diagnostic standardization, not a causal adjustment. Structure/domain/family training support may still differ. Seeds reuse pairs. No sequence/context/readout perturbation was performed.',
        'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    put(a.output/'summary.json',result)
    print(json.dumps({'status':'complete','pairs':len(pair),'elapsed_seconds':result['elapsed_seconds'],'summary_sha256':sha(a.output/'summary.json')}))


if __name__=='__main__':main()
