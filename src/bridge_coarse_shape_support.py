"""Native-bin finite-window shape support, not reconstruction of raw modes."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
sys.dont_write_bytecode=True


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def put(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def coarsen(q,edges,phase):
    import numpy as np
    groups=[];i=0
    if phase==1:groups.append((0,1));i=1
    while i<len(q):groups.append((i,min(i+2,len(q))));i+=2
    mass=np.array([q[a:b].sum() for a,b in groups])
    e=np.array([edges[a] for a,b in groups]+[edges[-1]])
    assert abs(mass.sum()-q.sum())<1e-12 and len(e)==len(mass)+1
    return mass,e


def peaks(y,relative_prominence):
    """Interior flat-top peaks with finite-window topographic prominence."""
    import numpy as np
    eps=max(float(max(y)),1e-300)*1e-12;out=[];i=0
    while i<len(y):
        j=i
        while j+1<len(y) and abs(y[j+1]-y[i])<=eps:j+=1
        height=float(max(y[i:j+1]))
        if i>0 and j<len(y)-1 and height>y[i-1]+eps and height>y[j+1]+eps:
            left=height;right=height
            for k in range(i-1,-1,-1):
                if y[k]>height+eps:break
                left=min(left,float(y[k]))
            for k in range(j+1,len(y)):
                if y[k]>height+eps:break
                right=min(right,float(y[k]))
            prominence=height-max(left,right)
            if prominence+eps>=relative_prominence*float(max(y)):
                out.append({'left_bin':i,'right_bin':j,'height':height,'prominence':prominence})
        i=j+1
    return out


def view(q,edges,native_width,plan):
    import numpy as np
    mass=float(q.sum());density=q/np.diff(edges);reasons=[]
    if mass<=0:return {'class':'uncertain','reasons':['finite_mass_zero'],'finite_mass':mass,'major_peaks':[],'candidates':[],'valleys':[]}
    threshold=plan['major_basin_full_distribution_mass'];tol=plan['mass_threshold_uncertainty_tolerance']
    if mass<threshold+tol:reasons.append('finite_mass_insufficient_for_major_basin')
    maximum=float(max(density));eps=maximum*plan['density_plateau_relative_tolerance']
    edge_rising=any(density[i]>=plan['prominence_relative_to_view_max']*maximum and density[i]>=density[j]-eps for i,j in [(0,1),(-1,-2)])
    if edge_rising:reasons.append('significant_boundary_density_not_resolved_as_interior_peak')
    candidates=peaks(density,plan['prominence_relative_to_view_max']);valleys=[]
    for left,right in zip(candidates[:-1],candidates[1:]):
        lo=left['right_bin']+1;hi=right['left_bin']
        assert hi>lo
        segment=density[lo:hi];minimum=float(min(segment));v=lo+int(np.argmin(segment));last=v
        while last+1<hi and abs(density[last+1]-minimum)<=eps:last+=1
        valleys.append({'left_bin':v,'right_bin':last,'density':minimum,'mass':float(q[v:last+1].sum())})
    major=[]
    for i,peak in enumerate(candidates):
        lower_start=valleys[i-1]['right_bin']+1 if i else 0
        lower_end=valleys[i]['left_bin'] if i<len(valleys) else len(q)
        lower=float(q[lower_start:lower_end].sum())
        upper=lower+(valleys[i-1]['mass'] if i else 0)+(valleys[i]['mass'] if i<len(valleys) else 0)
        peak.update(interval=[float(edges[peak['left_bin']]),float(edges[peak['right_bin']+1])],basin_mass_lower=lower,basin_mass_upper=upper)
        if lower>=threshold+tol:major.append(peak)
        elif upper>=threshold-tol:reasons.append('major_basin_threshold_crossed_by_valley_mass_bounds')
    if not major:reasons.append('no_certified_major_basin')
    boundary_width=plan['boundary_guard_native_widths']*native_width
    if any(p['interval'][0]-edges[0]<=boundary_width+1e-10 or edges[-1]-p['interval'][1]<=boundary_width+1e-10 for p in major):
        reasons.append('major_peak_in_finite_boundary_guard_zone')
    pairs=[]
    for left,right in zip(major[:-1],major[1:]):
        minimum=float(min(density[left['right_bin']:right['left_bin']+1]))
        depth=1-minimum/min(left['height'],right['height'])
        separation=right['interval'][0]-left['interval'][1]
        supported=depth>=plan['valley_depth_relative_lower_major_peak'] and separation+1e-10>=plan['minimum_peak_interval_separation_in_native_widths']*native_width
        pairs.append({'valley_depth':depth,'minimum_interval_separation':separation,'pass':bool(supported)})
        if not supported:reasons.append('adjacent_major_pair_has_insufficient_valley_or_interval_separation')
    count=len(major)
    label='one_major_peak' if count==1 else 'two_major_peaks' if count==2 else 'multiple_major_peaks' if count>=3 else 'uncertain'
    return {'class':'uncertain' if reasons else label,'reasons':sorted(set(reasons)),'finite_mass':mass,'major_peaks':major,'major_count':count,'candidates':candidates,'valleys':valleys,'adjacent_major_pairs':pairs}


def common_intervals(views):
    if not all(v['class']!='uncertain' for v in views):return False
    if len({len(v['major_peaks']) for v in views})!=1:return False
    return all(min(v['major_peaks'][i]['interval'][1] for v in views)-max(v['major_peaks'][i]['interval'][0] for v in views)>1e-10 for i in range(len(views[0]['major_peaks'])))


def classify(p,edges,plan):
    import numpy as np
    q=np.asarray(p[1:-1],dtype='float64');w=float(np.median(np.diff(edges)))
    configs=[(q,edges),coarsen(q,edges,0),coarsen(q,edges,1)]
    vv=[view(m,e,w,plan) for m,e in configs]
    reasons=[r for v in vv for r in v['reasons']]
    if len({v.get('major_count',0) for v in vv})!=1:reasons.append('major_count_changes_with_exact_coarsening')
    if all(v['class']!='uncertain' for v in vv) and not common_intervals(vv):reasons.append('ordered_peak_intervals_lack_positive_common_overlap')
    label=vv[0]['class'] if not reasons and common_intervals(vv) else 'uncertain'
    return vv,label,sorted(set(reasons))


def selftest(plan):
    import numpy as np
    e=np.arange(62,dtype=float)*2.5-75;x=(e[:-1]+e[1:])/2
    checks=[]
    for name,centres,label in [('one',[1.25],'one_major_peak'),('two',[-23.75,26.25],'two_major_peaks'),('three',[-48.75,1.25,51.25],'multiple_major_peaks')]:
        q=sum(np.exp(-.5*((x-c)/3)**2) for c in centres);q/=q.sum();p=np.r_[0,q,0]
        vv,actual,reason=classify(p,e,plan);assert actual==label,(name,actual,reason);checks.append(name)
    q=np.exp(-np.arange(61));q/=q.sum();assert classify(np.r_[0,q,0],e,plan)[1]=='uncertain';checks.append('boundary_not_invented_interior_peak')
    assert classify(np.r_[1,np.zeros(61),0],e,plan)[2]==['finite_mass_zero'];checks.append('zero_finite_mass_retained')
    q=np.diff(e)/(e[-1]-e[0])
    for phase in [0,1]:
        m,ee=coarsen(q,e,phase);assert np.allclose(m/np.diff(ee),1/(e[-1]-e[0]),rtol=0,atol=1e-14)
    checks.append('actual_singleton_width_preserves_uniform_density')
    v=view(np.array([0,.14,.03,.83,0]),np.arange(6,dtype=float),1,plan)
    assert 'major_basin_threshold_crossed_by_valley_mass_bounds' in v['reasons'];checks.append('valley_mass_threshold_ambiguity')
    vv=[{'class':'one_major_peak','major_peaks':[{'interval':z}]} for z in [[0,1],[1,2],[0,2]]]
    assert not common_intervals(vv);checks.append('endpoint_touch_not_peak_match')
    return {'status':'passed','synthetic_checks':checks,'real_labels_read':0,'models_loaded':0}


def run(a):
    import resource,signal
    os.sched_setaffinity(0,{174})
    resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s wall budget')));signal.alarm(600)
    import numpy as np,pandas as pd
    from collections import Counter
    start=time.monotonic();plan=load(a.plan)
    assert sha(a.plan)==a.plan_sha256 and sha(__file__)==plan['source_code_sha256']
    assert os.sched_getaffinity(0)=={174} and not os.environ['CUDA_VISIBLE_DEVICES']
    assert not a.output.exists();a.output.mkdir(parents=True)
    put(a.output/'plan.json',plan);put(a.output/'selftest.json',selftest(plan))
    data=a.atlas_root/'data';release_path=data/'release.json'
    assert sha(release_path)==plan['source_release_sha256'];release=load(release_path)
    assert sha(data/'samples.parquet')==release['artifacts']['samples.parquet']
    meta=pd.read_parquet(data/'samples.parquet',columns=['sample_id','protein_id','group_id','sequence_sha256'])
    assert len(meta)==plan['expected_rows'] and meta.sample_id.is_unique
    assert meta.protein_id.nunique()==plan['expected_unique_T'] and meta.group_id.nunique()==plan['expected_groups']
    assert sha(a.roles_root/'code/bridge_roles_guard.py')==plan['guard_code_sha256']
    sys.path.insert(0,str(a.roles_root/'code'));from bridge_roles_guard import check
    request={'operation':'read_target','stage':'development','sequence_sha256':sorted(meta.sequence_sha256.unique().tolist()),'aliases':[]}
    guard=check(a.roles_root/'registry_v1/guard_contract.json',plan['guard_contract_sha256'],request)
    put(a.output/'role_preflight.json',guard)
    binding_path=data/'label_contract_addendum.json';binding=load(binding_path)
    assert binding['release_sha256']==plan['source_release_sha256']
    assert binding['label_contract_sha256']==plan['label_contract_sha256']==sha(binding['label_contract_path'])
    contract=load(binding['label_contract_path']);assert contract==binding['label_contract']
    assert contract['phase']=='historical' and contract['measure']=='equal_retained_vertices' and contract['bin_count']==63
    edges=np.asarray(contract['edges'],dtype='float64');assert edges.shape==(62,) and np.isfinite(edges).all() and (np.diff(edges)>0).all()
    assert sha(data/'p.npy')==release['artifacts']['p.npy'];p=np.load(data/'p.npy',mmap_mode='r')
    assert p.shape==(len(meta),63) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
    records=[]
    def tail_category(x):return 'zero' if x==0 else 'positive_up_to_0.01' if x<=.01 else 'above_0.01_up_to_0.05' if x<=.05 else 'above_0.05'
    for i,row in enumerate(meta.itertuples(index=False)):
        vv,label,reasons=classify(p[i],edges,plan)
        record={'sample_id':row.sample_id,'protein_id':row.protein_id,'group_id':row.group_id,'native_class':vv[0]['class'],'phase0_class':vv[1]['class'],'phase1_class':vv[2]['class'],'consensus_class':label,'negative_open_tail_mass':float(p[i,0]),'positive_open_tail_mass':float(p[i,-1]),'negative_tail_category':tail_category(float(p[i,0])),'positive_tail_category':tail_category(float(p[i,-1])),'both_open_tails_nonzero':bool(p[i,0]>0 and p[i,-1]>0),'uncertain_reasons':json.dumps(reasons),'view_details':json.dumps(dict(zip(plan['views'],vv)),separators=(',',':'))}
        records.append(record)
    per=pd.DataFrame(records);target=a.output/'per_row_shape_support.parquet';per.to_parquet(target,index=False)
    labels=['one_major_peak','two_major_peaks','multiple_major_peaks','uncertain']
    def support(mask):
        d=per.loc[mask];return {'rows':len(d),'unique_T':int(d.protein_id.nunique()),'groups_with_presence':int(d.group_id.nunique())}
    counts={name:{label:support(per[name]==label) for label in labels} for name in ['native_class','phase0_class','phase1_class','consensus_class']}
    reasons=Counter(r for s in per.uncertain_reasons for r in json.loads(s))
    byT=per.groupby('protein_id').consensus_class.agg(lambda x:x.iloc[0] if x.nunique()==1 else 'mixed_across_rows')
    tails={side:{cat:support(per[side+'_tail_category']==cat) for cat in plan['open_tail_mass_categories']} for side in ['negative','positive']}
    tail_by_consensus={label:{'negative_nonzero':support((per.consensus_class==label)&(per.negative_open_tail_mass>0)),'positive_nonzero':support((per.consensus_class==label)&(per.positive_open_tail_mass>0)),'both_nonzero':support((per.consensus_class==label)&per.both_open_tails_nonzero)} for label in labels}
    boundary_reasons={}
    for name in plan['views']:
        ct=Counter()
        for s in per.view_details:ct.update(json.loads(s)[name]['reasons'])
        boundary_reasons[name]=dict(ct)
    summary={'schema':'native63_coarse_shape_support_summary_v1','status':'complete','data_rows':len(meta),'unique_T':int(meta.protein_id.nunique()),'groups':int(meta.group_id.nunique()),'counts':counts,'unique_T_consensus_classes_without_averaging_rows':byT.value_counts().to_dict(),'uncertainty_reason_row_counts_nonexclusive':dict(reasons),'per_view_uncertainty_reasons_nonexclusive':boundary_reasons,'open_tail_mass_support':tails,'both_open_tails_nonzero':support(per.both_open_tails_nonzero),'open_tail_presence_by_consensus':tail_by_consensus,'support_count_note':'Rows partition by class; unique-T and group presence counts within categories may overlap. No average distribution or group-majority class was formed.','finite_window':[float(edges[0]),float(edges[-1])],'finite_bins':61,'native_width_range':[float(np.diff(edges).min()),float(np.diff(edges).max())],'classification_scope':plan['scope'],'unconfirmed_is_not_absence':True,'code_sha256':sha(__file__),'plan_sha256':sha(a.plan),'source_release_sha256':sha(release_path),'label_contract_sha256':plan['label_contract_sha256'],'source_target_sha256':sha(data/'p.npy'),'guard_contract_sha256':plan['guard_contract_sha256'],'role_preflight_sha256':sha(a.output/'role_preflight.json'),'private_per_row_sha256':sha(target),'resource':{'CPU_affinity':sorted(os.sched_getaffinity(0)),'BLAS_threads':1,'GPU':False,'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'address_space_limit_GiB':8,'wall_budget_seconds':600},'new_fits':0,'model_reads':0,'raw_phi_reads':0,'protected_labels_read':0,'gallery_modified':False,'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}}
    put(a.output/'summary.json',summary)
    print(json.dumps({'status':'complete','rows':len(meta),'elapsed_seconds':summary['resource']['elapsed_seconds'],'summary_sha256':sha(a.output/'summary.json'),'native':counts['native_class'],'consensus':counts['consensus_class']}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256');ap.add_argument('--atlas-root',type=Path);ap.add_argument('--roles-root',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:print(json.dumps(selftest(load(a.plan))));return
    assert a.plan_sha256 and a.atlas_root and a.roles_root and a.output;run(a)


if __name__=='__main__':main()
