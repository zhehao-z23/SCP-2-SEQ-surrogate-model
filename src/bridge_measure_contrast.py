"""Same frozen scalar field, different probability measure; no solver or model."""
from pathlib import Path
import argparse, datetime, hashlib, json, platform, time
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def empirical_distance(values, area, vertex):
    order=np.argsort(values,kind='stable'); x=values[order]
    starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1]
    unique=x[starts]
    # Collapse exact ties before CDF sup: no artificial order within a tied atom.
    signed=np.add.reduceat(area[order]-vertex[order],starts)
    delta=np.cumsum(signed,dtype=np.float64)
    w1=float(np.dot(np.abs(delta[:-1]),np.diff(unique)))
    cdf=float(np.max(np.abs(delta)))
    # Independent arithmetic route: each CDF is accumulated separately.
    fa=np.cumsum(np.add.reduceat(area[order],starts),dtype=np.float64)
    fv=np.cumsum(np.add.reduceat(vertex[order],starts),dtype=np.float64)
    w1_check=float(np.dot(np.abs((fa-fv)[:-1]),np.diff(unique)))
    cdf_check=float(np.max(np.abs(fa-fv)))
    if not np.allclose([w1,cdf],[w1_check,cdf_check],atol=1e-9,rtol=1e-9):
        raise RuntimeError('CDF arithmetic crosscheck failed')
    return {'raw_W1_kBT_per_e':w1,'raw_CDF_sup':cdf,'check_W1_abs_error':abs(w1-w1_check),'check_CDF_abs_error':abs(cdf-cdf_check),'unique_potential_values':int(len(unique))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--output',required=True);ap.add_argument('--public-output',required=True);a=ap.parse_args()
    start=time.perf_counter();d=Path(a.inputs);mp=d/'input_manifest.json'
    if sha(mp)!=a.manifest_sha256:raise RuntimeError('manifest hash mismatch')
    manifest=json.loads(mp.read_text());checks=[]
    for name in ['phi.npy','weights.npy','shared_bins.json','sample_record.json']:
        p=d/name
        if p.stat().st_size>32*1024**2:raise RuntimeError('bounded small-array audit exceeded')
        observed=sha(p);expected=manifest['files'][name]['sha256']
        if observed!=expected:raise RuntimeError('frozen input changed: '+name)
        checks.append({'file':name,'sha256':observed,'bytes':p.stat().st_size})
    phi=np.load(d/'phi.npy',allow_pickle=False);w=np.load(d/'weights.npy',allow_pickle=False)
    if phi.ndim!=1 or w.shape!=phi.shape or not len(phi) or not np.isfinite(phi).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise RuntimeError('invalid complete empirical inputs')
    # A tied-support example with known W1=.25 and CDFsup=.125.
    toy=empirical_distance(np.array([0.,0.,2.]),np.array([.25,.25,.5]),np.array([.125,.25,.625]))
    if not np.allclose([toy['raw_W1_kBT_per_e'],toy['raw_CDF_sup']],[.25,.125],atol=1e-14):raise RuntimeError('analytic tie check failed')
    bins=json.loads((d/'shared_bins.json').read_text());edges=np.asarray(bins['internal_edges'],dtype=np.float64)
    if len(edges)!=62 or not np.isfinite(edges).all() or not (np.diff(edges)>0).all():raise RuntimeError('expected frozen 63-bin schema')
    area=w.astype(np.float64)/np.sum(w,dtype=np.float64);vertex=np.full(len(phi),1./len(phi));index=np.searchsorted(edges,phi,side='right')
    pa=np.bincount(index,weights=area,minlength=63);pv=np.bincount(index,weights=vertex,minlength=63)
    if not np.allclose([area.sum(),vertex.sum(),pa.sum(),pv.sum()],1.,atol=2e-12,rtol=0):raise RuntimeError('probability normalization failed')
    metrics=empirical_distance(phi,area,vertex);metrics['Hellinger_shared63']=float(np.linalg.norm(np.sqrt(pa)-np.sqrt(pv))/np.sqrt(2.))
    def stats(weight,p):
        return {'mean_kBT_per_e':float(np.dot(phi,weight)),'positive_fraction':float(weight[phi>0].sum()),'left_open_tail_mass':float(p[0]),'right_open_tail_mass':float(p[-1]),'total_open_tail_mass':float(p[0]+p[-1]),'mass_sum':float(weight.sum())}
    ast=stats(area,pa);vst=stats(vertex,pv);delta={k:vst[k]-ast[k]for k in ast}
    scientific={'case_count':1,'comparison':'equal vertex minus normalized incident-triangle area; same frozen corrected field and mesh','bin_count':63,'finite_boundary_range_kBT_per_e':[float(edges[0]),float(edges[-1])],'tie_rule':'left-closed/right-open; value at upper central edge enters right tail','metrics':{k:v for k,v in metrics.items()if not k.startswith('check_') and k!='unique_potential_values'},'area_measure':ast,'equal_vertex_measure':vst,'delta_vertex_minus_area':delta,'scope':['Measure-only mathematical contrast on one exposed existing numerically validated PDB development reference.','Not the complete historical Atlas-to-new-C target shift, not structural-source causality, not a surrogate-generalization or production-authorization result.','Existing PDB qualification does not transfer to any AF object.'],'PB_calls':0,'model_predictions':0,'raw_DX_reads':0}
    private={'schema':'private_same_field_measure_contrast_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_case':manifest['case'],'source_manifest_sha256':a.manifest_sha256,'code_sha256':sha(__file__),'verified_inputs':checks,'query_count':len(phi),'total_area_A2':float(w.sum()),'zero_area_vertices':int((w==0).sum()),'phi_minmax_kBT_per_e':[float(phi.min()),float(phi.max())],'arithmetic_crosscheck':{k:v for k,v in metrics.items()if k.startswith('check_')},'analytic_tie_selfcheck_passed':True,'environment':{'python':platform.python_version(),'numpy':np.__version__},'wall_seconds':time.perf_counter()-start,'result':scientific}
    # Public serialization excludes absolute single-object target summaries.
    public_result={k:v for k,v in scientific.items() if k not in ['area_measure','equal_vertex_measure']}
    public_result['delta_vertex_minus_area']={k:v for k,v in delta.items() if k in ['mean_kBT_per_e','positive_fraction']}
    public_result['tail_scope']='A single reference cannot establish tail generalization; an empty observed tail does not establish interchangeable tail measures.'
    public={'schema':'anonymous_same_field_measure_contrast_v1','status':'prepared_for_parent_review_not_published','execution_code_sha256':private['code_sha256'],'result':public_result}
    for path,obj in [(Path(a.output),private),(Path(a.public_output),public)]:
        if path.exists():raise RuntimeError('refuse receipt overwrite')
        path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':scientific['metrics'],'area':ast,'vertex':vst,'wall_seconds':private['wall_seconds']}))

if __name__=='__main__':main()
