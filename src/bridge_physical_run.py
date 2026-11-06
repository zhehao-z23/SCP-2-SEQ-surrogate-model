"""Isolated AF/PDB physical development pilots using immutable production components.

Every APBS launch is registered to this branch's independent ledger. This runner
never edits production code, starts its controller, or promotes a probability
vector to qualified merely because it is finite. Outputs retain full fields.
"""
from pathlib import Path
import os, sys, json, hashlib, time, sqlite3, argparse, traceback
sys.dont_write_bytecode=True
from bridge_physical_paths import consume_paths
PATHS=consume_paths();P=PATHS.production_project;R=PATHS.bridge_root
sys.path[:0]=[str(P/'code'),str(P/'code/vendor_revision003'),str(P/'code/vendor_chemistry_v3')]
import numpy as np
from workflow_core import ResourceReservations
from archived_execution import runtime, run_command
from project_paths import ProjectPaths
from solver_surface import read_pqr_atoms, write_pqr_atoms
from nanoshaper15_adapter_v3_runtime import run_nanoshaper15
from checked_surface_normals_v2 import checked_surface_normals_v2
from operational_grid_geometry import fine_box, rotation_matrix
from capacity_coarse_geometry import capacity_coarse_box
from adaptive_query_boxes import query_box_plan
from SES_local_map_preparation_v4_resources import prepare_local_maps
import SES_map_solver_v3_runtime as solver
from surface_field_sampling import sample
from export_representation import export
from distribution_metrics import compare

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def frozen_command(entry,args):
    paths=[str(P/'software/python_runtime_snapshot_v3/site-packages'),str(P/'code/vendor_chemistry_v3')]
    boot='import sys,runpy;sys.dont_write_bytecode=True;sys.path[:0]='+repr(paths)+';sys.argv='+repr([str(entry),*map(str,args)])+';runpy.run_path(sys.argv[0],run_name="__main__")'
    return runtime(P).command('python',['-B','-I','-S','-c',boot])
def initialize():
    for d in ['records/SES_map_fields','records/budget','contracts','runs','cases','results']:(R/d).mkdir(parents=True,exist_ok=True)
    contract={'schema':'AF_bridge_development_budget_v2','authority':'Historical bounded development recipe; this portable archive does not grant execution authorization','new_PB_attempt_cap':120,'phases':{'AF_bridge_development':120},'max_source_objects':4,'maximum_CPU_threads':4,'max_concurrent_cases':1,'memory_reservation_GiB':96,'host_headroom_GiB':128,'per_APBS_timeout_s':1800,'full_run_enabled':False,'production_controller_untouched':True}
    cp=R/'contracts/authority_and_budget.json'
    if cp.exists()and load(cp)!=contract:
        old=load(cp)
        if old.get('schema')!='AF_bridge_development_budget_v1':raise RuntimeError('budget binding changed')
        write(R/'contracts/authority_and_budget_v1_preserved.json',old)
    write(cp,contract)
    with sqlite3.connect(R/'records/budget/PB_attempts.sqlite')as c:c.execute('CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, phase TEXT NOT NULL, job_key TEXT UNIQUE NOT NULL, reserved_at_utc TEXT NOT NULL, launched_at_utc TEXT, pid INTEGER, host TEXT, status TEXT NOT NULL, command_json TEXT NOT NULL, input_hash TEXT NOT NULL, exit_code INTEGER, result_path TEXT)')
    cfg=load(P/'config/production_candidate_SES_v2.json');bins=ProjectPaths(P).reference(cfg['representation']['shared_bins'])
    write(R/'records/ledger_registration.json',{'authority':ref(cp),'ledger':str(R/'records/budget/PB_attempts.sqlite'),'scope':'three exposed development sources; independent branch, no old AF106/prod ledger mutation','production_protocol':ref(P/'config/production_candidate_SES_v2.json'),'acceptance_contract':ref(P/'contracts/representation_acceptance_v1.json'),'shared_bins':ref(bins),'code':ref(__file__)})
    solver.N=R
    return cfg,bins,ResourceReservations(R/'records/budget/resources.json',cap_bytes=96*1024**3,max_jobs=1)
def resource_preflight():
    available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    rows=[]
    for p in Path('/sys/fs/cgroup').glob('atlas_full_dC_*/memory.current'):
        group=p.parent;events=(group/'cgroup.events').read_text()
        if 'populated 1'not in events:continue
        current=int(p.read_text());mx=(group/'memory.max').read_text().strip();maximum=None if mx=='max'else int(mx)
        rows.append({'group':str(group),'current_bytes':current,'max_bytes':maximum,'headroom_bytes':None if maximum is None else maximum-current})
    if available<(96+128)*1024**3:raise RuntimeError('host memory reserve would encroach on production headroom')
    if any(x['headroom_bytes']is not None and x['headroom_bytes']<16*1024**3 for x in rows):raise RuntimeError('production cgroup memory headroom below16GiB')
    write(R/'records'/('resource_check_'+str(time.time_ns())+'.json'),{'MemAvailable':available,'required_free_bytes':224*1024**3,'active_production_cgroups':rows,'production_cgroups_modified':False})
def prepare(sid):
    rows=load(R/'inputs/pilots.json')['cases'];obj=next(x for x in rows if x['sample_id']==sid)
    d=R/'cases'/sid;d.mkdir(parents=True,exist_ok=True);done=d/'prepared_case.json'
    if done.exists():
        r=load(done)
        if sha(r['pqr_path'])!=r['PQR']['pqr_sha256']:raise RuntimeError('PQR changed')
        return r
    src=R/'inputs'/obj['file'];assert sha(src)==obj['source_sha256']
    V=P/'code/vendor_chemistry_v3';sr=d/'structure_result.json'
    if not sr.exists():
        req={'implementation':{x:sha(V/x)for x in ['structure_prepare.py','structure_worker.py']},'source':ref(src),'auth_chain':obj['auth_chain'],'model_id':'1','sample_id':sid,'output_directory':str(d/'structure'),'result_path':str(sr),'segment_policy':'joint_neutral_artificial','terminal_policy':'selective_neutral_artificial'}
        write(d/'structure_request.json',req);x=run_command(frozen_command(V/'structure_worker.py',['--request',d/'structure_request.json']),d,'structure',600,1)
        if x['returncode']:raise RuntimeError('structure worker failed')
    context=load(sr)
    if context['status']!='ready_for_parameterization':raise RuntimeError('structure ineligible')
    if obj.get('sequence_sha256')and context['prepared_observed_sequence_sha256']!=obj['sequence_sha256']:raise RuntimeError('AF coordinate sequence differs from frozen API sequence')
    p=d/'PQR';p.mkdir(exist_ok=True)
    if not(p/'chemistry_result.json').exists():
        write(p/'context.json',context);write(p/'config.json',load(P/'config/chemistry_revision15_source.json'))
        x=run_command(frozen_command(V/'chemistry_adapter.py',['--worker',context['prepared_pdb_path'],p/'context.json',p/'config.json',p]),p,'parameterization',1800,1,env_extra={'PYTHONPATH':str(P/'software/python_runtime_snapshot_v3/site-packages')+':'+str(V),'PYTHONNOUSERSITE':'1'})
        if x['returncode']:raise RuntimeError('PARSE parameterization failed')
    chem=load(p/'chemistry_result.json')
    if chem.get('status')!='ok'or chem.get('chemical_integrity_status')!='passed':raise RuntimeError('chemical integrity failed')
    pqr=chem['pqr_path'];mesh=d/'surface8'
    if not(mesh/'surface_manifest.json').exists():run_nanoshaper15(pqr,mesh,grid_scale=8,threads=2,timeout=1800)
    r={'sample_id':sid,'source':obj,'source_cif':ref(src),'structure':context,'PQR':chem,'pqr_path':pqr,'mesh_root':str(mesh),'source_origin':obj['origin'],'production_protocol':ref(P/'config/production_candidate_SES_v2.json'),'numerical_qualified':False}
    write(done,r);return r
def geometry(pre,variant):
    d=R/'cases'/pre['sample_id']/variant;d.mkdir(exist_ok=True);pqr=Path(pre['pqr_path']);mesh=Path(pre['mesh_root']);atoms=read_pqr_atoms(pqr)
    if variant=='rotation':
        center=atoms[['x','y','z']].values.mean(0);rot=rotation_matrix();rotp=d/'rotated.pqr'
        if not rotp.exists():
            aa=atoms.copy();aa[['x','y','z']]=(aa[['x','y','z']].values-center)@rot.T+center;write_pqr_atoms(aa,rotp);write(d/'rotation.json',{'center':center.tolist(),'rotation':rot.tolist(),'source_pqr':ref(pqr),'rotated_pqr':ref(rotp),'reparameterized':False})
        pqr=rotp;atoms=read_pqr_atoms(pqr);mesh=d/'surface8'
    elif variant=='surface16':mesh=d/'surface16'
    if not(mesh/'surface_manifest.json').exists():run_nanoshaper15(pqr,mesh,grid_scale=16 if variant=='surface16'else 8,threads=2,timeout=1800)
    geom=d/'geometry';geom.mkdir(exist_ok=True);normal=geom/'surface_normals.npy'
    if normal.exists():q=np.load(mesh/'vertices.npy');nn=np.load(normal)
    else:q,nn,_=checked_surface_normals_v2(mesh,geom)
    w=np.load(mesh/'vertex_area_weights.npy');h=.125 if variant in ['refined','overlap12']else .0625 if variant=='further'else .25
    fine=fine_box(atoms[['x','y','z']].values,atoms.radius.values,q,h=h,padding_A=12. if variant=='expanded'else 8.)
    if variant=='shifted':
        for key in ['center','origin']:fine[key]=(np.array(fine[key])+h/2).tolist()
    parent=capacity_coarse_box(atoms[['x','y','z']].values,atoms.radius.values,fine,maximum_spacing_A=.5 if variant in ['refined','further','overlap12','parent_refined']else 1.,dimension_cap=513 if variant in ['refined','further','overlap12','parent_refined']else 385)
    # Retain original base partition if already committed. Other grids use the
    # approved larger reservation to avoid repeating the same halo hundreds of times.
    oldbase=(d/'outcome.json').exists()and variant=='base'
    plans=query_box_plan(q,fine['origin'],h,halo_A=12. if variant in ['overlap12','expanded']else 8.,cap_bytes=(25 if oldbase else 80)*1024**3,max_points=20000000 if oldbase else 100000000)
    meta={'fine':fine,'parent':parent,'parts':[{k:v for k,v in x.items()if k!='query_indices'}for x in plans],'planned_PB_attempts':1+len(plans),'pqr':ref(pqr),'mesh':str(mesh),'h':h}
    write(d/'geometry_plan.json',meta)
    return d,pqr,mesh,q,nn,w,h,parent,plans
def run_case(sid,variant,cfg,bins,pool,plan_only=False):
    resource_preflight()
    pre=prepare(sid);d,pqr,mesh,q,nn,w,h,parent,plans=geometry(pre,variant)
    if plan_only:return {'case':sid,'variant':variant,'planned_calls':1+len(plans)}
    done=d/'outcome.json'
    if done.exists():return load(done)
    md=d/'maps';md.mkdir(exist_ok=True);key=sid+'_'+variant
    with sqlite3.connect(R/'records/budget/PB_attempts.sqlite')as db:count=db.execute('select count(*)from attempts').fetchone()[0]
    if count+1+len(plans)>120:raise RuntimeError('complete variant exceeds remaining bound')
    physical=cfg['PB_physical_parameters'];maps=prepare_local_maps(pqr,mesh,parent['origin'],parent['spacing_A'],parent['dime'],md/'parent',resource_pool=pool)
    resource_preflight();parentfield=solver.run_manual_job(key+'_parent',str(pqr),parent['dime'],parent['length'],parent['center'],physical,phase='AF_bridge_development',timeout=1800,coefficient_maps=[x['path']for x in maps['maps']],resource_pool=pool)
    phi=np.full(len(q),np.nan);raw=phi.copy();accepted=np.zeros(len(q),bool);fields=[ref(R/'records/SES_map_fields'/(key+'_parent.json'))]
    for part in plans:
        rank=part['rank'];idx=part['query_indices'];pd=d/('part_'+str(rank));pd.mkdir(exist_ok=True);np.save(pd/'query_indices.npy',idx)
        origin=np.array(part['global_origin'])+np.array(part['compute_start'])*h;length=(np.array(part['local_dime'])-1)*h;center=origin+length/2
        maps=prepare_local_maps(pqr,mesh,origin,h,part['local_dime'],md/('fine_'+str(rank)),resource_pool=pool)
        resource_preflight();field=solver.run_manual_job(key+'_fine_'+str(rank),str(pqr),part['local_dime'],length.tolist(),center.tolist(),physical,parentfield['dx_path'],phase='AF_bridge_development',timeout=1800,coefficient_maps=[x['path']for x in maps['maps']],resource_pool=pool)
        v,rv,fb=sample(field['dx_path'],q[idx],nn[idx],w[idx],pd);phi[idx]=v;raw[idx]=rv;accepted[idx]=np.load(pd/'query_accepted.npy');fields.append(ref(R/'records/SES_map_fields'/(key+'_fine_'+str(rank)+'.json')))
    if not np.isfinite(phi).all():raise RuntimeError('incomplete/nonfinite samples')
    for name,arr in [('phi',phi),('raw',raw),('weights',w)]:np.save(d/(name+'.npy'),arr)
    case={**pre,'pqr_path':str(pqr)};write(d/'source_case.json',case)
    inputs={'corrected_phi':ref(d/'phi.npy'),'raw_trilinear_phi':ref(d/'raw.npy'),'area_weights':ref(d/'weights.npy'),'vertices':ref(mesh/'vertices.npy'),'faces':ref(mesh/'faces.npy'),'shared_bins':ref(bins),'protocol_configuration':ref(P/'config/production_candidate_SES_v2.json'),'source_case_record':ref(d/'source_case.json')}
    spec={'sample_id':sid,'source_role':'AF_bridge_exposed_development','inputs':inputs,'PB_field_receipts':fields}
    export(spec,d/'representation')
    r={'status':'completed','sample_id':sid,'variant':variant,'representation':ref(d/'representation/manifest.json'),'phi':ref(d/'phi.npy'),'weights':ref(d/'weights.npy'),'fields':fields,'fallback_area_fraction':float(w[~accepted].sum()/w.sum()),'numerical_qualified':False,'qualification_requires_all_frozen_axes':True}
    write(done,r);return r
def comparisons(sid,bins):
    root=R/'cases'/sid;edges=load(bins)['internal_edges'];pairs=[('base','refined','spacing'),('refined','further','further_refinement'),('base','shifted','phase'),('base','rotation','rotation'),('base','surface16','surface_resolution'),('base','expanded','expanded_domain'),('base','parent_refined','coarse_parent'),('refined','overlap12','overlap')];rows=[]
    for a,b,axis in pairs:
        if not all((root/x/'outcome.json').exists()for x in [a,b]):continue
        m=compare(np.load(root/a/'phi.npy'),np.load(root/a/'weights.npy'),np.load(root/b/'phi.npy'),np.load(root/b/'weights.npy'),edges)
        rows.append({'axis':axis,'test':a,'reference':b,'metrics':m,'pass':m['Hellinger']<=.02 and m['W1_kBT_per_e']<=.1 and m['CDF_sup']<=.01})
    r={'sample_id':sid,'comparisons':rows,'expected_axes':[x[2]for x in pairs],'numerical_qualified':len(rows)==len(pairs)and all(x['pass']for x in rows),'not_experimental_accuracy':True,'not_global_AFDB_qualification':True};write(R/'results'/(sid+'_qualification.json'),r)
    return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variants',nargs='+',default=['base']);p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    if 'further' in args.variants:
        p.error("--variants further is not an archived execution route; use bridge_physical_further_parallel.py with its nested-reference evidence and preflight")
    cfg,bins,pool=initialize();start=time.time()
    write(R/'records'/('launch_'+args.case+'.json'),{'pid':os.getpid(),'start_epoch':start,'args':vars(args),'code':ref(__file__),'affinity':sorted(os.sched_getaffinity(0))})
    try:
        for variant in args.variants:
            print('START',args.case,variant,flush=True);r=run_case(args.case,variant,cfg,bins,pool,args.plan_only);print('DONE',args.case,variant,r.get('status',r),flush=True);comparisons(args.case,bins)
        write(R/'records'/('completion_'+args.case+'.json'),{'status':'completed_requested_stages','elapsed_s':time.time()-start,'args':vars(args)})
    except BaseException as e:
        write(R/'records'/('failure_'+args.case+'_'+str(time.time_ns())+'.json'),{'error':str(e),'traceback':traceback.format_exc(),'args':vars(args)});raise
if __name__=='__main__':main()
