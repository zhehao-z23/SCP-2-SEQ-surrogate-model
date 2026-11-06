"""Bounded two-field AF reference execution; unchanged native solver/central ledger.

Heavy artifacts start on a separate configured NVMe root. Both workers share atomically
locked memory/disk reservations and the exact original SQLite attempt ledger.
Each worker owns one physical core's SMT pair; pinned APBS requests one thread.
"""
from pathlib import Path
import argparse,sys,os,json,time,sqlite3,fcntl,shutil,multiprocessing,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import bridge_physical_run as b
import bridge_physical_further as f
import bridge_physical_resume_support as reuse
from project_paths import ProjectPaths
from PB_ledger import Ledger as NativeLedger

V=b.PATHS.heavy_root;GIB=1024**3
if V is None:raise RuntimeError('--heavy-root is required for nested execution')
WORK={}
def bounded_dispatch(pool,fn,metadata,on_success,on_failure):
    """At most two in-flight; first error prevents all future dispatches."""
    pending=iter(metadata);futures={};results=[];errors=[];dispatched=[]
    def submit(row):
        futures[pool.submit(fn,row)]=row['rank'];dispatched.append(row['rank'])
    for _ in range(2):
        row=next(pending,None)
        if row is not None:submit(row)
    while futures:
        completed,_=wait(futures,return_when=FIRST_COMPLETED)
        for future in completed:
            rank=futures.pop(future)
            try:
                result=future.result();results.append(result);on_success(result,len(results))
            except BaseException as e:
                errors.append({'rank':rank,'error':str(e),'traceback':traceback.format_exc()});on_failure(errors,list(futures.values()))
        if not errors:
            while len(futures)<2:
                row=next(pending,None)
                if row is None:break
                submit(row)
    return results,errors,dispatched
def branch_bytes():return sum(p.stat().st_size for p in V.rglob('*')if p.is_file())
def check_host():
    cpu=b.load(b.R/'records/CPU_isolation_contract.json');pilotkeys={tuple(x)for x in cpu['pilot_physical_cores']}
    for role,ref in cpu['bindings'].items():
        if b.sha(ref['path'])!=ref['sha256']:raise RuntimeError('production CPU contract changed; re-audit before dispatch: '+role)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            cmd=(proc/'cmdline').read_bytes();cwd=(proc/'cwd').resolve()
            if b.P.parent.name.encode()not in cmd or cwd.is_relative_to(b.R)or cwd.is_relative_to(V):continue
            if b'/bin/apbs'not in cmd and b'production_controller.py'not in cmd:continue
            keys=[]
            for c in sorted(os.sched_getaffinity(int(proc.name))):
                top=Path('/sys/devices/system/cpu')/('cpu'+str(c))/'topology';key=(int((top/'physical_package_id').read_text()),int((top/'core_id').read_text()))
                if key not in keys:keys.append(key)
            if b'production_controller.py'in cmd:keys=keys[:64]
            if set(keys)&pilotkeys:raise RuntimeError('live production slot/SMT sibling overlaps fixed pilot island')
        except (OSError,ProcessLookupError):pass
    free=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    if free<320*GIB:raise RuntimeError('MemAvailable below192+128GiB')
    if shutil.disk_usage(V).free<512*GIB:raise RuntimeError('NVMe free below512GiB')
    prod=[]
    for p in Path('/sys/fs/cgroup').glob('atlas_full_dC_*/memory.current'):
        if 'populated 1'not in(p.parent/'cgroup.events').read_text():continue
        used=int(p.read_text());mx=(p.parent/'memory.max').read_text().strip();head=None if mx=='max'else int(mx)-used
        if head is not None and head<16*GIB:raise RuntimeError('production cgroup headroom below16GiB')
        prod.append({'path':str(p.parent),'current_bytes':used,'headroom_bytes':head})
    return {'MemAvailable':free,'NVMe_free_bytes':shutil.disk_usage(V).free,'production_cgroups':prod,'fixed_CPU_contract':b.ref(b.R/'records/CPU_isolation_contract.json'),'automatic_low_memory_serial_fallback':False}
def disk_reserve(key,nodes):
    # Four maps + raw DX + lossless zstd/temp output, <=32 ASCII bytes/value
    # plus conservative framing/metadata. Existing bytes and in-flight bounds
    # are double-counted deliberately so two dispatches cannot over-admit.
    amount=int(nodes)*202+64*1024**2;p=V/'records/disk_reservations.json'
    with open(str(p)+'.lock','a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);d=b.load(p)if p.exists()else {};check=check_host()
        if key in d:raise RuntimeError('existing disk reservation requires process reconciliation')
        pending=sum(x['bytes']for x in d.values());actual=branch_bytes()
        if actual+pending+amount>650*GIB:raise RuntimeError('650GiB NVMe branch hard admission cap')
        if check['NVMe_free_bytes']<512*GIB+pending+amount:raise RuntimeError('NVMe512GiB floor with in-flight peak reserve')
        d[key]={'bytes':amount,'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'epoch':time.time()};b.write(p,d)
        b.write(V/'records'/('dispatch_'+key+'.json'),{'reservation':d[key],'branch_actual_bytes':actual,'other_reserved_bytes':pending,'host':check,'affinity':sorted(os.sched_getaffinity(0)),'actual_APBS_threads':1})
def disk_release(key):
    p=V/'records/disk_reservations.json'
    with open(str(p)+'.lock','a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);d=b.load(p)
        if d[key]['pid']!=os.getpid():raise RuntimeError('disk reservation owner differs')
        del d[key];b.write(p,d)
        if branch_bytes()>650*GIB:raise RuntimeError('NVMe branch exceeded hard cap')
class CanonicalLedger(NativeLedger):
    def __init__(self,path,contract):
        assert Path(path)==V/'records/budget/PB_attempts.sqlite'
        super().__init__(b.R/'records/budget/PB_attempts.sqlite',contract)
def init_worker(core_queue,sid):
    cpus=core_queue.get(timeout=10);os.sched_setaffinity(0,cpus)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    pre=b.load(b.R/'cases'/sid/'prepared_case.json');mesh=Path(pre['mesh_root']);geom=b.R/'cases'/sid/'overlap12/geometry'
    WORK.update(sid=sid,pre=pre,pqr=Path(pre['pqr_path']),mesh=mesh,q=np.load(mesh/'vertices.npy'),nn=np.load(geom/'surface_normals.npy'),weights=np.load(mesh/'vertex_area_weights.npy'),cfg=b.load(b.P/'config/production_candidate_SES_v2.json'),pool=b.ResourceReservations(b.R/'records/budget/further_resources.json',cap_bytes=192*GIB,max_jobs=2))
    b.solver.N=V;b.solver.Ledger=CanonicalLedger
def one_field(row):
    sid=WORK['sid'];rank=row['rank'];key=sid+'_further_nested_fine_'+str(rank);pqr=WORK['pqr'];mesh=WORK['mesh'];pool=WORK['pool'];cfg=WORK['cfg'];idx=np.load(b.R/'cases'/sid/'further'/('query_indices_'+str(rank)+'.npy'))
    parentpath=b.R/'records/SES_map_fields'/(row['parent_source_key']+'.json');parent=b.load(parentpath)
    with sqlite3.connect('file:'+str(b.R/'records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)as db:r=db.execute('select status,result_path from attempts where id=?',(parent['PB_attempt_id'],)).fetchone()
    if not r or r[0]!='completed'or Path(r[1])!=parentpath:raise RuntimeError('intermediate parent ledger mismatch')
    if parent['status']!='completed'or b.sha(parent['dx_path'])!=parent['dx_sha256']or parent['source']['source_PQR_sha256']!=b.sha(pqr)or parent['source']['physical']!=cfg['PB_physical_parameters']:raise RuntimeError('parent field hash/physics mismatch')
    pp=row['parent_query_plan'];expected_po=np.array(pp['global_origin'])+np.array(pp['compute_start'])*.125
    if parent['dims']!=pp['local_dime']or not np.allclose(parent['h'],.125,rtol=0,atol=5e-6)or not np.allclose(parent['origin'],expected_po,rtol=0,atol=5e-5):raise RuntimeError('intermediate parent layer/grid differs from frozen plan')
    header={};delta=[]
    with open(parent['dx_path'])as dx:
        for _ in range(100):
            line=dx.readline();a=line.split()
            if 'gridpositions'in a:header['dims']=list(map(int,a[-3:]))
            elif a and a[0]=='origin':header['origin']=list(map(float,a[1:]))
            elif a and a[0]=='delta':delta.append(list(map(float,a[1:])))
            if 'data follows'in line:break
    if header.get('dims')!=pp['local_dime']or not np.allclose(header.get('origin',[]),expected_po,rtol=0,atol=5e-5)or not np.allclose(delta,np.eye(3)*.125,rtol=0,atol=5e-6):raise RuntimeError('actual parent DX header/grid layer differs')
    po=np.array(parent['origin']);pu=po+(np.array(parent['dims'])-1)*parent['h'];co=np.array(row['origin']);cu=co+row['length'];margin=np.minimum(co-po,pu-cu)
    if np.any(margin<.125-1e-5):raise RuntimeError('actual parent DX lacks one grid-step child support')
    disk_reserve(key,int(np.prod(row['local_dime'])))
    try:
        md=V/'maps'/key;pd=V/'parts'/key;pd.mkdir(parents=True,exist_ok=True);check_host()
        maps=b.prepare_local_maps(pqr,mesh,co,.0625,row['local_dime'],md,resource_pool=pool)
        check_host();center=co+np.array(row['length'])/2
        field=b.solver.run_manual_job(key,str(pqr),row['local_dime'],row['length'],center.tolist(),cfg['PB_physical_parameters'],parent['dx_path'],phase='AF_bridge_development',timeout=1800,coefficient_maps=[x['path']for x in maps['maps']],resource_pool=pool)
        # Sampling memory is in the same two-slot global pool as maps/APBS.
        skey=key+'_sampling';need=6*GIB+int(np.prod(row['local_dime']))*16
        while not pool.acquire(skey,need):time.sleep(1)
        try:val,rv,_=b.sample(field['dx_path'],WORK['q'][idx],WORK['nn'][idx],WORK['weights'][idx],pd)
        finally:pool.release(skey)
        fr=V/'records/SES_map_fields'/(key+'.json');out={'rank':rank,'key':key,'field':b.ref(fr),'parent_field':b.ref(parentpath),'query_indices':b.ref(b.R/'cases'/sid/'further'/('query_indices_'+str(rank)+'.npy')),'phi':b.ref(pd/'phi_jump.npy'),'raw':b.ref(pd/'phi_raw.npy'),'accepted':b.ref(pd/'query_accepted.npy'),'actual_parent_margin_A':margin.tolist(),'worker_pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),'APBS_threads':1};b.write(pd/'sampling_receipt.json',out);return out
    finally:disk_release(key)

def geometry_gate(sid):
    """Read-only admission validation; existing normal filenames are insufficient."""
    path=b.R/'records/geometry_v3'/sid/'all_variant_admissions.json';index=b.load(path)
    expected={'base','refined','shifted','rotation','expanded','parent_refined','overlap12','surface16'}
    if index.get('eight_variants_bound')is not True or index.get('operator_changed')is not False or {x['variant']for x in index['rows']}!=expected:raise RuntimeError('geometry revision variant closure incomplete')
    checked={}
    def verify(ref):
        p=ref['path']
        if p not in checked:checked[p]=b.sha(p)
        if checked[p]!=ref['sha256']:raise RuntimeError('geometry admission binding changed:'+p)
    for row in index['rows']:
        verify(row['admission']);r=b.load(row['admission']['path'])
        if r.get('pass')is not True or r.get('all_original_probes_represented')is not True or r.get('normals_equal_to_certified_derivation')is not True or r.get('physical_geometry_or_PQR_changed')is not False:raise RuntimeError('geometry admission did not pass:'+row['variant'])
        for key in ['certificate','revision_contract','original_orientation','coherent_normals','gate_code']:verify(r[key])
        for ref in [*r['mesh_identity'].values(),*r['original_qualification_code_bindings'].values()]:verify(ref)
        cert=b.load(r['certificate']['path'])
        if cert.get('pass')is not True or cert['binding']!=r['mesh_identity']:raise RuntimeError('geometry certificate mismatch:'+row['variant'])
        verify(cert['new_coherent_normals'])
        if r['coherent_normals']['sha256']!=cert['new_coherent_normals']['sha256']:raise RuntimeError('normal bytes differ from certified derivation')
    return {'admissions':b.ref(path),'verified_unique_files':len(checked),'all_eight_variants_passed':True,'operator_changed':False}

def preflight(sid,summary,rows):
    for sub in ['records/SES_map_fields','records/budget','contracts','runs','maps','parts']:(V/sub).mkdir(parents=True,exist_ok=True)
    q=b.load(b.R/'results'/(sid+'_qualification.json'));required=set(q['expected_axes'])-{'further_refinement'}
    if {x['axis']for x in q['comparisons']}!=required:raise RuntimeError('all other frozen AF axes must be complete before further dispatch')
    for row in q['comparisons']:
        values=[row['metrics'][k]for k in ['Hellinger','W1_kBT_per_e','CDF_sup']]
        if row.get('pass')is not True or not all(np.isfinite(v)and 0<=v<=limit for v,limit in zip(values,[.02,.1,.01])):raise RuntimeError('prior required AF axis not passed; further dispatch blocked:'+row['axis'])
    geometry=geometry_gate(sid);blockers=[];remaining=[];reused=[];cached=[];reuse_evidence=[];verifier=None
    with sqlite3.connect('file:'+str(b.R/'records/budget/PB_attempts.sqlite')+'?mode=ro',uri=True)as db:
        used=db.execute('select count(*)from attempts').fetchone()[0];running=db.execute("select job_key from attempts where status in ('running','reserved')").fetchall()
        if running:blockers.append('prior running/reserved attempt requires verified wait/reconciliation')
        for row in rows:
            key=sid+'_further_nested_fine_'+str(row['rank']);r=db.execute('select id,status,result_path from attempts where job_key=?',(key,)).fetchone();rp=V/'records/SES_map_fields'/(key+'.json')
            if r:
                if r[1]!='completed'or not rp.exists()or Path(r[2])!=rp:blockers.append('existing noncompleted/misbound field:'+key);continue
                field=b.load(rp)
                if field.get('status')!='completed'or field['PB_attempt_id']!=r[0]:blockers.append('invalid completed reuse:'+key);continue
                if verifier is None:verifier=reuse.ReuseVerifier(sid,V)
                sample,evidence=verifier.sampling(row);cached.append(sample);reuse_evidence.append(evidence);reused.append(key)
            else:
                if rp.exists()or(V/'runs'/key).exists():blockers.append('unreceipted field directory:'+key)
                remaining.append(key)
    if used+len(remaining)>120:blockers.append('all prior/failed attempts plus remaining exceeds120')
    for p in [b.R/'records/budget/resources.json',b.R/'records/budget/further_resources.json',V/'records/disk_reservations.json']:
        if p.exists()and b.load(p):blockers.append('nonempty reservation requires owner/child reconciliation:'+str(p))
    check=check_host();projected=summary['predicted_incremental_retained_bytes'];actual=branch_bytes()
    # Full conservative bound is preserved; remaining resumption can use the
    # frozen650GiB admission bound without charging already retained data twice.
    if not reused and (projected is None or projected>650*GIB or check['NVMe_free_bytes']<projected+512*GIB):blockers.append('initial whole-reference storage bound fails')
    report={'schema':'AF_nested_further_execution_preflight_v1','ready':not blockers,'blockers':blockers,'case':sid,'heavy_root':str(V),'canonical_ledger':str(b.R/'records/budget/PB_attempts.sqlite'),'existing_all_attempts':used,'required_new_attempts':len(remaining),'verified_reusable_fields':len(reused),'total_attempts_if_success':used+len(remaining),'other_frozen_axes_completed':sorted(required),'per_field_logical_CPU_capacity':2,'per_field_physical_cores':1,'actual_native_APBS_threads':1,'concurrent_fields':2,'total_logical_CPU_limit':4,'total_physical_cores':2,'total_memory_reservation_GiB':192,'host_margin_GiB':128,'automatic_low_memory_serial_fallback':False,'NVMe_branch_hard_limit_GiB':650,'NVMe_global_free_floor_GiB':512,'projected_total_new_retained_bytes':projected,'current_branch_bytes':actual,'host':check,'geometry_plan':b.ref(b.R/'cases'/sid/'further/nested_plan.json'),'runner':b.ref(__file__),'source_numerical_solver':b.ref(b.P/'code/SES_map_solver_v3_runtime.py'),'native_ledger':b.ref(b.P/'code/PB_ledger.py'),'failed_attempts_not_deleted':True}
    report['geometry_admission']=geometry
    report['cached_sampling_receipts']=cached;report['completed_reuse_evidence']=reuse_evidence;report['reuse_validator']=b.ref(reuse.__file__);report['reuse_unique_files_actually_hashed']=len(verifier.hashes)if verifier else 0
    b.write(b.R/'records/further_execution_preflight.json',report);return report

def execute(sid):
    summary,rows,pre,pqr,mesh,q,nn,w=load_plan(sid);metadata=[{k:v for k,v in x.items()if k!='query_indices'}for x in rows];gate=preflight(sid,summary,metadata)
    if not gate['ready']:raise RuntimeError('further preflight: '+str(gate['blockers']))
    authority=b.load(b.R/'contracts/authority_and_budget.json');authority={**authority,'schema':'AF_bridge_further_execution_authority_v1','supersedes_resource_limits_for_further_only':True,'memory_reservation_GiB':192,'max_concurrent_fields':2,'native_APBS_threads':1,'CPU_capacity_per_field':2,'NVMe_branch_hard_limit_GiB':650,'NVMe_global_free_floor_GiB':512,'canonical_ledger':str(b.R/'records/budget/PB_attempts.sqlite'),'execution_preflight_sha256':b.sha(b.R/'records/further_execution_preflight.json')};b.write(V/'contracts/authority_and_budget.json',authority)
    # Preserve the already authorized island: four logical CPUs, two physical
    # cores. Each worker receives one complete SMT pair and APBS stays1thread.
    check_host();cpu=b.load(b.R/'records/CPU_isolation_contract.json');cpus=cpu['fixed_logical_CPUs'];pairs=cpu['worker_pairs']
    if not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('fixed authorized CPUs unavailable')
    ctx=multiprocessing.get_context('spawn');cq=ctx.Queue();cq.put(pairs[0]);cq.put(pairs[1]);os.sched_setaffinity(0,cpus)
    b.write(b.R/'records/further_launch.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'epoch':time.time(),'affinity':cpus,'worker_SMT_pairs':pairs,'logical_CPUs':4,'physical_cores':2,'actual_native_APBS_threads_per_field':1,'preflight':b.ref(b.R/'records/further_execution_preflight.json'),'runner':b.ref(__file__)})
    cached=gate['cached_sampling_receipts'];dispatch=reuse.pending_rows(metadata,cached)
    if cached:
        stage=b.R/'records/resume_stages'/('resume_'+str(time.time_ns()));stage.mkdir(parents=True)
        copies={}
        for name,path in {'CPU_contract':b.R/'records/CPU_isolation_contract.json','preflight':b.R/'records/further_execution_preflight.json','launch':b.R/'records/further_launch.json','runner':Path(__file__),'reuse_validator':Path(reuse.__file__),'prior_failure':b.R/'records/further_dispatch_failure.json'}.items():
            if path.exists():target=stage/(name+path.suffix);shutil.copyfile(path,target);copies[name]=b.ref(target)
        receipt={'schema':'verified_sampling_reuse_resume_v1','status':'validated_for_resume','completed_sampling_reused_without_rewrite':len(cached),'remaining_new_fields':len(dispatch),'total_attempts_if_success':gate['total_attempts_if_success'],'snapshots':copies,'resolved_pre_PB_failure_sha256':copies.get('prior_failure',{}).get('sha256'),'cached_sampling_evidence':gate['completed_reuse_evidence'],'numerical_one_field_function_unchanged':True,'epoch':time.time()}
        b.write(stage/'resume_receipt.json',receipt);b.write(b.R/'records/further_resume_receipt.json',receipt)
        hp=b.R/'records/further_resume_history.json';history=b.load(hp)if hp.exists()else {'schema':'further_execution_stage_history_v1','original_interrupted_stage':b.ref(b.R/'records/resume_stages/stage01_end_runtime_upgrade_guard/manifest.json'),'resumed_stages':[]}
        history['resumed_stages'].append(b.ref(stage/'resume_receipt.json'));b.write(hp,history)
        b.write(b.R/'records/further_progress.json',{'completed_sampled_fields':len(cached),'expected_fields':len(rows),'resumed_completed_base':len(cached),'epoch':time.time()})
    def success(result,count):
        total=len(cached)+count;b.write(b.R/'records/further_progress.json',{'completed_sampled_fields':total,'expected_fields':len(rows),'resumed_completed_base':len(cached),'last_rank':result['rank'],'epoch':time.time()});print('FURTHER_DONE',result['rank'],total,'/',len(rows),flush=True)
    def failure(errors,inflight):b.write(b.R/'records/further_dispatch_failure.json',{'errors':errors,'new_dispatch_stopped':True,'already_inflight_allowed_to_finish':inflight,'epoch':time.time()})
    results=[];errors=[];dispatched=[]
    if dispatch:
        with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=init_worker,initargs=(cq,sid))as pool:
            results,errors,dispatched=bounded_dispatch(pool,one_field,dispatch,success,failure)
    b.write(b.R/'records/further_dispatch_order.json',{'dispatched_ranks':dispatched,'errors':errors,'at_most_two_inflight':True})
    if errors:raise RuntimeError('further stopped after field failure; pending fields not dispatched: '+str(errors))
    results=reuse.merge_results(metadata,cached,results)
    d=b.R/'cases'/sid/'further';phi=np.full(len(q),np.nan);raw=phi.copy();accepted=np.zeros(len(q),bool);fields={};ancestor=b.R/'records/SES_map_fields'/(sid+'_overlap12_parent.json');fields[str(ancestor)]=b.ref(ancestor)
    for row in results:
        idx=np.load(row['query_indices']['path']);phi[idx]=np.load(row['phi']['path']);raw[idx]=np.load(row['raw']['path']);accepted[idx]=np.load(row['accepted']['path']);fields[row['field']['path']]=row['field'];fields[row['parent_field']['path']]=row['parent_field']
    assert np.isfinite(phi).all()
    for name,arr in [('phi',phi),('raw',raw),('weights',w)]:np.save(d/(name+'.npy'),arr)
    cfg=b.load(b.P/'config/production_candidate_SES_v2.json');bins=ProjectPaths(b.P).reference(cfg['representation']['shared_bins']);b.write(d/'source_case.json',pre)
    ins={'corrected_phi':b.ref(d/'phi.npy'),'raw_trilinear_phi':b.ref(d/'raw.npy'),'area_weights':b.ref(d/'weights.npy'),'vertices':b.ref(mesh/'vertices.npy'),'faces':b.ref(mesh/'faces.npy'),'shared_bins':b.ref(bins),'protocol_configuration':b.ref(b.P/'config/production_candidate_SES_v2.json'),'source_case_record':b.ref(d/'source_case.json')};spec={'sample_id':sid,'source_role':'AF_bridge_full_nested_further_development','inputs':ins,'PB_field_receipts':list(fields.values())};b.export(spec,d/'representation')
    b.write(d/'outcome.json',{'status':'completed','sample_id':sid,'variant':'further','representation':b.ref(d/'representation/manifest.json'),'phi':b.ref(d/'phi.npy'),'weights':b.ref(d/'weights.npy'),'fields':list(fields.values()),'nested_reference_plan':b.ref(d/'nested_plan.json'),'execution_preflight':b.ref(b.R/'records/further_execution_preflight.json'),'field_sampling_receipts':results,'fallback_area_fraction':float(w[~accepted].sum()/w.sum()),'numerical_qualified':False,'qualification_requires_all_frozen_axes':True})
    qual=b.comparisons(sid,bins);extra=[]
    for a,z,axis in [('parent_refined','refined','matched_parent_spacing'),('overlap12','further','matched_intermediate_further')]:
        da=b.R/'cases'/sid/a;dz=b.R/'cases'/sid/z;m=b.compare(np.load(da/'phi.npy'),np.load(da/'weights.npy'),np.load(dz/'phi.npy'),np.load(dz/'weights.npy'),b.load(bins)['internal_edges']);extra.append({'axis':axis,'metrics':m,'pass':m['Hellinger']<=.02 and m['W1_kBT_per_e']<=.1 and m['CDF_sup']<=.01})
    b.write(b.R/'results'/(sid+'_decomposed_comparisons.json'),{'rows':extra,'changes_do_not_replace_original_required_pairs':True});b.write(b.R/'records/further_complete.json',{'strict_per_instance_qualified':qual['numerical_qualified'],'completed_child_fields':len(results),'NVMe_branch_bytes':branch_bytes(),'epoch':time.time()})
def load_plan(sid):
    return reuse.frozen_plan(sid)if(b.R/'records/further_launch.json').exists()else f.plan(sid)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute(a.case)
    else:
        s,rows,*_=load_plan(a.case);report=preflight(a.case,s,[{k:v for k,v in x.items()if k!='query_indices'}for x in rows]);print(json.dumps({k:v for k,v in report.items()if k not in ['cached_sampling_receipts','completed_reuse_evidence']}))
if __name__=='__main__':main()
