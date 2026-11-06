"""Complete uncertainty accounting for frozen ranking budgets; no reranking or new fits."""
import hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(os.environ['BIOLOGY_ROOT']);S=Path(os.environ['SEQ2SCP_ROOT'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert __debug__ and os.sched_getaffinity(0)=={178};resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30));start=time.monotonic()
out=R/'uncertainty_addendum_v3';assert not out.exists();out.mkdir()
plan=load(R/'plan_v3.json')
assert sha(R/'plan_v3.json')=='355f07057b60a002eea5c62d61f8df2a56ef335e538bc800e34b981bd5f8d711'
assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
assert not os.environ.get('CUDA_VISIBLE_DEVICES')
def bind(p,expected):
 assert sha(p)==expected,str(p)
 return p
# Fixed production worker-slot separation, not OS-wide exclusivity.
bind(R/'records/cpu_inspection_v3.json',plan['cpu_inspection_sha256']);host=load(R/'records/cpu_inspection_v3.json');phase=Path(os.environ['SEQ2SCP_PRODUCTION_PHASE'])
review=R/'records/runtime_metadata_revision_review_v1.json';assert sha(review)=='5cc34976b4312c95d063036068973d280261fc8aa1a85d42dfc31909a5efc8eb'
review_data=load(review);assert review_data['pass'] and review_data['prior_runtime_sha256']==host['runtime']['upgrade_sha256']
host['runtime']['upgrade_sha256']=review_data['runtime_sha256']
assert sha(phase/'runtime_upgrade.json')==host['runtime']['upgrade_sha256'];assert host['runtime']['phase_plan'].get('host_resource_policy') is None
for source_ref in host['static_code_refs']:bind(source_ref['path'],source_ref['sha256'])
def core(cpu):
 tp=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
 return (int((tp/'physical_package_id').read_text()),int((tp/'core_id').read_text()))
pilot={core(178),core(179)};live=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  argv=(proc/'cmdline').read_bytes().split(b'\0');cmd=b' '.join(argv).decode(errors='replace')
  controller='production_controller.py' in cmd;worker=any(k in cmd for k in ['science_worker.py','worker_bootstrap.py']) or any(Path(v.decode(errors='replace')).name=='apbs' for v in argv if v)
  if not (controller or worker):continue
  mask=sorted(os.sched_getaffinity(int(proc.name)));tick=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
  if controller:
   assert host['runtime']['release'] in cmd, 'New unreviewed production controller release'
   reps={}
   for cpu in mask:reps.setdefault(core(cpu),cpu)
   reserved=set(list(reps)[:host['runtime']['phase_plan']['cpu_slots']]);assert not reserved&pilot
  if worker:
   for task in (proc/'task').iterdir():assert not {core(cpu) for cpu in os.sched_getaffinity(int(task.name))}&pilot
  live.append({'pid':int(proc.name),'start_ticks':tick,'controller':controller,'worker':worker,'affinity':mask})
 except (FileNotFoundError,ProcessLookupError):continue
assert sha(phase/'runtime_upgrade.json')==host['runtime']['upgrade_sha256']
put(out/'live_cpu_gate.json',{'observed_unix':time.time(),'pass':True,'live_processes':live,'old_controller_absent_is_not_failure':True,'not_OS_exclusive':True,'no_production_restart':True})
mem=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'));assert mem>=32*1024**2
assert sha(R/'select_v3/summary.json')=='501696471a9779356fb3143f0588dfde6a2f1a3997f75658a01baf1ddf29ef80'
sel=load(R/'select_v3/summary.json');ev=load(R/'evaluate_v3/summary.json');assert sha(R/'evaluate_v3/summary.json')=='a78182e94cbfc66cc2a23ddc2e1af0cc03702fbce7f5db35c09976696a59d2c5'
path=R/'select_v3/development_annotations_and_physical_targets_private.parquet';assert sha(path)==sel['private_table_sha256']
# Read identity columns first, guard, then derived target flags.
ids=pd.read_parquet(path,columns=['sample_id','sequence_sha256']);guard=S/'bridge_20261106/roles/code/bridge_roles_guard.py'
assert sha(guard)=='5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3';sys.path.insert(0,str(guard.parent));from bridge_roles_guard import check
req={'stage':'development','sequence_sha256':sorted(ids.sequence_sha256.unique().tolist()),'aliases':[]}
put(out/'role_preflight.json',check(guard.parent.parent/'registry_v1/guard_contract.json','71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c',dict(operation='read_target',**req)))
d=pd.read_parquet(path).sort_values(['sequence_sha256','sample_id']).drop_duplicates('sequence_sha256').set_index('sample_id')
rp=R/'evaluate_v3/rankings_private.parquet';assert sha(rp)==ev['ranking_table_sha256'];rank=pd.read_parquet(rp);rows=[]
for old in ev['ranking_results']:
 q=(rank.population==old['population'])&(rank.side==old['side'])&(rank.model==old['model'])
 q &= rank.seed.isna() if old['seed'] is None else rank.seed.eq(old['seed'])
 r=rank[q];assert len(r)==old['N_T'] and r['selected'].sum()==old['K_T'] and r.sample_id.is_unique
 full=d.loc[r.sample_id];selected=d.loc[r.loc[r['selected'],'sample_id']];side=old['side'];P=full[side+'_mass_lower']>=.8;U=(full[side+'_mass_lower']<.8)&(full[side+'_mass_upper']>=.8)
 ps=selected[side+'_mass_lower']>=.8;us=(selected[side+'_mass_lower']<.8)&(selected[side+'_mass_upper']>=.8);M=int(P.sum());UU=int(U.sum());tp=int(ps.sum());uu=int(us.sum());K=len(selected);N=len(full)
 assert tp==old['TP_certified_T'] and M==old['certified_target_T']
 row={**old,'selected_certified_positive_T':tp,'selected_uncertain_T':uu,'selected_certified_negative_T':K-tp-uu,'unselected_certified_positive_T':M-tp,'unselected_uncertain_T':UU-uu,'total_uncertain_T':UU,
  'selected_certified_positive_groups':int(selected.loc[ps,'group_id'].nunique()),'selected_uncertain_groups':int(selected.loc[us,'group_id'].nunique()),'unselected_certified_positive_groups':int(full.loc[P&~full.index.isin(selected.index),'group_id'].nunique()),
  'true_positive_precision_bounds_from_labels':[tp/K,(tp+uu)/K],
  'true_positive_recall_bounds_from_labels':[tp/(M+UU-uu) if M+UU-uu else None,(tp+uu)/(M+uu) if M+uu else None],
  'physical_review_count_if_all_selected_rechecked':K,'not_reviewed_T':N-K,'physical_review_fraction':K/N,'nominal_all_case_review_count_reduction_fraction':1-K/N}
 assert tp+uu+row['selected_certified_negative_T']==K;rows.append(row)
# Frozen four-case compact biological/mapping facts; existing prediction arrays stay Byte.
packet_path=R/'evaluate_v3/case_plot_packet_private.json';assert sha(packet_path)==ev['case_packet_sha256'];packet=load(packet_path)
oldpath=S/'data/qualification/release_v1/sample_manifest.parquet';assert sha(oldpath)=='d108028ecc33c40af9402dbf841734ad7d1fe22f4d275f598ebc663f7525d59c'
old=pd.read_parquet(oldpath,columns=['sample_id','teacher_length','U_length','strict_sequence_equal','T_identity','T_query_coverage','T_U_coverage','binding_status','missing_residue_count','missing_acidic_DE','missing_basic_KRH'],filters=[('sample_id','in',[c['sample_id'] for c in packet['cases']])]).set_index('sample_id')
cases=[]
for c in packet['cases']:
 sid=c['sample_id'];item={k:v for k,v in c.items() if k not in ['true_p','true_counts','models']};item['T_U']=json.loads(old.loc[sid].to_frame().T.to_json(orient='records'))[0]
 item['models']={k:{j:v for j,v in z.items() if j!='p'} for k,z in c['models'].items()}
 item['budget_ranks']=[{'population':r['population'],'side':r['side'],'model':r['model'],'seed':None if pd.isna(r['seed']) else int(r['seed']),'rank':int(r['rank']),'selected':bool(r['selected'])} for r in rank[rank.sample_id==sid].to_dict('records')]
 cases.append(item)
put(out/'case_facts_private.json',{'cases':cases,'source_packet_sha256':sha(packet_path)})
result={'status':'complete','schema':'biological_use_budget_uncertainty_addendum_v3','runtime_review_sha256':sha(review),'predecessor':'v1 stopped before target reads at runtime drift; root explicitly accepted reviewed revision9ccdd8d3, scientific formulas/ranking inputs unchanged; v2 completed checks then failed serializing missing seed for non-model comparator, v3 serializes that missing seed as null','results':rows,'source_evaluation_summary_sha256':sha(R/'evaluate_v3/summary.json'),'ranking_table_sha256':sha(rp),'cohort_table_sha256':sha(path),'code_sha256':sha(__file__),
 'definitions':{'certified_positive':'true mass lower>=0.8','certified_negative':'true mass upper<0.8','uncertain':'lower<0.8<=upper, remains in ranking pool and budget','precision':'certified target hit/K is not exact true physical precision; bounds include unknown selected labels.','recall_bounds':'Min assign all unselected uncertain positive, selected uncertain negative; max reverse. These are conservative label-identification bounds, not statistical confidence intervals. Null only if denominator can be0.','cost':'N-to-K counts only, not unmeasured wall-time savings; known unselected certified positives remain missed. Same historical target/cohort, not a new-C or universe guarantee.'},
 'case_facts_sha256':sha(out/'case_facts_private.json'),'wall_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_fits':0,'raw_phi_reads':0}
put(out/'summary.json',result);print(json.dumps({'status':'complete','summary_sha256':sha(out/'summary.json'),'case_facts_sha256':sha(out/'case_facts_private.json'),'wall_seconds':result['wall_seconds']}))
