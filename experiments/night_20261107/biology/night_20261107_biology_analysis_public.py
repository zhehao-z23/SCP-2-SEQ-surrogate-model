"""Frozen, retrospective physical-screening utility on existing development OOF.

Metadata-only legacy annotation projection; no new fit, raw phi, protected target,
or spatial/functional inference from a histogram. Case selection precedes OOF reads.
"""
import argparse,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--stage',choices=['select','evaluate'],required=True);a=ap.parse_args()
 assert __debug__ and os.sched_getaffinity(0)=={178} and not os.environ.get('CUDA_VISIBLE_DEVICES')
 assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
 resource.setrlimit(resource.RLIMIT_AS,(8<<30,8<<30))
 import numpy as np
 import pandas as pd
 import pyarrow.parquet as pq
 plan=load(a.plan);assert sha(__file__)==plan['code_sha256'];R=Path(plan['root']);out=R/(a.stage+'_v3');assert not out.exists();out.mkdir();start=time.monotonic();refs=[]
 def bind(p,expected):
  p=Path(p);actual=sha(p);assert actual==expected,str(p);refs.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size});return p
 def sizes(d):return {'rows':len(d),'unique_T':int(d.sequence_sha256.nunique()),'groups':int(d.group_id.nunique())}
 try:
  put(out/'started.json',{'pid':os.getpid(),'time':time.time(),'stage':a.stage,'code_sha256':sha(__file__),'plan_sha256':sha(a.plan),'CPU':[178],'BLAS_threads':1,'AS_GiB':8,'wall_budget_seconds':600})
  # Fixed production worker-slot separation, not OS-wide exclusivity.
  bind(R/'records/cpu_inspection_v3.json',plan['cpu_inspection_sha256']);host=load(R/'records/cpu_inspection_v3.json');phase=Path(os.environ['SEQ2SCP_PRODUCTION_PHASE'])
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
  S=Path(plan['project_root']);data=S/'atlas_scale_20261105/data';supp=S/'bridge_20261106/diagnostic_supplements'
  bind(data/'release.json',plan['release_sha256']);release=load(data/'release.json');bind(data/'samples.parquet',release['artifacts']['samples.parquet'])
  m=pq.read_table(data/'samples.parquet',columns=['sample_id','sequence_sha256','group_id','fold','length','coverage','target_row','source','sequence']).to_pandas()
  assert m.sample_id.is_unique and len(m)==21921 and m.sequence_sha256.nunique()==21915 and m.group_id.nunique()==7999
  role=S/'bridge_20261106/roles';bind(role/'code/bridge_roles_guard.py',plan['guard_code_sha256']);sys.path.insert(0,str(role/'code'));from bridge_roles_guard import check
  request={'stage':'development','sequence_sha256':sorted(m.sequence_sha256.unique().tolist()),'aliases':[]}
  put(out/'role_preflight.json',[check(role/'registry_v1/guard_contract.json',plan['guard_contract_sha256'],dict(operation=op,**request)) for op in ['read_target','model_eval']])
  # No physical/cluster archive column is ever requested or decoded.
  annpath=R/'source_private/final_scp_atlas_table_local_backfill.parquet';bind(annpath,plan['annotation_sha256'])
  acols=['sample_id','entry_id','chain_id','entity_id','source_entity_id','rcsb_entity_id','protein_description_rcsb','protein_names_uniprot','primary_uniprot_accession','uniprot_accessions_rcsb','organism_scientific_name_rcsb','organism_uniprot','go_ids_uniprot','go_terms','Function [CC]','entry_title_rcsb','resolution_combined_rcsb','auth_asym_ids','asym_ids']
  ann=pq.read_table(annpath,columns=acols,filters=[('sample_id','in',m.sample_id.tolist())]).to_pandas();assert ann.sample_id.is_unique
  ann=ann.set_index('sample_id').reindex(m.sample_id).reset_index()
  import re
  def text(v):
   if v is None:return ''
   if isinstance(v,(list,tuple,np.ndarray)):return ' '.join(map(str,v))
   if isinstance(v,float) and not np.isfinite(v):return ''
   return str(v)
  for c in acols[1:]:ann[c]=ann[c].map(text)
  d=m.drop(columns='sequence').copy()
  for c in acols[1:]:d[c]=ann[c].to_numpy()
  gotext=d.go_ids_uniprot+' '+d.go_terms;names=d.protein_description_rcsb+' '+d.protein_names_uniprot
  d['GO_present']=gotext.str.contains(r'GO:\d{7}',regex=True)
  d['RNA_direct_GO']=gotext.str.contains(r'(?<!\w)GO:0003723(?!\d)',regex=True)
  d['DNA_direct_GO']=gotext.str.contains(r'(?<!\w)GO:0003677(?!\d)',regex=True)
  d['ribosomal_name']=names.str.contains(r'\bribosomal protein\b',regex=True,case=False)
  d['RNA_annotation_cohort']=d.RNA_direct_GO|d.ribosomal_name
  d['nominal_KR_DE_density']=m.sequence.map(lambda s:(s.count('K')+s.count('R')-s.count('D')-s.count('E'))/len(s))
  assert m.apply(lambda row:len(row.sequence)==row.length and hashlib.sha256(row.sequence.encode()).hexdigest()==row.sequence_sha256,axis=1).all()
  bind(data/'p.npy',release['artifacts']['p.npy']);y=np.load(data/'p.npy',allow_pickle=False)
  cpath=S/'source/labels/source_snapshots/vectors_q05_995_bins63_raw_count.parquet';bind(cpath,plan['counts_sha256'])
  counts=pq.read_table(cpath,filters=[('sample_id','in',m.sample_id.tolist())]).to_pandas().set_index('sample_id').loc[m.sample_id][[f'bin_{i:03d}' for i in range(63)]].to_numpy()
  assert np.issubdtype(counts.dtype,np.integer) and (counts>=0).all() and np.array_equal((counts/counts.sum(1)[:,None]).astype(np.float32),y)
  contract=S/'data/labels/release.pending.json';bind(contract,plan['label_contract_sha256']);edges=np.asarray(load(contract)['edges']);assert len(edges)==62
  helper=supp/'code/bridge_tail_fraction_bounds.py';bind(helper,plan['fraction_helper_sha256']);sys.path.insert(0,str(helper.parent));from bridge_tail_fraction_bounds import fraction_bounds
  for side,interval in [('positive',(5.,np.inf)),('negative',(-np.inf,-5.))]:
   lo,hi=fraction_bounds(counts,edges,interval);d[side+'_mass_lower']=lo;d[side+'_mass_upper']=hi;d[side+'_certified_concentrated']=lo>=.8;d[side+'_uncertain_concentration']=(lo<.8)&(hi>=.8)
  rep=supp/'representation_audit_v1';bind(rep/'summary.json',plan['representation_summary_sha256']);rs=load(rep/'summary.json');bind(rep/'per_sample.parquet',rs['private_table']['sha256'])
  raw=pq.read_table(rep/'per_sample.parquet',columns=['sample_id','q25_raw_QC','q50_raw_QC','q75_raw_QC','raw_IQR','raw_Bowley','raw_negative_absolute_shape','raw_positive_absolute_shape']).to_pandas().set_index('sample_id').loc[m.sample_id]
  for c in raw.columns:d[c]=raw[c].to_numpy()
  # Canonical measurement view fixed lexicographically; never average p between rows.
  canonical=d.sort_values(['sequence_sha256','sample_id']).drop_duplicates('sequence_sha256').copy()
  assert canonical.groupby('group_id').fold.nunique().max()==1
  q=canonical.RNA_annotation_cohort&(canonical.length>=80)&(canonical.length<=500)&(canonical.coverage>=.9)&canonical.primary_uniprot_accession.ne('')
  candidates=canonical[q].copy()
  if a.stage=='select':
   selected=[]
   if len(candidates):
    positive=candidates[candidates.positive_certified_concentrated].sort_values(['positive_mass_lower','sequence_sha256','sample_id'],ascending=[False,True,True])
    if len(positive):
     first=positive.iloc[0];selected.append({'alias':'RNA_high_positive','sample_id':first.sample_id,'rule':'Highest true positive mass lower bound among fixed annotated/length/coverage/UniProt-eligible canonical T with lower>=0.8; ties by sequence SHA then ID.'})
     contrast=candidates[(candidates.group_id!=first.group_id)&(candidates.length/first.length>=.8)&(candidates.length/first.length<=1.25)&(abs(candidates.nominal_KR_DE_density-first.nominal_KR_DE_density)<=.05)].sort_values(['positive_mass_upper','sequence_sha256','sample_id'])
     if len(contrast):selected.append({'alias':'RNA_charge_matched_contrast','sample_id':contrast.iloc[0].sample_id,'rule':'Different group; same fixed eligible RNA annotation cohort, length ratio0.8..1.25, nominal KR-DE density difference<=0.05; lowest true positive mass upper bound; ties SHA/ID. Contrast, not established biological negative.'})
   # Existing frozen failure/control remain fixed regardless of new ranking results.
   old=supp/'shape_case_facts_v2/private_case_packet.json';bind(old,plan['previous_case_packet_sha256']);oldpacket=load(old)
   for c in oldpacket['cases']:
    if c['case_type']=='raw_absolute_positive':selected.append({'alias':'previous_positive_shape_case' if c['role']=='frozen_badcase' else 'previous_shape_control','sample_id':c['sample_id'],'rule':'Previously frozen distribution-shape case/control; biologically annotated after selection, not chosen for model success.'})
   chosen=d.set_index('sample_id').loc[[x['sample_id'] for x in selected]].reset_index()
   put(out/'case_selection.json',{'status':'frozen_before_current_OOF_reads','cases':selected,'all_rules':plan['case_rules'],'selected_rows':json.loads(chosen.to_json(orient='records',double_precision=15)),'existing_prior_exposure':'Two reused cases and global model summaries were already inspected in earlier development; two new RNA cases selected solely from current truth/annotation rules.'})
   d.to_parquet(out/'development_annotations_and_physical_targets_private.parquet',index=False)
   summary={'status':'complete','cohort':sizes(d),'canonical_T_rows':len(canonical),'annotation_rows_missing_entry_id':int(d.entry_id.eq('').sum()),
    'RNA_direct_GO':sizes(d[d.RNA_direct_GO]),'DNA_direct_GO':sizes(d[d.DNA_direct_GO]),'ribosomal_name':sizes(d[d.ribosomal_name]),'RNA_union':sizes(d[d.RNA_annotation_cohort]),'GO_present':sizes(d[d.GO_present]),'RNA_case_eligible':sizes(candidates),
    'physical_targets':{side:{'canonical_certified':int(canonical[side+'_certified_concentrated'].sum()),'canonical_uncertain':int(canonical[side+'_uncertain_concentration'].sum()),'RNA_canonical_certified':int(canonical.loc[canonical.RNA_annotation_cohort,side+'_certified_concentrated'].sum())} for side in ['positive','negative']},
    'case_selection_sha256':sha(out/'case_selection.json'),'private_table_sha256':sha(out/'development_annotations_and_physical_targets_private.parquet'),'no_current_OOF_prediction_reads':True,
    'annotation_scope':'Cached explicit GO:0003723 OR ribosomal protein name; no GO ancestor expansion, no functional-negative inference from missing annotation.'}
  else:
   select=load(R/'select_v3/summary.json');bind(R/'select_v3/case_selection.json',select['case_selection_sha256']);selection=load(R/'select_v3/case_selection.json');assert selection['status']=='frozen_before_current_OOF_reads'
   bind(R/'select_v3/development_annotations_and_physical_targets_private.parquet',select['private_table_sha256']);prior=pd.read_parquet(R/'select_v3/development_annotations_and_physical_targets_private.parquet');assert d.equals(prior)
   selected_ids=[x['sample_id'] for x in selection['cases']];rowpos=pd.Series(np.arange(len(m)),index=m.sample_id)
   base=supp/'shape_model_comparison_v1';bind(base/'summary.json',plan['shape_summary_sha256']);ss=load(base/'summary.json');bind(base/'identity_and_prediction_replay.json',ss['identity_sha256']);si=load(base/'identity_and_prediction_replay.json');pins={x['path']:x['sha256'] for x in si['refs']}
   metric_helper=supp/'code/bridge_shape_model_comparison.py';bind(metric_helper,plan['shape_helper_sha256']);from bridge_shape_model_comparison import probability_errors,macro,uniform_quantiles
   fits_path=S/'overnight_20261106/confirmation/results/all_fits.csv';bind(fits_path,plan['fit_index_sha256']);fits=pd.read_csv(fits_path)
   predictions={};replay=[];fitrefs=[]
   for (family,seed),ff in fits.groupby(['family','seed']):
    assert family in ['composition_charge_deep','mean_pre','residue_mean'] and int(seed) in [1701,3407] and set(ff.fold)==set(range(5))
    pred=np.full_like(y,np.nan)
    for _,f in ff.iterrows():
     run=Path(f.artifact_path)
     for n in ['receipt.json','roles.json','evaluation_prediction.npy']:bind(run/n,pins[str(run/n)])
     rec=load(run/'receipt.json');roles=load(run/'roles.json');p=np.load(run/'evaluation_prediction.npy',allow_pickle=False);ids=rowpos.loc[roles['evaluation_ids']].to_numpy();assert np.array_equal(ids,np.flatnonzero(m.fold==f.fold))
     assert not set(m.set_index('sample_id').loc[roles['train_ids']].group_id)&set(m.iloc[ids].group_id)
     assert p.shape==(len(ids),63) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
     z=m.iloc[ids][['sample_id','sequence_sha256','group_id']].copy()
     for key,val in probability_errors(y[ids],p).items():z[key]=val
     result=macro(z,['H','CDF_MSE','edge_MAE']);diff=abs(result['H']['group_macro']-float(f.H));assert diff<1e-7
     replay.append({'family':family,'seed':int(seed),'fold':int(f.fold),'H_replay_absdiff':diff});pred[ids]=p;fitrefs.append({'path':str(run),'receipt_sha256':sha(run/'receipt.json'),'prediction_sha256':sha(run/'evaluation_prediction.npy')})
    assert np.isfinite(pred).all();predictions[(family,int(seed))]=pred
   # All fits are retained, two seeds separate; no best-seed or checkpoint selection.
   cidx=rowpos.loc[canonical.sample_id].to_numpy();units=canonical.reset_index(drop=True)
   ranks=[];ranktables=[];error=[]
   def budget_record(pop,name,scores,truth_score,side,fraction,model,seed):
    z=units.loc[pop].copy();s=np.asarray(scores)[pop];true=np.asarray(truth_score)[pop];K=int(np.ceil(len(z)*fraction));assert len(z)>0
    z['score']=s;z['true_score']=true;z['hash_tie']=z.sequence_sha256
    order=z.sort_values(['score','hash_tie','sample_id'],ascending=[False,True,True]);truthorder=z.sort_values(['true_score','hash_tie','sample_id'],ascending=[False,True,True]);sel=order.head(K);toptruth=set(truthorder.head(K).sequence_sha256);cert=side+'_certified_concentrated'
    M=int(z[cert].sum());hits=int(sel[cert].sum());top_hits=int(sel.sequence_sha256.isin(toptruth).sum());ann=int(sel.RNA_annotation_cohort.sum())
    r={'population':name,'side':side,'model':model,'seed':seed,'N_T':len(z),'N_groups':int(z.group_id.nunique()),'budget_fraction':fraction,'K_T':K,'selected_groups':int(sel.group_id.nunique()),'certified_target_T':M,'TP_certified_T':hits,'TP_certified_groups':int(sel.loc[sel[cert],'group_id'].nunique()),'recall_certified':hits/M if M else None,'precision_certified':hits/K,'random_expected_TP_certified':K*M/len(z),'true_topK_lower_bound_hits_T':top_hits,'true_topK_recall':top_hits/K,'random_expected_topK_hits':K*K/len(z),'selected_RNA_annotated_T':ann,'selected_GO_present_T':int(sel.GO_present.sum()),'RNA_annotation_background_T':int(z.RNA_annotation_cohort.sum()),'RNA_annotation_yield':ann/K,'truth_lower_mean_selected':float(sel[side+'_mass_lower'].mean()),'truth_upper_mean_selected':float(sel[side+'_mass_upper'].mean())}
    ranks.append(r);order['rank']=np.arange(1,len(order)+1);order['selected']=order['rank']<=K;order['population']=name;order['side']=side;order['model']=model;order['seed']=seed;ranktables.append(order[['sample_id','sequence_sha256','group_id','population','side','model','seed','rank','selected','score','true_score']])
   allpop=np.ones(len(units),bool);rnapop=units.RNA_annotation_cohort.to_numpy(bool)
   for side,interval in [('positive',(5.,np.inf)),('negative',(-np.inf,-5.))]:
    true=units[side+'_mass_lower'].to_numpy();simple=units.nominal_KR_DE_density.to_numpy()*(1 if side=='positive' else -1)
    for pop,name,fraction in [(allpop,'all_development_canonical_T',.01),(rnapop,'explicit_RNA_annotation_canonical_T',.10)]:
     if not pop.any():continue
     budget_record(pop,name,true,true,side,fraction,'oracle_true63_lower_bound',None)
     budget_record(pop,name,simple,true,side,fraction,'nominal_KR_DE_density',None)
     for (family,seed),p in predictions.items():
      lo,hi=fraction_bounds(p,edges,interval);budget_record(pop,name,lo[cidx],true,side,fraction,family,seed)
   pd.concat(ranktables,ignore_index=True).to_parquet(out/'rankings_private.parquet',index=False)
   packet=[]
   for selected in selection['cases']:
    sid=selected['sample_id'];i=int(rowpos[sid]);r=d.iloc[i];views={}
    for (family,seed),p in predictions.items():
     q,_=uniform_quantiles(p[[i]],edges);iqr=float(q[0,2]-q[0,0]);bow=float((q[0,0]+q[0,2]-2*q[0,1])/iqr) if np.isfinite(q).all() and iqr>=1 else None
     metrics={k:float(v[0]) for k,v in probability_errors(y[[i]],p[[i]]).items()}
     views[family+'_s'+str(seed)]={'p':p[i].tolist(),'metrics':metrics,'uniform_proxy_Bowley':bow,'positive_mass_bounds':[float(v[0]) for v in fraction_bounds(p[[i]],edges,(5.,np.inf))],'negative_mass_bounds':[float(v[0]) for v in fraction_bounds(p[[i]],edges,(-np.inf,-5.))]}
    packet.append({**selected,'metadata_and_truth':json.loads(r.to_frame().T.to_json(orient='records',double_precision=15))[0],'true_p':y[i].tolist(),'true_counts':counts[i].tolist(),'models':views})
   put(out/'case_plot_packet_private.json',{'cases':packet,'finite_edges':edges.tolist(),'target':'Historical equal-retained-vertex potential histogram, not new-C area PMF; open-tail mass shown separately.','selection_sha256':sha(R/'select_v3/case_selection.json')})
   duplicate=[]
   for tsha,g in m.groupby('sequence_sha256'):
    if len(g)>1:
     ii=g.index.to_numpy();duplicate.append({'T':tsha,'rows':len(g),'same_fold':bool(g.fold.nunique()==1),'truth_maxdiff':float(np.max(abs(y[ii]-y[ii[0]]))),'prediction_maxdiff':max(float(np.max(abs(p[ii]-p[ii[0]]))) for p in predictions.values())})
   summary={'status':'complete','ranking_results':ranks,'case_selection_sha256':sha(R/'select_v3/case_selection.json'),'case_packet_sha256':sha(out/'case_plot_packet_private.json'),'ranking_table_sha256':sha(out/'rankings_private.parquet'),'OOF_fits':30,'max_H_replay_absdiff':max(x['H_replay_absdiff'] for x in replay),'canonical_T':len(units),'duplicate_T_count':len(duplicate),'duplicate_prediction_maxdiff':max(x['prediction_maxdiff'] for x in duplicate),'duplicate_truth_maxdiff':max(x['truth_maxdiff'] for x in duplicate),
    'random_reference':'Expected counts under uniformly choosing K distinct T, not a new random-fit model or significance test.','annotation_limit':'Missing direct GO or name term is not known functional absence. Annotation yield is descriptive and dependent on deposited/cached annotation coverage.','target_limit':'Certified concentration is a sufficient lower-bound condition for >=80% mass beyond +/-5. Other true cases can be unclassified by63 bins. true_topK is the specific lower-bound-score ranking, not an exact full-phi ranking.','seeds':'Two realizations on the same T/groups, not independent sample-size doubling.','case_counts':len(packet)}
   put(out/'OOF_replay_and_duplicate_checks_private.json',{'replays':replay,'duplicate_T':duplicate,'fit_refs':fitrefs})
  put(out/'identity.json',{'refs':refs,'annotation_column_allowlist':acols,'metadata_projection_only':True,'old_archive_physical_and_cluster_columns_decoded':False})
  summary.update(code_sha256=sha(__file__),plan_sha256=sha(a.plan),identity_sha256=sha(out/'identity.json'),resources={'CPU':[178],'threads':1,'wall_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},new_fits=0,models_loaded=0,raw_phi_reads=0,protected_targets_read=0)
  put(out/'summary.json',summary);print(json.dumps({'status':'complete','stage':a.stage,'summary_sha256':sha(out/'summary.json'),'resources':summary['resources']}))
 except Exception as exc:put(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'refs_so_far':refs});raise

if __name__=='__main__':main()
