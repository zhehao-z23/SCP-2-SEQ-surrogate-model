"""Explicitly project completed private diagnostics into aggregate public drafts.

No arrays, models, network, sample memberships or per-case records are read.
This is an archival filter, not a recomputation of the underlying diagnostics.
"""
import argparse,hashlib,json,shutil
from pathlib import Path

PINS={
 'representation':'995a2e7b6502226ea39540a752b0574af0cf498d2b90919732447cf6f87ad41f',
 'coarse':'4fec38480424061afc9ca4207682564da20758e95406d07190c60ac18cebbfb3',
 'duplicate':'40d97475b6dbdce650add4d10e1c39162eec3fd6f273cd3a9c80a14a4ff6595a',
 'input':'c12bf8ae57550127e2d3c3ead0c3ac8c7eaa963e1d675f2d0a09c23fa7a54eb2',
 'oof':'675c87743ed787135fa99b1d25c98ebfa0fa1bb427a06681c96f5b54b7e292cd',
 'optimization':'9d3d7855f7841c593ac745a549e5362ead50c9c8ec39ca5bd94cdd5014686575'}
NAMES={'representation':'bridge_representation_audit_summary.json','coarse':'bridge_coarse_shape_support_summary.json',
 'duplicate':'bridge_duplicate_T_audit_summary.json','input':'bridge_duplicate_T_input_closure_summary.json',
 'oof':'bridge_shape_model_comparison_summary.json','optimization':'bridge_shape_optimization_addendum_summary.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def take(x,keys):return {k:x[k] for k in keys}
def put(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def result(x):
 return take(x,['family','seed','metrics','support','negative','positive','fixed_one_percent_ranking'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists();a.output.mkdir(parents=True)
 for n in ['src','results','docs','configs']:(a.output/n).mkdir()
 data={}
 for key,name in NAMES.items():
  path=a.input/name;assert sha(path)==PINS[key];data[key]=json.loads(path.read_text());assert data[key]['status'] in ['complete','passed']
 r=data['representation'];c=data['coarse'];d=data['duplicate'];i=data['input'];o=data['oof'];opt=data['optimization']
 representation=take(r,['outcome','identity_status','cohort','quantiles','any_quantile_mismatch','quantile_tolerance','shape','finite_certifiable_Bowley_interval_width','method','limits','mathematical_preflight','elapsed_seconds','peak_RSS_KiB'])
 representation['current_count_normalization_bit_equal_FP32']=True
 coarse=take(c,['data_rows','unique_T','groups','counts','unique_T_consensus_classes_without_averaging_rows','uncertainty_reason_row_counts_nonexclusive','per_view_uncertainty_reasons_nonexclusive','open_tail_mass_support','both_open_tails_nonzero','open_tail_presence_by_consensus','support_count_note','finite_window','finite_bins','native_width_range','classification_scope','unconfirmed_is_not_absence'])
 coarse['resource']=take(c['resource'],['BLAS_threads','GPU','elapsed_seconds','peak_RSS_KiB','address_space_limit_GiB','wall_budget_seconds'])
 duplicate=take(d,['selection','cohort','duplicate_support','nonidentical_p_support','same_identity_pair_counts','unavailable_identity_comparison_pair_counts','pair_H','max_abs_p_difference','max_abs_saved_raw_QC_differences','affected_max_metric_contribution_support_bounds','triangle_inequality_lower_bounds_if_one_deterministic_T_prediction','controls','not_identified'])
 duplicate['actual_input_closure']=take(i,['status','rows','duplicate_T_pairs','charge_shape_per_row','charge_dtype','charge_byte_identical_pairs','NaN_mask_identical_pairs','same_cache_index_offset_length_pairs','current_model_input_scope','embedding_arrays_loaded','embedding_bytes_rehashed','cache_check'])
 duplicate['elapsed_seconds']={'target_QC_audit':d['resource']['elapsed_seconds'],'input_closure':i['resource']['elapsed_seconds']}
 oof=take(o,['cohort','OOF_fits','seeds','seeds_are_not_independent_sample_replicates','definitions','limits','elapsed_seconds','peak_RSS_KiB'])
 oof['results']=[result(z) for z in o['results']]
 # Only aggregate residual statistics, not the per-run replay/member list.
 oof['probability_replay']={k:v for k,v in o['probability_replay'].items() if isinstance(v,(str,int,float,bool)) or v is None}
 pairs=[]
 for z in opt['pairs']:
  p=take(z,['family','fold','seed','cohort','reference_cohort','roles_and_order_identical','same_cache_receipt','scalers_loaded_in_this_analysis','common_support_paired_metrics','interpretation'])
  p['different_configuration_fields']=sorted(z['different_config_values'])
  p['plateau']=result(z['plateau_summary']);p['reference']=result(z['reference_summary'])
  pairs.append(p)
 optimization=take(opt,['optimization_observations','old_fixed_LR_references','saved_prediction_files','paired_comparisons','complete_fivefold_OOF','new_fits','model_loads','max_abs_H_replay_difference','limits','elapsed_seconds','peak_RSS_KiB'])
 optimization['pairs']=pairs
 optimization['oracle_fold_references']=[{'fold':z['fold'],'cohort':z['cohort'],**result(z)} for z in opt['oracle_fold_references']]
 optimization['preserved_failed_attempt']={'stage':'v1','failure':'A source-cohort string was overwritten by a same-named cohort-count field at final aggregation, after 16 prediction/role/metric checks.','repair':'Use source_cohort for provenance, retaining cohort for counts; successful v2 changes no mathematical or scientific definition.','old_records_preserved_private':True}
 for name,payload in [('representation',representation),('coarse_shape',coarse),('duplicate_T',duplicate),('shape_OOF',oof),('shape_optimization',optimization)]:
  put(a.output/'results'/(name+'.json'),{'schema':'public_aggregate_diagnostic_v1','status':'completed_development_diagnostic','diagnostic':name,'new_fits':0,'new_physical_solves':0,'raw_phi_reads':0,'model_loads':0,'public_sample_memberships':False,'results':payload})
 sources={
  'bridge_binned_quantile_bounds.py':r['bounds_code_sha256'],
  'bridge_representation_audit.py':r['code_sha256'],
  'bridge_coarse_shape_support.py':c['code_sha256'],
  'bridge_duplicate_T_audit.py':d['source_code_sha256'],
  'bridge_duplicate_T_input_closure.py':i['source_code_sha256'],
  'bridge_shape_model_comparison.py':o['code_sha256'],
  'bridge_shape_optimization_addendum.py':opt['code_sha256']}
 records=[]
 for name,pin in sources.items():
  assert sha(a.input/name)==pin;shutil.copyfile(a.input/name,a.output/'src'/name)
  records.append({'path':'src/'+name,'actual_execution_source_sha256':pin,'archive_sha256':pin,'adaptation':'byte_for_byte; original completed execution, not rerun by this archive filter'})
 shutil.copyfile(__file__,a.output/'src'/Path(__file__).name)
 records.append({'path':'src/'+Path(__file__).name,'actual_execution_source_sha256':sha(__file__),'archive_sha256':sha(__file__),'adaptation':'New archival projection executed locally on pinned aggregate JSON only; does not recompute any scientific measurement'})
 plan=a.input/'bridge_coarse_shape_support_plan.json';assert sha(plan)==c['plan_sha256'];shutil.copyfile(plan,a.output/'configs/coarse_shape_plan.json')
 provenance={'schema':'diagnostic_public_projection_provenance_v1','status':'draft_not_published','sources':records,'private_source_summary_sha256':PINS,'source_release_sha256':d['release_sha256'],'historical_label_contract_sha256':d['label_contract_sha256'],'projection':'Explicitly selected aggregate fields only. No per-sample tables, identities, run membership, prediction arrays, weights, cache addresses, paths, PNG, SVG or HTML.','executed_original_sources_preserved':True,'source_normalization_or_path_rewrite':False,'private_reproduction_inputs_required':True,'failed_optimization_attempt_preserved':optimization['preserved_failed_attempt']}
 put(a.output/'docs/source_provenance.json',provenance)
 print(json.dumps({'status':'draft_prepared','source_count':len(records),'aggregate_result_files':5,'new_scientific_computation':False}))
if __name__=='__main__':main()
