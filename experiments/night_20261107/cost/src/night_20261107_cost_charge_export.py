"""Aggregate-only supplement export; no model loading or target access."""
import argparse,hashlib,json,statistics,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--measurement',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads(a.previous.read_text());r=json.loads(a.measurement.read_text());assert r['status']=='complete' and r['rows']==95 and r['repeats']==3
strata=s['length_frame']['strata'];N=s['length_frame']['unique_sequences'];rows=[]
for k,st in enumerate(strata):
 key=f"{st['lo']}_{st['hi_exclusive']}"
 t=[x for x in r['timings'] if x['stratum']==key];c=[x for x in r['cached_singleton'] if x['stratum']==key]
 assert len(t)==len(c)==st['sample_n']*3
 z={'length_lo':st['lo'],'length_hi_exclusive':st['hi_exclusive'],'population_n':st['population_n'],'sample_n':st['sample_n']}
 for name in ['pipeline_seconds','feature_seconds','head_seconds']:
  z[name+'_by_repeat']=[statistics.mean(x[name] for x in t if x['repeat']==rep) for rep in range(3)]
 z['cached_head_seconds_by_repeat']=[statistics.mean(x['seconds'] for x in c if x['repeat']==rep) for rep in range(3)];rows.append(z)
def weighted(key):return [sum(x[key][rep]*x['population_n']/N for x in rows) for rep in range(3)]
w=weighted('pipeline_seconds_by_repeat');wh=weighted('cached_head_seconds_by_repeat');wf=weighted('feature_seconds_by_repeat');wn=weighted('head_seconds_by_repeat')
warm=statistics.mean(w[1:]);plm=statistics.mean(s['measured_surrogate']['historical_frame_weighted_seconds_per_sequence_by_repeat'][1:])
supp={'schema':'matched_charge_cost_aggregate_v1','measurement_sha256':sha(a.measurement),'code_sha256':r['code_sha256'],'plan_sha256':r['plan_sha256'],'trigger':'Bounded followup prompted by the independently frozen positive-tail screening comparison; no model or threshold selection in this cost benchmark.',
 'same_inputs_as_primary_benchmark':95,'repeats':3,'model':'Historical composition_charge_deep seed1701 exact five OOF heads; each input routed to one original fold, not an ensemble. Timing is not an accuracy comparison with the fold0 residue head.','encoder_loaded':False,'new_fits':0,'targets_read':False,
 'strata':rows,'historical_frame_weighted_pipeline_seconds_by_repeat':w,'historical_frame_weighted_cached_singleton_seconds_by_repeat':wh,'historical_frame_weighted_descriptor_seconds_by_repeat':wf,'historical_frame_weighted_head_seconds_by_repeat':wn,
 'warm_pipeline_seconds_per_sequence':warm,'warm_cached_singleton_seconds_per_sequence':statistics.mean(wh[1:]),'warm_feature_plus_head_seconds_per_sequence':statistics.mean(wf[1:])+statistics.mean(wn[1:]),
 'pipeline_definition':'Feature extraction, normalization/missing flags, transfer, exact head, output transfer and correctness-validation overhead. Feature+head sum excludes the per-input archived-feature assertion; additional full-library input/output IO is not included.',
 'primary_ESMC_residue_pipeline_divided_by_charge_pipeline':plm/warm,'ratio_scope':'Measured engineering route ratio on the same 95 inputs and historical-frame weights, singleton GPU1, not a matched-accuracy speedup; Python/PyTorch environments differ and batching can change this ratio.',
 'three_passes':r['end_to_end_passes'],'saved_OOF_prediction_maxdiff':max(x['saved_OOF_prediction_maxdiff'] for x in r['end_to_end_passes']),
 'cached_five_fold_batches_seconds':r['cached_five_fold_batches_seconds'],'warm_cached_batch_seconds_per_sequence':statistics.mean(r['cached_five_fold_batches_seconds'][1:])/95,
 'cached_batch_scope':'Small in-memory 95-input panel partitioned into five fold batches; not a sustained full-cache throughput measurement.',
 'load_seconds':r['load_seconds'],'new_process_to_first_95_predictions_seconds':r['cold_process_to_first_predictions_seconds'],'process_wall_seconds':r['process_wall_seconds'],'CPU_seconds':r['CPU_seconds'],'GPU_busy_seconds_measured':False,'peak_GPU_bytes':r['peak_GPU_bytes'],'peak_RSS_bytes':r['peak_RSS_bytes'],
 'failed_start':'First interpreter lacked pandas and stopped during imports before model/data loading. Original failure receipt retained; successful rerun used the existing training runtime. No installation or scientific-code change.',
 'conditional_scan_scenarios':[{'sequences':n,'core_singleton_process_hours':n*warm/3600,'warm_repeat_spread_hours_not_CI':[n*min(w[1:])/3600,n*max(w[1:])/3600],'plus25pct_unmeasured_overhead_hours':n*warm*1.25/3600,'plus100pct_unmeasured_overhead_hours':n*warm*2/3600} for n in [1000000,100000000]],
 'screening_link':{'independent_summary_sha256':'63cfcacc492495be8a006c6fb7772fa2b6af961f563c4b158354607e77a37a97','population_T':21915,'budget_T':220,'certified_positive_T':135,'unknown_T':374,'charge1701_selected':{'certified_positive':124,'unknown':90,'certified_negative':6},'charge3407_selected':{'certified_positive':121,'unknown':96,'certified_negative':3},'residue1701_selected':{'certified_positive':118,'unknown':99,'certified_negative':3},'residue3407_selected':{'certified_positive':117,'unknown':99,'certified_negative':4},'simple_KRDE_selected':{'certified_positive':46,'unknown':121,'certified_negative':53},'scope':'Historical vertex-measure label task and existing exposed T pool. Unknowns are not false positives; certified-positive recall is not recall against all physically positive objects. No new-C, AFDB or per-protein physical walltime guarantee. Selected-count reduction is an object-count comparison only.'},
 'limitations':['Same small deterministic length-stratified panel, not AFDB-random; population-weighting does not remove within-stratum bias.','Three repeated timings quantify repeat variability, not population uncertainty.','The primary benchmark uses FP32 ESMC plus residue head; this supplement is the independently specified historical charge-only architecture. Cost alone does not select a final model.','No inference above the observed maximum length, networked IO, sustained cache scan or multi-GPU scaling measured.']}
s['schema']='night_20261107_cost_public_v3';s['v2_summary_sha256']=sha(a.previous);s['revision_scope']='Adds the bounded same-input exact historical charge-only OOF route cost; all earlier measured values are retained.';s['created_unix']=time.time();s['charge_only_followup']=supp
with a.output.open('x') as f:json.dump(s,f,indent=2,allow_nan=False)
print(json.dumps({'output_sha256':sha(a.output),'warm_pipeline_ms':warm*1000,'warm_cached_singleton_ms':statistics.mean(wh[1:])*1000,'plm_route_ratio':plm/warm,'scenarios':supp['conditional_scan_scenarios']}))
