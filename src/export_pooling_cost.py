"""Aggregate recorded compute/storage costs without sample or group identifiers.

Run after the final numerical summary is frozen. This exports existing evidence;
it does not claim to revalidate model predictions or measure new throughput.
"""
import argparse,csv,hashlib,json,statistics,time
from collections import defaultdict,Counter
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(values):
 values=list(values)
 return {'min':min(values),'median':statistics.median(values),'max':max(values)}
def directory_size(p):return sum(f.stat().st_size for f in Path(p).rglob('*') if f.is_file())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--expected',type=int,default=20);a=ap.parse_args();r=Path(a.root)
 receipts=[read(f) for f in sorted((r/'runs').glob('*/receipt.json'))]
 summary=read(r/'results/summary.json')
 queues=[(f,read(f)) for f in sorted((r/'records').glob('queue*.json'))]
 ready=len(receipts)==a.expected and all(x['status']=='complete' for x in receipts) and summary.get('status')=='complete' and summary.get('fits_completed')==a.expected and summary.get('expected_fits')==a.expected and queues and all(x.get('status')=='complete' for _,x in queues)
 if not ready:
  print(json.dumps({'status':'not_ready','receipts':len(receipts),'expected':a.expected,'summary_status':summary.get('status'),'queue_statuses':{f.name:x.get('status') for f,x in queues}}));return
 with (r/'results/selected_fits.csv').open(newline='') as f:selected_rows=list(csv.DictReader(f))
 selected_ids={x['run_id'] for x in selected_rows};assert len(selected_ids)==summary['selected']
 keys=set();groups=defaultdict(list);per_fit=[];queue_rows=[];queue_child_identities=set()
 stop_selection={}
 for x in receipts:
  c=x['config'];key=(c['mode'],c['seed'],c['fold'],c['lr']);assert key not in keys;keys.add(key)
  skey=key[:3];previous=stop_selection.get(skey)
  if previous is None or (x['best_stop_H'],c['lr'])<(previous['best_stop_H'],previous['config']['lr']):stop_selection[skey]=x
  groups[(c['mode'],c['seed'])].append(x)
  hfields=['release_sha256','cache_receipt_sha256','cache_index_sha256','script_sha256','model_sha256','parent_metrics_and_split_sha256']
  per_fit.append({'run_id':x['run_id'],'artifact_directory':'runs/'+x['run_id'],'mode':c['mode'],'seed':c['seed'],'fold':c['fold'],'lr':c['lr'],'selected_by_stop':x['run_id'] in selected_ids,'parameters':x['parameters'],'epochs':x['epochs'],'best_epoch':x['best_epoch'],'updates':x['updates'],'sample_presentations':x['sample_presentations'],'residue_presentations':x['residue_presentations'],'training_samples':x['training_samples'],'training_unique_T':x['training_proteins'],'training_groups':x['training_groups'],'stop_reason':x['stop_reason'],'fit_wall_seconds':x['fit_wall_seconds'],'cached_readout_sequences_per_second':x['cached_readout_sequences_per_second'],'cache_on_GPU':x['cache_on_GPU'],'peak_GPU_bytes':x['peak_GPU_bytes'],'peak_RSS_bytes':x['peak_RSS_bytes'],'hashes':{k:c[k] for k in hfields},'checkpoint_sha256':x['checkpoint_sha256'],'prediction_sha256':x['prediction_sha256'],'environment':x['environment']})
 assert selected_ids=={x['run_id'] for x in stop_selection.values()},'selected CSV does not match stop-only selection'
 for f,q in queues:
  attempts=q['attempts'];seed=int(q.get('seed',1701));mode=q['mode'];child_wall=[]
  for z in attempts:
   assert z['returncode']==0 and z['ended_epoch']>=z['started_epoch'];key=(mode,seed,z['fold'],z['lr']);assert key not in queue_child_identities;queue_child_identities.add(key);child_wall.append(z['ended_epoch']-z['started_epoch'])
  queue_rows.append({'record':'records/'+f.name,'record_sha256':sha(f),'mode':mode,'seed':seed,'status':q['status'],'attempts':len(attempts),'queue_wall_seconds':q['ended_epoch']-q['started_epoch'],'child_process_wall_seconds_sum':sum(child_wall),'child_process_wall_seconds_range':interval(child_wall)})
 assert queue_child_identities==keys,'queue and completed fit identities differ'
 aggregates=[]
 for (mode,seed),part in sorted(groups.items()):
  sel=[x for x in part if x['run_id'] in selected_ids]
  aggregates.append({'mode':mode,'seed':seed,'fits':len(part),'selected_fits':len(sel),'fit_wall_seconds_sum':sum(x['fit_wall_seconds'] for x in part),'fit_wall_seconds_range':interval(x['fit_wall_seconds'] for x in part),'epochs_sum':sum(x['epochs'] for x in part),'epochs_range':interval(x['epochs'] for x in part),'updates_sum':sum(x['updates'] for x in part),'updates_range':interval(x['updates'] for x in part),'sample_presentations_sum':sum(x['sample_presentations'] for x in part),'residue_presentations_sum':sum(x['residue_presentations'] for x in part),'stop_reasons':dict(Counter(x['stop_reason'] for x in part)),'cached_readout_sequences_per_second_range':interval(x['cached_readout_sequences_per_second'] for x in part),'peak_GPU_bytes_max':max(x['peak_GPU_bytes'] for x in part),'peak_RSS_bytes_max':max(x['peak_RSS_bytes'] for x in part),'selected_parameters':sorted({x['parameters'] for x in sel}),'selected_cached_readout_sequences_per_second_range':interval(x['cached_readout_sequences_per_second'] for x in sel),'selected_run_ids':[x['run_id'] for x in sel]})
 cache=read(r/'cache/receipt.json');cache_bytes={name:(r/'cache'/name).stat().st_size for name in ['flat.npy','index.parquet','mean.npy','max.npy','receipt.json']}
 plans={name:sha(r/'records'/name) for name in ['plan.json','amendment.json','amendment_timing_note.json','confirmation_selection.json'] if (r/'records'/name).exists()}
 used_hashes={key:sorted({row['hashes'][key] for row in per_fit}) for key in per_fit[0]['hashes']}
 assert used_hashes['cache_receipt_sha256']==[sha(r/'cache/receipt.json')]
 total_fit=sum(x['fit_wall_seconds'] for x in receipts);assert abs(total_fit-summary['fit_wall_seconds_total'])<1e-6
 result={'schema':'pooling.cost.aggregate.v1','stage':'pooling_20261105','status':'complete','expected_fits':a.expected,'completed_fits':len(receipts),'selected_fits':len(selected_ids),'source_summary_sha256':sha(r/'results/summary.json'),'source_selected_fits_sha256':sha(r/'results/selected_fits.csv'),'exporter_sha256':sha(__file__),'plan_sha256':plans,'used_source_hashes':used_hashes,'time_definitions':{'fit_wall_seconds':'Trainer timer starts after imports, target/cache loading, normalization and GPU-cache transfer; includes epochs, stop evaluation, final checkpoint/prediction serialization and repeated cached-readout timing; excludes final receipt checksum/write and process cleanup. It is elapsed time, not GPU busy time.','queue_wall_seconds':'Queue process elapsed time; includes child startup, data/cache loading, fit, checksum and teardown. Sum across concurrent queues is not stage end-to-end elapsed time.','head_queue_span_seconds':'Last queue end minus first queue start, including inter-queue analysis/selection gaps; not an isolated compute benchmark.','cached_readout_sequences_per_second':'Single repeated prediction pass on existing cached features; vector summaries or full residue cache already resident as recorded. Includes readout batching/normalization as implemented and device-to-host prediction output, excludes encoder inference, cache building/loading and general sequence-to-p deployment I/O. Very short timings are microbenchmarks.','encoder_seconds':'Reported launch-to-receipt sum includes startup and artifact hashing for both workers and pilot; it is not profiled GPU busy time.','residue_presentations':'Input sequence lengths summed over training presentations for every mode; vector modes do not perform residue-wise neural readout work, so equal counts do not imply equal FLOPs.'},'totals':{'fit_wall_seconds_sum':total_fit,'queue_wall_seconds_sum':sum(z['queue_wall_seconds'] for z in queue_rows),'child_process_wall_seconds_sum':sum(z['child_process_wall_seconds_sum'] for z in queue_rows),'head_queue_span_seconds':max(q['ended_epoch'] for _,q in queues)-min(q['started_epoch'] for _,q in queues),'stop_reasons':dict(Counter(x['stop_reason'] for x in receipts)),'selected_budget_limited':sum(x['stop_reason']!='early_stop' for x in receipts if x['run_id'] in selected_ids)},'by_mode_seed':aggregates,'queues':queue_rows,'fits':per_fit,'encoder_and_cache':{'rows':cache['rows'],'residues':cache['residues'],'width':cache['width'],'dtype':cache['dtype'],'compute':cache['encoder_compute'],'worker_reported_seconds_sum':cache['encoder_worker_reported_seconds'],'launch_to_receipt_seconds_sum':cache['encoder_launch_to_receipt_seconds_sum'],'publisher_seconds':cache['publisher_seconds'],'encoder_peak_GPU_bytes':cache['encoder_peak_GPU_bytes'],'encoder_peak_RSS_bytes':cache['encoder_peak_RSS_bytes'],'publisher_peak_RSS_bytes':cache['publisher_peak_RSS_bytes'],'published_file_bytes':cache_bytes,'published_directory_bytes':directory_size(r/'cache'),'work_cache_directory_bytes':directory_size(r/'cache_work'),'artifact_sha256':cache['artifacts'],'receipt_sha256':sha(r/'cache/receipt.json'),'source_sequence_snapshot_sha256':cache['source_sha256'],'worker_identities':cache['worker_identities'],'quantized_mean_vs_legacy':cache['quantized_mean_vs_legacy']},'qualification':{'retrospective_development':True,'evaluated_folds':sorted({x['config']['fold'] for x in receipts}),'seeds':sorted({x['config']['seed'] for x in receipts}),'no_end_to_end_deployment_throughput_measured':True,'new_C_transfer_permitted':False,'full_U_deployment_validated':False,'summary_only_no_sample_or_group_identifiers':True}}
 plan=read(r/'records/plan.json');amendment=read(r/'records/amendment.json');cache_budget=read(r/'records/cache_launch_full.json')
 head_cap=amendment.get('same_total_head_GPU_hours_cap',plan['total_head_GPU_hours_cap'])*3600;encoder_cap=plan['encoder_GPU_hours_cap']*3600
 cache_total=result['encoder_and_cache']['published_directory_bytes']+result['encoder_and_cache']['work_cache_directory_bytes']
 result['budget_observation']={'head_seconds_cap':head_cap,'head_queue_wall_seconds_sum':result['totals']['queue_wall_seconds_sum'],'head_queue_wall_sum_within_cap':result['totals']['queue_wall_seconds_sum']<=head_cap,'head_queue_sum_interpretation':'Conservative allocated-time proxy from one queue per assigned GPU; includes process overhead, not measured GPU busy time.','encoder_seconds_cap':encoder_cap,'encoder_launch_to_receipt_seconds_sum':cache['encoder_launch_to_receipt_seconds_sum'],'encoder_within_cap':cache['encoder_launch_to_receipt_seconds_sum']<=encoder_cap,'cache_disk_bytes_cap':cache_budget['residue_cache_disk_bytes_cap'],'cache_total_bytes':cache_total,'cache_disk_within_cap':cache_total<=cache_budget['residue_cache_disk_bytes_cap']}
 text=json.dumps(result,indent=2,allow_nan=False)+'\n';assert '/data/' not in text and 'sequence\"' not in text
 target=r/'results/cost_public.json';tmp=target.with_suffix('.tmp');tmp.write_text(text);tmp.replace(target)
 print(json.dumps({'status':'complete','fits':len(receipts),'selected':len(selected_ids),'totals':result['totals'],'by_mode_seed':[{k:v for k,v in x.items() if k in ['mode','seed','fit_wall_seconds_sum','epochs_range','selected_cached_readout_sequences_per_second_range','peak_GPU_bytes_max','peak_RSS_bytes_max']} for x in aggregates],'encoder_launch_to_receipt_seconds_sum':cache['encoder_launch_to_receipt_seconds_sum']},indent=2))
if __name__=='__main__':main()
