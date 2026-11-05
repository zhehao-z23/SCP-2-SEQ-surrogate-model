"""Export aggregate-only evidence; never export sequence/sample/group identifiers."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);a=p.parse_args();r=Path(a.root)
sys.path.insert(0,str(Path(a.source)/'package/src'))
from seq2scp_training.models import SequenceModel
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=read(r/'results/summary.json');assert summary['status']=='complete' and summary['fits_complete']==52
selection=pd.read_csv(r/'results/selected_fits.csv');allfits=pd.read_csv(r/'results/all_fits.csv');meta=pd.read_parquet(r/'data/samples.parquet')
rows=[]
for z in selection.itertuples():
 x=read(Path(z.artifact_path)/'receipt.json');c=x['config']
 model=SequenceModel(input_dim=c['input_dim'],bins=63,kind='vector',pool='mean',hidden=c['hidden'],dropout=c['dropout'])
 rows.append({'family':z.family,'scale':str(z.scale),'fold':z.fold,'run_id':z.run_id,'lr':z.lr,'seed':c['seed'],'input_dim':c['input_dim'],'hidden':c['hidden'],'dropout':c['dropout'],'parameters':sum(p.numel() for p in model.parameters()),'training_samples':x['training_samples'],'training_proteins':x['training_proteins'],'training_groups':x['training_groups'],'epochs':x['epochs'],'best_epoch':x['best_epoch'],'stop_reason':x['stop_reason'],'result':x['result'],'wall_seconds':x['wall_seconds'],'head_sequences_per_second':x['head_inference_sequences_per_second'],'peak_GPU_bytes':x['peak_gpu_allocated_bytes'],'peak_RSS_bytes':x['peak_rss_bytes'],'checkpoint_sha256':x['checkpoint_sha256'],'script_sha256':c['script_sha256']})
metrics=[]
for c in summary['learning_curves']:
 part=[z for z in rows if z['family']==c['family'] and z['scale']==str(c['scale'])]
 value=dict(c)
 for m in ['H','cutpoint_CDF_MSE','edge_bin_mass_MAE']:
  value['protein_macro_'+m]=sum(z['result']['protein_macro_'+m]*z['result']['proteins'] for z in part)/sum(z['result']['proteins'] for z in part)
 value['per_fold_H']={str(z['fold']):z['result']['group_macro_H'] for z in part}
 metrics.append(value)
cache=read(r/'data/esmc_300m_mean/receipt.json');queues=[]
for f in (r/'records').glob('queue_*.json'):
 q=read(f)
 if 'failed_environment' in f.name:continue
 if 'attempts' not in q:continue
 assert q['status']=='complete'
 queues.append({'family':q.get('family','composition_charge_deep'),'queue_seconds':q['ended_epoch']-q['started_epoch'],'fits':len(q['attempts'])})
replay=read(r/'results/checkpoint_replay.json');assert len(replay)==26 and all(z['passed'] for z in replay)
source=[]
for k,g in meta.groupby('source',dropna=False):source.append({'source':str(k),'rows':len(g),'unique_T':g.protein_id.nunique(),'groups':g.group_id.nunique()})
out={'schema':'atlas.scale.aggregate.v1','stage':'atlas_scale_20261105','status':'complete','fits':52,'selected_fits':26,'learning_curves':metrics,'paired':summary['paired'],'data':{'rows':len(meta),'unique_T':meta.protein_id.nunique(),'groups':meta.group_id.nunique(),'source_composition':source,'source_groups_may_overlap':True,'input':'teacher-retained T','target':'historical vertex probability p','bins':63,'release_sha256':sha(r/'data/release.json'),'label_contract_addendum_sha256':sha(r/'data/label_contract_addendum.json')},'cost':{'sum_of_individual_fit_wall_seconds':summary['fit_wall_seconds_total'],'fit_time_is_not_end_to_end_wall':True,'encoder_allocated_GPU_seconds_including_pilot':cache['extraction_GPU_allocated_seconds'],'encoder_peak_GPU_bytes':cache['peak_GPU_bytes'],'encoder_peak_RSS_bytes':cache['peak_RSS_bytes'],'mean_cache_bytes':(r/'data/esmc_300m_mean/mean.npy').stat().st_size,'queues':queues},'selected':rows,'checks':{'predictions_recomputed':summary['all_recomputed'],'checkpoint_reload_subset_checks':len(replay),'checkpoint_reload_rows_each':128,'checkpoint_reload_max_abs_error':max(z['CPU_fresh_reload_vs_saved_GPU_max_abs_error'] for z in replay),'selected_budget_limited':sum(z['stop_reason']!='early_stop' for z in rows),'all_fit_stop_counts':allfits.stop_reason.value_counts().to_dict(),'exact_legacy_encoder_overlap':cache['legacy_overlap_parity']['exact_sequence_overlap'],'legacy_encoder_max_abs_difference':cache['legacy_overlap_parity']['max_absolute_difference']},'engineering_repairs':[{'stage':'encoder','reason':'missing launch tools directory; exited before extraction','scientific_data_affected':False},{'stage':'capacity_control','reason':'encoder Python lacked pandas; exited before training; resumed using original head runtime wrapper','scientific_data_affected':False}],'qualification':{'retrospective_exposed_development':True,'seeds':[1701],'evaluation_folds':[0,1],'new_C_transfer_permitted':False,'initialization_benefit_tested':False,'full_U_deployment_validated':False,'capacity_control':'same hidden widths/depth/dropout, not exact total parameter matching','CDF':'unweighted mean squared error across 62 internal cutpoints, not continuous Wasserstein','tails':'mean absolute error of first and last bin masses'}}
(r/'results/aggregate_public.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'fits':out['fits'],'selected':out['selected_fits'],'cost':out['cost'],'checks':out['checks'],'source':source,'large_scale':[{k:v for k,v in c.items() if k!='selected_run_ids'} for c in metrics if str(c['scale'])=='all'],'paired_control':summary['paired'][-1]},indent=2))
