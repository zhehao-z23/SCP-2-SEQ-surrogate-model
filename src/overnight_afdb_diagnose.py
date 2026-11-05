"""Inference and feature-shift diagnostic only; AFDB physical accuracy is not measured."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(x):return {str(q):float(np.quantile(x,q)) for q in [0,.5,.9,.95,.99,1]}
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--pooling',required=True);ap.add_argument('--parent',required=True);ap.add_argument('--project',required=True);a=ap.parse_args()
r=Path(a.root);pool=Path(a.pooling);parent=Path(a.parent);project=Path(a.project);start=time.perf_counter()
sys.path.insert(0,str(pool/'code'));sys.path.insert(0,str(project/'package/src'))
from pooling_model import PoolingModel
from seq2scp_training.charge_features import extract_features,VERSION
from seq2scp_training.features import VIEWS
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
source=r/'data/overnight_afdb_private_panel.json';panel=json.loads(source.read_text());rows=panel['rows'];cache=r/'cache/worker0'
cr=json.loads((cache/'receipt_afdb01.json').read_text());assert cr['status']=='complete' and cr['completed_rows']==len(rows)==64
for name,h in cr['artifacts'].items():assert sha(cache/name)==h
index=json.loads((cache/'index.json').read_text());flat=np.load(cache/'flat.npy');means=np.load(cache/'mean.npy');assert flat.nbytes<1024**3
assert sum(x['length'] for x in rows)==flat.shape[0] and np.isfinite(flat).all()
order={x['row_index']:i for i,x in enumerate(index)};means=means[[order[i] for i in range(64)]]
for i,x in enumerate(rows):assert hashlib.sha256(x['sequence'].encode()).hexdigest()==x['sequence_sha256'] and x['length']==len(x['sequence'])
features=[extract_features(x['sequence'],sequence_view='legacy_AF_annotation_sequence',full_chain=True) for x in rows]
charge=np.asarray([[np.nan if f[k] is None else f[k] for k in VIEWS['composition_charge']] for f in features],dtype='float32');assert np.isfinite(charge).all()
selection=pd.read_csv(pool/'results/selected_fits.csv');sel=selection[(selection.seed==1701)&(selection.fold==0)&selection['mode'].isin(['mean_pre','residue_mean'])];assert len(sel)==2
data=pd.read_parquet(parent/'data/samples.parquet');pos=pd.Series(np.arange(len(data)),index=data.sample_id)
ci=pd.read_parquet(pool/'cache/index.parquet');lookup=pd.Series(np.arange(len(ci)),index=ci.sequence_sha256);atlas_mean=np.load(pool/'cache/mean.npy')[lookup.loc[data.sequence_sha256].to_numpy()];atlas_charge=np.load(parent/'data/composition_charge.npy')
outputs={};costs=[];identities=[];shifts=[];prediction_checks=[]
device_flat=torch.tensor(flat,device='cuda');gpu_means=torch.tensor(means,device='cuda')
for rec in sel.itertuples():
 path=pool/'runs'/rec.run_id;receipt=json.loads((path/'receipt.json').read_text());assert sha(path/'best.pt')==receipt['checkpoint_sha256'];t=time.perf_counter()
 ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);assert ck['config']['mode']==rec.mode
 model=PoolingModel(rec.mode).cuda().eval();model.load_state_dict(ck['state_dict']);torch.cuda.synchronize();load_sec=time.perf_counter()-t
 em=ck['embedding_mean'];es=ck['embedding_std'];cm=ck['charge_mean'];cs=ck['charge_std'];med=ck['charge_median']
 extra=torch.tensor(np.concatenate([(np.where(np.isnan(charge),med,charge)-cm)/cs,np.isnan(charge).astype('float32')],1),device='cuda')
 emg=torch.tensor(em,device='cuda');esg=torch.tensor(es,device='cuda');vec=(gpu_means-emg)/esg
 def predict():
  with torch.inference_mode():
   if rec.mode=='mean_pre':return model(vec,extra).softmax(1).cpu().numpy()
   result=[]
   for i in range(64):
    ix=index[order[i]];x=(device_flat[ix['offset']:ix['offset']+ix['length']].float()-emg)/esg
    result.append(model(x[None],extra[i:i+1],torch.ones((1,ix['length']),dtype=torch.bool,device='cuda')).softmax(1).cpu().numpy())
   return np.concatenate(result)
 torch.cuda.synchronize();t=time.perf_counter();p=predict();torch.cuda.synchronize();first=time.perf_counter()-t
 assert p.shape==(64,63) and np.isfinite(p).all() and (p>=0).all() and np.max(abs(p.sum(1)-1))<1e-6
 times=[];max_repeat=0.
 for _ in range(5):
  torch.cuda.synchronize();t=time.perf_counter();again=predict();torch.cuda.synchronize();times.append(time.perf_counter()-t);max_repeat=max(max_repeat,float(abs(again-p).max()))
 assert max_repeat<1e-6;outputs[rec.mode]=p;np.save(r/'data'/('predictions_'+rec.mode+'.npy'),p,allow_pickle=False)
 prediction_checks.append({'mode':rec.mode,'probability_shape':[64,63],'finite_nonnegative':True,'sum_error_max':float(abs(p.sum(1)-1).max()),'five_repeats_max_abs_difference':max_repeat})
 costs.append({'mode':rec.mode,'head_load_seconds':load_sec,'first_forward_seconds':first,'cached_forward_5_repeats_seconds':times,'cached_forward_median_sequences_per_second':64/np.median(times),'inference_batching':'all64' if rec.mode=='mean_pre' else 'singleton_no_padding','comparison_limit':'Different inference batching; observed implementation cost, not intrinsic model speed ratio'})
 roles=json.loads((path/'roles.json').read_text());tr=pos.loc[roles['train_ids']].to_numpy();az=(atlas_mean[tr]-em)/es;tz=(means-em)/es;ac=(np.where(np.isnan(atlas_charge[tr]),med,atlas_charge[tr])-cm)/cs;tc=(charge-cm)/cs
 for name,fit,test in [('mean_embedding_standardized_RMS',np.sqrt((az**2).mean(1)),np.sqrt((tz**2).mean(1))),('charge_standardized_RMS',np.sqrt((ac**2).mean(1)),np.sqrt((tc**2).mean(1)))]:
  cut=float(np.quantile(fit,.99));shifts.append({'mode':rec.mode,'diagnostic':name,'atlas_fit_rows':len(fit),'atlas_quantiles':quant(fit),'afdb_quantiles':quant(test),'AF_above_training_99th_percentile_count':int((test>cut).sum()),'interpretation':'Descriptive feature shift, not calibrated OOD probability or electrostatic error; length and candidate sampling differ'})
 identities.append({'mode':rec.mode,'checkpoint_sha256':receipt['checkpoint_sha256'],'config_sha256':sha(path/'config.json'),'role_sha256':sha(path/'roles.json'),'seed':1701,'fold':0})
h=np.sqrt(.5*((np.sqrt(outputs['mean_pre'].astype('float64'))-np.sqrt(outputs['residue_mean'].astype('float64')))**2).sum(1));cdf=((outputs['mean_pre'].cumsum(1)[:,:-1]-outputs['residue_mean'].cumsum(1)[:,:-1])**2).mean(1);tail=abs(outputs['mean_pre'][:,[0,-1]]-outputs['residue_mean'][:,[0,-1]]).mean(1)
strata=[]
for b in sorted({x['selection_bin'] for x in rows}):
 mask=np.asarray([x['selection_bin']==b for x in rows]);strata.append({'length_bin':b,'rows':int(mask.sum()),'prediction_disagreement_H_mean':float(h[mask].mean()),'prediction_disagreement_H_quantiles':quant(h[mask]),'prediction_disagreement_CDF_mean':float(cdf[mask].mean()),'prediction_disagreement_tail_mean':float(tail[mask].mean())})
summary={'schema':'AFDB.sequence.engineering_diagnostic.v1','status':'complete','scope':'Frozen biased legacy AFDB annotation/candidate sequences; no paired qualified physical labels and no accuracy metric','source':{k:v for k,v in panel['source'].items() if k!='path'},'source_panel_sha256':sha(source),'script_sha256':sha(__file__),'sequence_checks':{'rows':64,'unique_sequences':len({x['sequence_sha256'] for x in rows}),'canonical_AA20':True,'length_range':[min(x['length'] for x in rows),max(x['length'] for x in rows)],'length_quantiles':quant([x['length'] for x in rows]),'residues':len(flat),'no_truncation':True,'exact_sequence_overlap_with_Atlas':sum(x['sequence_sha256'] in set(data.sequence_sha256) for x in rows),'AF_v4_coordinate_sequence_mapping_verified':False},'feature_version':VERSION,'identities':identities,'probability_checks':prediction_checks,'feature_shift':shifts,'prediction_disagreement':{'meaning':'Difference between two models, not error against physics or calibrated uncertainty','mean_H':float(h.mean()),'H_quantiles':quant(h),'strata':strata},'cost':{'encoder_receipt_elapsed_seconds':cr['elapsed_seconds'],'encoder_peak_GPU_bytes':cr['peak_GPU_bytes'],'encoder_peak_RSS_bytes':cr['peak_RSS_bytes'],'residue_cache_bytes':flat.nbytes,'mean_cache_bytes':means.nbytes,'head_costs':costs,'diagnostic_wall_seconds':time.perf_counter()-start,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'boundary':'Encoder and diagnostic are measured separate processes; sum is stage-duration estimate, not observed continuous end-to-end wall time; no whole-AFDB throughput claim'},'new_training_fits':0,'physical_calls':0,'accuracy_evaluated':False,'next_decision':'Input/model interfaces work; preserve mean as cheap deployment reference. Before an AF accuracy challenge, bind exact structure-sequence/protocol and trustworthy compatible labels; current disagreement cannot select a winner.'}
(r/'results/summary_public.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))
