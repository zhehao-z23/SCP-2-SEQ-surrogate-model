"""One bounded cold-process sequence-to-two-head replay, no training or physical labels."""
import time
started=time.perf_counter()
import argparse,csv,hashlib,json,os,resource,sys
from pathlib import Path
import numpy as np,torch

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--pooling',required=True);ap.add_argument('--project',required=True);ap.add_argument('--backend-src',required=True);ap.add_argument('--checkpoint-manifest',required=True);a=ap.parse_args()
r=Path(a.root);pool=Path(a.pooling);project=Path(a.project)
sys.path.insert(0,a.backend_src);sys.path.insert(0,str(pool/'code'));sys.path.insert(0,str(project/'package/src'))
from seq2scp_training.hf_backend import LocalHFBackend
from seq2scp_training.charge_features import extract_features
from seq2scp_training.features import VIEWS
from pooling_model import PoolingModel
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=json.loads((r/'data/overnight_afdb_sequences.json').read_text());assert len(rows)==64
manifest=json.loads(Path(a.checkpoint_manifest).read_text());backend=LocalHFBackend(manifest,device='cuda:0')
assert not backend.model.training and not any(p.requires_grad for p in backend.model.parameters())
selected=list(csv.DictReader((pool/'results/selected_fits.csv').open()))
selected=[x for x in selected if x['seed']=='1701' and x['fold']=='0' and x['mode'] in ['mean_pre','residue_mean']];assert len(selected)==2
models={};checkpoints={};predictions={};cost={}
for item in selected:
 mode=item['mode'];path=pool/'runs'/item['run_id']/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);m=PoolingModel(mode).cuda().eval();m.load_state_dict(ck['state_dict']);models[mode]=m
 checkpoints[mode]={k:torch.tensor(ck[k],device='cuda') for k in ['embedding_mean','embedding_std','charge_mean','charge_std']};predictions[mode]=[];cost[mode]=[]
torch.cuda.synchronize();load=time.perf_counter()-started;encoder=[];feature=[];lengths=[]
for row in rows:
 assert time.perf_counter()-started<1800
 assert hashlib.sha256(row['sequence'].encode()).hexdigest()==row['sequence_sha256']
 t=time.perf_counter();f=extract_features(row['sequence'],sequence_view='legacy_AF_annotation_sequence',full_chain=True);charge=np.asarray([f[k] for k in VIEWS['composition_charge']],dtype='float32');assert np.isfinite(charge).all();feature.append(time.perf_counter()-t)
 torch.cuda.synchronize();t=time.perf_counter();z=backend.extract_batch([row['sequence']],[30],precision='fp32')[0];torch.cuda.synchronize();encoder.append(time.perf_counter()-t)
 assert z['sequence_sha256']==row['sequence_sha256'] and z['positions']==list(range(row['length']))
 mat=z['layers'][30];assert mat.shape==(row['length'],960) and np.isfinite(mat).all();mat=mat.astype('float16').astype('float32');lengths.append(row['length'])
 with torch.inference_mode():
  for mode,model in models.items():
   torch.cuda.synchronize();t=time.perf_counter();ck=checkpoints[mode];ch=torch.tensor(charge,device='cuda');extra=torch.cat([(ch-ck['charge_mean'])/ck['charge_std'],torch.zeros(39,device='cuda')])[None]
   if mode=='mean_pre':
    x=(torch.tensor(mat.mean(0,dtype=np.float32),device='cuda')-ck['embedding_mean'])/ck['embedding_std'];p=model(x[None],extra).softmax(1).cpu().numpy()
   else:
    x=(torch.tensor(mat,device='cuda')-ck['embedding_mean'])/ck['embedding_std'];mask=torch.ones((1,row['length']),dtype=torch.bool,device='cuda');p=model(x[None],extra,mask).softmax(1).cpu().numpy()
   torch.cuda.synchronize();cost[mode].append(time.perf_counter()-t);predictions[mode].append(p)
torch.cuda.synchronize();elapsed=time.perf_counter()-started;checks=[]
for mode,parts in predictions.items():
 p=np.concatenate(parts);prior=np.load(r/'data'/('predictions_'+mode+'.npy'))
 assert p.shape==(64,63) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-6
 err=float(abs(p-prior).max());assert np.allclose(p,prior,atol=1e-5,rtol=1e-4),(mode,err)
 checks.append({'mode':mode,'previous_cache_route_prediction_max_abs_difference':err,'probability_sum_error_max':float(abs(p.sum(1)-1).max()),'all_rows_pass':True})
result={'schema':'AFDB.cold_process_e2e.v1','status':'complete','rows':64,'residues':sum(lengths),'length_range':[min(lengths),max(lengths)],'cold_process_sequence_to_two_predictions_seconds':elapsed,'scope':'Includes Python scientific-library imports, local encoder and two trained-head loads, descriptors, fresh full-sequence encoder inference, both head predictions and CPU transfer. Excludes remote source discovery/transfer and posthoc feature-shift analysis. Local OS filesystem may be warm; no network/model download. Two heads share one extraction. Singleton batching for both.','cold_import_and_model_load_seconds':load,'encoder_forward_seconds_sum':sum(encoder),'descriptor_seconds_sum':sum(feature),'head_including_normalization_transfer_seconds_sum':{k:sum(v) for k,v in cost.items()},'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'checks':checks,'script_sha256':sha(__file__),'source_sha256':sha(r/'data/overnight_afdb_sequences.json'),'checkpoint_manifest_sha256':sha(a.checkpoint_manifest),'precision':'FP32 extraction/math; residue FP16 roundtrip matches training cache','accuracy_evaluated':False,'warning':'Measured engineering pipeline only. Prediction parity and throughput do not establish AFDB physical accuracy or whole-library throughput.'}
(r/'results/e2e_public.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
