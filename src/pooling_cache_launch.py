"""Explicit bounded cache pilot/full launch with process identities."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--python',required=True);p.add_argument('--backend-src',required=True);p.add_argument('--checkpoint-manifest',required=True);p.add_argument('--attempt',choices=['pilot','full'],required=True);p.add_argument('--gpu-uuids',nargs=2,required=True,help='Two distinct assigned GPU UUIDs in worker-rank order');a=p.parse_args();r=Path(a.root)
gpus=a.gpu_uuids;assert len(set(gpus))==2 and all(g.startswith('GPU-') for g in gpus),'assign two distinct GPU UUIDs'
(r/'cache_work').mkdir(parents=True,exist_ok=True);(r/'records').mkdir(exist_ok=True)
record=r/'records'/('cache_launch_'+a.attempt+'.json');assert not record.exists(),'existing launch; inspect handles before any restart'
if a.attempt=='full':assert json.loads((r/'records/cache_pilot_check.json').read_text())['full_extraction_permitted'] is True
budget={'residue_cache_disk_bytes_cap':40*1024**3,'encoder_GPU_seconds_cap':7200,'worker_wall_seconds_cap':3600,'workers':2,'worker_RSS_bytes_cap':16*1024**3,'threads':2,'pilot_sequences':16,'labels_read':False,'attempt':a.attempt,'created_epoch':time.time(),'processes':[]}
record.write_text(json.dumps(budget,indent=2)+'\n')
for rank,gpu in enumerate(gpus):
 if a.attempt=='full':
  prior=json.loads((r/'cache_work'/f'worker{rank}'/'receipt_pilot.json').read_text());assert prior['status']=='pilot_complete' and prior['completed_rows']==8
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_MODULES_CACHE=str(r/'cache_work'/f'modules{rank}'))
 cmd=[a.python,str(r/'code/pooling_cache_extract.py'),'--source',a.source,'--output',str(r/'cache_work'),'--backend-src',a.backend_src,'--checkpoint-manifest',a.checkpoint_manifest,'--rank',str(rank),'--gpu-uuid',gpu,'--attempt',a.attempt,'--max-seconds','3600']
 if a.attempt=='pilot':cmd+=['--max-items','8']
 log=r/'records'/f'cache_{a.attempt}_{rank}.log'
 with log.open('w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
 pid=proc.pid;start_ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 budget['processes'].append({'rank':rank,'pid':pid,'start_ticks':start_ticks,'gpu_uuid':gpu,'log':str(log),'command':cmd});record.write_text(json.dumps(budget,indent=2)+'\n')
print(json.dumps(budget))
