"""Execute only the four preregistered amendment trials, retaining original code."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);a=p.parse_args()
r=Path(a.root);am=json.loads((r/'records/capacity_amendment.json').read_text());record=r/'records/queue_composition_charge_deep.json'
assert not record.exists()
state={'status':'running','pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time(),'amendment':am['amendment'],'attempts':[]}
def save():
 t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(record)
save()
try:
 for fold in am['evaluation_folds']:
  for lr in am['learning_rates']:
   remaining=am['queue_wall_seconds']-(time.time()-state['started_epoch'])
   if remaining<=0:raise TimeoutError('queue budget')
   log=r/'records'/f'capacity_fold{fold}_lr{lr}.log';item={'fold':fold,'lr':lr,'log':str(log),'started_epoch':time.time()};state['attempts'].append(item)
   cmd=[sys.executable,str(r/'code/train_atlas_capacity.py'),'--root',str(r),'--source',a.source,'--family',am['family'],'--fold',str(fold),'--scale','all','--lr',str(lr)]
   with log.open('w') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
    try:rc=child.wait(timeout=min(660,remaining))
    except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20);raise
   item.update(returncode=rc,ended_epoch=time.time());save()
   if rc:raise RuntimeError('fit failed; no automatic retry')
 state['status']='complete'
except BaseException as e:state['status']='failed';state['error']=repr(e);raise
finally:state['ended_epoch']=time.time();save()
