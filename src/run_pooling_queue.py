"""One queue per readout family; immutable PID records and no automatic retries."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',required=True);p.add_argument('--mode',required=True);a=p.parse_args();r=Path(a.root);record=r/'records'/('queue_'+a.mode+'.json');assert not record.exists()
state={'status':'running','pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time(),'mode':a.mode,'attempts':[]}
def put():
 t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(record)
put()
try:
 for fold in [0,1]:
  for lr in [.001,.0003]:
   log=r/'records'/f'{a.mode}_f{fold}_lr{lr}.log';item={'fold':fold,'lr':lr,'log':str(log),'started_epoch':time.time()};state['attempts'].append(item)
   with log.open('w') as f:
    c=subprocess.Popen([sys.executable,str(r/'code/train_pooling.py'),'--root',a.root,'--parent',a.parent,'--mode',a.mode,'--fold',str(fold),'--lr',str(lr)],stdout=f,stderr=subprocess.STDOUT);item['pid']=c.pid;put()
    try:rc=c.wait(timeout=1100)
    except subprocess.TimeoutExpired:c.terminate();c.wait(timeout=30);raise
   item.update(returncode=rc,ended_epoch=time.time());put()
   if rc:raise RuntimeError('fit failure; inspect before recovery')
 state['status']='complete'
except BaseException as e:state['status']='failed';state['error']=repr(e);raise
finally:state['ended_epoch']=time.time();put()
