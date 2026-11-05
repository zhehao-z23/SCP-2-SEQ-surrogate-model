"""Two explicit, disjoint vector-head queues with bounded process lifetime."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--family',required=True);a=p.parse_args()
 r=Path(a.root);record=r/'records'/('queue_'+a.family+'.json');started=time.time();attempts=[]
 assert not record.exists(),'queue already has an owner; inspect before resume'
 state={'status':'running','pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':started,'family':a.family,'attempts':attempts,'max_wall_seconds':7200,'max_fits':12,'automatic_retries':0}
 def save():
  t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(record)
 save()
 try:
  for fold in [0,1]:
   for scale in ['200','1000','all']:
    for lr in [.001,.0003]:
     if time.time()-started>=7200:raise TimeoutError('queue budget exhausted')
     cmd=[sys.executable,str(r/'code/train_atlas_scale.py'),'--root',str(r),'--source',a.source,'--family',a.family,'--fold',str(fold),'--scale',scale,'--lr',str(lr)]
     log=r/'records'/f'{a.family}_fold{fold}_g{scale}_lr{lr}.log'
     item={'fold':fold,'scale':scale,'lr':lr,'started_epoch':time.time(),'log':str(log),'status':'running'};attempts.append(item);save()
     with log.open('w') as f:
      child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;save()
      try:rc=child.wait(timeout=min(660,7200-(time.time()-started)))
      except subprocess.TimeoutExpired:
       child.terminate();child.wait(timeout=20);raise
     item.update(status='complete' if rc==0 else 'failed',returncode=rc,ended_epoch=time.time());save()
     if rc!=0:raise RuntimeError('fit failure; stop affected family for diagnosis')
  state['status']='complete'
 except BaseException as e:
  state['status']='failed';state['error']=repr(e);raise
 finally:state['ended_epoch']=time.time();save()
if __name__=='__main__':main()
