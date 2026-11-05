"""Run only the registered missing confirmation fits; one owner per queue."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--project',required=True);p.add_argument('--queue',type=int,required=True);a=p.parse_args();r=Path(a.root);plan=json.loads((r/'records/confirmation_plan.json').read_text());record=r/'records'/f'queue_{a.queue}.json';assert not record.exists()
state={'status':'running','pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time(),'queue':a.queue,'attempts':[]}
def put():
 t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(record)
put()
try:
 for job in plan['jobs']:
  if job['queue']!=a.queue:continue
  name=f"{job['family']}_f{job['fold']}_s{job['seed']}";log=r/'records'/(name+'.log');item=dict(job,log=str(log),started_epoch=time.time());state['attempts'].append(item)
  if job['family']=='composition_charge_deep':cmd=[sys.executable,str(r/'code/train_atlas_capacity.py'),'--root',a.root,'--source',a.project,'--family',job['family'],'--scale','all']
  else:cmd=[sys.executable,str(r/'code/train_pooling.py'),'--root',a.root,'--parent',plan['atlas_source'],'--mode',job['family']]
  cmd+=['--fold',str(job['fold']),'--seed',str(job['seed']),'--lr',str(job['lr'])]
  with log.open('w') as f:
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);item['pid']=child.pid;put()
   try:rc=child.wait(timeout=1100)
   except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);raise
  item.update(returncode=rc,ended_epoch=time.time());put()
  if rc:raise RuntimeError('fit failed; stop lane for diagnosis')
 state['status']='complete'
except BaseException as e:state['status']='failed';state['error']=repr(e);raise
finally:state['ended_epoch']=time.time();put()
