"""Two bounded queues; explicit UUIDs, at most eight fits, no automatic retry."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'plan.json').read_text());started=time.time()
def worker(rank):
 path=root/'records'/f'queue_{rank}.json';assert not path.exists()
 state={'status':'running','pid':os.getpid(),'rank':rank,'gpu_uuid':plan['gpus'][rank],'attempts':[],'started_epoch':started}
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
 save()
 try:
  for k in range(rank,len(plan['fits']),2):
   spec=plan['fits'][k];log=root/'records'/f'fit{k:02d}_{spec["mode"]}_lr{spec["lr"]}.log'
   cmd=[sys.executable,str(root/'code/overnight_explore_train.py'),'--root',str(root),'--parent',plan['parent'],'--cache-root',plan['cache_root'],'--mode',spec['mode'],'--fold',str(plan['fold']),'--seed',str(plan['seed']),'--lr',str(spec['lr']),'--max-seconds',str(plan['fit_max_seconds'])]
   if spec['warmstart']:cmd+=['--warmstart',plan['warmstart']]
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=plan['gpus'][rank]
   item={'fit_index':k,'spec':spec,'start_epoch':time.time(),'status':'running','log':str(log)};state['attempts'].append(item)
   with log.open('w') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);item['pid']=child.pid;save()
    try:rc=child.wait(timeout=plan['fit_process_max_seconds'])
    except subprocess.TimeoutExpired:
     child.terminate()
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     raise RuntimeError('fit process budget exceeded')
   item.update(status='complete' if rc==0 else 'failed',returncode=rc,end_epoch=time.time());save()
   if rc:raise RuntimeError(f'fit {k} failed; no auto retry')
  state['status']='complete'
 except BaseException as e:state.update(status='failed',error=repr(e))
 finally:state['end_epoch']=time.time();save()
 return state['status']

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:states=list(e.map(worker,[0,1]))
 (root/'records/queue_complete.json').write_text(json.dumps({'statuses':states,'elapsed':time.time()-started})+'\n')
 if states!=['complete','complete']:raise SystemExit(1)
