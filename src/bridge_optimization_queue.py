"""Two bounded GPU workers; stages sequenced before warm branches."""
import concurrent.futures,json,os,subprocess,sys,time,threading
from pathlib import Path
r=Path(__file__).resolve().parents[1];plan=json.loads((r/'plan.json').read_text());lock=threading.Lock();spent=0
def run(rank):
 global spent
 logpath=r/'records'/f'queue_{rank}.json';assert not logpath.exists();state={'status':'running','pid':os.getpid(),'rank':rank,'attempts':[]}
 def save():
  tmp=logpath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(logpath)
 save()
 # Even/odd policy assignment: every source fresh completes before its retain/reset sibling needs it.
 jobs=[(family,policy) for family in plan['families'] for policy in plan['policies']][rank::2]
 try:
  for family,policy in jobs:
   source=None
   if policy in ['reset','retain']:
    deadline=time.monotonic()+1100
    while source is None:
     for f in (r/'runs').glob('*/receipt.json'):
      z=json.loads(f.read_text());c=z['config']
      if c['mode']==family and c['policy']=='fresh' and c['fold']==plan['fold'] and c['seed']==plan['seed']:source=f.parent/'best.pt';break
     if source is None:
      if time.monotonic()>deadline:raise RuntimeError('source wait timed out')
      time.sleep(1)
   with lock:
    if spent+plan['process_max_seconds']>plan['max_GPU_hours']*3600:raise RuntimeError('remaining allocated budget insufficient for new fit')
   args=[sys.executable,str(r/'code/bridge_optimization_train.py'),'--root',str(r),'--parent',plan['parent'],'--cache-root',plan['cache_root'],'--mode',family,'--policy',policy,'--fold',str(plan['fold']),'--seed',str(plan['seed']),'--lr',str(plan['phase2_lr'] if source else plan['phase1_lr']),'--max-seconds',str(plan['fit_max_seconds'])]
   if source:args+=['--warmstart',str(source)]
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=plan['gpus'][rank];item={'family':family,'policy':policy,'start_epoch':time.time(),'status':'running'};state['attempts'].append(item)
   with (r/'records'/f'{family}_{policy}.log').open('w') as f:
    child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env);item['pid']=child.pid;save()
    try:rc=child.wait(timeout=plan['process_max_seconds'])
    except subprocess.TimeoutExpired:
     child.terminate()
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     rc=124
   item.update(status='complete' if rc==0 else 'failed',returncode=rc,end_epoch=time.time());save()
   with lock:spent+=item['end_epoch']-item['start_epoch']
   if rc:raise RuntimeError('failed fit; no automatic retry')
  state['status']='complete'
 except BaseException as e:state.update(status='failed',error=repr(e))
 finally:save()
 return state['status']
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:status=list(e.map(run,[0,1]))
 (r/'records/complete.json').write_text(json.dumps({'statuses':status,'allocated_seconds':spent})+'\n')
