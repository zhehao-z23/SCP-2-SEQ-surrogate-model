import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=json.loads((r/'plan.json').read_text())
def worker(rank):
 record=r/'records'/f'queue_{rank}.json';assert not record.exists();state={'status':'running','rank':rank,'attempts':[]}
 def save():
  t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(record)
 save()
 try:
  for k in range(rank,len(p['fits']),2):
   spec=p['fits'][k];args=[sys.executable,str(r/'code/bridge_optimization_train.py'),'--root',str(r),'--parent',p['parent'],'--cache-root',p['cache_root'],'--mode',spec['family'],'--policy',spec['policy'],'--fold',str(spec['fold']),'--seed',str(spec['seed']),'--lr',str(p['initial_lr']),'--max-seconds',str(p['fit_max_seconds'])];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=p['gpus'][rank]
   item={'spec':spec,'start_epoch':time.time(),'status':'running'};state['attempts'].append(item)
   with (r/'records'/f'fit{k}.log').open('w') as f:
    child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env);item['pid']=child.pid;save()
    try:rc=child.wait(timeout=p['process_max_seconds'])
    except subprocess.TimeoutExpired:
     child.terminate()
     try:child.wait(timeout=15)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     rc=124
   item.update(status='complete' if rc==0 else 'failed',returncode=rc,end_epoch=time.time());save()
   if rc:raise RuntimeError('fit failed; no automatic retry')
  state['status']='complete'
 except BaseException as e:state.update(status='failed',error=repr(e))
 finally:save()
 return state['status']
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:status=list(e.map(worker,[0,1]))
 (r/'records/complete.json').write_text(json.dumps({'statuses':status})+'\n')
