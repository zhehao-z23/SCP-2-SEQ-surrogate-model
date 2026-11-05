import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=json.loads((r/'plan.json').read_text())
def run(k):
 spec=p['fits'][k];record=r/'records'/f'fit{k}.json';assert not record.exists()
 cmd=[sys.executable,str(r/'code/overnight_explore_train.py'),'--root',str(r),'--parent',p['parent'],'--cache-root',p['cache_root'],'--mode',p['mode'],'--lr',str(p['lr']),'--fold',str(spec['fold']),'--seed',str(spec['seed']),'--warmstart',spec['warmstart'],'--max-seconds',str(p['max_seconds_per_fit'])]
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=spec['gpu_uuid'];state={'status':'running','spec':spec,'start_epoch':time.time()}
 with (r/'records'/f'fit{k}.log').open('w') as f:
  child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);state['pid']=child.pid;record.write_text(json.dumps(state,indent=2)+'\n')
  try:rc=child.wait(timeout=p['max_process_seconds_per_fit'])
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   rc=124
 state.update(status='complete' if rc==0 else 'failed',returncode=rc,end_epoch=time.time());record.write_text(json.dumps(state,indent=2)+'\n');return state['status']
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:status=list(e.map(run,[0,1]))
 (r/'records/complete.json').write_text(json.dumps({'statuses':status})+'\n')
 if status!=['complete','complete']:raise SystemExit(1)
