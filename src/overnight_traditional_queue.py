"""One sequential CPU queue, with a single persisted two-hour deadline."""
import os

import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def put(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['screen','confirm'],required=True);p.add_argument('--candidates',nargs='*');a=p.parse_args()
 r=Path(a.root);(r/'logs').mkdir(parents=True,exist_ok=True);(r/'records').mkdir(exist_ok=True)
 budget=r/'records/budget.json'
 if not budget.exists():put(budget,dict(start_epoch=time.time(),deadline_epoch=time.time()+7200,cpu_threads=8,max_address_space_GiB=32,GPU=False,max_screen_fits=6,max_confirm_fits=8,max_fit_seconds=1200,max_HGB_seconds=1800))
 b=json.loads(budget.read_text());records=[]
 candidates=a.candidates or ['et_q_leaf1','et_q_leaf3','rf_q_leaf3','et_p_leaf3','et_q_esmc_charge','hgb_q']
 folds=[0] if a.phase=='screen' else [1,2,3,4]
 assert len(candidates)<= (6 if a.phase=='screen' else 2)
 for candidate in candidates:
  for fold in folds:
   remain=b['deadline_epoch']-time.time()
   if remain<90:put(r/'records'/f'{a.phase}_queue.json',dict(status='budget_exhausted',fits=records));return
   env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
   cmd=[sys.executable,str(r/'tools/overnight_traditional.py'),'--root',str(r),'--atlas',str(Path(os.environ["PROJECT_ROOT"]) / 'atlas_scale_20261105'),'--candidate',candidate,'--fold',str(fold)]
   log=r/'logs'/f'{candidate}_fold{fold}.log'
   assert not log.exists(),'Preserve earlier attempt log; reconcile before retry'
   started=time.time()
   with log.open('w') as stream:
    child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    current=dict(candidate=candidate,fold=fold,pid=child.pid,start_epoch=started,command=cmd)
    put(r/'records'/f'{a.phase}_queue.json',dict(status='running',current=current,fits=records))
    try:
     code=child.wait(timeout=min(remain,1800 if candidate=='hgb_q' else 1200));state='complete' if code==0 else 'failed'
    except subprocess.TimeoutExpired:
     child.terminate()
     try:child.wait(timeout=20)
     except subprocess.TimeoutExpired:child.kill();child.wait()
     code=child.returncode;state='wall_budget'
   current.update(exit_code=code,status=state,wall_seconds=time.time()-started,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest());records.append(current)
   put(r/'records'/f'{a.phase}_queue.json',dict(status='running',fits=records));print(json.dumps(current),flush=True)
 put(r/'records'/f'{a.phase}_queue.json',dict(status='complete',fits=records))
if __name__=='__main__':main()
