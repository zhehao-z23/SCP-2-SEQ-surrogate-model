"""Fixed-LR second-seed replication from the seed-1701 stop-selected trials."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--parent',required=True);p.add_argument('--mode',choices=['mean_pre','residue_mean','residue_attention'],required=True);a=p.parse_args();r=Path(a.root);selection=r/'results/selected_fits.csv';raw=selection.read_bytes();tab=pd.read_csv(selection);tab=tab[(tab['mode']==a.mode)&(tab.seed==1701)];assert set(tab.fold)=={0,1} and len(tab)==2
record=r/'records'/('queue_'+a.mode+'_seed3407.json');assert not record.exists();state={'status':'running','pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'started_epoch':time.time(),'mode':a.mode,'seed':3407,'LR_source_selection_sha256':hashlib.sha256(raw).hexdigest(),'LR_sources':tab[['run_id','fold','lr']].to_dict('records'),'attempts':[]}
def put():
 t=record.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(record)
put()
try:
 for row in tab.sort_values('fold').itertuples():
  log=r/'records'/f'{a.mode}_f{row.fold}_seed3407.log';item={'fold':row.fold,'lr':row.lr,'log':str(log),'started_epoch':time.time()};state['attempts'].append(item)
  with log.open('w') as f:
   c=subprocess.Popen([sys.executable,str(r/'code/train_pooling.py'),'--root',a.root,'--parent',a.parent,'--mode',a.mode,'--fold',str(row.fold),'--lr',str(row.lr),'--seed','3407'],stdout=f,stderr=subprocess.STDOUT);item['pid']=c.pid;put()
   try:rc=c.wait(timeout=1100)
   except subprocess.TimeoutExpired:c.terminate();c.wait(timeout=30);raise
  item.update(returncode=rc,ended_epoch=time.time());put()
  if rc:raise RuntimeError('fit failure; no automatic retry')
 state['status']='complete'
except BaseException as e:state['status']='failed';state['error']=repr(e);raise
finally:state['ended_epoch']=time.time();put()
