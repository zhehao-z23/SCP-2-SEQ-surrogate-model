"""Aggregate-only audit of predictions, saved optimizer identities and paired policies."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np,pandas as pd,torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root);plan=json.loads((r/'plan.json').read_text());p=Path(plan['parent']);d=pd.read_parquet(p/'data/samples.parquet').set_index('sample_id');y=np.load(p/'data/p.npy');pos=pd.Series(np.arange(len(d)),index=d.index);rows=[];tables={};private={}
 for q in sorted((r/'runs').glob('*/receipt.json')):
  x=json.loads(q.read_text());c=x['config'];roles=json.loads(q.with_name('roles.json').read_text());ids=roles['evaluation_ids'];truth=y[pos.loc[ids].to_numpy()].astype('float64');pred=np.load(q.with_name('evaluation_prediction.npy')).astype('float64');fold=c['fold']
  assert pred.shape==truth.shape and np.isfinite(pred).all() and (pred>=0).all() and max(abs(pred.sum(1)-1))<1e-6
  assert sha(q.with_name('best.pt'))==x['checkpoint_sha256'] and sha(q.with_name('evaluation_prediction.npy'))==x['prediction_sha256'];assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==x['run_id']
  for field in ['group_id','sequence_sha256']:
   parts={k:set(d.loc[v,field]) for k,v in roles.items()}
   for i,j in itertools.combinations(parts,2):assert not parts[i]&parts[j]
  assert set(d.loc[roles['train_ids']].fold)==set(range(5))-{fold,(fold+1)%5} and set(d.loc[roles['stop_ids']].fold)=={(fold+1)%5} and set(d.loc[ids].fold)=={fold}
  cp=torch.load(q.with_name('best.pt'),map_location='cpu',weights_only=False);assert cp['config']==c and 'optimizer_state' in cp and 'rng_state' in cp
  t=d.loc[ids,['group_id','protein_id']].copy();t['H']=np.sqrt(.5*((truth**.5-pred**.5)**2).sum(1));t['CDF']=((truth.cumsum(1)[:,:-1]-pred.cumsum(1)[:,:-1])**2).mean(1);t['tail']=abs(truth[:,[0,-1]]-pred[:,[0,-1]]).mean(1);g=t.groupby(['group_id','protein_id'])[['H','CDF','tail']].mean().groupby('group_id').mean();assert abs(g.H.mean()-x['result']['group_macro_H'])<1e-7;tables[x['run_id']]=g
  source=None
  if c['warmstart_path']:
   src=Path(c['warmstart_path']);assert sha(src)==c['warmstart_sha256'];source=json.loads(src.with_name('receipt.json').read_text());assert roles==json.loads(src.with_name('roles.json').read_text())
   assert abs(x['initial_stop_H']-source['best_stop_H'])<1e-7 and x['rng_restored']
   if c['policy']=='retain':assert x['source_optimizer_digest']==x['initial_optimizer_digest'] and min(x['initial_optimizer_steps'])>0
   else:assert not x['initial_optimizer_steps'] and x['initial_optimizer_digest']!=x['source_optimizer_digest']
  row={'run_id':x['run_id'],'family':c['mode'],'policy':c['policy'],'fold':fold,'seed':c['seed'],'stop_H':x['best_stop_H'],'H':float(g.H.mean()),'CDF':float(g.CDF.mean()),'tail':float(g['tail'].mean()),'initial_stop_H':x['initial_stop_H'],'best_epoch':x['best_epoch'],'epochs':x['epochs'],'stop_reason':x['stop_reason'],'final_LR_stage':x['final_LR_stage'],'fit_seconds':x['fit_wall_seconds'],'pipeline_fit_seconds':x['fit_wall_seconds']+(source['fit_wall_seconds'] if source else 0),'parameters':x['parameters'],'source_run_id':source['run_id'] if source else None,'source_optimizer_digest':x['source_optimizer_digest'],'initial_optimizer_digest':x['initial_optimizer_digest'],'initial_optimizer_steps':x['initial_optimizer_steps'],'data_and_scaler_identity_validated':True,'prediction_and_checkpoint_validated':True,'checkpoint_sha256':x['checkpoint_sha256'],'prediction_sha256':x['prediction_sha256'],'script_sha256':c['script_sha256'],'model_sha256':c['model_sha256'],'plan_sha256':c['plan_sha256']}
  rows.append(row);private[x['run_id']]=str(q.parent)
 pairs=[]
 for family in set(z['family'] for z in rows):
  subset=[z for z in rows if z['family']==family]
  for left,right in [('plateau','fresh'),('reset','fresh'),('retain','fresh'),('retain','reset')]:
   ls=[z for z in subset if z['policy']==left];rs=[z for z in subset if z['policy']==right]
   for x in ls:
    match=[z for z in rs if (z['fold'],z['seed'])==(x['fold'],x['seed'])]
    if not match:continue
    z=match[0];g=tables[x['run_id']];b=tables[z['run_id']];assert set(g.index)==set(b.index);delta=g.loc[b.index].H.to_numpy()-b.H.to_numpy();rng=np.random.default_rng(1701);boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]);pairs.append({'family':family,'fold':x['fold'],'seed':x['seed'],'comparison':left+' minus '+right,'delta_stop_H':x['stop_H']-z['stop_H'],'delta_H':float(delta.mean()),'descriptive_group_95':np.quantile(boot,[.025,.975]).tolist(),'selection_adjusted':False})
 trajectories=[]
 for z in rows:
  if z['policy']!='plateau':continue
  matches=[x for x in rows if x['family']==z['family'] and x['policy']=='fresh' and (x['fold'],x['seed'])==(z['fold'],z['seed'])]
  if not matches:continue
  ac=json.loads((Path(private[matches[0]['run_id']])/'curve.json').read_text());bc=json.loads((Path(private[z['run_id']])/'curve.json').read_text());shared=[(x,y) for x,y in zip(ac,bc) if y['LR_stage']==0];lossdiff=max(abs(x['train_H2']-y['train_H2']) for x,y in shared);stopdiff=max(abs(x['stop_H']-y['stop_H']) for x,y in shared)
  trajectories.append({'family':z['family'],'fold':z['fold'],'seed':z['seed'],'shared_pre_decay_epochs':len(shared),'max_train_loss_difference':lossdiff,'max_stop_H_difference':stopdiff})
 summary={'status':'complete' if len(rows)==plan.get('expected_fits',12) else 'partial','fits_complete':len(rows),'expected_fits':plan.get('expected_fits',12),'all_fits':rows,'paired':pairs,'shared_initial_trajectories':trajectories,'fit_seconds_total':sum(z['fit_seconds'] for z in rows),'qualifications':'exposed historical T/vertex-p development; policy selection on stop; stage budgets differ; no new C claim'}
 (r/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');(r/'results/private_run_paths.json').write_text(json.dumps(private,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
