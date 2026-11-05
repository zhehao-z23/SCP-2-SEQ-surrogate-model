"""Bind reusable trials and create an isolated, no-copy confirmation workspace."""
import argparse,hashlib,json,shutil
from pathlib import Path
import pandas as pd
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pooling',required=True);p.add_argument('--atlas',required=True);a=p.parse_args();r=Path(a.root);pool=Path(a.pooling);atlas=Path(a.atlas)
for name in ['code','records','runs','results']:(r/name).mkdir(parents=True,exist_ok=True)
for name,target in [('cache',pool/'cache'),('data',atlas/'data')]:
 dest=r/name
 if dest.exists():assert dest.resolve()==target.resolve()
 else:dest.symlink_to(target,target_is_directory=True)
for name in ['train_pooling.py','pooling_model.py']:shutil.copyfile(pool/'code'/name,r/'code'/name)
shutil.copyfile(atlas/'code/train_atlas_capacity.py',r/'code/train_atlas_capacity.py');shutil.copyfile(pool/'records/cache_acceptance.json',r/'records/cache_acceptance.json')
reuse=[]
for row in pd.read_csv(pool/'results/selected_fits.csv').itertuples():
 if row.mode not in ['mean_pre','residue_mean']:continue
 path=pool/'runs'/row.run_id;rc=json.loads((path/'receipt.json').read_text());assert row.lr==.0003 and row.fold in [0,1] and row.seed in [1701,3407]
 assert rc['config']['script_sha256']==sha(r/'code/train_pooling.py') and rc['config']['model_sha256']==sha(r/'code/pooling_model.py')
 reuse.append({'family':row.mode,'fold':row.fold,'seed':row.seed,'path':str(path),'receipt_sha256':sha(path/'receipt.json')})
for row in pd.read_csv(atlas/'results/selected_fits.csv').itertuples():
 if row.family!='composition_charge_deep':continue
 path=atlas/'runs'/row.run_id;rc=json.loads((path/'receipt.json').read_text());assert row.lr==.0003 and row.fold in [0,1] and rc['config']['seed']==1701 and rc['config']['script_sha256']==sha(r/'code/train_atlas_capacity.py')
 reuse.append({'family':row.family,'fold':row.fold,'seed':1701,'path':str(path),'receipt_sha256':sha(path/'receipt.json')})
assert len(reuse)==10 and len({(x['family'],x['fold'],x['seed']) for x in reuse})==10
jobs=[]
for seed in [1701,3407]:
 for fold in range(5):
  for family in ['mean_pre','residue_mean','composition_charge_deep']:
   if any(z['family']==family and z['fold']==fold and z['seed']==seed for z in reuse):continue
   jobs.append({'family':family,'seed':seed,'fold':fold,'lr':.0003,'queue':1 if family=='residue_mean' else 0})
assert len(jobs)==20
record={'status':'prepared','reuse':reuse,'jobs':jobs,'expected_combined':30,'expected_new':20,'pooling_source':str(pool),'atlas_source':str(atlas),'executed_code_hashes':{name:sha(r/'code'/name) for name in ['train_pooling.py','pooling_model.py','train_atlas_capacity.py']},'qualification':'five-fold/two-seed retrospective development confirmation, not new blind test'}
out=r/'records/confirmation_plan.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'reused':10,'new':20,'jobs_per_queue':{i:sum(z['queue']==i for z in jobs) for i in [0,1]}}))
