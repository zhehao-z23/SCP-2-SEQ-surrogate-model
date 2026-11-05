"""Apply the registered alternative-selection rule once all initial trials finish."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root);f=r/'results/summary.json';s=json.loads(f.read_text());t=pd.read_csv(r/'results/all_fits.csv')
for name in ['mean_pre','max_pre','residue_mean','residue_attention']:
 rows=t[(t['mode']==name)&(t.seed==1701)];assert len(rows)==4 and set(zip(rows.fold,rows.lr))=={(i,lr) for i in [0,1] for lr in [.001,.0003]}
eligible=[z for z in s['scores'] if z['mode'] in ['residue_mean','residue_attention'] and z['seed']==1701];assert len(eligible)==2 and all(z['folds']==[0,1] for z in eligible)
chosen=min(eligible,key=lambda z:z['group_macro_H']);record={'rule':'lower seed1701 two-fold exposed development group_macro_H among residue_mean and residue_attention','summary_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'chosen_mode':chosen['mode'],'candidates':{z['mode']:z['group_macro_H'] for z in eligible},'reference':'mean_pre','confirmation_seed':3407,'confirmation_LR':'fixed seed1701 stop-selected LR per mode/fold','not_blind':True}
out=r/'records/confirmation_selection.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
