"""Read-only post-hoc bridge contrasts on the completed Atlas scale experiment.

Print aggregate JSON. No sequences, sample/group identifiers or predictions are exported.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--script-sha256');a=ap.parse_args();r=Path(a.root)
p=r/'results/strata_audit/selected_per_sample_strata.parquet'
s=pd.read_parquet(p);s=s[(s.scale=='all')&s.family.isin(['esmc_charge','composition_charge_deep'])].copy()
s['length_bin']=pd.cut(s.length,[0,150,300,600,float('inf')],labels=['1-150','151-300','301-600','601+']).astype(str)
s['subset']=np.where(s.stratum=='strict_T_equals_U','exact_U',np.where((s.stratum=='T_differs_from_U')&(s.coverage>.98),'near_full_T','other'))
rows=[]
for (subset,length),g in s[s.subset!='other'].groupby(['subset','length_bin']):
 vals={family:gg.groupby(['group_id','protein_id']).H.mean().groupby('group_id').mean() for family,gg in g.groupby('family')}
 assert set(vals['esmc_charge'].index)==set(vals['composition_charge_deep'].index)
 delta=vals['esmc_charge']-vals['composition_charge_deep'];rng=np.random.default_rng(1701)
 boots=np.asarray([rng.choice(delta.to_numpy(),len(delta),replace=True).mean() for _ in range(2000)])
 rows.append({'subset':subset,'length_bin':length,'unique_T':int(g.protein_id.nunique()),'groups':len(delta),'delta_H':float(delta.mean()),'descriptive_95_interval':np.quantile(boots,[.025,.975]).tolist(),'fusion_H':float(vals['esmc_charge'].mean()),'deep_H':float(vals['composition_charge_deep'].mean())})
source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if Path(__file__).is_file() else a.script_sha256
output={'analysis':'posthoc_length_by_U_relation','input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_sha256':source_hash,'folds':[0,1],'seed':1701,'contrast':'esmc_charge minus composition_charge_deep','scope':'Two selected original scale-run fits per family, all scale; same saved predictions, no retraining. Near-full coverage >0.98 is descriptive and does not establish full-U equivalence. Paired conditional group-bootstrap intervals exclude selection/multiplicity/seed uncertainty; group counts overlap across cells.','rows':rows}
print(json.dumps(output,indent=2))
