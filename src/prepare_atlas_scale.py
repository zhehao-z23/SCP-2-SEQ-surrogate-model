"""Bind a retrospective T-input Atlas development release without modifying sources."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p=Path(p);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--output',required=True);args=a.parse_args()
 s=Path(args.source);o=Path(args.output);o.mkdir(parents=True,exist_ok=True)
 if (o/'release.json').exists():raise RuntimeError('release exists')
 sys.path.insert(0,str(s/'package/src'))
 from seq2scp_training.charge_features import extract_features,VERSION
 from seq2scp_training.features import VIEWS
 manifest=s/'data/qualification/release_v1/sample_manifest.parquet'
 allgroups=s/'data/qualification/release_v1/all_sequence_groups.parquet'
 m=pd.read_parquet(manifest);g=pd.read_parquet(allgroups).set_index('sequence_sha256').group_id
 take=(m.role=='development')&m.static_eligible&m.label_join_valid
 d=m[take].copy().reset_index(drop=True)
 assert len(d)>1061 and d.sample_id.is_unique
 assert d.teacher_sequence.map(lambda x:isinstance(x,str) and len(x)>0).all()
 assert (d.teacher_sequence.map(lambda x:hashlib.sha256(x.encode()).hexdigest())==d.teacher_sequence_sha256).all()
 assert d.teacher_sequence_sha256.isin(g.index).all(),'T not covered in existing full relation graph'
 assert (d.teacher_sequence_sha256.map(g)==d.group_id).all(),'T group mismatch'
 assert d.groupby('group_id').outer_fold.nunique().max()==1
 assert set(d.outer_fold.astype(int))==set(range(5))
 d['protein_id']='T_'+d.teacher_sequence_sha256
 d['sequence']=d.teacher_sequence;d['sequence_sha256']=d.teacher_sequence_sha256
 d['length']=d.sequence.str.len();d['fold']=d.outer_fold.astype(int)
 d['source']='PDB_historical_vertex_p';d['coverage']=d.T_U_coverage
 index=pd.read_parquet(s/'data/labels/index.parquet').set_index('sample_id')
 assert (index.loc[d.sample_id,'row'].to_numpy()==d.target_row.to_numpy()).all()
 y=np.load(s/'data/labels/p.float32.npy',allow_pickle=False)[d.target_row.astype(int)]
 assert y.shape==(len(d),63) and np.isfinite(y).all() and (y>=0).all()
 assert np.max(np.abs(y.sum(1)-1))<=1e-6
 np.save(o/'p.npy',y,allow_pickle=False)
 unique=d.drop_duplicates('sequence_sha256');cache={}
 for r in unique.itertuples():cache[r.sequence_sha256]=extract_features(r.sequence,sequence_view='T',full_chain=True)
 for view in ['composition_length','composition_charge']:
  arr=np.array([[np.nan if cache[h][k] is None else cache[h][k] for k in VIEWS[view]] for h in d.sequence_sha256],dtype=np.float32)
  assert not np.isinf(arr).any()
  np.save(o/(view+'.npy'),arr,allow_pickle=False)
 cols=['sample_id','protein_id','sequence','sequence_sha256','length','group_id','fold','source','coverage','target_row','source_structure_sha256','source_origin']
 d[cols].to_parquet(o/'samples.parquet',index=False)
 rows=unique[['sequence','sequence_sha256','length']].sort_values(['length','sequence_sha256']).to_dict('records')
 put(o/'sequences.json',rows)
 put(o/'exposure.json',{'role':'retrospective_development','all_historical_exposed':True,'new_C_transfer_allowed':False,'reason':'new C protected closure not yet joined; weights are development-only until fold-specific exclusion is established'})
 receipt={'schema':'atlas.scale.T.v1','task':'prepared teacher-retained sequence T -> historical vertex-count probability p','not_claim':'not full natural U target, not new C/area-p release, not independent blind test','sources':{str(p):sha(p) for p in [manifest,allgroups,s/'data/labels/p.float32.npy',s/'data/labels/index.parquet',s/'records/label_integrity.json',s/'data/labels/release.pending.json']},'selection':'existing development & static_eligible & label_join_valid; all such rows retained','all_source_rows':len(m),'selected_samples':len(d),'independent_T_sequences':d.protein_id.nunique(),'homology_groups':d.group_id.nunique(),'excluded_by_existing_role_or_static_or_label':int((~take).sum()),'fold_counts':d.groupby('fold').agg(samples=('sample_id','size'),proteins=('protein_id','nunique'),groups=('group_id','nunique')).to_dict('index'),'split':'existing fullU successor components including all sequence hashes; new retrospective 3fit/1stop/1evaluation roles','confirmed_cross_fold_groups':0,'p_rowsum_max_error':float(np.max(abs(y.sum(1)-1))),'label_measure':'vertex_count','bins':63,'feature_version':VERSION,'terminal_feature_convention':'sequence-only free-chain charge descriptor on concatenated T; not exact multisegment physical charge','code_sha256':sha(__file__),'created_epoch':time.time()}
 receipt['artifacts']={p.name:sha(p) for p in o.iterdir() if p.is_file()}
 put(o/'release.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
