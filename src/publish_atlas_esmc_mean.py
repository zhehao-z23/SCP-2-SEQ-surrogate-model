"""Verify full singleton shards and publish a lightweight hash-joined mean cache."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--legacy-cache',required=True);args=a.parse_args()
    root=Path(args.root); cache=root/'cache/esmc300_T'; target=root/'data/esmc_300m_mean'
    source=root/'data/sequences.json'; rows=json.loads(source.read_text()); hashes=[r['sequence_sha256'] for r in rows]
    assert len(hashes)==len(set(hashes)); arr=np.empty((len(rows),960),dtype=np.float32);seen=set(); inputs=[]; receipts=[]
    for rank in [0,1]:
        p=cache/f'worker{rank}/receipt_full01.json'; r=json.loads(p.read_text());receipts.append(r)
        assert r['status']=='complete' and r['identity']['source_sha256']==sha(source)
        assert r['identity']['rank']==rank and r['identity']['workers']==2
        for part in r['shards']:
            f=p.parent/part['path']; assert sha(f)==part['sha256']
            with np.load(f,allow_pickle=False) as z:
                ids=z['row_indices'];v=z['vectors']
                assert v.shape==(len(ids),960) and np.isfinite(v).all()
                assert not seen.intersection(ids.tolist()) and all(int(i)%2==rank for i in ids)
                assert all(0<=int(i)<len(rows) for i in ids)
                arr[ids]=v;seen.update(ids.tolist())
            inputs.append(dict(path=str(f),sha256=part['sha256']))
    assert seen==set(range(len(rows))) and np.isfinite(arr).all()
    # Independent replay: compare every exact-sequence overlap with the earlier
    # singleton FP32 cache, whose mean file is checked against its original hash.
    old=Path(args.legacy_cache); oi=json.loads((old/'index.json').read_text()); ident=oi['identity']
    assert ident['checkpoint_manifest_sha256']==receipts[0]['identity']['checkpoint_manifest_sha256']
    assert ident['code']['hf_backend.py']==receipts[0]['identity']['backend_sha256']
    assert ident['extraction_convention']=='singleton_unpadded_last_hidden_state_v1'
    assert ident['compute_dtype']=='float32' and ident['TF32'] is False
    assert sha(old/'mean.layer30.npy')==oi['files_sha256']['mean.layer30.npy']
    oldmean=np.load(old/'mean.layer30.npy',allow_pickle=False); mapping={h:i for i,h in enumerate(hashes)}
    pairs=[(mapping[r['sequence_sha256']],r['mean']['30']['row']) for r in oi['records'] if r['sequence_sha256'] in mapping]
    assert pairs
    left=arr[[x[0] for x in pairs]];right=oldmean[[x[1] for x in pairs]]
    difference=np.abs(left-right); parity=dict(exact_sequence_overlap=len(pairs),max_absolute_difference=float(difference.max()),
        atol=1e-5,rtol=1e-4,passed=bool(np.allclose(left,right,rtol=1e-4,atol=1e-5)),
        legacy_index_sha256=sha(old/'index.json'),legacy_mean_sha256=sha(old/'mean.layer30.npy'))
    assert parity['passed'],parity
    target.mkdir(parents=True,exist_ok=True)
    assert not (target/'receipt.json').exists(),'refuse to overwrite published cache'
    tmp=target/'mean.npy.tmp'
    with tmp.open('wb') as f:np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
    tmp.replace(target/'mean.npy')
    index=pd.DataFrame(dict(sequence_sha256=hashes,length=[r['length'] for r in rows],row=np.arange(len(rows))))
    index.to_parquet(target/'index.parquet',index=False)
    assert np.array_equal(np.load(target/'mean.npy',allow_pickle=False),arr)
    assert pd.read_parquet(target/'index.parquet').sequence_sha256.tolist()==hashes
    receipt=dict(schema='atlas.T.mean.publish.v1',status='complete',rows=len(rows),width=960,source_sha256=sha(source),
        encoder='esmc_300m',layer=30,sequence_view='teacher_retained_T',dtype='float32',no_truncation=True,
        mean_sha256=sha(target/'mean.npy'),index_sha256=sha(target/'index.parquet'),
        worker_identities=[r['identity'] for r in receipts],shards=inputs,legacy_overlap_parity=parity,
        extraction_GPU_allocated_seconds=sum(r['elapsed_seconds'] for r in receipts)+sum(json.loads((cache/f'worker{i}/receipt_pilot02.json').read_text())['elapsed_seconds'] for i in [0,1]),
        peak_GPU_bytes=max(r['peak_GPU_bytes'] for r in receipts),peak_RSS_bytes=max(r['peak_RSS_bytes'] for r in receipts))
    temp=target/'receipt.json.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(target/'receipt.json')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['shards','worker_identities']}))

if __name__=='__main__':main()
