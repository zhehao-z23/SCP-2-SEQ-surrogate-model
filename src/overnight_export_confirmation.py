"""Publish aggregate confirmation evidence and measured cost, without sample IDs."""
import argparse,hashlib,json,statistics
from collections import Counter
from pathlib import Path
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def span(x):
    x=list(x);return dict(min=min(x),median=statistics.median(x),max=max(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();r=Path(a.root)
    read=lambda p:json.loads(Path(p).read_text())
    summary=read(r/'results/summary.json');replay=read(r/'results/checkpoint_replay.json')
    assert summary['status']=='complete' and summary['fits_total']==30 and summary['new_fits']==20
    assert replay['status']=='complete' and replay['replayed_new']==20
    table=pd.read_csv(r/'results/all_fits.csv');receipts=[]
    for row in table.itertuples():
        z=read(Path(row.artifact_path)/'receipt.json');z['reused']=row.reused
        z['family']=row.family;receipts.append(z)
    queues=[read(f) for f in sorted((r/'records').glob('queue_*.json'))]
    assert len(queues)==2 and all(q['status']=='complete' for q in queues)
    attempts=[z for q in queues for z in q['attempts']]
    assert len(attempts)==20 and all(z['returncode']==0 for z in attempts)
    identities={(x['family'],x['fold'],x['seed']) for x in attempts}
    assert identities=={(x.family,x.fold,x.seed) for x in table.loc[~table.reused].itertuples()}
    groups=[]
    for family in sorted({z['family'] for z in receipts}):
        for seed in [1701,3407]:
            part=[z for z in receipts if z['family']==family and z['config']['seed']==seed]
            groups.append(dict(family=family,seed=seed,configs=len(part),reused=sum(x['reused'] for x in part),fit_wall_seconds=span(x.get('fit_wall_seconds',x.get('wall_seconds')) for x in part),new_fit_wall_seconds_sum=sum(x.get('fit_wall_seconds',x.get('wall_seconds')) for x in part if not x['reused']),cached_head_sequences_per_second=span(x.get('cached_readout_sequences_per_second',x.get('head_inference_sequences_per_second')) for x in part),peak_GPU_bytes=max(x.get('peak_GPU_bytes',x.get('peak_gpu_allocated_bytes')) for x in part),peak_RSS_bytes=max(x.get('peak_RSS_bytes',x.get('peak_rss_bytes')) for x in part),epochs=span(x['epochs'] for x in part),stop_reasons=dict(Counter(x['stop_reason'] for x in part))))
    cost=dict(status='complete',by_family_seed=groups,new_fit_wall_seconds_sum=summary['new_fit_wall_seconds_sum'],new_child_wall_seconds_sum=sum(z['ended_epoch']-z['started_epoch'] for z in attempts),new_queue_wall_seconds_sum=sum(q['ended_epoch']-q['started_epoch'] for q in queues),confirmation_elapsed_span_seconds=max(q['ended_epoch'] for q in queues)-min(q['started_epoch'] for q in queues),allocated_seconds_cap=5*3600,encoder_new_extractions=0,interpretation='All use previously frozen features. Cached head throughput excludes encoder/cache loading. Fit wall time is recorded by the original trainer; one queue per GPU gives a conservative allocation proxy, not GPU busy time. Charge and pooling timing code differ; not a rigorous cross-family deployment benchmark.')
    assert cost['new_queue_wall_seconds_sum']<cost['allocated_seconds_cap']
    public=table.drop(columns=['artifact_path'])
    public.to_csv(r/'results/all_fits_public.csv',index=False)
    (r/'results/cost_public.json').write_text(json.dumps(cost,indent=2)+'\n')
    provenance=dict(status='complete',release_sha256=sha(r/'data/release.json'),cache_receipt_sha256=sha(r/'cache/receipt.json'),confirmation_plan_sha256=sha(r/'records/confirmation_plan.json'),records={f.name:sha(f) for f in (r/'results').glob('*') if f.name in ['summary.json','checkpoint_replay.json','cost_public.json','all_fits_public.csv']},executed_code={f.name:sha(f) for f in (r/'code').glob('*.py')},original_checkpoint_checks='Ten reused fits retain replay evidence from atlas_scale_20261105 and pooling_20261105. The new replay report covers all twenty additional fits.',exposure='All five evaluation folds are development OOF, separately for each seed. No independently protected final newC test results.')
    (r/'results/provenance_public.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(cost,indent=2))

if __name__=='__main__':main()
