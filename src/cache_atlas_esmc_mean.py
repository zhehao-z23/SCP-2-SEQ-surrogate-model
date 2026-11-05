"""Bounded singleton ESMC means for a hash-bound sequence-only snapshot.

Uses the reviewed project backend; model and data roots are explicit arguments.
No labels are read. Each worker owns disjoint rows and resumable small shards.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, resource, signal, sys, time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def atomic(p, value):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--source',required=True); a.add_argument('--output',required=True)
    a.add_argument('--backend-src',required=True); a.add_argument('--checkpoint-manifest',required=True)
    a.add_argument('--rank',type=int,required=True); a.add_argument('--workers',type=int,default=2)
    a.add_argument('--gpu-uuid',required=True); a.add_argument('--max-seconds',type=int,default=3600)
    a.add_argument('--max-items',type=int,default=0); a.add_argument('--attempt',required=True)
    args=a.parse_args(); out=Path(args.output)/f'worker{args.rank}'; out.mkdir(parents=True,exist_ok=True)
    lock=(out/'writer.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic(); rows=json.loads(Path(args.source).read_text())
    assert 0<=args.rank<args.workers
    assert len({r['sequence_sha256'] for r in rows})==len(rows)
    for r in rows:
        assert sha_text(r['sequence'])==r['sequence_sha256'] and len(r['sequence'])==r['length']
    ids=list(range(args.rank,len(rows),args.workers))
    # The first 50 evenly spaced length-ordered rows form a representative pilot.
    pilot=[ids[round(i*(len(ids)-1)/49)] for i in range(50)] if len(ids)>=50 else ids
    order=pilot+[i for i in ids if i not in set(pilot)]
    identity=dict(schema='atlas.T.esmc_mean.v1',source_sha256=sha(args.source),
        script_sha256=sha(__file__),checkpoint_manifest_sha256=sha(args.checkpoint_manifest),
        backend_sha256=sha(Path(args.backend_src)/'seq2scp_training/hf_backend.py'),
        encoder='esmc_300m',layer=30,sequence_view='teacher_retained_T',
        extraction='singleton_unpadded_last_hidden_state',compute='fp32',pooling='all residue arithmetic mean fp32',
        special_tokens_excluded=True,truncation=False,TF32=False,rank=args.rank,workers=args.workers,
        row_order_sha256=sha_text(json.dumps(order)))
    ip=out/'identity.json'
    if ip.exists(): assert json.loads(ip.read_text())==identity, 'cache identity changed'
    else: atomic(ip,identity)
    sys.path.insert(0,args.backend_src)
    import numpy as np, torch
    from seq2scp_training.hf_backend import LocalHFBackend
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu_uuid and torch.cuda.device_count()==1
    actual=str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
    assert actual==args.gpu_uuid.lower().removeprefix('gpu-')
    done=[]; parts=[]
    for p in sorted(out.glob('part*.npz')):
        with np.load(p,allow_pickle=False) as z:
            ri=z['row_indices'].tolist(); v=z['vectors']
            assert v.shape==(len(ri),960) and np.isfinite(v).all()
            assert ri==order[len(done):len(done)+len(ri)]
            done+=ri; parts.append(dict(path=p.name,sha256=sha(p),rows=len(ri)))
    assert len(done)<=len(order)
    state=dict(status='loading',attempt=args.attempt,pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        gpu_uuid=args.gpu_uuid,completed_rows=len(done),total_rows=len(order),
        source_rows=len(rows),memory_budget_GiB=16,threads=2,max_seconds=args.max_seconds,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),labels_read=False)
    pending=[]; vectors=[]; stop=False
    def update(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_GPU_bytes=torch.cuda.max_memory_allocated())
        atomic(out/'progress.json',state)
    def flush():
        if not pending:return
        p=out/f'part{len(parts):05d}.npz'; tmp=p.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez(f,row_indices=np.asarray(pending,dtype=np.int64),vectors=np.asarray(vectors,dtype=np.float32)); f.flush(); os.fsync(f.fileno())
        tmp.replace(p); parts.append(dict(path=p.name,sha256=sha(p),rows=len(pending)))
        done.extend(pending); pending.clear(); vectors.clear(); update(completed_rows=len(done))
    def on_signal(*_):
        nonlocal stop; stop=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    update()
    try:
        manifest=json.loads(Path(args.checkpoint_manifest).read_text())
        assert manifest['encoder']=='esmc_300m' and manifest['compute_precision']=='fp32'
        backend=LocalHFBackend(manifest,device='cuda:0'); assert backend.layers==30
        assert not backend.model.training and not any(p.requires_grad for p in backend.model.parameters())
        update(status='running',torch=torch.__version__,numpy=np.__version__,python=sys.version)
        start_count=len(done)
        for ri in order[len(done):]:
            if stop or time.monotonic()-started>=args.max_seconds: break
            if args.max_items and len(done)+len(pending)-start_count>=args.max_items:break
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>16*1024**3: raise RuntimeError('RSS budget exceeded')
            r=rows[ri]; result=backend.extract_batch([r['sequence']],[backend.layers],precision='fp32')[0]
            matrix=result['layers'][30]
            assert matrix.shape==(r['length'],960) and np.isfinite(matrix).all()
            mean=matrix.mean(axis=0,dtype=np.float32); assert np.isfinite(mean).all()
            pending.append(ri);vectors.append(mean)
            if len(pending)==50: flush()
        flush()
        status='complete' if len(done)==len(order) else ('pilot_complete' if args.max_items and len(done)-start_count>=args.max_items else 'budget_stopped')
        update(status=status,completed_rows=len(done),new_rows=len(done)-start_count)
        atomic(out/f'receipt_{args.attempt}.json',dict(state,identity=identity,shards=parts))
        print(json.dumps(state),flush=True)
    except BaseException as exc:
        flush(); update(status='failed',error=repr(exc))
        atomic(out/f'receipt_{args.attempt}.json',dict(state,identity=identity,shards=parts)); raise

def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()

if __name__=='__main__':main()
