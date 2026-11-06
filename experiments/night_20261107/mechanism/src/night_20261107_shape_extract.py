"""Freeze an existing H2 model and extract diagnostic features; no fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

BASE = Path(os.environ['SCP_DATA_ROOT'])
BRIDGE = BASE/'bridge_20261106'
RUN = Path(os.environ['SCP_H2_REFERENCE_RUN'])
RECEIPT_PIN = 'adbdfc561f2c1969f28032f5d62785578dcf36a0e8523c36d154701ff81c4426'
MODEL_PIN = '41b81d67c67a55898324f20436c487548cc45c19b0d9991f3592b9e9487fdd7d'
RELEASE_PIN = '67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b'
GUARD_PIN = '71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c'
GUARD_CODE_PIN = '5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_bytes())


def put(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True); ap.add_argument('--plan-sha256', required=True)
    a = ap.parse_args(); start = time.monotonic()
    assert __debug__ and sha(a.plan) == a.plan_sha256
    plan = load(a.plan); assert plan['extract_code_sha256'] == sha(__file__)
    assert sorted(os.sched_getaffinity(0)) == [172, 173]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['GPU']
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.mem_get_info()[0] >= 18*(1 << 30)
    assert not a.out.exists(); a.out.mkdir(parents=True)
    put(a.out/'started.json', {'PID':os.getpid(), 'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'code_sha256':sha(__file__), 'plan_sha256':sha(a.plan), 'new_fits':0})
    refs = {}
    def bind(path, pin=None):
        h = sha(path)
        if pin: assert h == pin, str(path)
        refs[str(path)] = h
        return h
    bind(RUN/'receipt.json', RECEIPT_PIN); rec = load(RUN/'receipt.json')
    assert rec['status']=='complete' and rec['best_epoch']==104 and rec['config']['loss']=='H2'
    data = BASE/'atlas_scale_20261105/data'
    bind(data/'release.json', RELEASE_PIN); release = load(data/'release.json')
    bind(data/'samples.parquet', release['artifacts']['samples.parquet']); meta = pd.read_parquet(data/'samples.parquet')
    guards = BRIDGE/'roles'; bind(guards/'code/bridge_roles_guard.py', GUARD_CODE_PIN)
    sys.path.insert(0,str(guards/'code')); from bridge_roles_guard import check
    request={'stage':'development','sequence_sha256':sorted(meta.sequence_sha256.unique()),'aliases':[]}
    put(a.out/'role_preflight.json',[check(guards/'registry_v1/guard_contract.json',GUARD_PIN,{'operation':op,**request}) for op in ['predict','model_eval']])
    bind(RUN/'roles.json'); roles=load(RUN/'roles.json')
    bind(RUN/'fixed_best_partitions.json',rec['fixed_best_partitions_sha256']); parts=load(RUN/'fixed_best_partitions.json')
    partition_ids={}
    for name, p in parts.items():
        bind(RUN/(name+'_source_indices.npy'),p['indices_sha256']); ids=np.load(RUN/(name+'_source_indices.npy'))
        assert roles[name+'_ids']==meta.iloc[ids].sample_id.tolist(); partition_ids[name]=ids
    assert np.array_equal(np.sort(np.concatenate(list(partition_ids.values()))),np.arange(len(meta)))
    for left,right in [('train','stop'),('train','evaluation'),('stop','evaluation')]:
        for col in ['sequence_sha256','group_id']:
            assert not set(meta.iloc[partition_ids[left]][col]) & set(meta.iloc[partition_ids[right]][col])
    cache=BASE/'pooling_20261105/cache'; bind(cache/'receipt.json',rec['config']['cache_receipt_sha256'])
    bind(cache/'index.parquet',rec['config']['cache_index_sha256']); acceptance=load(cache.parent/'records/cache_acceptance.json')
    assert acceptance['status']=='accepted' and acceptance['receipt_sha256']==sha(cache/'receipt.json')
    # Rehash all existing immutable cache files before the new extraction.
    for name,p in acceptance['files'].items():
        assert (cache/name).stat().st_size==p['size']; bind(cache/name,p['sha256'])
    ix=pd.read_parquet(cache/'index.parquet'); assert ix.sequence_sha256.is_unique
    ri=pd.Series(np.arange(len(ix)),index=ix.sequence_sha256).loc[meta.sequence_sha256].to_numpy()
    length=ix.length.to_numpy()[ri]; offset=ix.offset.to_numpy()[ri]
    bind(RUN/'best.pt',rec['checkpoint_sha256']); bundle=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False)
    assert bundle['config']==rec['config']
    model_path=BRIDGE/'optimization/code/bridge_optimization_model.py'; bind(model_path,MODEL_PIN)
    sys.path.insert(0,str(model_path.parent)); from bridge_optimization_model import PoolingModel
    model=PoolingModel('residue_mean').cuda().eval(); model.load_state_dict(bundle['state_dict'])
    for p in model.parameters(): p.requires_grad_(False)
    bind(data/'composition_charge.npy',release['artifacts']['composition_charge.npy']); raw=np.load(data/'composition_charge.npy')
    tr=partition_ids['train']; med=np.nan_to_num(np.nanmedian(raw[tr],axis=0)); assert np.array_equal(med,bundle['charge_median'])
    charge=np.where(np.isnan(raw),med,raw)
    extra=np.concatenate([(charge-bundle['charge_mean'])/bundle['charge_std'],np.isnan(raw).astype('float32')],axis=1).astype('float32')
    means=np.load(cache/'mean.npy')[ri]
    assert np.array_equal(means[tr].mean(0),bundle['embedding_mean'])
    input_mean=np.concatenate([(means-bundle['embedding_mean'])/bundle['embedding_std'],extra],axis=1).astype('float32')
    flat=np.load(cache/'flat.npy',mmap_mode='r'); assert flat.dtype==np.float16 and flat.nbytes < 16*(1 << 30)
    flat=torch.from_numpy(np.array(flat)).cuda(); ex=torch.tensor(extra,device='cuda')
    em=torch.tensor(bundle['embedding_mean'],device='cuda'); es=torch.tensor(bundle['embedding_std'],device='cuda')
    captured=[]
    def hook(module,args): captured.append(args[0].detach().cpu().numpy().copy())
    handle=model.head[-1].register_forward_pre_hook(hook)
    z=np.empty((len(meta),256),np.float32); pred=np.empty((len(meta),63),np.float32); checks={}
    inference_start=time.monotonic()
    for name,ids in partition_ids.items():
        batches=[]
        for group in np.array_split(ids,max(1,int(np.ceil(len(ids)/128)))):
            chunk=[]; longest=0
            for i in group:
                nxt=max(longest,int(length[i]))
                if chunk and nxt*(len(chunk)+1)>16384: batches.append(chunk); chunk=[]; longest=0
                chunk.append(int(i)); longest=max(longest,int(length[i]))
            if chunk:batches.append(chunk)
        with torch.inference_mode():
            for rows in batches:
                pos=torch.arange(int(max(length[rows])),device='cuda')
                mask=pos[None] < torch.tensor(length[rows],device='cuda')[:,None]
                ind=torch.tensor(offset[rows],device='cuda')[:,None]+pos[None]
                ind=torch.where(mask,ind,torch.zeros_like(ind)); before=len(captured)
                pred[rows]=model((flat[ind].float()-em)/es,ex[rows],mask).softmax(1).cpu().numpy()
                assert len(captured)==before+1; z[rows]=captured.pop()
                assert time.monotonic()-start < 600
        bind(RUN/(name+'_prediction.npy'),parts[name]['prediction_sha256'])
        reference=np.load(RUN/(name+'_prediction.npy'))
        difference=float(abs(reference-pred[ids]).max()); assert difference<=1e-6
        checks[name]={'prediction_max_abs_difference':difference,'rows':len(ids)}
    torch.cuda.synchronize(); inference_seconds=time.monotonic()-inference_start; handle.remove()
    assert np.isfinite(z).all() and np.isfinite(input_mean).all() and not captured
    outputs={}
    for name,array in [('hidden256',z),('predicted_PMF',pred),('input_mean_charge',input_mean),('charge',extra)]:
        np.save(a.out/(name+'.npy'),array); outputs[name]={'sha256':sha(a.out/(name+'.npy')),'shape':list(array.shape)}
    for name,ids in partition_ids.items():
        np.save(a.out/(name+'_indices.npy'),ids); outputs[name+'_indices']={'sha256':sha(a.out/(name+'_indices.npy')),'shape':list(ids.shape)}
    meta[['sample_id','sequence_sha256','group_id','protein_id','fold']].to_parquet(a.out/'private_meta.parquet',index=False)
    summary={'status':'complete','schema':'frozen_H2_shape_feature_extraction.v1','source_refs':refs,'outputs':outputs,
        'private_meta_sha256':sha(a.out/'private_meta.parquet'),'checks':checks,'code_sha256':sha(__file__),'plan_sha256':sha(a.plan),
        'model_weights_changed':False,'PMF_replay_tolerance':1e-6,'new_fits':0,'GPU_inference_wall_seconds':inference_seconds,
        'process_seconds':time.monotonic()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'interpretation':'Features of a fixed stop-selected development model. Linear accessibility is not a test of information-theoretic absence or biological causality.'}
    put(a.out/'summary.json',summary)
    print(json.dumps({'status':'complete','checks':checks,'process_seconds':summary['process_seconds'],'summary_sha256':sha(a.out/'summary.json')}))


if __name__=='__main__': main()
