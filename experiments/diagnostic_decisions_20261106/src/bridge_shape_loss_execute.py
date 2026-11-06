"""Sequential bounded loss probe launcher; refuse CE unless H2 control replays."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text())


def put(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--plan-sha256', required=True); ap.add_argument('--arm', choices=['H2','CE_quarter'], required=True)
    ap.add_argument('--preflight-only', action='store_true'); a = ap.parse_args()
    assert __debug__ and sha(a.plan) == a.plan_sha256
    plan = load(a.plan); root = Path(plan['root']); code = root/'code'
    assert sorted(os.sched_getaffinity(0)) == plan['CPU_affinity']
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'probe.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for file, key in [('bridge_shape_loss_train.py','trainer_sha256'), ('bridge_optimization_model.py','model_sha256')]:
        assert sha(code/file) == plan[key]
    assert sha(Path(plan['parent'])/'code/train_atlas_scale.py') == plan['metrics_code_sha256']
    ref = Path(plan['reference_directory'])
    assert sha(ref/'receipt.json') == plan['reference_receipt_sha256'] and sha(ref/'curve.json') == plan['reference_curve_sha256']
    for file,key in [('receipt.json','cache_receipt_sha256'),('index.parquet','cache_index_sha256')]:
        assert sha(Path(plan['cache_root'])/'cache'/file) == plan[key]
    acceptance = load(Path(plan['cache_root'])/'records/cache_acceptance.json')
    assert acceptance['status'] == 'accepted' and acceptance['receipt_sha256'] == plan['cache_receipt_sha256']
    rehash = []
    for name, item in acceptance['files'].items():
        f = Path(plan['cache_root'])/'cache'/name
        assert f.stat().st_size == item['size'] and f.stat().st_mtime_ns == item['mtime_ns']
        assert sha(f) == item['sha256']
        rehash.append({'name':name,'sha256':item['sha256']})
    info = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0])*1024
    assert available >= 320*1024**3
    gpu = subprocess.check_output(['nvidia-smi','--id='+plan['GPU'], '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    assert int(gpu) >= 18*1024
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    assert not any(plan['GPU'] in line for line in apps.splitlines())
    preflight = {'status':'passed','code_sha256':sha(__file__),'trainer_sha256':plan['trainer_sha256'],
        'plan_sha256':a.plan_sha256,'arm':a.arm,'time_unix':time.time(),
        'cache_files_rehashed':rehash,'MemAvailable_bytes':available,'GPU_free_MiB':int(gpu),
        'no_competing_GPU_process':True,'CPU_affinity':sorted(os.sched_getaffinity(0))}
    path = root/(a.arm+('_prepare_preflight.json' if a.preflight_only else '_execution_preflight.json'))
    put(path, preflight)
    if a.preflight_only:
        print(json.dumps({'status':'preflight_passed','arm':a.arm,'receipt_sha256':sha(path),'new_fits':0})); return
    if a.arm == 'CE_quarter':
        control = load(root/'H2_control_replay.json')
        assert control['status'] == 'passed' and control['plan_sha256'] == a.plan_sha256
        assert sha(Path(control['control_directory'])/'receipt.json') == control['control_receipt_sha256']
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=plan['GPU'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    cmd = [sys.executable,str(code/'bridge_shape_loss_train.py'),'--root',str(root),'--parent',plan['parent'],
           '--cache-root',plan['cache_root'],'--mode','residue_mean','--fold',str(plan['fold']),'--seed',str(plan['seed']),
           '--lr','0.0003','--policy','plateau','--max-seconds','900','--max-epochs','134','--loss',a.arm]
    t = time.monotonic()
    with (root/(a.arm+'_training.log')).open('x') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1020)
    process_seconds = time.monotonic()-t
    assert result.returncode == 0, 'Training failed; retain log and no automatic retry'
    import numpy as np
    runs = [(p.parent,load(p)) for p in (root/'runs').glob('*/receipt.json')]
    chosen = [(p,r) for p,r in runs if r['config']['loss'] == a.arm]
    assert len(chosen) == 1
    directory, rec = chosen[0]; curve = load(directory/'curve.json')
    assert rec['status']=='complete' and rec['epochs']==134 and len(curve)==134
    assert sha(directory/'evaluation_prediction.npy') == rec['prediction_sha256']
    if a.arm == 'H2':
        previous = load(ref/'receipt.json'); old = load(ref/'curve.json')
        limits = plan['control_replay_gate']
        d0 = abs(rec['initial_stop_H']-previous['initial_stop_H'])
        ds = max(abs(n['stop_H']-o['stop_H']) for n,o in zip(curve,old))
        dt = max(abs(n['online_train_loss']-o['train_H2']) for n,o in zip(curve,old))
        assert all(n['epoch']==o['epoch'] and n['lr']==o['lr'] and n['LR_stage']==o['LR_stage'] for n,o in zip(curve,old))
        assert sha(ref/'evaluation_prediction.npy') == previous['prediction_sha256']
        dp = float(abs(np.load(directory/'evaluation_prediction.npy')-np.load(ref/'evaluation_prediction.npy')).max())
        observed = {'initial_stop_difference':d0,'max_epoch_stop_difference':ds,'max_online_H2_difference':dt,
                    'max_evaluation_prediction_difference':dp,'best_epoch':rec['best_epoch']}
        valid = (d0<=limits['initial_stop_absolute_tolerance'] and ds<=limits['each_epoch_stop_absolute_tolerance']
            and dt<=limits['each_epoch_online_H2_absolute_tolerance'] and dp<=limits['prediction_max_absolute_tolerance']
            and rec['best_epoch']==limits['selected_epoch'])
        put(root/'H2_control_replay.json', {'status':'passed' if valid else 'failed','observed':observed,
            'plan_sha256':a.plan_sha256,'control_directory':str(directory),'control_receipt_sha256':sha(directory/'receipt.json'),
            'reference_receipt_sha256':sha(ref/'receipt.json'),'process_seconds':process_seconds,
            'old_initial_state_hash_available':False,'new_initial_state_and_RNG_hashes_saved':True})
        assert valid, 'Control replay failed; CE not authorized by execution gate'
    else:
        base = Path(control['control_directory']); br = load(base/'receipt.json'); bc = load(base/'curve.json')
        assert rec['initial_model_digest']==br['initial_model_digest'] and rec['initial_rng_digest']==br['initial_rng_digest']
        assert rec['initial_stop_H']==br['initial_stop_H']
        assert all(n['batch_order_sha256']==o['batch_order_sha256'] and n['end_epoch_RNG_digest']==o['end_epoch_RNG_digest'] for n,o in zip(curve,bc))
        put(root/'paired_path_checks.json', {'status':'passed','epochs_checked':134,'same_initial_model':True,
            'same_initial_RNG':True,'same_all_batch_orders':True,'same_all_end_epoch_RNG':True,
            'CE_receipt_sha256':sha(directory/'receipt.json'),'H2_receipt_sha256':sha(base/'receipt.json'),
            'plan_sha256':a.plan_sha256,'process_seconds':process_seconds})
    print(json.dumps({'status':'complete','arm':a.arm,'process_seconds':process_seconds,'receipt_sha256':sha(directory/'receipt.json')}))


if __name__ == '__main__': main()
