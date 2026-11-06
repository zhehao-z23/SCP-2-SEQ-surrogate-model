"""Fixed-best train/stop diagnostics; no fitting or historical-epoch reconstruction.

Only the two pinned residue-mean checkpoints are eligible. Predictions remain
private; aggregate results use identical eval mode and row/protein/group rules.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch

PINS = {
    'fresh': ('7cb3e6ef30f17bd413cac6afe9b7e2ecf345859c04857af291261d798fb37b74',
              '8e8ce2af7592298cc6b31c6ca58c639d919ffbd09b66a0c55973cf62a31b5a7e', 68),
    'plateau': ('a256424dfe60f8a445d4c71bf3f4fff0f6c7698a920e14e442d985925f267330',
                'd21e8a32ab08bc3725455939f8c8d9c8eec8d5a151754073a614b66bcd74d5f3', 104),
}
GUARD_PIN = '71359d1a06e18ab5dec65e825c82a07c3d7e0401c35266e8645957b943bb903c'
RELEASE_PIN = '67053baa3e80d8c30fda95c0686b8f56c141391bed4ebe880de11cb24c47a49b'
MODEL_PIN = '41b81d67c67a55898324f20436c487548cc45c19b0d9991f3592b9e9487fdd7d'
GUARD_CODE_PIN = '5401723f38305496ab11ab71d07825b2f1fe56476d753593d7929ed12a693cd3'
METRICS_CODE_PIN = '8f5d1ae625abc8b7b636fc1bdd6febbcb436ba0d3e0a1f1f5034d2a69621878b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def put(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def measures(target, prediction, meta):
    p, q = target.astype('float64'), prediction.astype('float64')
    assert p.shape == q.shape and p.shape[1] == 63
    assert np.isfinite(p).all() and np.isfinite(q).all() and (p >= 0).all() and (q >= 0).all()
    assert max(abs(p.sum(1)-1).max(), abs(q.sum(1)-1).max()) < 1e-6
    rows = meta[['sample_id', 'protein_id', 'group_id']].reset_index(drop=True).copy()
    rows['H2'] = .5 * ((np.sqrt(p)-np.sqrt(q))**2).sum(1)
    rows['H'] = np.sqrt(rows.H2)
    rows['CDF_MSE'] = ((p.cumsum(1)[:, :-1]-q.cumsum(1)[:, :-1])**2).mean(1)
    rows['edge_mass_MAE'] = abs(p[:, [0, -1]]-q[:, [0, -1]]).mean(1)
    rows['negative_edge_bias'] = q[:, 0]-p[:, 0]
    rows['positive_edge_bias'] = q[:, -1]-p[:, -1]
    cols = ['H', 'H2', 'CDF_MSE', 'edge_mass_MAE', 'negative_edge_bias', 'positive_edge_bias']
    proteins = rows.groupby(['group_id', 'protein_id'])[cols].mean()
    groups = proteins.groupby('group_id').mean()
    return {'rows': len(rows), 'proteins': len(proteins), 'groups': len(groups),
            'group_macro': groups.mean().to_dict(), 'protein_macro': proteins.mean().to_dict()}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--optimization-root', type=Path, required=True)
    ap.add_argument('--roles-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert __debug__ and not a.output.exists()
    a.output.mkdir(parents=True)
    start = time.monotonic()
    put(a.output/'started.json', {'pid': os.getpid(), 'start_unix': time.time(),
                                'new_fits': 0, 'code_sha256': sha(__file__)})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = load(a.optimization_root/'plan.json')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == plan['gpus'][0]
    parent = Path(plan['parent']); data = parent/'data'; cache_root = Path(plan['cache_root'])
    cache = cache_root/'cache'
    assert sha(data/'release.json') == RELEASE_PIN
    release = load(data/'release.json')
    assert sha(data/'samples.parquet') == release['artifacts']['samples.parquet']
    meta = pd.read_parquet(data/'samples.parquet')
    assert meta.sample_id.is_unique
    # Identity policy gate runs before labels, checkpoint or prediction arrays.
    assert sha(a.roles_root/'code/bridge_roles_guard.py') == GUARD_CODE_PIN
    sys.path.insert(0, str(a.roles_root/'code'))
    from bridge_roles_guard import check
    identities = sorted(meta.sequence_sha256.unique().tolist())
    guard_receipts = []
    for op in ['read_target', 'model_eval', 'predict']:
        request = {'operation': op, 'stage': 'development', 'sequence_sha256': identities, 'aliases': []}
        guard_receipts.append(check(a.roles_root/'registry_v1/guard_contract.json', GUARD_PIN, request))
    put(a.output/'role_preflight.json', guard_receipts)
    for name in ['p.npy', 'composition_charge.npy']:
        assert sha(data/name) == release['artifacts'][name]
    target = np.load(data/'p.npy'); raw_charge = np.load(data/'composition_charge.npy')
    acceptance = load(cache_root/'records/cache_acceptance.json')
    assert acceptance['status'] == 'accepted' and sha(cache/'receipt.json') == acceptance['receipt_sha256']
    assert load(cache/'receipt.json')['status'] == 'complete'
    cache_checks = []
    for name, bound in acceptance['files'].items():
        f = cache/name
        assert f.stat().st_size == bound['size'] and f.stat().st_mtime_ns == bound['mtime_ns']
        assert sha(f) == bound['sha256'], name
        cache_checks.append({'name': name, 'sha256': bound['sha256']})
    ix = pd.read_parquet(cache/'index.parquet')
    assert ix.sequence_sha256.is_unique
    ri = pd.Series(np.arange(len(ix)), index=ix.sequence_sha256).loc[meta.sequence_sha256].to_numpy()
    length = ix.length.to_numpy()[ri]; offset = ix.offset.to_numpy()[ri]
    raw_flat = np.load(cache/'flat.npy', mmap_mode='r')
    assert raw_flat.dtype == np.float16 and raw_flat.shape[1] == 960 and raw_flat.nbytes < 16 * 1024**3
    assert (length > 0).all() and (offset >= 0).all() and (offset+length <= len(raw_flat)).all()
    flat = torch.from_numpy(np.array(raw_flat)).cuda()
    sys.path.insert(0, str(a.optimization_root/'code'))
    sys.path.insert(0, str(parent/'code'))
    assert sha(a.optimization_root/'code/bridge_optimization_model.py') == MODEL_PIN
    assert sha(parent/'code/train_atlas_scale.py') == METRICS_CODE_PIN
    from bridge_optimization_model import PoolingModel
    from train_atlas_scale import select, metrics
    tr, st, ev = select(meta, 0, 'all')
    partitions = dict(train=tr, stop=st, evaluation=ev)
    assert len(np.unique(np.concatenate(list(partitions.values())))) == len(meta)
    for x, y in [('train', 'stop'), ('train', 'evaluation'), ('stop', 'evaluation')]:
        for key in ['sequence_sha256', 'group_id']:
            assert not set(meta.iloc[partitions[x]][key]) & set(meta.iloc[partitions[y]][key])
    results = []
    for policy, (run_id, checkpoint_pin, epoch) in PINS.items():
        run = a.optimization_root/'runs'/run_id
        receipt = load(run/'receipt.json'); cfg = receipt['config']; roles = load(run/'roles.json')
        assert receipt['status'] == 'complete' and receipt['best_epoch'] == epoch
        assert cfg['mode'] == 'residue_mean' and cfg['fold'] == 0 and cfg['seed'] == 1701 and cfg['policy'] == policy
        assert cfg['release_sha256'] == RELEASE_PIN and cfg['model_sha256'] == MODEL_PIN
        assert cfg['cache_receipt_sha256'] == sha(cache/'receipt.json') and cfg['cache_index_sha256'] == sha(cache/'index.parquet')
        assert cfg['parent_metrics_and_split_sha256'] == sha(parent/'code/train_atlas_scale.py')
        assert sha(run/'best.pt') == checkpoint_pin
        assert all(roles[name+'_ids'] == meta.iloc[ids].sample_id.tolist() for name, ids in partitions.items())
        bundle = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        assert bundle['config'] == cfg
        model = PoolingModel('residue_mean').cuda().eval(); model.load_state_dict(bundle['state_dict'])
        charge = np.where(np.isnan(raw_charge), bundle['charge_median'], raw_charge)
        extra = torch.tensor(np.concatenate([(charge-bundle['charge_mean'])/bundle['charge_std'],
                                             np.isnan(raw_charge).astype('float32')], 1), device='cuda')
        em = torch.tensor(bundle['embedding_mean'], device='cuda')
        es = torch.tensor(bundle['embedding_std'], device='cuda')
        item = {'policy': policy, 'run_id': run_id, 'best_epoch': epoch, 'checkpoint_sha256': checkpoint_pin,
                'receipt_sha256': sha(run/'receipt.json'), 'roles_sha256': sha(run/'roles.json'), 'partitions': {}}
        destination = a.output/policy; destination.mkdir()
        for name, ids in partitions.items():
            batches = []
            for group in np.array_split(ids, max(1, int(np.ceil(len(ids)/128)))):
                part = []; longest = 0
                for i in group:
                    nxt = max(longest, int(length[i]))
                    if part and nxt*(len(part)+1) > 16384:
                        batches.append(part); part = []; longest = 0
                    part.append(int(i)); longest = max(longest, int(length[i]))
                if part: batches.append(part)
            predictions = []
            with torch.inference_mode():
                for z in batches:
                    positions = torch.arange(int(max(length[z])), device='cuda')
                    mask = positions[None] < torch.tensor(length[z], device='cuda')[:, None]
                    indices = torch.tensor(offset[z], device='cuda')[:, None] + positions[None]
                    indices = torch.where(mask, indices, torch.zeros_like(indices))
                    predictions.append(model((flat[indices].float()-em)/es, extra[z], mask).softmax(1).cpu().numpy())
            prediction = np.concatenate(predictions)
            scores, rows = measures(target[ids], prediction, meta.iloc[ids])
            legacy = metrics(target[ids], prediction, meta.iloc[ids])[0]
            assert abs(scores['group_macro']['H']-legacy['group_macro_H']) < 1e-7
            if name == 'stop':
                assert abs(legacy['group_macro_H']-receipt['best_stop_H']) < 1e-7
                scores['recorded_best_stop_H'] = receipt['best_stop_H']
            if name == 'evaluation':
                assert sha(run/'evaluation_prediction.npy') == receipt['prediction_sha256']
                saved = np.load(run/'evaluation_prediction.npy')
                scores['saved_prediction_max_abs_difference'] = float(abs(saved-prediction).max())
                assert np.allclose(saved, prediction, atol=1e-6, rtol=1e-5)
            np.save(destination/(name+'_prediction.npy'), prediction)
            np.save(destination/(name+'_source_indices.npy'), ids)
            rows.to_parquet(destination/(name+'_metrics.parquet'), index=False)
            scores['prediction_sha256'] = sha(destination/(name+'_prediction.npy'))
            scores['indices_sha256'] = sha(destination/(name+'_source_indices.npy'))
            item['partitions'][name] = scores
        results.append(item)
        del model
    summary = {'schema': 'fixed_best_comparable_diagnostics_v1', 'status': 'complete', 'results': results,
               'new_fits': 0, 'optimizer_updates': 0, 'epoch_selection_performed': False,
               'measurement': 'Fixed saved best; dropout off; row to protein to group to equal groups. Historical vertex 63-bin target.',
               'tail_definition': 'First and last open-bin mass; not historical neg5/pos5.',
               'limits': 'Two existing selected states only; no arbitrary epoch trajectory, final test, causal attribution or new-C claim.',
               'elapsed_seconds': time.monotonic()-start, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
               'max_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'cache_files_rehashed': cache_checks, 'code_sha256': sha(__file__),
               'release_sha256': RELEASE_PIN, 'roles_contract_sha256': GUARD_PIN,
               'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                               'GPU': os.environ['CUDA_VISIBLE_DEVICES'], 'cpu_affinity': sorted(os.sched_getaffinity(0))}}
    put(a.output/'summary.json', summary)
    print(json.dumps({'status': 'complete', 'checkpoints': len(results), 'new_fits': 0,
                      'elapsed_seconds': summary['elapsed_seconds'], 'summary_sha256': sha(a.output/'summary.json')}))


if __name__ == '__main__':
    main()
