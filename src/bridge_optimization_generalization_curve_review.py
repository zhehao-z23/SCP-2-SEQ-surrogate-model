"""Read existing small logs only; no arrays, checkpoints or model imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def analyze(curve, receipt, initial):
    cfg = receipt['config']
    tol = cfg['min_delta']
    assert tol == 1e-5 and receipt['status'] == 'complete'
    assert len(curve) == receipt['epochs']
    assert [r['epoch'] for r in curve] == list(range(1, len(curve) + 1))
    assert initial['stop_H'] == receipt['initial_stop_H']
    assert initial['policy'] == cfg['policy']
    best = stage_best = initial['stop_H']
    best_ep = stage_best_ep = stage_start = stage = 0
    lrs = cfg['plateau_lrs'] if cfg['policy'] == 'plateau' else [cfg['lr']]
    best_updates = []
    early = False
    for r in curve:
        e, score = r['epoch'], r['stop_H']
        assert r['LR_stage'] == stage and r['lr'] == lrs[stage]
        if score < best - tol:
            best, best_ep = score, e
            best_updates.append(e)
        if cfg['policy'] == 'plateau':
            if score < stage_best - tol:
                stage_best, stage_best_ep = score, e
            if stage < 2 and e - stage_best_ep >= cfg['plateau_patience']:
                stage += 1
                stage_start, stage_best, stage_best_ep = e, score, e
            early = stage == 2 and e-stage_start >= cfg['final_stage_min_epochs'] and e >= cfg['min_epochs'] and e-best_ep >= cfg['patience']
        else:
            early = e >= cfg['min_epochs'] and e-best_ep >= cfg['patience']
        if early:
            assert e == len(curve), 'record continued past reproduced early stop'
    assert early and receipt['stop_reason'] == 'early_stop'
    assert best_ep == receipt['best_epoch'] and best == receipt['best_stop_H']
    assert stage == receipt['final_LR_stage']
    prev = statistics.mean(r['stop_H'] for r in curve[-20:-10])
    tail = statistics.mean(r['stop_H'] for r in curve[-10:])
    delta = tail-prev
    tail_class = 'late_window_lower' if delta < -tol else 'late_window_higher' if delta > tol else 'late_window_near_flat'
    raw_min = min([{'epoch': 0, 'stop_H': initial['stop_H']}] + curve, key=lambda r:r['stop_H'])
    stages = []
    for s in sorted(set(r['LR_stage'] for r in curve)):
        rows = [r for r in curve if r['LR_stage'] == s]
        m = min(rows, key=lambda r:r['stop_H'])
        stages.append({'stage': s, 'lr': rows[0]['lr'], 'first_epoch': rows[0]['epoch'],
                       'last_epoch': rows[-1]['epoch'], 'epochs': len(rows),
                       'raw_min_stop_H': m['stop_H'], 'raw_min_epoch': m['epoch'],
                       'selected_best_updates': [e for e in best_updates if rows[0]['epoch'] <= e <= rows[-1]['epoch']]})
    return {'run_id': receipt['run_id'], 'family': cfg['mode'], 'policy': cfg['policy'],
            'fold': cfg['fold'], 'seed': cfg['seed'], 'epochs': len(curve), 'stop_reason': receipt['stop_reason'],
            'initial_stop_H': initial['stop_H'], 'epoch0_is_fresh_best_warmstart': cfg['policy'] in ('reset', 'retain'),
            'selected_best_epoch': best_ep, 'selected_best_stop_H': best,
            'selected_best_stage': 0 if best_ep == 0 else curve[best_ep-1]['LR_stage'],
            'last_selected_improvement_gap_epochs': len(curve)-best_ep,
            'selected_best_updates': best_updates, 'raw_min_epoch': raw_min['epoch'],
            'raw_min_stop_H': raw_min['stop_H'], 'selected_best_minus_raw_min': best-raw_min['stop_H'],
            'last_stop_H': curve[-1]['stop_H'], 'last_minus_selected_best': curve[-1]['stop_H']-best,
            'prior_10_epoch_mean_stop_H': prev, 'last_10_epoch_mean_stop_H': tail,
            'late_window_delta_H': delta, 'late_window_description': tail_class,
            'last_10_minus_selected_best': tail-best, 'LR_stages': stages,
            'train_H2_separate': {'first': curve[0]['train_H2'], 'last': curve[-1]['train_H2'],
                                  'prior_10_mean': statistics.mean(r['train_H2'] for r in curve[-20:-10]),
                                  'last_10_mean': statistics.mean(r['train_H2'] for r in curve[-10:])},
            'reproduced_selection_and_early_stop': True,
            'epoch_CDF_or_edge_metrics_available': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mirror', type=Path, required=True)
    ap.add_argument('--training-source', type=Path, required=True)
    ap.add_argument('--output-prefix', type=Path, required=True)
    a = ap.parse_args()
    receipt_path = a.mirror/'mirror_receipt.json'
    mirror = load(receipt_path)
    assert mirror['run_count'] == 18 and mirror['file_count'] == 54
    refs = {}
    for r in mirror['files']:
        p = a.mirror/r['run_id']/r['name']
        assert sha(p) == r['sha256'] and p.stat().st_size == r['bytes']
        refs.setdefault(r['run_id'], {})[r['name']] = {'path': r['path'], 'sha256': r['sha256'], 'bytes': r['bytes']}
    source_sha = sha(a.training_source)
    rows = []
    for run_id in sorted(refs):
        base = a.mirror/run_id
        r = load(base/'receipt.json')
        assert r['config']['script_sha256'] == source_sha
        item = analyze(load(base/'curve.json'), r, load(base/'initialization.json'))
        item['cohort'] = 'confirmation6' if '/confirmation/runs/' in refs[run_id]['receipt.json']['path'] else 'screen12'
        item['source_refs'] = refs[run_id]
        rows.append(item)
    rows.sort(key=lambda r:(r['family'],r['fold'],r['seed'],r['policy']))
    summary = {'schema': 'existing_optimization_curve_review_v1', 'status': 'complete',
               'code_sha256': sha(__file__), 'training_source_sha256': source_sha,
               'mirror_receipt_sha256': sha(receipt_path), 'run_count': len(rows),
               'cohort_counts': dict(Counter(r['cohort'] for r in rows)),
               'stop_reason_counts': dict(Counter(r['stop_reason'] for r in rows)),
               'tail_window_counts': dict(Counter(r['late_window_description'] for r in rows)),
               'selected_best_in_final_30_epochs': sum(r['last_selected_improvement_gap_epochs'] < 30 for r in rows),
               'all_train_H2_last_below_first': all(r['train_H2_separate']['last'] < r['train_H2_separate']['first'] for r in rows),
               'all_train_H2_last10_mean_below_prior10': all(r['train_H2_separate']['last_10_mean'] < r['train_H2_separate']['prior_10_mean'] for r in rows),
               'training_source_line_evidence': {'protein_weighting': [82], 'eval_predict': [101, 106],
                                                  'epoch0_and_snapshot': [114, 116], 'online_train_H2': [118, 124],
                                                  'stop_and_selection': [125, 127], 'LR_and_early_stop': [129, 136],
                                                  'best_restore_and_saved_evaluation_only': [137, 140]},
               'method': {'late_window_comparison': 'mean last 10 recorded epochs minus mean preceding 10; descriptive, not a convergence test',
                          'near_flat_band': 'absolute difference <= 1e-5, reusing training checkpoint min_delta only as a descriptive scale',
                          'selected_best': 'replay original sequential score < previous selected best - 1e-5, including eligible epoch 0; not necessarily raw curve minimum',
                          'train_H2': 'online dropout-on changing-parameter minibatch loss, protein inverse-duplicate weights; not group macro evaluation H',
                          'stop_H': 'end-of-epoch eval/dropout-off H then row-to-protein-to-group equal-group mean',
                          'not_inferred': 'No epoch CDF/tail trajectories; no causal overfitting/generalization claim; no new fitting or checkpoint load'},
               'runs': rows}
    a.output_prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = ['既有 18 次优化曲线的独立复核（私有详细记录）', '',
             '只分析冻结日志：12 次初筛、6 次确认；54 份原始 JSON 的 SHA 全部匹配。逐 epoch 重放实际选 best 与 LR/早停状态机，18/18 与收据一致。没有加载模型、预测数组或启动训练。',
             '全部 18 次由 early_stop 结束；最后达到选模改善阈值的 epoch 距终点 30–38 epoch。没有一次在终点仍刷新按 min_delta=1e-5 选中的 best。早停只是本次策略停止条件，不证明全局收敛或排除其它优化方案。',
             '末段仅作描述：最后 10 epoch 均值与前 10 epoch 均值比较，以 1e-5 为描述性近平台带（并非预声明科学显著性或收敛门槛）。末段下降也可能仍显著高于先前最佳点，不能与持续刷新 best 混同。', '',
             'family policy fold seed | epoch0 H | selected best(epoch/stage) | end H(epoch) | last10-prev10 | description']
    for r in rows:
        lines.append(f"{r['family']} {r['policy']} f{r['fold']} s{r['seed']} | {r['initial_stop_H']:.8f} | {r['selected_best_stop_H']:.8f}({r['selected_best_epoch']}/s{r['selected_best_stage']}) | {r['last_stop_H']:.8f}({r['epochs']}) | {r['late_window_delta_H']:+.8f} | {r['late_window_description']}")
    lines += ['', '解读边界：',
              '1. 原始 fresh/plateau epoch0 约 0.525–0.528，reset/retain epoch0 为同一 fresh 已选最佳点约 0.0485–0.0522。warm-start 的 epoch1 并不相当于从头训练 epoch1。reset/retain 的额外训练预算须加上源 fresh 成本。',
              '2. 所有 plateau 都实际进入三档 LR。多数选中 best 位于 3e-5 阶段；mean_pre f0 s3407 的 best 在第二档 epoch66，第三档没有超过 1e-5 的新选中改善，最后 epoch101 是末档最小运行长度约束与全局耐心共同作用。',
              '3. residue_mean f0 s1701 的 plateau 从 fresh best H=0.04851890 到 0.04758426；末段仍小幅下行但未刷新 epoch104 的 best。residue warm reset/retain 类似，末段下降却未达到 epoch42 之后再选新 best 的 1e-5 改善阈值。',
              '4. mean_pre reset/retain 在 warm-start 后 epoch2 取得 best，随后到 epoch40 的末段均值上升；charge_deep reset/retain best 在 epoch10，末段均值上升。此为这些固定设置停止集上的最佳后回升，不足以宣称所有泛化失效或确定因果性过拟合。',
              '5. train_H2 的首尾下降在全部 18 次可见，但它是 dropout-on、参数沿 minibatch 变化、蛋白重复数加权的 H²；stop_H 是固定 epoch 末 eval 下按组宏平均 H。二者既量纲/变换不同，也聚合与模式不同；不能直接相减、开方后声称可比 gap。',
              '6. 原日志没有逐 epoch train eval-H/训练 PMF、CDF 或边尾误差曲线。最佳 checkpoint 的固定状态补测可填同口径 train/stop 缺口，但不能追溯未存 epoch 轨迹。',
              '7. 原来的 raw 最小值可比选中 best 低不足 1e-5；本报告显式同时保留，绝不把 raw min epoch 改成保存 checkpoint 的 epoch。', '',
              f"Counts: {summary['tail_window_counts']}", f"Training source SHA256: {source_sha}", f"Mirror receipt SHA256: {sha(receipt_path)}", f"Review code SHA256: {sha(__file__)}"]
    a.output_prefix.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('runs','method')},indent=2))


if __name__ == '__main__':
    main()
