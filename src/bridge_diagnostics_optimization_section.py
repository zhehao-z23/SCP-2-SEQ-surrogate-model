"""Append existing optimization logs to a private offline gallery. No model IO."""
from pathlib import Path
import argparse, csv, hashlib, html, json, os, re, shutil, sys, time
from html.parser import HTMLParser

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def table(rows, columns):
    def cell(v):
        if isinstance(v, float):
            return f'{v:.7g}'
        return '' if v is None else str(v)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(cell(r.get(c))) + '</td>' for c in columns) + '</tr>' for r in rows) + '</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--logs', type=Path, required=True)
    ap.add_argument('--gallery', type=Path, required=True)
    ap.add_argument('--plot-packages', type=Path)
    args = ap.parse_args()
    if args.plot_packages:
        sys.path.insert(0, str(args.plot_packages.resolve()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from PIL import Image
    start = time.monotonic()
    review, logs, gallery = args.review.resolve(), args.logs.resolve(), args.gallery.resolve()
    section = gallery / 'optimization18'
    private = gallery / 'private/optimization18'
    for p in [section / 'plots', private / 'logs', private / 'source_summaries']:
        p.mkdir(parents=True, exist_ok=True)
    mirror = load(logs / 'mirror_receipt.json')
    assert mirror['run_count'] == 18 and mirror['file_count'] == 54 and mirror['all_hashes_match'] is True
    for item in mirror['files']:
        path = logs / item['run_id'] / item['name']
        assert sha(path) == item['sha256'] == item['local_sha256'] and path.stat().st_size == item['bytes']
    shutil.copyfile(logs / 'mirror_receipt.json', private / 'source_summaries/mirror_receipt.json')
    previous_manifest_path = private / 'artifact_manifest_before.json'
    if not previous_manifest_path.exists():
        shutil.copyfile(gallery / 'artifact_manifest.json', previous_manifest_path)
        shutil.copyfile(gallery / 'index.html', private / 'index_before.html')
    old_manifest = load(previous_manifest_path)
    assert len(old_manifest) == 66
    for name, pin in old_manifest.items():
        path = private / 'index_before.html' if name == 'index.html' else gallery / name
        assert sha(path) == pin, name
    sources = {}
    for name in ['bridge_optimization_screen_results.json', 'bridge_optimization_confirmation_results.json', 'bridge_optimization_references_results.json', 'bridge_optimization_public.json', 'bridge_optimization_report.md', 'bridge_roles_training_assets_private.json', 'bridge_optimization_generalization_curve_review.json', 'bridge_optimization_generalization_curve_review.txt']:
        shutil.copyfile(review / name, private / 'source_summaries' / name)
        sources[name] = sha(review / name)
    screen = load(review / 'bridge_optimization_screen_results.json')
    confirmation = load(review / 'bridge_optimization_confirmation_results.json')
    refs = load(review / 'bridge_optimization_references_results.json')
    public = load(review / 'bridge_optimization_public.json')
    inventory = {r['run_id']: r for r in load(review / 'bridge_roles_training_assets_private.json')['cohorts']['optimization18']}
    old_receipt = load(gallery / 'public/receipt.json')
    trajectory_review = load(review / 'bridge_optimization_generalization_curve_review.json')
    assert trajectory_review['status'] == 'complete' and trajectory_review['run_count'] == 18
    assert trajectory_review['mirror_receipt_sha256'] == sha(logs / 'mirror_receipt.json')
    rows = screen['all_fits'] + confirmation['all_fits']
    assert len(rows) == len(inventory) == 18 and len({r['run_id'] for r in rows}) == 18
    families = ['residue_mean', 'mean_pre', 'charge_deep']
    family_name = {'residue_mean': 'Residue mean + charge', 'mean_pre': 'Mean MLP + charge', 'charge_deep': 'Charge MLP (common runner)'}
    policies = ['fresh', 'plateau', 'reset', 'retain']
    policy_color = {'fresh': '#62778d', 'plateau': '#cf562c', 'reset': '#3985b6', 'retain': '#6e58a8'}
    stage_color = ['#e8eff9', '#e5f2e8', '#fff0d7']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.hashsalt': 'optimization18'})
    records, curves, log_refs = {}, {}, []
    sorted_rows = sorted(rows, key=lambda r: (families.index(r['family']), r['fold'], r['seed'], policies.index(r['policy'])))
    run_summary = []
    metric_key = {'H': 'group_macro_H', 'CDF': 'group_macro_cutpoint_CDF_MSE', 'tail': 'group_macro_edge_bin_mass_MAE'}
    for number, row in enumerate(sorted_rows, 1):
        rid = row['run_id']; run = logs / rid
        for filename in ['curve.json', 'receipt.json', 'initialization.json']:
            src = run / filename; dest = private / 'logs' / rid / filename
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dest)
            log_refs.append({'run_id': rid, 'name': filename, 'source_path': inventory[rid]['directory'] + '/' + filename, 'sha256': sha(src), 'mirror': str(dest.relative_to(gallery)), 'bytes': src.stat().st_size})
        rec, curve, init = load(run / 'receipt.json'), load(run / 'curve.json'), load(run / 'initialization.json')
        cfg = rec['config']
        assert rec['status'] == 'complete' and rec['stop_reason'] == 'early_stop'
        assert rec['run_id'] == rid == hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        assert (cfg['mode'], cfg['policy'], cfg['fold'], cfg['seed']) == (row['family'], row['policy'], row['fold'], row['seed'])
        assert cfg['release_sha256'] == old_receipt['source_release_sha256']
        assert rec['checkpoint_sha256'] == row['checkpoint_sha256'] == inventory[rid]['checkpoint_sha256']
        assert rec['prediction_sha256'] == row['prediction_sha256']
        assert cfg['script_sha256'] == row['script_sha256'] and cfg['model_sha256'] == row['model_sha256']
        assert len(curve) == rec['epochs'] == row['epochs']
        assert [r['epoch'] for r in curve] == list(range(1, rec['epochs'] + 1))
        assert init['stop_H'] == rec['initial_stop_H'] == row['initial_stop_H']
        best, best_epoch = init['stop_H'], 0
        for point in curve:
            assert all(np.isfinite(point[k]) and point[k] >= 0 for k in ['train_H2', 'stop_H', 'lr'])
            if point['stop_H'] < best - 1e-5:
                best, best_epoch = point['stop_H'], point['epoch']
        assert best_epoch == rec['best_epoch'] == row['best_epoch'] and best == rec['best_stop_H'] == row['stop_H']
        for key, result_key in metric_key.items():
            assert abs(rec['result'][result_key] - row[key]) < 1e-7, (rid, key)
        stage = 0; stage_start = 0; stage_best = init['stop_H']; stage_best_epoch = 0
        stages = []
        for point in curve:
            assert point['LR_stage'] == stage
            expected_lr = [3e-4, 1e-4, 3e-5][stage] if row['policy'] == 'plateau' else (1e-4 if row['policy'] in ['reset', 'retain'] else 3e-4)
            assert point['lr'] == expected_lr
            if not stages or stages[-1]['stage'] != stage:
                stages.append({'stage': stage, 'first_epoch_using_lr': point['epoch'], 'lr': point['lr']})
            if row['policy'] == 'plateau':
                if point['stop_H'] < stage_best - 1e-5:
                    stage_best, stage_best_epoch = point['stop_H'], point['epoch']
                if stage < 2 and point['epoch'] - stage_best_epoch >= 15:
                    stage += 1; stage_start = point['epoch']; stage_best = point['stop_H']; stage_best_epoch = point['epoch']
        assert rec['epochs'] >= 40 and rec['epochs'] - best_epoch >= 30
        if row['policy'] == 'plateau':
            assert stage == 2 and rec['epochs'] - stage_start >= 20
        best_lr = curve[best_epoch - 1]['lr'] if best_epoch else curve[0]['lr']
        extra = {'plot_number': number, 'best_lr': best_lr, 'LR_stages': stages, 'best_checkpoint': 'best.pt (written once at fit end)', 'source_best_epoch': None}
        records[rid], curves[rid] = rec, curve
        run_summary.append({**row, **extra})
    for row in run_summary:
        source = row['source_run_id']
        if source:
            assert source in records and records[source]['config']['policy'] == 'fresh'
            assert records[row['run_id']]['initial_stop_H'] == records[source]['best_stop_H']
            row['source_best_epoch'] = records[source]['best_epoch']
    fixed_path = review / 'bridge_best_state_diagnostics_result.json'
    assert sha(fixed_path) == '292f35d956933b8ff0706a37b2ecf8ad12ff693b07c62c10f5e39801d8083742'
    fixed = load(fixed_path)
    assert fixed['status'] == 'complete' and fixed['new_fits'] == fixed['optimizer_updates'] == 0
    assert fixed['release_sha256'] == old_receipt['source_release_sha256']
    fixed_rows = []
    for result in fixed['results']:
        prior = records[result['run_id']]
        assert prior['checkpoint_sha256'] == result['checkpoint_sha256'] and prior['best_epoch'] == result['best_epoch']
        assert result['partitions']['evaluation']['saved_prediction_max_abs_difference'] == 0
        assert abs(result['partitions']['stop']['group_macro']['H'] - prior['best_stop_H']) < 4e-9
        assert abs(result['partitions']['evaluation']['group_macro']['H'] - prior['result']['group_macro_H']) < 1e-7
        for partition in ['train', 'stop', 'evaluation']:
            fixed_rows.append({'policy': result['policy'], 'best_epoch': result['best_epoch'], 'partition': partition, **result['partitions'][partition]['group_macro']})
    assert len(fixed_rows) == 6
    shutil.copyfile(fixed_path, private / 'fixed_best_summary.json')
    sources[fixed_path.name] = sha(fixed_path)
    checks = []
    for row in screen['shared_initial_trajectories'] + confirmation['shared_initial_trajectories']:
        match = [r for r in rows if (r['family'], r['fold'], r['seed']) == (row['family'], row['fold'], row['seed'])]
        fresh = next(r for r in match if r['policy'] == 'fresh'); plateau = next(r for r in match if r['policy'] == 'plateau')
        common = [(a, b) for a, b in zip(curves[fresh['run_id']], curves[plateau['run_id']]) if b['LR_stage'] == 0]
        assert len(common) == row['shared_pre_decay_epochs']
        for key, checkkey in [('train_H2', 'max_train_loss_difference'), ('stop_H', 'max_stop_H_difference')]:
            assert max(abs(a[key] - b[key]) for a, b in common) == row[checkkey] == 0
        checks.append(row)
    plots = []
    def save(fig, name):
        for ext in ['png', 'svg']:
            fig.savefig(section / 'plots' / (name + '.' + ext), dpi=150, bbox_inches='tight', metadata={'Date': None} if ext == 'svg' else None)
        plt.close(fig); plots.append(name)
    for row in run_summary:
        curve = curves[row['run_id']]; rec = records[row['run_id']]
        ep = np.array([x['epoch'] for x in curve]); stop = np.r_[rec['initial_stop_H'], [x['stop_H'] for x in curve]]
        xstop = np.r_[0, ep]; train = [x['train_H2'] for x in curve]
        fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
        ax[0, 0].plot(ep, train, color='#2673b8'); ax[0, 0].set(yscale='log', title='Online training loss (dropout ON)', ylabel='Protein-weighted online H-squared')
        for panel in [ax[0, 1], ax[1, 0]]:
            panel.plot(xstop, stop, color='#cf562c', lw=1.5)
            panel.scatter([0], [rec['initial_stop_H']], marker='s', facecolors='white', edgecolors='black', s=45, zorder=6, label='Epoch 0 stop')
            panel.scatter([row['best_epoch']], [row['stop_H']], marker='D', color='#942743', s=45, zorder=7, label='Saved best checkpoint')
            panel.scatter([row['epochs']], [stop[-1]], marker='x', color='#111', s=45, zorder=7, label='Last logged / early stop')
        ax[0, 1].set(yscale='log', title='Stop set: fixed model.eval(), full range', ylabel='Group-macro H')
        zoom = stop[1:] if len(stop) < 12 else stop[11:]
        low, high = float(min(zoom.min(), row['stop_H'])), float(zoom.max())
        pad = max((high - low) * .12, .00025)
        ax[1, 0].set(ylim=(max(0, low-pad), high+pad), title='Same stop curve: explicit late-range zoom', ylabel='Group-macro H (zoom)')
        clipped = int(((stop < max(0, low-pad)) | (stop > high+pad)).sum())
        ax[1, 0].text(.02, .98, f'{clipped} initial/high points outside zoom; full range above', transform=ax[1, 0].transAxes, va='top', fontsize=8)
        ax[1, 1].step(np.r_[0, ep], np.r_[curve[0]['lr'], [x['lr'] for x in curve]], where='post', color='#333')
        ax[1, 1].set(yscale='log', ylim=(2e-5, 5e-4), title=f'Actual LR used at each epoch; best LR={row["best_lr"]:.0e}', ylabel='Learning rate')
        ax[1, 1].set_yticks([3e-5, 1e-4, 3e-4], ['3e-5', '1e-4', '3e-4'])
        ax[1, 1].yaxis.set_minor_formatter(NullFormatter())
        for panel in ax.flat:
            for i, stage in enumerate(row['LR_stages']):
                left = 0 if i == 0 else stage['first_epoch_using_lr'] - .5
                right = row['LR_stages'][i+1]['first_epoch_using_lr'] - .5 if i+1 < len(row['LR_stages']) else row['epochs'] + .5
                panel.axvspan(left, right, color=stage_color[stage['stage']], alpha=.45, zorder=0)
            panel.axvline(row['best_epoch'], color='#942743', ls=':', alpha=.7, lw=1)
            panel.set_xlabel('Fit-local epoch (0 = initialized or restored best)'); panel.grid(alpha=.18)
        ax[0, 1].legend(fontsize=8)
        source_text = f' | branch starts from fresh best epoch {row["source_best_epoch"]}' if row['source_run_id'] else ''
        fig.suptitle(f'{family_name[row["family"]]} | {row["policy"]} | fold {row["fold"]}, seed {row["seed"]}{source_text}\nBest epoch {row["best_epoch"]}; last epoch {row["epochs"]}; online training H² and stop H are NOT a generalization gap', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .93)); save(fig, f'run_{row["plot_number"]:02d}')
    metrics = ['H', 'CDF', 'tail']; metric_labels = ['Group-macro H', 'Group-macro CDF MSE', 'Group-macro edge-bin mass MAE']
    fig, ax = plt.subplots(3, 3, figsize=(13, 10))
    for i, family in enumerate(families):
        fits = [next(r for r in screen['all_fits'] if r['family'] == family and r['policy'] == policy) for policy in policies]
        for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
            vals = [r[metric] for r in fits]
            ax[i, j].scatter(range(4), vals, c=[policy_color[p] for p in policies], s=65)
            ax[i, j].set_xticks(range(4), policies); ax[i, j].set_ylabel(label); ax[i, j].set_title(family_name[family]); ax[i, j].grid(alpha=.2)
            ax[i, j].set_xlim(-.55, 3.55)
            for k, v in enumerate(vals): ax[i, j].annotate(f'{v:.6g}', (k, v), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
            lo, hi = min(vals), max(vals); pad=max((hi-lo)*.35, hi*.005);ax[i,j].set_ylim(lo-pad,hi+pad)
    fig.suptitle('Initial screen | fold 0, seed 1701 | evaluation at stop-selected checkpoint\nReset/retain add a second fit to the fresh source: unequal total optimization budgets', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94)); save(fig, 'screen_metrics')
    comparisons = []
    for r in refs['comparisons']:
        if r['comparison'] != 'plateau minus existing fixed LR3e-4': continue
        comparisons.append({'family': r['family'], 'fold': r['fold'], 'seed': r['seed'], 'H': r['delta_H'], 'CDF': r['new_CDF']-r['reference_CDF'], 'tail': r['new_tail']-r['reference_tail'], 'H_CI': r['descriptive_group_95'], 'reference': 'existing fixed-LR result; same roles/scalers'})
    for r in screen['paired'] + confirmation['paired']:
        if r['family'] != 'charge_deep' or r['comparison'] != 'plateau minus fresh': continue
        fits = [x for x in rows if (x['family'], x['fold'], x['seed']) == (r['family'], r['fold'], r['seed'])]
        new, old = next(x for x in fits if x['policy']=='plateau'),next(x for x in fits if x['policy']=='fresh')
        comparisons.append({'family':r['family'],'fold':r['fold'],'seed':r['seed'],'H':r['delta_H'],'CDF':new['CDF']-old['CDF'],'tail':new['tail']-old['tail'],'H_CI':r['descriptive_group_95'],'reference':'same common-runner fresh result'})
    comparisons.sort(key=lambda r:(families.index(r['family']),r['fold'],r['seed']));assert len(comparisons)==8
    def delta_plot(data, name, title, labels):
        fig, axes = plt.subplots(1,3,figsize=(14,max(3.8,len(data)*.58+1.6)))
        y=np.arange(len(data)); text=[labels(r) for r in data]
        for ax,metric,label in zip(axes,metrics,metric_labels):
            vals=np.array([r[metric] for r in data]);ax.axvline(0,color='#555',lw=1)
            if metric=='H':
                ci=np.array([r['H_CI'] for r in data]);ax.hlines(y,ci[:,0],ci[:,1],color='#8c94a0',lw=2)
            ax.scatter(vals,y,c=np.where(vals<=0,'#2673b8','#bd4d35'),s=45,zorder=3)
            ax.set_yticks(y,text if metric=='H' else ['']*len(data));ax.invert_yaxis();ax.set_xlabel('Delta '+label.replace('Group-macro ','')+'\n(new − reference)',labelpad=10);ax.grid(alpha=.18)
            ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
        fig.suptitle(title+'\nNegative = lower error. H intervals are descriptive paired-group intervals, unadjusted for selection; no CDF/tail intervals inferred.',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.9));save(fig,name)
    delta_plot(comparisons,'replication_deltas','Plateau versus fixed-LR references | partial fold/seed replication, not complete fivefold',lambda r:f'{family_name[r["family"]]}\nf{r["fold"]} / seed {r["seed"]}')
    legacy=[]
    for r in public['legacy_charge_reference']['comparisons']:
        legacy.append({'fold':r['fold'],'H':r['new_minus_legacy_H'],'CDF':r['new_CDF']-r['legacy_CDF'],'tail':r['new_tail']-r['legacy_tail'],'H_CI':r['descriptive_group_95']})
    delta_plot(legacy,'legacy_charge','Retain the earlier capacity charge baseline | new plateau minus old capacity\nDifferent batching/RNG/microbatch paths: practical reference, not an isolated LR experiment',lambda r:f'Charge / fold {r["fold"]}\nseed 1701')
    label_contract={k:old_receipt[k] for k in ['historical_label_contract_sha256','historical_label_binding_sha256','CDF_axis_cutpoint_range','source_release_sha256']}
    dump(private/'run_summary.json',run_summary);dump(private/'source_log_manifest.json',log_refs)
    dump(private/'comparison_values.json',{'screen':screen['all_fits'],'replication':comparisons,'legacy_charge':legacy,'source_summaries':sources,'historical_contract':label_contract})
    with (private/'training_curves.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['run_id','family','policy','fold','seed','epoch','train_H2','stop_H','lr','LR_stage','elapsed_seconds','updates','residue_presentations'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in run_summary:
            for point in curves[row['run_id']]:w.writerow({**{k:row[k] for k in fields[:5]},**{k:point[k] for k in fields[5:]}})
    columns=['family','policy','fold','seed','initial_stop_H','best_epoch','best_lr','epochs','stop_reason','stop_H','H','CDF','tail','fit_seconds','pipeline_fit_seconds','source_best_epoch']
    style='body{margin:0;background:#f4f6f8;color:#17212b;font:16px/1.65 Arial,"Microsoft YaHei",sans-serif}main{max-width:1250px;margin:auto;padding:28px}section{background:white;border:1px solid #dde3e9;border-radius:12px;padding:22px;margin:20px 0}h1,h2{line-height:1.3}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:7px 10px;border-bottom:1px solid #e2e6ea;text-align:right}th{background:#eef3f7}.scroll{overflow:auto}a{color:#17679a}.note{border-left:5px solid #cb6335;padding:14px;background:#fff5e9}code{font-size:13px}summary{cursor:pointer;font-weight:bold}.tags{color:#566371}'
    intro='''<h1>18次优化：既有训练轨迹与结果</h1><p class="tags">12 初筛 + 6 有限复制 · 历史 T → 顶点测度 63 箱 · 私有离线审阅 · 不含新训练或模型推理</p><p><a href="../index.html">← 返回原图册（全部旧案例保留）</a> · <a href="#allruns">18 个逐 run 曲线</a></p><section class="note"><b>这不是 train–stop 泛化 gap 图。</b> train_H2 是 dropout 开启、参数随 minibatch 更新的在线 protein 逆重复加权 H²；stop_H 是 epoch 结束后固定模型 eval 的逐行 H → protein → group → 组宏平均。两者不能相减，sqrt(train_H2) 也不是组宏 H。没有保存逐 epoch 的 train eval H、stop CDF/尾误差或任意 epoch checkpoint，此处不会补画这些缺失数据。</section>'''
    method='''<section><h2>时点、学习率与 checkpoint</h2><p>每个 fit 以局部 epoch 计数。空心方块是初始化/恢复后的 epoch 0 stop H（参与 best 选择）；菱形为 stop 按改善门槛 1e−5 选出的 best；叉号为最后记录的 early-stop epoch。训练中只更新内存快照，结束后写一次 best.pt；没有 last 或周期 checkpoint。低 LR 续训的 epoch 0 从其 fresh best 出发，不等于源 fit 最后 epoch。</p><p>背景颜色与 LR 面板显示实际用于该 epoch 的 LR。连续 plateau 在阶段 15 epoch 未改善后降至 1e−4、3e−5，不回滚模型/AdamW/RNG；LR 切换在该 epoch 更新之后发生，下一 epoch 才使用新 LR。最终阶段至少 20 epoch，整体至少 40 epoch，global best 30 epoch 未达改善后 early stop。reset/retain 都追加 fit，不能将总成本当作 fresh 的等预算单因素比较。</p><p>stop 图含完整范围及明确标注的局部放大；被放大范围排除的早期点数量单列。仅画真实记录，未做平滑或插值。评价 H/CDF/尾误差来自既有独立复算结果，此次只核日志/收据一致性，不重新载入模型或预测数组。</p></section>'''
    summary_html='<section><h2>全部 18 fit 的选择与评价</h2><p>全部已 early-stop。CDF 是 62 个有限切点的 MSE，tail 是首末开放 bin 质量 MAE；均按 protein 再 group 归约。它不是 neg5/pos5 筛选准确率。pipeline time 对 reset/retain 包含源 fresh，源训练在不同分支表中重复展示，资源总数只计一次。</p><details><summary>展开 18 项完整数值表</summary>'+table(run_summary,columns)+'</details></section>'
    conclusions='''<section><h2>比较边界与本轮读出</h2><p>共同 runner 内三家庭均按 stop 选出连续 plateau。Residue mean 在三个有限配置上均改善 H；mean_pre 的第二 seed 与 charge 的 fold1 增益较小。H 改善不意味着 CDF 与尾箱误差全部改善，图中正值会标为红色。区间没有修正选型、多重比较或 seed 不确定性，不能称完整五折复制或盲测。</p><p>旧 capacity 电荷结果仍比新共同 runner 的 charge plateau 更强，应保留为基线。两者 batching、随机数路径、microbatch 与预算上限不同；不能把差异单独归因于 LR，也不能声称这次优化已解决泛化问题。历史 T/顶点 p 不授予新 C/面积标签资格。</p></section>'''
    comparisons_html='<section><h2>初筛四种流程：三指标并列</h2><img src="plots/screen_metrics.png" alt="Three families four optimization policies and three evaluation metrics"></section><section><h2>有限复制：评价指标增量</h2><img src="plots/replication_deltas.png" alt="Partial fold seed replication H CDF tail differences"></section><section><h2>保留旧强电荷参照</h2><img src="plots/legacy_charge.png" alt="New charge plateau versus old capacity baseline"></section>'
    per_run=['<section id="allruns"><h2>18 个独立 fit：原始记录曲线</h2><p>每图均包含在线训练 H²、stop H 全范围和放大图、LR；展开查看。原始日志与绑定位于 private/optimization18/。</p>']
    for row in run_summary:
        title=f'{row["plot_number"]:02d} · {family_name[row["family"]]} · {row["policy"]} · fold {row["fold"]} / seed {row["seed"]} · best {row["best_epoch"]} / last {row["epochs"]}'
        per_run.append('<details><summary>'+html.escape(title)+'</summary><p><code>'+row['run_id']+'</code></p><img src="plots/run_'+f'{row["plot_number"]:02d}'+'.png" loading="lazy" alt="Independent fit training loss stop metric LR and checkpoint markers"></details>')
    per_run.append('</section>')
    trajectory_html='''<section><h2>现在能采取的步骤，以及曲线实际支持什么</h2><p><b>保留连续 plateau 作为下一阶段待验证的读出优化默认；保留旧强电荷基线与 mean_pre 对照。</b> 不需要为继续整理结果追加这 18 个 fit。未来新 C 任务仍需单独确认标签与划分资格。</p><p>独立逐 epoch 状态机复核 18/18 一致；全部 early-stop，所选 best 距终点 30–38 epoch，没有一个在最后仍刷新达到 1e−5 门槛的新 best。最后 10 epoch 与前 10 epoch 的 stop 均值比较：10 项回升、3 项近平台、5 项下降。这里 ±1e−5 仅是描述尺度，不是新的科学收敛判据；末段下降也不等于仍在刷新 best。</p><p>8 个 plateau 中 7 个 best 位于末档；mean_pre / fold0 / seed3407 的 best 在第二档 epoch66，最终到101是末档最少时长与全局耐心共同约束。4 项原始曲线最低点比保存 best 低不足1e−5，图中仍标真实保存 checkpoint，不篡改为 raw-min epoch。在线 train_H2 末段下降不能据此证明过拟合或收敛；同口径固定 best 的 train/stop 测量是另一个补充，不能追溯未保存的历史 epoch。</p><p><a href="../private/optimization18/source_summaries/bridge_optimization_generalization_curve_review.txt">查看独立轨迹短文及18项表</a></p></section>'''
    fixed_html='<section><h2>补齐同口径缺口：两个实际保存 best 的固定状态评估</h2><p>另一个已完成补充只恢复 residue_mean / fold0 / seed1701 的 fresh epoch68 与 plateau epoch104，统一 dropout 关闭、固定参数、逐行 → protein → group → 组宏归约。0新fit、0优化更新；测量耗时49.77秒。两份 evaluation 与原保存预测最大差为0；stop与原收据差小于4e−9。原数组留Byte，仅引用汇总。</p>'+table(fixed_rows,['policy','best_epoch','partition','H','H2','CDF_MSE','edge_mass_MAE'])+'<p><b>train、stop、evaluation 的H与CDF都下降，但evaluation首末开放bin质量MAE由0.002021698605略升至0.002023031852。</b> 同口径train–stop H差扩大，但stop/evaluation H同时改善，因此不能仅由差距扩大判定有害过拟合。这是两个已选状态的有限对照，不是epoch轨迹、最终独立测试或新C结论。尾箱误差不等于neg5/pos5筛选准确率。</p><p><a href="../private/optimization18/fixed_best_summary.json">固定best评估汇总与源码/角色/输入SHA</a></p></section>'
    intro=intro.replace('不含新训练或模型推理','无新训练；曲线来自日志；另附两个best固定评估')
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>18次优化补充 · 私有图册</title><style>'+style+'</style></head><body><main>'+intro+trajectory_html+fixed_html+conclusions+method+summary_html+comparisons_html+''.join(per_run)+'<section><h2>来源与未执行项</h2><p>沿用原图册绑定的历史 63 箱合同，有限切点范围 −44.6206059 至 108.0536255 kBT/e；本集成程序不重新画 PMF/CDF 分布坐标、不读取保护标签。原始 66 文件快照逐 hash 校验，旧 index 留存，旧案例和数据字节不变。图册集成没有新 fit、embedding、模型加载/预测或物理计算；另一个独立执行的两个best评估仅引用其完成汇总。未公开发布图片。</p></section></main></body></html>'
    (section/'index.html').write_text(doc,encoding='utf-8')
    old_index=(private/'index_before.html').read_text(encoding='utf-8')
    old_index=old_index.replace('无新训练或GPU','原OOF图册构建无新训练或GPU；新增补测两份既有best约49.77秒GPU进程内计时，0新fit/0update')
    old_index=old_index.replace('没有网络请求、坐标下载、新同源搜索、模型训练或物理计算。','原OOF/续训图册构建没有网络请求、坐标下载、新同源搜索、模型训练或物理计算。新增18曲线集成仍只读日志；另附两份既有best的GPU固定评估（约49.77秒进程内计时、0新训练），不是历史曲线重建。')
    banner='<section id="optimization18-update"><h2>新增：18次优化轨迹与两个best固定评估</h2><p><a href="optimization18/index.html">先看行动结论、同口径小表与证据限制，再展开18个fit曲线 →</a></p><p>复用已完成优化；online train H² 与 stop 组宏 H 分开显示，不称泛化 gap。两个既有best的固定eval表单列，不混入训练轨迹。以下原30 OOF、3续训与10案例保留。</p></section>'
    assert old_index.count('<main>')==1
    (gallery/'index.html').write_text(old_index.replace('<main>','<main>'+banner,1),encoding='utf-8')
    class Assets(HTMLParser):
        def __init__(self,root):super().__init__();self.root=root;self.count=0
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key=='src':
                    assert not value.startswith(('http:','https:','//')) and (self.root/value).is_file(),value
                    self.count+=1
    for page in [gallery/'index.html',section/'index.html']:
        parser=Assets(page.parent);parser.feed(page.read_text(encoding='utf-8'))
    for path in (section/'plots').glob('*.png'):
        with Image.open(path) as image:image.verify()
    assert len(plots)==21
    for name,pin in old_manifest.items():
        if name!='index.html':assert sha(gallery/name)==pin,name
    # Contact sheet supports human QA; keep it private and out of source summaries.
    thumb_w=420;thumb_h=275;sheet=Image.new('RGB',(thumb_w*3,thumb_h*7),'white')
    for i,name in enumerate(plots):
        with Image.open(section/'plots'/(name+'.png')) as image:
            image.thumbnail((thumb_w,thumb_h));sheet.paste(image,((i%3)*thumb_w,(i//3)*thumb_h))
    sheet.save(private/'QA_optimization18.png')
    receipt={'schema':'private_optimization18_gallery_update_v1','status':'rendered_numeric_checks_passed_visual_QA_pending','code_sha256':sha(__file__),'existing_artifact_count':66,'existing_files_unchanged':65,'existing_index_preserved_sha256':sha(private/'index_before.html'),'old_manifest_sha256':sha(previous_manifest_path),'optimization_fits':18,'new_run_plots':18,'new_comparison_plots':3,'curve_points':sum(len(c) for c in curves.values()),'source_summaries':sources,'source_log_manifest_sha256':sha(private/'source_log_manifest.json'),'historical_contract':label_contract,'checks':{'receipts_config_hash_and_model_prediction_pins_match_prior_audit':True,'consecutive_epoch_logs':True,'best_selection_replayed_from_epoch0':True,'LR_schedule_and_early_stop_rule_replayed':True,'reported_eval_metrics_match_prior_recomputation_within_1e_minus7':True,'shared_pre_LR_trajectories_exact':checks,'old_case_and_evidence_bytes_unchanged':True,'all_HTML_images_exist_offline':True,'PNG_decode_integrity':True},'metric_contract':{'train_H2':'online model.train/dropout, inverse-protein-repeat weights; not fixed model eval or group macro H','stop_H':'epoch-end model.eval; per-row H -> protein -> group -> group macro','missing':['epochwise train eval metrics','epochwise stop CDF/tail','last or arbitrary epoch checkpoints'],'generalization_gap_not_computed':True},'new_training':0,'model_loads':0,'prediction_array_reads':0,'physical_calls':0,'protected_labels_read':0,'execution_location':'local computer','elapsed_seconds':time.monotonic()-start,'environment':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    receipt['fixed_best_external_measurement']={'status':'complete','source_summary_sha256':sha(fixed_path),'code_sha256':fixed['code_sha256'],'measured_saved_checkpoints':2,'new_fits':0,'optimizer_updates':0,'arrays_copied_to_gallery':False,'summary_only':True,'separate_execution_not_the_gallery_builder':True}
    dump(private/'integration_receipt.json',receipt)
    shutil.copyfile(__file__, private/'bridge_diagnostics_optimization_section.py')
    (section/'REPORT.md').write_text('''# 18次优化图册集成（私有）

本集成复用已执行的12初筛、6复制，18份curve、receipt与initialization共54个原始JSON。图册程序只在本机渲染，没有新fit、模型加载、预测数组读取、物理调用或保护标签读取。另附独立完成的两个best固定评估汇总，不在图册内执行模型。先前66个图册文件中，65个保持逐字节不变，原index单独保留后增加补充入口。

入口按可行动结论、证据限制、比较和逐run展开排列；18张run图分别展示online train H²、stop组宏H全范围/明确放大范围、epoch0、实际LR阶段、选中best及early-stop终点。另有3张初筛/有限复制/旧强电荷比较图。train与stop不做差，也不伪造训练eval或逐epoch CDF/尾部曲线。

全部54文件SHA、receipt身份、epoch连续性、epoch0参与的best选择、逐epoch LR、原早停规则均已核对；评价H/CDF/首末开放bin质量MAE与既有独立复算一致（1e−7容差），未重新读取预测数组。此前同初始轨迹的相同段逐值一致。所有HTML图像均为存在的离线资源，PNG解码通过；人工视觉检查及最终镜像状态见private/optimization18/visual_QA.json及独立同步收据。

独立轨迹复核18/18与状态机一致。所选best距终点30–38epoch；末10对前10的stop均值为10回升、3近平台、5下降，±1e−5仅作描述尺度，不能证明全局收敛。8plateau中7best在末档；mean_pre f0/3407是第二档epoch66。4项raw-min低于保存best不足min_delta，不改真实checkpoint时点。

连续plateau可作为下一阶段待验证的工程默认；旧capacity电荷强基线保留。H改善不保证CDF或首末开放bin质量误差均改善，末者不等于neg5/pos5筛选准确率。所有结果仍属已曝光历史T/顶点63箱开发域，部分fold/seed复制，不授予新C/面积标签或全sequence-space泛化资格。

两个既有residue best（fresh68/plateau104）的实际固定状态评估完成49.77秒、0fit/0update；train/stop/evaluation采用相同eval与组宏口径。H/CDF均降低，但evaluation首末开放bin质量MAE由0.002021698605略升至0.002023031852。train–stop差扩大不足以单独判定有害过拟合，因为stop/evaluation本身改善；这仍不是完整epoch轨迹或最终测试。summary的SHA为292f35d956933b8ff0706a37b2ecf8ad12ff693b07c62c10f5e39801d8083742，模型数组仅留Byte原补充目录。

逐run目录与SHA、54日志、比较数值、独立短文、训练资产缺失及代码副本均保留在private/optimization18/。没有发布图片、公开私有表或Git提交。
''',encoding='utf-8')
    manifest={p.relative_to(gallery).as_posix():sha(p) for p in sorted(gallery.rglob('*')) if p.is_file() and p.name not in {'.gitignore','artifact_manifest.json'}}
    dump(gallery/'artifact_manifest.json',manifest)
    print(json.dumps({'status':receipt['status'],'files_in_current_manifest':len(manifest),'new_plots':len(plots),'manifest_sha256':sha(gallery/'artifact_manifest.json'),'elapsed_seconds':receipt['elapsed_seconds']}))


if __name__=='__main__':
    main()
