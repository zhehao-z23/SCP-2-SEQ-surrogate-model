"""Render a private decision entry and project only reviewed aggregate evidence.

No training, inference, target arrays, new case selection or metric recomputation.
The prior index is preserved by exact bytes before a single navigation insertion.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_bytes((json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode())


def table(head, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--gallery',type=Path,required=True)
    p.add_argument('--public-draft',type=Path,required=True)
    a=p.parse_args()
    assert sha(a.plan)==a.plan_sha256
    plan=json.loads(a.plan.read_bytes())
    assert sha(__file__)==plan['renderer_sha256']
    sources={}
    refs={}
    for name,ref in plan['sources'].items():
        path=Path(ref['path'])
        assert sha(path)==ref['sha256'],name
        sources[name]=json.loads(path.read_bytes())
        refs[name]={'sha256':ref['sha256'],'schema':sources[name].get('schema')}
    sp=sources['shape_partition'];tail=sources['tail_screening'];lm=sources['length_match'];cr=sources['curve_review']
    assert all(x['status']=='complete' for x in [sp,tail,lm,cr])
    assert sp['code_sha256']==plan['scientific_code']['bridge_shape_partition_diagnostics.py']['sha256']
    assert tail['code_sha256']==plan['scientific_code']['bridge_tail_screening_audit.py']['sha256']
    assert tail['bounds_helper_sha256']==plan['scientific_code']['bridge_tail_fraction_bounds.py']['sha256']
    assert lm['code_sha256']==plan['scientific_code']['bridge_length_matched_diagnostic.py']['sha256']
    shape=[]
    for r in sp['results']:
        v={k:r[k] for k in ['predictor','partition','rows','unique_T','groups','metrics','negative','positive','fixed_one_percent_ranking']}
        v['truth_strata']={key:{k:value[k] for k in ['rows','unique_T','groups','metrics']} for key,value in r['truth_strata'].items()}
        shape.append(v)
    projected={
        'schema':'diagnostic_decisions_aggregate_v1','status':'completed_read_only_evidence_loss_result_not_yet_integrated',
        'sources':refs,
        'curve_review':{k:cr[k] for k in ['run_count','stop_reason_counts','tail_window_counts','selected_best_in_final_30_epochs','all_train_H2_last10_mean_below_prior10']},
        'shape_partition':{'results':shape,'scope':sp['scope'],'constant_reference_weighting':sp['constant_reference_weighting'],'new_fits':sp['new_fits'],'model_loads':sp['model_loads']},
        'tail_screening':{k:tail[k] for k in ['schema','cohort','results','definition','not_the_historical_extreme_rule','no_false_precision','new_fits','model_loads','raw_phi_reads','code_sha256','bounds_helper_sha256','identity_sha256','elapsed_seconds']},
        'length_match':{k:lm[k] for k in ['cohort','canonical_T','near_complete_coverage','long_pool','short_pool','pairs','distinct_groups_used','unmatched_long_reasons','balance','results','residue_vs_charge_interaction','code_sha256','method_sha256','identity_sha256','new_fits','new_inference','limitation','elapsed_seconds']},
        'length_matching_interpretation_gate':{'minimum_pairs':20,'observed_pairs':lm['pairs'],'passed':lm['pairs']>=20,'posthoc_relaxation':False},
        'scope':'Historical exposed development data. No final test, new-C or AF generalization claim. Public projection excludes sample/member tables, case target summaries, probabilities and images.'}
    assert lm['pairs']==8 and projected['length_matching_interpretation_gate']['passed'] is False
    rows=[];shape_rows=[]
    for part in ['train','stop','evaluation']:
        f=next(r for r in sp['results'] if r['predictor']=='fresh' and r['partition']==part)
        q=next(r for r in sp['results'] if r['predictor']=='plateau' and r['partition']==part)
        rows.append([part,f['unique_T'],f['groups'],f"{f['metrics']['H']['group_macro']:.8f} → {q['metrics']['H']['group_macro']:.8f}",f"{f['metrics']['CDF_MSE']['group_macro']:.6g} → {q['metrics']['CDF_MSE']['group_macro']:.6g}",f"{f['metrics']['edge_MAE']['group_macro']:.9f} → {q['metrics']['edge_MAE']['group_macro']:.9f}"])
        shape_rows.append([part,f"{f['negative']['row_events']['TP']['unique_T']} → {q['negative']['row_events']['TP']['unique_T']} / {f['negative']['truth_positive']['unique_T']}",f"{f['positive']['row_events']['TP']['unique_T']} → {q['positive']['row_events']['TP']['unique_T']} / {f['positive']['truth_positive']['unique_T']}",f['fixed_one_percent_ranking']['budget_T'],f"{f['fixed_one_percent_ranking']['negative']['true_positive_T']} → {q['fixed_one_percent_ranking']['negative']['true_positive_T']}",f"{f['fixed_one_percent_ranking']['positive']['true_positive_T']} → {q['fixed_one_percent_ranking']['positive']['true_positive_T']}"])
    tailrows=[];tailsupport=[]
    for r in tail['results']:
        if r['window']!='whole_support':continue
        tailsupport.append([r['side'],r['truth_support']['certified_positive']['unique_T'],r['truth_support']['certified_positive']['groups'],r['truth_support']['unresolved']['unique_T']])
        for m in sorted(r['models'],key=lambda z:(z['family'],z['seed'])):
            tailrows.append([r['side'],m['family'],m['seed'],m['known_truth_TP']['unique_T'],m['known_truth_FP']['unique_T'],m['known_truth_FN_including_defer']['unique_T'],m['deferred_prediction']['unique_T']])
    optional={}
    for key in ['case_facts_panel','loss_panel','case_loss_panel']:
        if key in plan:
            ref=plan[key];assert sha(ref['path'])==ref['sha256'];panel=json.loads(Path(ref['path']).read_bytes())
            assert set(panel)<= {'status','headline','paragraphs','table_headers','table_rows','evidence_sha256','public_aggregate'}
            projected[key]=panel.get('public_aggregate',{'status':panel['status'],'evidence_sha256':panel['evidence_sha256']})
            frag='<div class="insert"><h3>'+html.escape(panel['headline'])+'</h3>'+''.join('<p>'+html.escape(t)+'</p>' for t in panel['paragraphs'])
            if panel.get('table_headers'):frag+=table(panel['table_headers'],panel['table_rows'])
            optional[key]=f'<section id="{key}">{frag}</div></section>'
    if 'loss_panel' in plan:
        projected['status']='completed_development_decisions_loss_probe_failed_promotion'
    if 'loss_plot' in plan:
        ref=plan['loss_plot'];assert sha(ref['path'])==ref['sha256']
        optional['loss_panel']=optional.get('loss_panel','')+'<section><h3>两臂真实轨迹</h3><p>不同定义的 online loss 分轴；共同 stop H、未缩放 CE 与 cut-CDF 单列。标出 epoch0、固定 LR 转折及各自保存 best。没有补造逐epoch train-eval轨迹。</p><a href="plots/loss_trajectories.png"><img alt="两臂134轮真实曲线与选模位置" style="width:100%;height:auto" src="plots/loss_trajectories.png"></a></section>'
    q4=('固定架构、初始化、角色、134 轮既定 LR 路径与 stop-H 选模的 H² 对 CE/4 受控比较已启动。先验证 H² 控制逐轮重放，才允许 CE。当前页面不预判结果。' if 'loss_panel' not in plan else '受控损失实验的实际结果与推广判定见本页下方验收补充；保留原预声明门槛，不通过后再调阈值救结果。')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模型诊断 · 四项验收结论</title><style>
    body{margin:0;background:#f2f5f7;color:#1c2d3a;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px 24px 60px}nav{background:#17364d;color:#fff;padding:14px 24px;position:sticky;top:0;z-index:2}nav a{color:#fff;margin-right:20px}h1,h2,h3{line-height:1.35}h1{font-size:32px}h2{font-size:24px}section,.intro{background:#fff;border-radius:12px;padding:24px;margin:20px 0;box-shadow:0 2px 10px #183e5710}.lead{font-size:19px;font-weight:600}.status{display:inline-block;font-size:13px;border-radius:20px;padding:3px 11px;background:#e0edf4;color:#174762}.limit{border-left:4px solid #c88625;padding:9px 15px;background:#fff8ed}.action{border-left:4px solid #287c66;padding:9px 15px;background:#edf8f4}a{color:#14648e}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}th,td{padding:9px 12px;border-bottom:1px solid #dce5eb;text-align:right}th{background:#eaf1f6}th:first-child,td:first-child{text-align:left}details{margin:15px 0}summary{cursor:pointer;font-weight:600}.small{font-size:13px;color:#566876}code{overflow-wrap:anywhere}footer{margin:28px 0;color:#566876}ul{padding-left:22px}@media(max-width:700px){main{padding:12px}section,.intro{padding:18px}h1{font-size:26px}nav{position:static}}</style><body><nav><a href="../index.html">图册首页</a><a href="#q1">继续训练</a><a href="#q2">筛选用途</a><a href="#q3">失配解释</a><a href="#q4">改进验证</a></nav><main>
    <div class="intro"><span class="status">开发证据 · 非最终独立测试</span><h1>模型诊断：四项验收结论</h1><p class="lead">整体 H 已改善，科学筛选仍有明确缺口。下一项改进由受控实验判定，不由更低的平均误差自动放行。</p><p>先读每项结论、证据与行动，再按需展开曲线和病例。本页复用已验收的日志、固定 best 预测及原始整数计数摘要；未重新挑选病例或重算基础统计。</p></div>
    <section id="q1"><h2>1. 能否判断继续训练是否改善泛化？</h2><p class="lead">能判断已发生的有限改善；不能承诺继续加 epoch 仍有收益。</p><p>18 个原始训练轨迹的 checkpoint/LR/early-stop 状态机均重放一致。最终都 early-stop，保存 best 距终点 30–38 epoch；末 10 epoch stop-H 相对前 10 的描述性趋势为 10 回升、3 近平台、5 下降。下降的 5 个也未再达到选新 best 的 1e−5 改善要求。</p><p>online train H² 在 dropout 开启、参数逐批更新和 protein 逆重复权重下累计；stop 是固定 epoch 末 eval 的组宏 H。二者不能直接相减或开平方后充当 train–stop gap。</p>'''
    doc=doc.replace('section,.intro{','section{scroll-margin-top:80px}section,.intro{')
    doc+=table(['分区','T','组','组宏 H：fresh → plateau','CDF MSE','开放尾箱 MAE'],rows)
    doc+='''<p class="limit">两个既有 best（同 fold0/seed1701，epoch68 与104）采用同一固定 eval 口径，H/CDF 在 train、stop、evaluation 都改善；evaluation 开放尾 MAE 微升。较大的 train–stop H 差距不能单独等同有害过拟合。这是两个状态对照，不是任意 epoch 的泛化轨迹或最终测试。</p><p class="action">保留已验证的 plateau 工程基线；不把 train loss 继续下降当作追加训练的充分理由。下一次改动必须同时通过 stop/evaluation 分布与任务指标。</p><p><a href="../optimization18/index.html">查看18轨迹、epoch0、LR阶段与best/last标记 →</a></p></section>
    <section id="q2"><h2>2. 能否判断特殊形态及极端筛选是否可用？</h2><p class="lead">绝对强偏态阈值筛选当前不可作为可靠自动筛选器；预算排名与尾部浓度需分别评估。</p><p>30-fit OOF 的三模型×两 seed 对强 Bowley 绝对阈值均为零真阳性。失配也存在于参与训练的部分：训练集38个强负偏、11个强正偏，fresh只检出1/0，plateau只检出2/0。它不是仅在未见组上出现的泛化损失。</p>'''
    doc+=table(['分区','负偏 TP / 真值：fresh→plateau','正偏 TP / 真值','每侧1%预算T','预算负偏TP','预算正偏TP'],shape_rows)
    doc+='''<p>真实63箱的声明式箱内均匀代理，在全OOF 220-T预算下恢复61/61负偏、16/17正偏，说明仍有可利用的排名信号；它不是raw分位数、最优性能上界或分箱损失的精确分解。原生有限窗口只有1例可见两主要峰，三视图共识不稳定；“一个主要峰”允许肩峰，也不等于raw单峰。</p><h3>±5质量≥0.8：原始整数计数给出的可辨识界</h3>'''
    doc+=table(['全支持方向','确定阳性 T','阳性组','真值不确定 T'],tailsupport)
    doc+='''<p>真值采用原始 integer counts，不使用FP32累计近似。确定负侧14 T、正侧135 T，另22/374 T无法由63箱确定；这些区间是箱内分配可行界，不是置信区间。预测不确定保留为defer，确定阳性上的defer计入FN。</p><details><summary>三模型、两seed的确定支持内TP/FP/FN与defer</summary>'''+table(['方向','模型','seed','确定真值TP','确定真值FP','确定阳性FN含defer','预测defer T'],tailrows)+'''</details><p class="action">总体H最优的residue并非每种任务最优：负侧charge检出11/10，residue检出4/7；正侧residue检出64/70，但仍漏71/65个确定阳性。保留charge，分任务报告FN/FP与预算召回，不用单一H宣布筛选可用。</p><p class="limit">该浓度flag不是历史完整极端规则：缺少来源总体百分位，物理协议也不同。条件[-60,60)仅改同一Atlas分布的测度，不能复刻旧AF场。开放尾质量误差也不等于±5筛选准确率。</p></section>
    <section id="q3"><h2>3. 是否对主要 bad case 形成了经过检验的解释？</h2><p class="lead">已排除部分简单解释，并确认一个小支持机制；尚不能给出主要失败的单一因果解释。</p><ul><li>标签/QC与分箱计数闭环一致，65763个type-7分位均落在正确支持界内；未认证的Bowley不等于已经证明信息丢失。</li><li>6对相同实际输入（charge字节及缓存地址相同）对应不同目标，输入到标签的多值性已证实；只涉及12行，直接支持量太少，且没有同结构同标签重复控制，不能指认构象或数值噪声。</li><li>在同折/同来源、覆盖≥0.98、目标H≤0.03及两尾caliper下，304长T与1907短T仅形成8对、16个不复用组，低于预定20对解释门槛；没有放宽条件。不能据此归因于长度、位置或结构域。</li></ul><p class="limit">8类病例都按冻结规则选择，不平均个体分布。两主要峰实例N=1，原生与三视图判定不一致已明确标注；病例图仅展示原生cut、有限箱density、开放尾mass及有声明的分位代理。病例事实与正常对照的独立核验只在完成后追加。</p><p><a href="../shape_support_v1/index.html">打开8类独立分布与局部放大 →</a> · <a href="../index.html#correlation">原始分层与相关证据 →</a></p><p class="action">将“支持不足”“未分辨输入/标签差异”和“已测模型失败”分开。继续测试最小可区分假设，而不是用相关性或一个病例归因。</p></section>
    <section id="q4"><h2>4. 是否据此提出并实际验证最值得做的下一项改进？</h2><p class="lead">唯一优先改动：在相同流程下检验损失中心，而非扩大架构搜索。</p><p>'''+html.escape(q4)+'''</p><p>H²的自由PMF风险最小中心通常不同于条件平均PMF；CE对应后者。这是已核数学性质，不是已证项目病因。训练形态也受压缩、真实63箱仍有排名信息，使该受控替换比再加encoder或无界调参更有信息量。</p><p>预声明推广要求：stop及evaluation的强形态联合组宏Bowley AE至少降低10%，或负开放尾偏差幅度至少降低20%且负尾绝对误差降低；同时整体H/CDF不得恶化超过1%，两方向预算TP和不得下降。只有通过才允许至多一组额外fold/seed确认。</p><p class="limit">CE/4只匹配局部二阶损失尺度，不保证梯度裁剪或AdamW等价。LR时间表与stop-H选模沿用H²流程；负结果只否定该有限替换，不证明CE没有更优训练解。不同online loss不能直接相减。</p></section>'''
    if 'case_facts_panel' in optional:
        doc=doc.replace('<section id="q4">',optional['case_facts_panel']+'<section id="q4">',1)
        doc=doc.replace('病例事实与正常对照的独立核验只在完成后追加。','两例冻结偏态代表与固定正常对照的核验已完成，见下方事实补充；病例图与选择规则均未改变。')
    doc+=optional.get('loss_panel','')+'''<section><h2>指标如何支撑决策</h2><p>H检查整体已分箱分布；固定cut上的CDF MSE检查累计位置失配；FN/FP与固定预算召回直接回答科学筛选用途。三者互补，不再堆叠相关指标作为总分。当前CDF MSE不是raw CRPS，也不是带电势单位的W1。</p><p>所有比例保留支持数、未定义值与defer。预测seed重复相同对象，不当作独立生物样本；表面顶点也不是独立生物重复。</p></section><footer>私有审阅入口；图像、HTML和病例数据仅留本地/Byte。来源JSON与SHA在本页private目录保留。公开候选仅代码、指标依据及匿名聚合，经单独复核后发布。</footer></main></body></html>'''
    doc=doc.replace('唯一优先改动：在相同流程下检验损失中心，而非扩大架构搜索。','本轮已测试的单项改动：在相同流程下检验更换损失能否改善形态恢复。')
    doc=doc.replace('继续测试最小可区分假设，而不是用相关性或一个病例归因。','已完成loss替换检验，结果见第4项；其余机制保持未定，不由相关性或一个病例归因。')
    doc=doc.replace('来源JSON与SHA在本页private目录保留。','聚合投影、计划与来源SHA在本页private目录保留；原完整证据留Byte既有目录。')
    doc=doc.replace('<section><h2>指标如何支撑决策</h2>',optional.get('case_loss_panel','')+'<section><h2>指标如何支撑决策</h2>',1)
    gallery=a.gallery.resolve();index=gallery/'index.html'
    old_manifest=gallery/'artifact_manifest_with_shape.json'
    assert sha(old_manifest)==plan['base_manifest_sha256']
    base=json.loads(old_manifest.read_bytes())
    expected={**base['base_files'],**base['added_files']}
    assert len(expected)==199
    backup=gallery/'history'/('index_before_decisions_'+plan['prior_index_sha256'][:12]+'.html')
    if not backup.exists():
        assert sha(index)==plan['prior_index_sha256']
        for rel,h in expected.items():assert sha(gallery/rel)==h,rel
        backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(index.read_bytes())
    assert sha(backup)==plan['prior_index_sha256']
    for rel,h in expected.items():
        if rel!='index.html':assert sha(gallery/rel)==h,rel
    out=gallery/'diagnostic_decisions';(out/'private').mkdir(parents=True,exist_ok=True)
    if 'loss_plot' in plan:
        (out/'plots').mkdir(exist_ok=True);shutil.copy2(plan['loss_plot']['path'],out/'plots/loss_trajectories.png')
    (out/'index.html').write_bytes(doc.encode())
    dump(out/'private/aggregate_projection.json',projected)
    shutil.copy2(a.plan,out/'private/render_plan.json');shutil.copy2(__file__,out/'private/bridge_diagnostic_decisions_page.py')
    entry='<section id="diagnostic-decisions-entry"><h2>当前结论：四项模型诊断验收</h2><p><a href="diagnostic_decisions/index.html">继续训练、形态/极端筛选、bad case解释与下一改进的证据和限制 →</a></p><p>先看行动结论；以下原图册和历史曲线继续保留。新增原生形态病例与当前受控实验状态在结论页统一入口。</p></section>'
    original=backup.read_text(encoding='utf-8')
    assert '<main>' in original and 'diagnostic-decisions-entry' not in original
    index.write_bytes(original.replace('<main>','<main>'+entry,1).encode())
    receipt={'schema':'private_diagnostic_decisions_render_v1','status':'rendered_pending_visual_QA','source_sha256':sha(__file__),'plan_sha256':sha(a.plan),'prior_index_sha256':sha(backup),'prior_index_backup':backup.relative_to(gallery).as_posix(),'current_index_sha256':sha(index),'decision_page_sha256':sha(out/'index.html'),'source_summary_refs':refs,'prior_non_index_content_unchanged':198,'new_fit_or_inference_in_builder':False,'public_HTML_or_case_upload':False}
    dump(out/'private/render_receipt.json',receipt)
    files={x.relative_to(gallery).as_posix():sha(x) for x in gallery.rglob('*') if x.is_file() and x.name not in ['.gitignore','artifact_manifest_current.json']}
    dump(gallery/'artifact_manifest_current.json',{'schema':'private_gallery_current_manifest_v1','prior_manifest_sha256':plan['base_manifest_sha256'],'prior_index_backup':receipt['prior_index_backup'],'files':files,'historical_manifests_retained_not_claimed_current':True})
    public=a.public_draft;public.mkdir(parents=True,exist_ok=True)
    dump(public/'results/diagnostic_decisions.json',projected)
    for name,ref in plan['scientific_code'].items():
        assert sha(ref['path'])==ref['sha256'],name
        target=public/'src'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ref['path'],target)
    shutil.copy2(__file__,public/'src/bridge_diagnostic_decisions_page.py')
    dump(public/'docs/source_provenance.json',{'schema':'diagnostic_decisions_sources_v1','status':'prepared_not_published','parent_public_commit':plan['parent_public_commit'],'sources':[{'path':'src/'+name,'actual_execution_source_sha256':ref['sha256'],'archive_sha256':sha(public/'src'/name),'adaptation':'byte_for_byte'} for name,ref in plan['scientific_code'].items()]+[{'path':'src/bridge_diagnostic_decisions_page.py','actual_execution_source_sha256':sha(__file__),'archive_sha256':sha(__file__),'scope':'Local HTML rendering and explicit aggregate projection only; no numerical recomputation.'}],'summary_inputs':refs,'private_plan_not_distributed':True,'actual_data_files_not_distributed':True})
    print(json.dumps({'status':receipt['status'],'page_sha256':receipt['decision_page_sha256'],'index_sha256':receipt['current_index_sha256'],'files':len(files),'prior_non_index_unchanged':198}))


if __name__=='__main__':
    main()
