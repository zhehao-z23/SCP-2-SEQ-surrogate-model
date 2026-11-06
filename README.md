# SCP sequence surrogate: bridge stage, 2026-11-06

**本阶段仍在进行：优化、匿名诊断和暂定保护登记已完成；AF新物理资格尚未在本次公开归档中验收。** 历史Atlas开发成绩不等于新C最终模型，也不授权全sequence space精度扫描。

18次新增优化拟合全部early-stop，独立预测复算及18个checkpoint恢复通过。冻结ESMC300＋共享残基非线性后mean＋电荷仍是主候选；下一阶段优先验证连续plateau降LR `3e-4 → 1e-4 → 3e-5`，mean_pre作匹配机制对照，并保留旧capacity runner更强的电荷基线。

| residue_mean配置 | 固定LR组宏H↓ | 连续plateau组宏H↓ |
|---|---:|---:|
| fold0 / seed1701 | 0.04910350 | 0.04804177 |
| fold1 / seed1701 | 0.04904011 | 0.04795070 |
| fold0 / seed3407 | 0.04915787 | 0.04815976 |

这只是部分fold/seed复制。与前夜低LR续训相比，增益较小、部分描述性区间跨0，而且尾误差有局部代价。新共同runner的charge结果没有取代旧强基线。训练进程总占用0.4793 GPU小时，恢复脚本122.94秒另列。[优化报告](docs/bridge_optimization_report.md)和[聚合结果](experiments/bridge_20261106/optimization/summary.json)保留比较口径、成本与限制。

[匿名诊断](docs/bridge_diagnostics_report.md)覆盖30个完整OOF配置与3个既有续训结果。H与长度相关很弱；尾箱误差与长度的关联部分来自真实尾质量的尺度，剩余关系仍须面对。真实目标尺度用于回顾诊断，不是可部署特征或因果解释。病例CDF轴已绑定历史切点，个例序列、真实/预测分布和私有图册均不公开。匿名图表已完成核验并保留在本地/服务器；具体图片上传被自动审批拒绝，本次公开提交仅包含代码、数值摘要和报告，不含PNG/SVG。

两对象、两组件已实际登记为 **reserved_pending_final_closure**，保护标签和模型预测未读。已完成小面板与187,939旧视图、133,315新C序列的双向关系搜索，并保留706,552别名与旧组件；另在原服务器完成143历史曝光ID、149序列视图的补查，无历史FASTA跨机复制。16项守卫检查通过。**这不是最终独立外测**：完整新release组图与profile闭包未齐时，最终放行仍拒绝；旧角色与训练资格gate也仍必须满足。[保护报告](docs/bridge_roles_report.md)和[匿名收据](experiments/bridge_20261106/roles/summary.json)给出边界。

创新主张继续收窄：Deep Sets、MIL attention、ReZero及序列到空间电荷proxy均有先例，包括DeepSP/DeepSCM。项目应以清楚定义的结构/序列/物理标签合同、可靠转移与验证证据建立贡献，不宣称首次从序列预测空间电荷或原创pooling架构。[研究核验记录](docs/round_20261105_research.md)包含本次增补。

## Reproduction and history

- [Bridge reproduction](docs/bridge_reproduction.md) and [executed/archive source provenance](docs/bridge_source_provenance.json).
- [Completed overnight stage](docs/overnight_20261106.md): 44 new fits, five-fold/two-seed development OOF, traditional baselines and AFDB engineering evidence.
- [Earlier 52-fit Atlas stage](experiments/atlas_scale_20261105/README.md), [20-fit pooling stage](experiments/pooling_20261105/README.md), [original source layout](docs/reproduction.md), [runtime requirements](requirements-stage.txt), and [contribution rules](CONTRIBUTING.md).

`src/` contains source, `configs/` portable plans, `docs/` scientific/reproduction notes, and `experiments/` anonymous aggregates. Raw data, private identities and roles, embeddings, predictions, weights and full logs stay in the execution workspace. Archived portable edits and actual executed source hashes are recorded separately. Publication of this partial stage does not mark the full bridge objective complete.
