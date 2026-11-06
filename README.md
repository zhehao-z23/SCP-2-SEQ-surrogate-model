# SCP sequence surrogate: completed bridge stage, 2026-11-06

**2026-11-07 夜间阶段已完成。** 冻结表示的校准、24 个解析岭系数拟合及两臂小型非线性读出均已实际测试；稀有误差改善未转化为预算筛选提升，未晋级。旧标签的固定预算任务中，学习型 charge 仍是低成本强基线。完整新 Atlas 输入/新 head/面积标签路径已用 1 个合格开发对象在 CPU 跑通；这不等于新协议泛化或最终模型。

[夜间结果、实际成本与新 Atlas 交接](experiments/night_20261107/README.md) · [源码/匿名摘要归档映射](experiments/night_20261107/archive_manifest.json)。组会 13 页演示、病例、图册及预测只保留私有。以下为原 2026-11-06 桥接与诊断记录；其中后续探针的实际结果见上述新章节。


本轮四项模型诊断已收束：既有 plateau 仅证实有限改善；强形态与极端筛选仍有缺口；病例核验排除了部分简单错配，但主要因果未分离。固定路径 H² 对 CE/4 的两臂 134 轮实验实际完成，H²精确重放和配对检查通过，**CE/4未达推广门槛，保留H²，不追加第二折或事后调参**。CE在预算末端仍改善，不能由此宣称其全局无用。任务仍限已曝光开发数据，最终模型和新C迁移尚未完成。

[本轮四项验收、支持数与负结果](experiments/diagnostic_decisions_20261106/README.md) · [指标依据与解释边界](experiments/diagnostic_decisions_20261106/docs/metric_rationale.md) · [独立决策复核](experiments/diagnostic_decisions_20261106/results/loss_independent_review.json)。该章节保留19个已审归档文件的原始字节与内部目录；私有图册和病例数值不在公开仓库。


**本轮开发阶段已闭环：历史 Atlas 优化与形态诊断完成，1 个 AF 坐标对象通过自身 8 项数值敏感性检查和最终开发标签验收。最终模型、新 C 全量训练和全 sequence space 精度扫描仍未完成。**

冻结 ESMC300＋共享残基非线性后 mean＋电荷仍是整体分布 H 的主候选，连续 plateau 降 LR 值得保留；charge 和 matched mean 对照继续保留。18 次优化拟合均 early-stop，预测复算和 checkpoint 恢复通过。这些部分 fold/seed 的优化收益不是最终独立测试，也没有解决形态检索失败。

| residue_mean 配置 | 固定 LR 组宏 H↓ | 连续 plateau 组宏 H↓ |
|---|---:|---:|
| fold0 / seed1701 | 0.04910350 | 0.04804177 |
| fold1 / seed1701 | 0.04904011 | 0.04795070 |
| fold0 / seed3407 | 0.04915787 | 0.04815976 |

[优化报告](docs/bridge_optimization_report.md)保留比较限制与成本。18 份日志复核和两个实际 best 的统一 eval 补测显示：train–stop H 差距扩大，同时 stop/evaluation H 改善；不能仅凭差距判有害过拟合。evaluation 首末开放箱 MAE 略升，未作显著性推断，也不等同 ±5 筛选准确率。[同口径补测](experiments/bridge_20261106/diagnostic_supplement/README.md)记录 0 新 fit、49.77 秒 GPU 进程内测量及原预测精确重放。

已有 30-fit OOF 预测中，三模型×两 seed 对预先定义的 raw 绝对强 Bowley 偏态条件均为零真阳性；固定候选预算下仍有排名信号，但明显弱于真实 63 箱的声明式均匀箱内代理。8 组后续匹配优化比较全部改善 H，却没有同步改善所有形态检索。标签/QC 配对内部一致；保守分箱证书不足不证明信息必然丢失。6 组完全相同的实际输入对应不同目标已验证，但只涉及 12 行，不能解释主要全局失败或指认构象/数值噪声原因。[表示与形态诊断](experiments/bridge_20261106/label_diagnostics/README.md)给出支持数、native 分辨率限制与精简结果。

物理桥接的 **1 个 AF 开发对象**通过主完整性审计、独立末轴/采样/嵌套审计和不可变接受收据。94 次 PB 全部完成；进一步参考的 68 场按 37＋2＋29 三阶段完成，保留了元数据变化和实际 v45→v48 升级造成的两次保护停机。所有 8 项 H/W1/CDF sup 分别满足 0.02/0.1/0.01 上限。它们是注册的复合数值敏感性检查，不能解释成单因素因果、连续极限误差上界、实验绝对精度或全 AFDB 有效性。接受目标仍为 base 分辨率的面积加权分布，细化结果仅作参考。[最终物理报告](docs/physical_bridge/physical_final_report.md)、[匿名摘要](experiments/bridge_20261106/physical/results/summary.json)与[依赖边界](docs/physical_bridge/physical_reproduction.md)明确 1 对象范围、实际成本及尚未形成独立复现包的限制。

两对象、两组件继续登记为 **reserved_pending_final_closure**，保护目标和模型预测未读。既有关系搜索和 16 项守卫检查不等于最终独立外测：完整新 release 组图/profile 闭包未齐时，最终放行仍拒绝；旧角色与训练资格 gate 仍须满足。[保护报告](docs/bridge_roles_report.md)记录范围。

创新主张保持收窄：Deep Sets、MIL attention、ReZero 和序列到空间电荷 proxy 均有先例，包括 DeepSP/DeepSCM。本项目应以明确的结构/序列/物理标签合同、可靠转移与验证证据建立贡献，不宣称首次序列电荷预测或原创 pooling 架构。[研究核验](docs/round_20261105_research.md)保留依据。下一模型目标研究尚未在本提交中执行或完成。

## Reproduction and history

- [Bridge reproduction](docs/bridge_reproduction.md) and [earlier executed/archive provenance](docs/bridge_source_provenance.json).
- [Physical execution/archive source provenance](docs/physical_bridge/source_archive_provenance.json), [diagnostic supplement provenance](docs/diagnostic_supplement/source_provenance.json), and [label diagnostic provenance](docs/label_diagnostics/source_provenance.json).
- [Completed overnight stage](docs/overnight_20261106.md): 44 new fits, five-fold/two-seed development OOF, traditional baselines and AFDB engineering evidence.
- [52-fit Atlas stage](experiments/atlas_scale_20261105/README.md), [20-fit pooling stage](experiments/pooling_20261105/README.md), [original reproduction](docs/reproduction.md), [runtime requirements](requirements-stage.txt), and [contribution rules](CONTRIBUTING.md).

公开内容仅源码、配置、文本和匿名汇总。原始数据、受保护身份、逐样本预测、权重、完整日志及私有图册留在执行工作区；本次没有上传 PNG/SVG/HTML。可移植源码改动与实际执行 SHA 分开记录，未声称改写后的路径版本已重新运行科学计算。
