# 偏态数值更准，为什么没有找回更多极端对象？

本轮保留现有 H² 分布模型。冻结表示上的线性及小型非线性偏态读出均未达到预先固定的筛选晋级标准，因此停止这条分支，不继续调整头宽、阈值或稀有样本权重来追逐当前开发折。

## 问题与最小检验

此前整体 H 改善而强偏态绝对阈值仍无真阳性。这里区分三件事：分数幅度是否受到压缩、原始形态监督是否比 63 箱代理提供更多可用信息、冻结表示中的这些信息是否能够改善极端排序。探针只改变读出；原始 PMF、输入序列、物理计算对象和标签协议均不变。

先从固定 H² 模型的 checkpoint 104 提取最后 256 维隐藏层，逐项复现既有预测，三个分区最大差均为 0。训练/stop/开发评价分别为 13,162/4,389/4,364 个唯一 T，4,800/1,600/1,599 个同源组，强偏态只有 49/13/16 个 T。stop 曾用于选择主模型，现在又选择探针；这些都是已暴露开发数据，不是最终独立测试。

1. **幅度控制：**只在训练集拟合正斜率仿射修正。斜率 1.14808、截距 0.00265；三分区全体对象的次序逐项完全保持。
2. **信息访问：**电荷、平均 ESMC 加电荷、最后隐藏层、完整预测 PMF 四种冻结输入，各做原始 Bowley 与真实 63 箱均匀内插 Bowley 两种监督。岭回归只比较三个固定正则系数，总计 24 个系数解；按 stop 原始 Bowley 组宏 MSE 选择。配对的分箱监督复用相同系数和标准化，明确承认这一选择方式偏向原始任务。
3. **有限容量检验：**隐藏层线性探针降低了稀有误差但损害排序，因此只追加一个 256→128→1 小型 GELU 头，原始与分箱监督两臂。初始化、200 个 epoch 的批次和 dropout RNG 配对一致；公共训练集标准化、学习率日程及优化预算相同。只按 stop 原始组宏 MSE 选 epoch；两臂训练结束后才计算开发评价。没有搜索其他宽度、dropout、学习率或阈值。

训练阶段使用等 T 权重的在线标准化目标 MSE。验证使用原始 Bowley 的组宏 MSE，两者不是同一统计口径，不能相减解释泛化差距。曲线分别呈现真实观测值；验证轨迹无需由历史训练日志反推。整个 PMF 不变，因此 H、CDF 等分布指标也不变，不能把标量探针改善包装成分布恢复。

## 结果及决定

| 方法 | stop 强偏态组 MAE | 开发评价强偏态组 MAE | stop 固定预算 TP | 开发评价固定预算 TP |
|---|---:|---:|---:|---:|
| 原 PMF 导出偏态 | 0.1901 | 0.1614 | 8负+1正=9 | 8负+3正=11 |
| 训练集仿射修正 | 0.1688 | 0.1355 | 8+1=9 | 8+3=11 |
| 隐藏层线性，原始监督 | 0.1627 | 0.1311 | 6+1=7 | 6+3=9 |
| 隐藏层非线性，原始监督 | 0.1529 | 0.1187 | 6+1=7 | 7+3=10 |
| 隐藏层非线性，分箱监督 | 0.1733 | 0.1452 | 7+1=8 | 7+2=9 |

预算是正、负两个方向**分别取 1%**，stop 与开发评价各为每方向 44 个 T，不是两个方向合计 44。真值为原始 IQR≥1 且 Bowley≤−0.30 / ≥+0.30，只是历史筛选规则的绝对条件部分。每个 T 的排序分数使用预先固定的单个视图；真值保留任一原始视图满足规则的口径，不平均抹平形态。

非线性原始监督将稀有 MAE 降低约 19.6%/26.4%，但预算 TP 从 9/11 降为 7/10。绝对阈值在两分区仍然 0 TP；stop 新增 1 FP，评价无正调用。其 stop 总体组 MSE 从 0.00343687 降到 0.00332425，评价为 0.00369491（原模型 0.00369485）。这说明更低的验证平均误差仍不足以满足科学筛选目标。

两臂均在第 3 个 epoch 达到最佳 stop MSE；在后续降低学习率并完成 200 epoch 后，训练目标继续下降，stop 指标变差。证据支持在这个冻结读出问题上出现早期过拟合；不支持继续训练可以改善泛化，也不代表主分布模型的所有配置都同样过拟合。保存及重载预测最大差为 0。

晋级条件在实验前已确定：两个开发分区稀有组 MAE 均至少下降 10%，总体组 MSE 均不恶化超过 5%，方向预算 TP 合计两分区均不下降且至少一处分区增加 2。没有候选满足。分箱监督是控制臂，本来就不作为独立晋级候选；不靠事后修改标准晋级。

## 失败机制：哪些解释已经区分，哪些尚未区分

| 解释 | 本轮可检验的证据 | 当前判断 |
|---|---|---|
| 仅仅是输出尺度太小 | 正仿射修正降低误差，完整排序严格不变，绝对阈值仍未恢复 | 幅度压缩解释部分数值误差，无法单独解释或解决漏检 |
| 63 箱监督与原始形态目标存在差异 | 相同隐藏输入及控制条件下，原始监督降低稀有误差；更低总体 MSE 的分箱臂却有更差稀有误差 | 监督定义影响误差分配；尚不能推断全部重要信息被分箱删除 |
| 当前隐藏表示完全没有形态信号 | 线性和非线性读出明显优于训练均值参照；但筛选劣于 PMF 导出分数 | 有可访问的数值信号；没有证据表明指定读出能够增加极端检索信息 |
| 只需增加标量头容量或继续训练 | 固定小型非线性头降低稀有误差，预算检索仍下降；epoch 3 后验证退化 | 本次有界容量及优化机会没有解决问题；停止继续扩头 |
| 稀有训练支持不足导致条件均值收缩 | 稀有训练支持仅 49 T；预测倾向中间值与这一解释兼容 | 仍是解释候选，不是因果证明；需要新的独立稀有支持或保持评价固定的数据干预才能区分 |

整体误差、数值校正和极端发现之间的脱节是本项目的实证现象，不能称为新的通用理论。探针失败也不能证明序列或表示没有信息；未考察更强读出、联合训练或其他原始目标，并不等于它们必然有效。

## 方法依据与创新边界

[Alain 与 Bengio，Understanding intermediate layers using linear classifier probes](https://arxiv.org/pdf/1610.01644) 的引言及第 3 节用于界定冻结探针的解释范围。本文是分类表示诊断，本项目采用加权连续回归，不能直接迁移其任务结果。

[Menon 等，Predicting accurate probabilities with a ranking loss](https://arxiv.org/pdf/1206.4661) 第 1–3 节支持区分数值估计和排序问题。其二元概率校准理论不是连续 Bowley 回归的保证；本轮仿射排序保持来自正斜率数学性质与实际逐项核验。

线性探针、仿射修正和小型 MLP 均非新方法。本项目的潜在研究贡献是：在明确物理分布任务上，区分表示限制、数值校正与发现效用，用配对监督、真实筛选和成本证据收束模型选择。是否达到可发表程度仍取决于新 Atlas 的协议一致评价、独立稀有形态支持及跨分布检验。

## 成本、证据与下一步

冻结特征提取 26.89 秒；全部岭解过程 8.03 秒；两臂非线性过程 36.80 秒，拟合各 15.87 秒、33,025 个参数、200 epoch，峰值新增 GPU 张量约 43.8 MB。这里是实测过程持续时间，不等于 CPU/GPU 活跃计费。没有重训主模型、启动新 PB 或扫描全库。

详细证据位于私有执行工作区的 mechanism 根：`features_v1`、`linear_probes_v1`、`nonlinear_probe_v1`。公开汇总和图源由 `night_20261107_mechanism_export.py` 从固定哈希的实测摘要自动生成；个人对象、序列、预测及权重均保留私有。公开 `results/summary.json` 与 `results/curves.json` 是可核查伴随材料；两张科学图仅保留私有。

本轮下一项实际验证转向**任务与成本匹配**：由原有强电荷基线承担正电势集中候选排序，配合完整未确定真值界和精确实现的推理实测，而不是假设更昂贵的 ESMC 一定更有用。新 Atlas 已完成合法标签接入及从头/历史隐藏层初始化工程比较；其性能价值需待合格新样本扩大后判断。主分布候选、强电荷基线和受保护数据关系继续保留，不将本折探针权重带入最终独立初始化。


## Public archive / execution boundary

The original execution sources, plans and private receipts remain unchanged. `source_archive_provenance.json` separates each original SHA from its public archive SHA. Three source adapters require `SCP_DATA_ROOT`; extraction also requires `SCP_H2_REFERENCE_RUN`. Configure `${GPU}` and the existing CPU/resource policy before any authorized rerun. Public plan templates were not executed and do not authorize protected-data access. The historical independent-review source is an execution snapshot requiring its original colocated source/plan files, not a new public-template validation.

The aggregate exporter is executable without private data: `python src/night_20261107_mechanism_export.py --root results`. It validates the public path-only extraction projection while preserving the original execution SHA in the output, and writes into `results/night_20261107_mechanism/`. The actual archive-only export check reproduced both aggregate summary and curves exactly, with no model, sample, GPU or fitting access. See `aggregate_export_validation.json`.
