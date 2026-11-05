# 夜间attention机制探索与续训复制

2026-11-06；本阶段已经结束。实际完成8次探索fit和2次复制fit，全部early-stop，零训练失败、零自动重试。结论是：**优先保留残基均值读出，并将较低学习率续训作为优化锚点；不因本轮微小差异晋级attention。** 本结论只适用于已曝光的历史T→顶点p开发任务，不授予新C迁移或最终模型资格。

## 原文、机制、实现和训练结果

现有attention使用两层`tanh`打分后softmax加权聚合，对应[Ilse等，ICML 2018](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)§2.4式7–8。阅读这一原始公式并核对现有代码后，本轮检验“随机聚合权重扰动了特征/读出的早期优化”，将score最后一层置零，从严格均匀权重开始，而不扩大主体网络。

[ReZero，UAI 2021](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf)式1的零初始化残差系数启发第二个对照：`mean+tanh(alpha)*(attention-mean)`，alpha=0且score保持随机。这让初始预测等于已训练均值模型；若同时清零score和alpha会形成死支路，CPU梯度检查排除了这个错误。这里只借用优化机制，不声称复现ReZero整套架构，更不将attention权重解释为物理表面积或因果贡献。

所有warm版本从同fold、同seed、仅按stop选择的原residue-mean checkpoint开始，另设`mean_continue`控制更多训练及AdamW重启。主体参数1,736,255；attention共1,801,920；门控版1,801,921。每模式两个LR，按stop选LR，未以evaluation挑checkpoint。

| fold0 / seed1701配置 | 最佳stop H | 评价group-macro H | 对照解释 |
|---|---:|---:|---|
| 旧随机attention | 0.051595 | 0.052573 | 已有结果，未重复训练 |
| uniform attention从头 | 0.050744 | 0.051483 | 均匀初始化改善原attention，仍落后均值 |
| 原residue-mean | 0.048519 | 0.049104 | 原合法初始化 |
| mean_continue，LR=1e-4 | 0.047697 | 0.048257 | 新的匹配续训锚点 |
| uniform attention warm，LR=1e-4 | 0.047661 | 0.048209 | 相对续训锚点只降低H约0.00004798 |
| gated attention warm，LR=1e-4 | 0.047695 | 0.048255 | 相对续训锚点只降低H约0.00000148 |

Uniform warm相对matched continuation：stop差-0.00003534，evaluation差-0.00004798，描述性配对组95%区间[-0.00008233,-0.00001413]；相对误差改善约0.10%。这个区间不含0，但效应很小、只有一个开发fold/seed且未计选型不确定性，不能据此宣布实用优势。其最终attention entropy为0.99896、有效残基比例为0.98881，几乎仍是均值。

Gated warm相对matched continuation：stop差约-0.00000103，evaluation差-0.00000148，描述性区间[-0.00000665,0.00000349]，最终有符号门控系数为-0.01451，残差贡献很小。该门控是表征残差，不是凸组合的概率解释。

旧attention的128条stop长度分位样本CPU诊断显示entropy约0.805、有效残基比例约0.244、约16.9%的`tanh`前激活绝对值超过3。这些数值提示学习了较集中的聚合，但只作描述；本轮没有证明“集中本身导致误差”。Uniform从头改善而仍落后、warm权重重新接近均值，共同说明初始化能解释部分问题，而当前目标下attention尚无足以支付复杂性的增量。

## 晋级后的两次实际复制

在8fit复算完成后登记了两次额外fit，固定mean_continue、LR=1e-4，分别使用对应fold/seed的原residue-mean stop最优checkpoint。没有追加attention搜索。

| fold / seed | 原H → 续训H | ΔH | 描述性配对组95%区间 |
|---|---|---:|---|
| 0 / 1701（探索） | 0.04910350 → 0.04825696 | -0.00084654 | [-0.00117945,-0.00048944] |
| 1 / 1701（复制） | 0.04904011 → 0.04799599 | -0.00104412 | [-0.00135944,-0.00072904] |
| 0 / 3407（复制） | 0.04915787 → 0.04831378 | -0.00084410 | [-0.00117121,-0.00052596] |

新增两配置的stop、CDF和首末bin质量误差也改善；三配置H相对下降约1.72%、2.13%、1.72%。这是跨fold和seed的方向一致性，不是完整2fold×2seed试验，也不是独立盲测。建议主线程将“在合法已选checkpoint上以1e-4重启AdamW续训，epoch0允许保留”为后续公平优化候选；对其他架构比较时也应给匹配优化机会，不能直接把额外续训模型与未续训模型的差全部归因于架构。

## 数值核验与身份

- 10fit全部从保存预测独立复算H；验证预测形状、有限值、非负与和误差<1e-6，checkpoint及预测SHA匹配。
- 每fit的train/stop/evaluation同源组与精确序列无交叉；每个warmstart与当前任务的角色表、fold/seed、data/cache身份和五项scaler数组完全一致。没有跨fold复用监督权重。
- 10fit的配置hash等于run identity，训练/模型源码hash及计划hash匹配，checkpoint内config与receipt一致；公开provenance记录相应SHA。
- 初始模型检查通过均值预测等价、padding不变、非零score/gate梯度。6个选定checkpoint在新进程中重新加载模型和保存scaler、从缓存重放完整评价预测；原batch边界下最大误差为0。
- 第一版恢复脚本采用不同microbatch边界时，在首个checkpoint观察到最大FP32差3.13e-7，超过当时严格阈值。恢复原边界后通过；未改模型或训练产物。这个量级应作为batch浮点敏感性记录，不解释成科学收益或模型错误。

## 成本与阶段关闭

10fit内计时合计840.73秒；按队列记录的训练进程占用合计1,091.48 GPU秒，即0.3032 GPU小时（包含初始化/缓存装载等进程时间；两卡并发）。共46,865 optimizer updates。峰值GPU allocated约12.64 GB、峰值RSS约25.50 GB；两者是单进程观测最大值。仅使用GPU2/3，全部进程与恢复作业已退出。

三份唯一warmstart来源run的既有fit时间合计440.95秒，属于此前成本；不包含此前选LR的完整搜索成本。此次不重新提取encoder。6checkpoint恢复的已装载缓存后前向计时3.15秒；首次改变batch边界的恢复尝试未单独记录精确GPU墙钟时间，因此不把0.3032小时包装成包含全部验证/历史训练/编码的总服务成本。

阶段收束：保留mean_continue作为下一轮强优化锚点；attention新增收益暂不晋级；不再增加fit。原文改变了初始化/残差对照，实际训练又把优先级转回低LR均值续训，已完成“原文/代码→假设→实现→训练→判断”的闭环。这些普通读出和优化组合不是本项目创新主张；新C标签/保护闭包、最终五折、科学贡献及全sequence-space部署仍各自需要证据。

公开证据只含汇总及身份：`overnight_explore_public.json`、`overnight_explore_public_provenance.json`与本报告。源码与计划可按归档映射附带；逐样本预测、序列/角色清单和诊断样本ID保留在私有执行目录，不入公开仓库。
