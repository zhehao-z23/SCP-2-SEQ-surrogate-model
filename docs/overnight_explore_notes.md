# Attention探索：机制与判定合同

2026-11-06；本轮只在历史T/vertex-p、fold0、seed1701开发。代码与8-fit配置分别为`overnight_explore_model.py`、`overnight_explore_train.py`及`overnight_explore_plan.json`。不改旧训练、缓存或生产；不触碰GPU0/1/6，不启动AFDB。

原实现的注意力是`tanh`两层打分、softmax后加权求和，属于[Ilse等，ICML 2018](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)§2.4式7–8的基本形式。本文已读相应原文公式；借鉴的是聚合机制，不能将注意力权重当因果解释或本任务的表面积权重。现有随机注意力在fold0较residue-mean差，可能是聚合权重与特征/读出联合优化扰动，也可能是该目标确实更需要总体统计，不能只凭旧成绩裁定。

[ReZero，UAI 2021](https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf)式1使用零初始化的可学习残差系数，让新分支起始不扰动原路径。本轮借用这一优化思想，采用`mean+tanh(alpha)*(attention-mean)`，alpha=0，score保持随机。不是原论文完整架构复现；不能同时把score和alpha都置零，否则该差分支路起点无梯度。

四个辨识性对照，各两个LR：

1. uniform_scratch：同主体、从头训练，score最后一层置零，检验均匀初始化本身；LR与旧attention相同。
2. mean_continue：同fold、仅按stop选择的既有residue-mean checkpoint，重启AdamW继续训练，控制额外训练与优化器重启。
3. uniform_warm：从相同checkpoint开始，用零score加入注意力，起始预测等于均值模型。
4. gated_warm：相同checkpoint，随机score与零残差门控，让注意力逐渐加入。门控为有符号tanh系数，整体不是正权重凸组合，解释应限于表征残差。

后三项LR共同为1e-4/3e-4，新增训练预算相同，epoch0 checkpoint允许保留为最佳；每个初始化仅用train2/3/4与stop1，eval0从未参与梯度或checkpoint选择。它仍是已曝光开发验证。CPU已检查起始均值等价、padding不变性和非零score/gate梯度；主体1,736,255参数，attention1,801,920、gate版1,801,921，不以更大主体冒充机制。

最多8fit，每fit训练900秒/300epoch、进程1020秒、两GPU并发、总4GPUh、无自动重试。固定原数据、norm、batch、损失与stop规则，缓存只读。诊断记录stop/评价中的归一化attention entropy、有效残基比例`1/(L*sum(a²))`与最大权重；它们是按行均值的描述统计，与group-macro误差分别解释。

判断顺序：先按stop选每模式LR，再共同评价并独立复算；uniform与原随机attention比较初始化效应，warm版本必须同时与mean_continue比较。若warm提升只等于mean_continue，不能称注意力有效。若优于continue且CDF/尾质量没有明显代价，向主线程提出独立fold/seed复制候选；不在本轮无限追参，也不因单fold区间就宣布最终模型。
