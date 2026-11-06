# 本轮调研决策证据：KaML-ESM 与 ESM3

核验日：2026-11-05。用途是改变下一轮 Atlas 实验选择，不是完整文献综述。未训练、未下载模型权重、未启动 API 付费请求。本文独立于实验结果，不能写成机制已经在本项目验证。

## 直接改变本轮实验的结论

1. **先扩大训练规模，再裁定 mean-pLM。** KaML 的电性预测使用目标残基的向量，而非全蛋白 mean；其成功支持 pLM 中可能有可利用电性信息，不能证明 mean 保留了本项目所需信息。先做共同固定验证集上的组成/电荷与 mean-pLM 学习曲线；若扩大数据后仍落后，下一项区分实验是同 encoder 的“非线性后再 pooling”，而不是立即换最大 encoder。
2. **将监督初始化作为独立实验轴。** KaML 有合成监督预训练→实验标签训练的先例，但其标签是残基 pKa。它支持尝试历史 Atlas 初始化，不证明旧顶点 p→新面积 p 必然正迁移；采用同结构任务头、同新数据量和预算的随机初始化对照，并按同源 fold 产生历史初始化。
3. **ESM3 保留为条件性结构输入消融。** 公共实现已能用同一 checkpoint 做序列输入与序列＋坐标输入。无需先生成全库结构或做全量 embedding。先让现有锚点比较稳定，随后小样本实测接口和成本；只有出现独立结构输入增益才考虑蒸馏。

## KaML：已核到的具体证据及边界

| 事项 | 本轮核验 | 对本项目的行动 |
|---|---|---|
| 输出与数据 | 最新论文摘要为逐残基 pKa；不是固定盐条件下的 SES 电势直方图。2026预印本正文段落描述29,457残基/9,945蛋白的合成监督预训练，并有实验标签、GAIN​​ES和挑战集 | 论文主张限定为表面分布及相应泛化/统计用途；不声称首次序列预测电性 |
| 输入/读出 | 固定SHA代码的`emb_extraction.ipynb`取ESM2-650M第31层的目标残基1280D向量；`esm2V1layerV1p2`为1280→512→256→32→1，未做全蛋白mean | 支持共享残基MLP后pooling作为后续机制对照；不直接复制其pKa头 |
| 划分 | 数据README提供50个随机10%holdout、EXP67S、OBTRUDE、时间外部集及CDHIT划分；检查到的split notebook以UniProt_ID＋uni_resid为组键 | 残基去重不等于蛋白/同源隔离；本项目保持同源组分割，避免拿不同验证难度的论文数字背书 |
| 代码可复用性 | 分析训练脚本含作者绝对路径和模板占位符；对应MLP在forward内临时创建Dropout | 借鉴机制和对照，不将公开分析脚本当可直接运行的可靠runner。临时Dropout不受外层eval自动管理；这里只裁定检查到的代码，不据此推翻论文全部结果 |
| 成本 | README给安装/推理环境要求；未取得可直接用于本项目的训练GPU小时/全库吞吐 | 编码器提取与小头训练分别测速，不以论文模型大小估算本项目墙钟时间 |

代码版本：`JanaShenLab/KaML-ESM@19bd1902819b0a124d8cb58bdc60f59605b912b3`。

本轮轻量CSV独立计数：`datasets/PKAD-3_V2.csv`为1,804数据行、153个不同`Uniprot_ID`字段值、307个PDB_ID字段值；`datasets/GAINES.csv`为4,220行、2,164个不同Uniprot_ID字段值。这是**当前仓库文件计数**，未重新解析实体身份/重复测量，不能等同独立蛋白数或某版论文的最终训练规模。旧版本正文出现1,167 pKa/247蛋白，不能与当前CSV无解释拼接。

关键原始来源：

- [最终论文摘要与发表信息](https://www.nature.com/articles/s41467-026-78406-y)：检索服务可读摘要；直接全文重定向失败，未声称读完最终版。
- [2026预印本正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13041848/)：检索缓存取得Results中模型、预训练、层比较和验证段落；直接网页出现验证页，SI PDF未成功取得。正文版本与最终版分开。
- [数据说明](https://github.com/JanaShenLab/KaML-ESM/blob/19bd1902819b0a124d8cb58bdc60f59605b912b3/datasets/README)、[划分代码](https://github.com/JanaShenLab/KaML-ESM/blob/19bd1902819b0a124d8cb58bdc60f59605b912b3/analysis/dataset_split.ipynb)：已读相关代码单元，未独立重放划分。
- [残基输入提取](https://github.com/JanaShenLab/KaML-ESM/blob/19bd1902819b0a124d8cb58bdc60f59605b912b3/analysis/emb_extraction.ipynb)、[模型类](https://github.com/JanaShenLab/KaML-ESM/blob/19bd1902819b0a124d8cb58bdc60f59605b912b3/analysis/model_class.py)、[训练分析脚本](https://github.com/JanaShenLab/KaML-ESM/blob/19bd1902819b0a124d8cb58bdc60f59605b912b3/analysis/train_model.py)：已读相应实现；未运行或核对全部发布权重对应关系。

## 三张最小决策卡

| 要决定什么 | 最强替代解释 | 最小区分实验 | 继续/暂挂条件 |
|---|---|---|---|
| 小数据导致pLM排名低吗 | 优化不足、输入对象不匹配，而非样本量 | 同一合法Atlas标签/输入/组划分；固定stop/评价集，嵌套训练组规模；组成、电荷及mean-pLM均有必要LR机会 | 观察误差斜率、交叉排名、分层误差及实际更新量；学习曲线未收敛只标未充分评价 |
| mean丢了有用信息吗 | 参数量不同或训练容量不同 | 入围encoder上，mean→MLP 对比逐残基共享MLP→masked mean；共同训练池、指标、相近容量并记额外成本 | 有信号再补attention/顺序读出；无信号只约束该encoder与预算，不否定所有pLM |
| 历史监督初始化值得吗 | 只是多训练、或泄漏了新验证对象 | 新开发面板上同任务模型随机头 vs 按fold历史预训练；旧输出定义不同则重置输出头 | 同新数据量/预算同时比较H、CDF/尾部和达到同误差的时间；识别负迁移分层后决定扩大 |

这些实验共同受曝光登记约束。KaML结果不规定本项目的5%阈值、最终模型数或永久fit上限。

## ESM3：可用接口、机制和成本

原始代码固定为 `Biohub/esm@8f0834cbc350aeaf507f755475873f7c61fa44a1`，不是建议自动升级现有Byte环境。

公共本地checkpoint入口为`esm3-sm-open-v1`；官方文档将small列为约1.4B，公开构造函数为48层、1536隐藏维、24 attention heads。当前Hugging Face页面从EvolutionaryScale重定向至biohub；实际运行必须绑定下载到的revision和权重hash。公共接口已核；Byte已完成下述限定目录盘点，但**未验实际加载**。参考：[官方ESM3说明](https://github.com/Biohub/esm/blob/main/_assets/ESM3_README.md)、[模型卡](https://huggingface.co/biohub/esm3-sm-open-v1)、[固定版本构造函数](https://github.com/Biohub/esm/blob/8f0834cbc350aeaf507f755475873f7c61fa44a1/esm/pretrained.py)。

源码证实`encode(ESMProtein)`将序列和可选坐标等转为输入，`logits(..., LogitsConfig(return_embeddings=True))`能返回表征。坐标经结构encoder生成structure tokens；forward同时接收structure tokens与坐标几何，缺失结构采用相应mask/default。只做表征提取不需要调用迭代`generate()`。参考：[固定版本ESM3实现](https://github.com/Biohub/esm/blob/8f0834cbc350aeaf507f755475873f7c61fa44a1/esm/models/esm3.py)、[官方调用示例](https://github.com/Biohub/esm/blob/8f0834cbc350aeaf507f755475873f7c61fa44a1/cookbook/snippets/esm3.py)。

条件消融设计：

- A：纯序列，其他track清空；同一数据/小头与现有入围序列表征对照。
- B：同一ESM3 checkpoint、同一批蛋白、同一小头，加既有结构坐标。实验结构和预测结构分开列；前者明确为结构辅助对照。除坐标所必需的结构tokens外，不额外喂真实功能、SASA或二级结构标签。
- C：只有B稳定有益后再设计结构教师→序列学生的蒸馏，教师也只在对应训练fold监督拟合。

“去掉coordinates”不一定足以构成纯序列：复用已经encode的tensor时还需清空structure tokens及其他非序列track；最稳妥是分别从干净的ESMProtein对象构造。比较时保留残基映射、缺失mask、链边界与实际结构来源。任意别的encoder对比不能隔离出“结构预训练”的因果作用。

成本估算仅作pilot预算：单层FP16残基缓存为`2×1536×总残基数`字节，mean池化FP32缓存为`4×1536×蛋白数`字节。例如1,000条、平均300残基：一视图约0.922 GB残基缓存，mean约6.14 MB；两个输入视图约加倍，均未含文件/索引开销。约1.4B参数的BF16纯权重下界约2.8 GB，**不是峰值显存预测**；结构encoder、attention、临时输出及软件开销须实测。

建议触发后先固定16条覆盖长度区间的输入做加载/前向测速；按实际峰值与吞吐把范围扩到约200条配对开发样本，保存冻结表征，沿用已校准小头。样本数是建议pilot值，不是永久限制。不做全库缓存、不启动生成或新付费API。若加载资格、映射或资源不满足，只暂挂ESM3分支；Atlas学习曲线继续。

## 核验关闭状态

已足够支持本轮学习曲线、读出机制与初始化对照设计，应转入实验。未解决的最终版SI、完整KaML发布训练复现、ESM3本地加载与吞吐，只限制相应强主张或pilot启动，不构成等待Atlas主线的理由。

### Byte本地ESM3可用性补核（2026-11-05）

只读盘点了已知`${MODEL_ROOT}`下3层，以及`${MODEL_ROOT}/huggingface`下3层（含模型revision目录）；没有全盘搜索、凭证读取、下载或模型加载。该范围列出`arcinstitute/evo2_1b_base`、`Synthyra/ESMplusplus_small`、`Synthyra/ESMplusplus_large`、`biohub/ESMC-6B`及`biohub/ESMFold2`，**未见`esm3-sm-open-v1`模型目录或对应manifest**。因此ESM3当前应登记为“已核官方接口；已知本地模型根未找到，需获取或由用户提供其他确切路径”，不能登记为“本地exists/可立即执行”。这不是全机不存在的证明。补齐checkpoint身份、必要tokenizer/结构encoder配套资产和固定环境后，才进入16条加载/前向pilot。

## 2026-11-06 补充：序列到空间电荷描述符的近邻先例

本次在AF物理计算等待期间做定向补核，不重新展开模型搜索，也不把这次检索称为穷尽式创新审查。

[DeepSP原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140563/)使用20,530条抗体Fv序列，预测MD平均的SCM/SAP共30项区域描述符；SCM来自暴露残基附近的局部电荷和，不是PB电势概率分布。正文方法与结果说明输入为IMGT对齐、one-hot后的重轻链序列，模型为CNN。[作者仓库](https://github.com/Lailabcode/DeepSP)也提供仅序列推理入口。[DeepSCM作者仓库及论文入口](https://github.com/Lailabcode/DeepSCM)证实2022年已有序列到SCM的代理模型。已核DeepSP主文相关方法/结果及两个README；没有重放其训练、读取完整SI或声称本项目优于其指标。

由这些证据作出的项目判断：论文不应声称“首次从序列预测空间电荷性质”“首次以代理模型替代结构计算”，也不能仅以扩大到全库作为新意。当前可检验差异是：面向一般蛋白、指定化学与盐条件下的SES面积加权电势分布、连续量/尾部的可追溯数值资格，以及低同源评估后的适用域。差异存在不等于贡献已被证明，仍需新标签和独立科学用途证据。

[Deep Sets原论文](https://arxiv.org/abs/1703.06114)已有逐元素变换后聚合范式；当前残基共享非线性后mean不承担通用架构首创主张。冻结pLM残基向量带上下文，所以本实验只检验读出位置，不能据此说序列顺序没有作用。

[2026年结构/力场DNN论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13080996/)使用结构、拓扑与电性特征预测Coulomb/溶剂化能，是另一类物理代理先例。该条本次取得原始页面检索正文，直接ACS全文访问受限，未完整审阅SI。它不能作为本项目同任务直接性能基线，但进一步限制“首次电性代理模型”的措辞。

收束动作：保留现有共享残基mean、mean_pre与强电荷头；不因这些先例新增CNN或大encoder扫描。后续只保留两类有具体问题的实验：新定义标签上确认当前三者排序；在合法同源闭包下比较随机初始化与历史监督初始化。若新面积标签也显示明确尾部偏差，再登记一个有限的尾部校准/复合loss对照。每项必须先固定数据/标签、训练与stop角色、主要指标及成本，再查看结果。
