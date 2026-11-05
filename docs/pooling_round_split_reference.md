# 当前 Atlas 划分与 DefensePredictor：限定核验

2026-11-05。只回答会影响本轮池化比较的问题；未新做同源搜索、未改分组、未读取生产标签。本轮实际 Atlas 数据仍绑定 `atlas_scale_20261105/data/release.json`。

**结论：项目确实专门查过 DefensePredictor，但当前历史 Atlas 划分不是该方法的复现。现行实现适合在声明的 sequence/身份闭包下做共同开发比较；不能称为 profile 隔离、全面远同源隔离或独立最终泛化验证。本轮未发现需要重划或暂停池化实验的实质错误。**

## 参照了什么，实际做了什么

Byte `planning/20261013_next_stage/RUN.json` 明确登记了用户提出的“homology fivefold using DefensePredictor”；同目录 `literature/` 已保存原论文 XML、作者代码及获取 hash。因此不是仅凭模型名字追认引用。不过，正在使用的历史组件先在 2026-09-27 冻结，2026-10-12 扩展 full-U；没有证据表明后来已按作者 profile/Louvain 程序重建。30%/80% 数字相同也不能证明其历史来源。

| 部分 | DefensePredictor 原始代码 | 当前 Atlas 实际实现 |
|---|---|---|
| 找关系 | 三轮 MMseqs profile 构建后，对候选作 profile→序列搜索；split notebook读该结果，过滤 `qcov>0.8`、`fident>0.3` | MMseqs sequence 搜索，identity≥0.30、较短序列coverage≥0.80、E≤1e-3、sensitivity7.5；旧视图全对全及教师视图补搜，后续增加U双向搜索；未执行这条profile流程 |
| 形成组 | 正负类分别做 Louvain 社区；防御类再合并社区和可互换防御组关系；按组五折 | 所有接受边与已知身份关系做不可拆连通分量；不按标签拆类，不用Louvain切断已接受边 |
| 分配/早停 | 正负类分别GroupKFold；作者CV脚本从外层其余四折再用10组分割选一份validation，约72/8/20 | 当前Atlas保留既有五折ID，评价=i、早停=(i+1)%5、其余三折训练，约60/20/20；不是照搬作者内部早停方式 |

作者固定版本：[`30efca649e32c1e1039c19221dbee92e06441cc1`](https://github.com/PeterDeWeirdt/defense_predictor_manuscript/tree/30efca649e32c1e1039c19221dbee92e06441cc1)。直接核读了[profile构建](https://github.com/PeterDeWeirdt/defense_predictor_manuscript/blob/30efca649e32c1e1039c19221dbee92e06441cc1/code/05_03_01_run_mmseqs_profile.py)、[split notebook](https://github.com/PeterDeWeirdt/defense_predictor_manuscript/blob/30efca649e32c1e1039c19221dbee92e06441cc1/code/05_04_02_split_data.ipynb)相关代码单元和[CV脚本](https://github.com/PeterDeWeirdt/defense_predictor_manuscript/blob/30efca649e32c1e1039c19221dbee92e06441cc1/code/11_05_00_cross_validate_def_pred.py)。原论文主文也明确采用profile分组，并用HMM/BLAST/结构检索评估跨折预测能力；这些实测比较不能简化成“作者已证明任何同源都无泄漏”。[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13092281/)

阅读范围：论文为Byte已有Europe PMC fullTextXML及本次在线原始来源核对，网页直接打开有验证页；原作者split、profile和CV代码已读。本轮没有声称读完全部SI或重放作者完整数据集。

## 不只检查 group ID 的实际依据

1. 已读原 `data_contract_v1_20260927/scripts/freeze_contract_split.py`：它对原始、legacy CA、教师/构建体序列视图接入同一owner映射；合并历史组、PDB entity、可靠UniProt关系；逐条接受命中的边作union，随后按完整组件分配角色/折号。教师序列修订有双向补搜，不是只覆盖原CA序列。
2. 对应完成收据列出69,033,600条命中、45,537,157条接受边、150,749个不同视图；最大单query输出10,428，低于声明结果上限。首次失败的旧receipt仍在，但后继 `search_complete.json` 与教师补搜receipt均完成，不能把旧失败状态当最终状态。本轮复用原已绑定的搜索证据，没有重新读完约11.7GB原命中表。
3. 已读 `tools/cohort_finalize.py` 与 `qualification.accept_homology_hit`：继承全部旧组件，再加入所有已解析U候选/根身份及新增U搜索双向接受边。即使样本不合格，其关系仍参与保护闭包；输出绑定两个实际搜索文件hash。受保护与巨组角色保留，不能把当前development子集外推为全Atlas总体。
4. 对当前实际manifest做小型只读验证：每个旧组件对应的新组件数最大值为 **1**，没有旧组件被拆开；本轮 **21,921行** 与source manifest按sample ID连接后的group完全一致。已有当前T→all_sequence_groups映射和跨角色精确T审计也保持有效。

因此“组不交叉”在这里有搜索和身份闭包的实现依据，并非任意给样本不同group ID。但 sequence搜索的敏感度边界仍然存在；组件更保守与profile更敏感是两回事，不能总体宣称当前比DefensePredictor更严格。

## 对本轮研究的决定

- **继续沿用当前共同划分做池化开发比较。** 未找到接受关系被拆开、T遗漏旧搜索身份或旧组件被拆开的证据；不要为“照搬论文”而重做同一组数据的折号。所有新候选仍是曝光开发结果。
- **限制结论，不扩大审计。** 允许报告既有sequence/已知亲本关系控制下的增量；不报告profile/结构家族完全隔离，也不将缺失的最近训练T相似度写作0。
- **profile是后续确认条件，不是本轮训练前置。** 若入围模型要形成远同源/全序列部署主张，在明确新C/外部确认release时补齐其规定的profile关系和直接跨折挑战。不能把新C合同中的profile要求写成本次历史Atlas已经执行。若后续发现实质跨折已确认边，冻结原结果为旧split结果，在新版本组件上复查受影响候选，不能静默移动当前折号。

## 可追溯入口

Byte主根：`${PROJECT_ROOT}`。

- 历史引用：`planning/20261013_next_stage/RUN.json`、`literature/author_code_retrieval.json`、`literature/epmc_fulltext.xml`。
- 原闭包：`source/g3/seq2scp/data_contract_v1_20260927/contracts/seq2scp_data_v1.json`、`scripts/freeze_contract_split.py`、`validation/search_complete.json`、`teacher_search_supplement.json`、`split_validation.json`。
- 实际继承：`tools/cohort_finalize.py`、`package/src/seq2scp_training/qualification.py`、`data/homology/search_plan.json`、`data/qualification/release_v1/qualification_receipt.json`。
- 当前source manifest SHA256：`d108028ecc33c40af9402dbf841734ad7d1fe22f4d275f598ebc663f7525d59c`；freeze脚本：`6a900b17a4c27e9ccf3cb7dac4bccea31e99bd5701f85693ebc9f9950f25ca38`；cohort_finalize：`ba785a93aaf147976402f393cfe48fadb212b7a45760156476f09067f5c5f278`。

## 附：本轮池化实现的限定复核

只读检查了 `pooling_model.py` 与 `train_pooling.py`：mean_pre/max_pre/residue_mean均为1,736,255参数；共同fit-only仿射标准化、charge项与bias的广播、有效batch内microbatch梯度权重和组角色使用未见使比较失效的问题。mean与仿射变换可交换，而GELU前后聚合正是被检验因素。attention若启用会新增参数，应单列。建议开fit前将实际cache payload与publisher receipt作一次hash验收并绑定；不必每个fit重复读取大cache。此复核不替代实际训练、恢复及预测结果验证。
