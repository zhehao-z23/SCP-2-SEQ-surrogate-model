# 模型研究：2026-11-06夜间结果

**已完成44次新增拟合，以及旧结果复用、预测重算、checkpoint回放和64条AF来源序列的实际推理。** Byte权威根：`${PROJECT_ROOT}/overnight_20261106`。

主确认：21,921行、21,915个teacher序列、7,999组；三家模型五折两seed，共30配置，其中10个复用、20个新增，全部早停完成。各seed独立聚合，不合并为更多蛋白；这里是历史Atlas已曝光开发OOF。

| 模型 | seed1701组宏H↓ | seed3407组宏H↓ |
|---|---:|---:|
| 共享残基非线性后mean＋电荷 | **0.049038** | **0.049027** |
| 先mean再MLP＋电荷 | 0.050601 | 0.050433 |
| 电荷深MLP | 0.052344 | 0.052358 |
| HistGBDT | 0.054149 | 未追加 |
| ExtraTrees | 0.054794 | 未追加 |

共享残基mean在10/10个折×seed对比中均胜过mean与电荷深头。相对mean总体H改善3.09%/2.79%，相对电荷深头改善6.32%/6.36%；总体CDF与尾误差也改善。**主架构方向因此更明确。**树模型实际做了14fit及两家完整五折，保留GBDT作论文参照，停止本轮扩搜。

新的有效改进是**降低LR到1e−4、重启AdamW继续训练原残基mean**：fold0/1701、fold1/1701、fold0/3407的H分别到0.048257、0.047996、0.048314，三配置H/CDF/尾误差均改善。尚未完整五折两seed，不能与上表OOF混排为最终第一。

attention做了均匀初始化、暖启动和零残差门控等8个辨识实验。暖启动最好的H仅比匹配的mean继续训练低0.000048，注意力接近均匀；主要收益来自额外低LR优化。暂不扩大attention搜索，也未否定所有attention。共享读出、MIL attention及ReZero均有文献先例，不作为原创架构主张。

**全空间扫描尚未通过的关键项：**严格T=U子群有1,164行/767组，其H优势比旧两折更明确；这只表示当前序列hash相等，不等同天然全长物理资格。>600 aa相对电荷头仍有seed/指标差异，T=U且>600仅48行/42组，H不稳、CDF/尾误差仍有劣化。新C物理目标、保护组与用途误差资格也没有被历史成绩替代。

64条既有AF候选序列已实际全长编码和双头推理，冷进程共7.77秒，独立回放最大概率差1.64×10⁻⁷。它们来自有偏候选池；旧240箱标签不直接兼容当前63箱，暂停新标签数值资格未通过，**本轮没有AF精度结论**。

下一步优先补齐继续训练的确认，并给必要强基线同等低LR优化机会，再用合格新C标签比较随机初始化与历史暖启动。未来保护组须从旧监督和stop选择阶段隔离；先迁移架构/配置，不能把旧权重直接称为合格最终初始化。随后补足长全长及跨来源物理挑战证据；不重开全部架构笛卡尔积。

详细结果与原始证据留在Byte各分支；公开代码与无样本明细摘要见[本轮Git分支](https://github.com/zhehao-z23/SCP-2-SEQ-surrogate-model/tree/codex/overnight-20261106)，总报告为`docs/overnight_20261106.md`。源码、配置和结果各有SHA；公开模板若改动环境路径，会区分归档SHA与实际执行SHA。

## Repository evidence and reproduction

- [Full overnight report](docs/overnight_20261106.md), [reproduction instructions](docs/overnight_reproduction.md), and [execution/archive provenance](docs/overnight_source_provenance.json).
- [Five-fold/two-seed confirmation](experiments/overnight_20261106/confirmation/summary.json), [mechanism exploration](experiments/overnight_20261106/explore/summary_public.json), [traditional baselines](experiments/overnight_20261106/traditional/summary.json), and [AFDB engineering diagnostic](experiments/overnight_20261106/afdb/summary_public.json).
- Earlier [52-fit Atlas scale stage](experiments/atlas_scale_20261105/README.md) and [20-fit pooling stage](experiments/pooling_20261105/README.md) remain unchanged historical evidence.
- [Original data/module layout](docs/reproduction.md), [pooling reproduction](docs/pooling_reproduction.md), [head runtime requirements](requirements-stage.txt), and [contribution rules](CONTRIBUTING.md).

`src/` holds source, `configs/` holds plans/templates, `docs/` holds scientific and reproduction notes, and `experiments/` holds aggregate evidence. Raw datasets, private sample/sequence identities, embeddings, predictions, checkpoints and full logs remain in the execution workspace. Portable settings must be materialized on the target machine; do not substitute public archive hashes for the recorded original execution hashes.
