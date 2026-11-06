# 新 Atlas 可执行交接：开发接口已验证，最终训练资格仍关闭

2026-11-07。已将唯一拥有新协议正式接受收据的 AF 开发标签，接入真实序列 → 冻结 ESMC → residue-mean＋charge → 新面积输出头 → 梯度更新 → 保存重载的完整 CPU 路径。随机初始化与历史隐藏层初始化各执行 8 步；没有用这一例比较迁移优劣、选择初始化、估计泛化或提前训练最终模型。

## 当前可以消费什么

| 资产 | 当前用途与限制 |
|---|---|
| 历史 Atlas：21,921 行、21,915 个精确 T、7,999 组 | 已曝光开发数据；历史 vertex 分布和原训练结果保留，不能直接充当新面积目标 |
| 新协议 accepted AF：1 例 | 固定 AF 坐标、化学/SES/NPBE 合同下已接受的 development-only 标签；本次真实消费对象 |
| 新生产全量 | 仅读取当前 runtime/release/固定槽位身份，没有读取或接纳任何生产样本标签；生产进度不等于可用于训练的冻结 release |
| 两个暂保护对象 | 不读取标签或模型预测；仍等待最终新 release 的全组件/profile 闭包，不能称最终独立外测已完成 |

本次 accepted 收据 SHA `aaf0517369b9be9f981dc8914efadbcacdd0df01aa7e2b060880b618e40a2e41` 绑定自身 8 轴、主完整性审计和独审；消费入口不把旧 `numerical_qualified=false` 字段改写成 true，也不因向量文件存在就消费它。原化学配置文件中的历史 candidate 状态不单独授予资格，资格来自后继接受证据。该资格不表示天然结构实验真值、所有 AFDB 或长序列都合格。

新 C/Atlas 是一个数据 release 概念；模型使用哪个序列视图必须单独声明。本次实际 prepared observed 序列、实体序列和已冻结 AF 序列相等。未来部分坐标、多片段或 T≠U 对象，不能仅按 accession 换成完整 U 输入后沿用原物理目标。当前演示适配器对此类未声明视图不自动放行。

## 保留与适配

| 项目 | 交接决定 |
|---|---|
| 工程候选 | 保留冻结 ESMC300 最后第 30 层、960 维残基表示；共享逐残基线性/GELU后求均值，拼接 39 项 composition/charge 与 39 个缺失标志；隐藏层 1024/512/256，dropout 0.3。保留 mean-pre 和 charge-deep 作为公平比较基线，不因本 smoke 改架构结论 |
| ESMC 缓存 | 只能按精确输入序列、模型/后端 SHA、层、token 排除、不截断、精度与量化约定复用。AF64 与历史 Atlas 的已绑定索引都没有本例精确命中；本次新做 1 次 CPU FP32 提取，保存 FP16 残基缓存 |
| Charge 特征 | 39 项顺序与旧源码不变；这是序列 pKa/自由端点代理，不是结构质子化或 PQR 真电荷。多片段输入须保留代理解释，不能虚构天然自由链。缺失标志保持，imputer/scaler 每个新 fit 分区独立重拟合 |
| 标签与输出 | 旧合同是历史 LPBE、无显式盐、vertex measure，有限边界约 −44.62 到 108.05。新合同为 NPBE、100 mM、area measure，有限边界 −20.0342930165395 到 24.757910640129058 kBT/e；都是 63 维并不表示同一任务。新末层必须重置，不将旧 p 插值或直接与新 p 比精度 |
| 优化与指标 | 当前保留 H²，后续相同 AdamW/plateau 预算做初始化比较；此前 CE 有限替换未通过晋级。本次不选学习率/损失。新 H、实际 cut CDF 与明确筛选任务互补；不能把旧开放尾误差当成新 tail/±5 命中率 |

61 个有限中心箱及两个开放尾按新固定边界使用，edge tie 归右侧箱，不截断电势。未重新估计边界、未读取 raw phi、未追加 PB。方法资格的 8 轴是注册复合数值敏感性检查，不是连续极限误差上界或结构来源因果实验。

## 初始化的公平条件已写成配置

两臂使用相同新 fit/stop/evaluation 角色、同一新输出头初始化、同一新 fit-only scalers、相同 batch 顺序和 dropout RNG 路径、同一优化器初始空状态及预算/stop 判据。历史臂不复制原 output、optimizer 或 RNG。

历史源总计 1,736,255 参数：1,720,064 个隐藏参数来自旧模型，16,191 个输出参数重新初始化。第一层 1,063,936 参数依据新 scaler 坐标作代数重标定；其后 524,800 与 131,328 个隐藏参数原值复制。新的标准化输入两臂完全相同。连续列变换为 `W_new = W_old × std_new / std_old`；bias 加旧 W 乘均值坐标差；missing flag 另加旧 W 乘 `(median_old − median_new) / std_old`，避免 imputer 改变时破坏旧函数。它是初始化来源适配，不是新架构。

合成混合缺失输入三种基线的第一层最大误差 ≤2.63×10⁻⁶；本例真实输入检查最大差 5.96×10⁻⁶（FP32 舍入）。新输出层始终保留；不同输出维数 17 的合成检查与 padding 检查通过。

历史权重曾监督使用 13,162 T/4,800 组，stop 使用 4,389 T/1,600 组；另有 4,364 T/1,599 组用于已曝光开发评估，而且模型架构/训练策略也据全开发证据选择。这些不能只按“没进梯度训练”就当未曝光。旧 train、stop、model-selection/evaluation 暴露都必须进入新 release 的排除和闭包审计；箱边校准与数值资格面板也须登记。当前历史权重 **仅限 development**。

后续五折采用冻结新同源组件：每轮 3 fold fit / 1 stop / 1 evaluation，两个初始化共享角色。最低 40 epoch、最多 300；LR 3e−4 → 1e−4 → 3e−5、stage plateau 15、最终 stage 最少 20 epoch、best 改善阈值 group-H 1e−5、stop patience 30、epoch0 可选。相同最大预算和规则不要求两臂实际恰好同 epoch；实际工作量与 epoch0 outcome 都要报告。这是预设方案，尚未执行全量比较。新 group/profile 闭包或 source/view 改变时旧角色配置失效，不能套用历史 fold 编号。

当前可执行 gate 对所有 `final_release` 请求直接拒绝，即使请求自称 closure ready；需要另行审核并绑定的新 release issuer 才能取得最终资格。7 类负例（旧 measure、equation、bins、视图、保护角色、未接受状态及 final_release）均实际拒绝。这一开发入口不替代原 release 的 protected/calibration gate。

## 已经实际完成的验证

| 步骤 | 实际结果 |
|---|---|
| Role/receipt/sequence/target 验证 | 在 target、checkpoint 读取前检查保护守卫；准确绑定 accepted 源和 corrected_area_p；1 例、63 箱、无 raw phi |
| CPU 编码 | 6.472 s 进程内总时长，其中 encoder forward 1.402 s；峰值 RSS 2,138,972,160 B；不使用 GPU |
| 完整模型 smoke | 两臂各 8 更新、总 16；均有限且非零梯度、合法概率，保存重载 maxdiff=0；不设置 stop/test 分区 |
| 数据准备与 smoke 时长 | 2.252 s 与 4.735 s；三阶段累计 13.459 s，不含 inventory/合成/交接复核和 shell startup，也不是 CPU 活跃时间计费 |
| 资源隔离 | CPU 176/177 单物理核、torch/BLAS 1 线程；每阶段 600 s/8 GiB 上限，实际均在内；当前 v48 固定科学槽位及活 worker 不交，但宽 affinity 控制进程可能共享，非 OS 独占 |

实际唯一执行适配器 SHA：`f160b22af2ddf01e599401ef3b5eb4a35c2d905b214c48ac5f70b08fc34bdba7`。本次真实有两条 N1 smoke 优化轨迹，不能描述成“零更新”；没有新泛化实验。N1 loss/target/prediction/weights 与历史暴露成员全部留私有目录，公开只保留接口、资源、计数与不含样本明细的验收结果。

## 数据到达后的入口与收束点

1. 先冻结 label release、精确 prepared/实体视图、化学/面积/箱边合同和接受证据，再完成旧监督与校准暴露加入后的全组件/profile 闭包。保护标签不能提前读取以挑数据或策略。
2. 对确定的新角色生成精确序列键缓存，拟合各 fold 的新 scalers；旧缓存只按内容合同复用。当前 N1 零方差回退 scaler 不得带入全量训练。
3. 按已写同预算方案跑 scratch vs 合资格 historical-hidden 两臂；如历史曝光不能满足排除，正式独立比较只采用 scratch，历史初始化仍作为开发参考。评价顺序为 stop 选点、evaluation 报告，不以 evaluation 调参。
4. 只有完整 release 上的重复折/种子、实际任务筛选、成本和失败边界通过后才决定最终模型与全 sequence-space 扫描；本 N1 已完成的是工程接线，无初始化胜负。

本轮不修改生产、旧公开仓库或旧 README。代码与模板只是可执行开发交接，不自称已打包私有生产、encoder 或完整模型依赖。
