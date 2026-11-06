# 优化流程确认：18 fit阶段结论

2026-11-06。18次新训练（12初筛＋6复制）全部完成、early-stop，零训练失败和自动重试。**encoder读出下一阶段优先采用连续plateau下降3e-4→1e-4→3e-5；电荷基线保留旧capacity实现的更强实测结果，mean_pre继续作为对照，不再把重置AdamW当必要步骤。** 这是已曝光历史T→顶点p开发域的工程选择，不是新C结果或最终模型冻结。

## 比较如何构成

固定三家庭：ESMC300残基共享非线性后均值＋电荷、ESMC300先均值后MLP＋电荷（mean_pre）、组成/电荷deep MLP。前两者各1,736,255参数；电荷头753,215参数。hidden均为[1024,512,256]、dropout0.3，电荷输入维不同，因此不宣称三者总参数相同。

fold0/seed1701先各跑四流程：fresh常数3e-4；连续plateau；同一fresh最佳模型/RNG分支后重置AdamW以1e-4续训；同一best状态保留AdamW矩和step以1e-4续训。后两者追加完整fit预算，整体训练时长与单次fresh/plateau不等价。旧checkpoint未存optimizer，不能假装恢复；本轮新做三个phase1源fit保存完整状态，原三个夜间mean_continue仅复用结果。

连续plateau预先固定：每阶段stop H在15epoch内未改善超过1e-5，则下降一次LR，不回滚模型、AdamW或RNG；最终阶段至少20epoch后，且全局最佳30epoch未改善才早停，整体min40/max300epoch、单fit900秒。所有选择依据stop，不以评价指标挑checkpoint。

## 初筛：连续下降在本轮共同runner内均按stop胜出

下表为同一评价集的group-macro H，括号为该流程完整fit时间；reset/retain时间包括源fresh fit，故同一源会在比较表中重复列出，资源总数只计一次。

| 家庭 | 常数fresh | best-state reset | best-state retain | 连续plateau |
|---|---:|---:|---:|---:|
| residue_mean＋电荷 | 0.049104（146.0s） | 0.048349（254.8s） | 0.048343（257.0s） | **0.048042（202.8s）** |
| mean_pre＋电荷 | 0.050862（44.0s） | 0.050077（65.7s） | 0.049991（65.4s） | **0.049708（54.8s）** |
| charge_deep | 0.053062（54.7s） | 0.052682（73.5s） | 0.052665（73.7s） | **0.052563（60.0s）** |

对应plateau的stop H依次为0.047584、0.049170、0.052002，均低于同家庭其他流程；它还比两种best-state续训的完整成本更低。相对fresh，初筛配对组ΔH分别为-0.001062、-0.001153、-0.000500，描述性区间均支持改善。

reset/retain采用同一源模型及RNG，是固定分支条件下更窄的对照。保留AdamW相对重置的ΔH分别为-0.00000605、-0.00008558、-0.00001670；可见增益较小且非所有主指标一致，不值得把复杂的best-state恢复作为默认优化流程。该观察不推出“AdamW状态永远不重要”，也不把连续plateau与续训的全部差异说成单因素因果。

三家庭在首次降LR前的83、51、71个epoch中，fresh与plateau的训练损失/stop H逐值相同。三个plateau最佳checkpoint都来自3e-5阶段，最佳epoch分别104、82、90；第三阶段确实参与了最优结果。

## 旧强电荷基线仍须保留

上述charge_deep只说明在**新共同pooling runner内部**plateau优于其fresh参照，不能称超越旧capacity runner。已读取旧4fit，按各fold的stop在两个LR中选定实际旧最佳（两折均3e-4），重新计算同角色、同scaler下的指标：

| seed1701 | 旧capacity charge H | 新共同runner plateau H | 新−旧ΔH / 描述性95%区间 |
|---|---:|---:|---|
| fold0 | **0.05235366** | 0.05256278 | +0.00020911 / [-0.00012879,+0.00055433] |
| fold1 | **0.05202003** | 0.05233937 | +0.00031934 / [+0.00002486,+0.00061192] |

新charge plateau的CDF和首末bin质量MAE在两折也比旧capacity结果略高，因此旧强基线不降级、不以新fresh较差的分数替代。

两runner的隐藏层、dropout、输入、AdamW/weight decay及每次更新的global-norm clipping=1.0相同；CPU核验seed1701下初始参数张量也完全一致。不同点是：旧版对全训练行直接随机排列、连续128行一次forward/backward；新版先随机1024块、按长度排序成128，再打乱batch，并执行16384-token限制的microbatch梯度累积（向量头也沿用）。因此numpy随机数消耗、dropout随机路径及梯度累积边界不同，**同seed不等于同训练轨迹**。旧版600秒、新版900秒上限也不同，但本次各fit均early-stop。跨runner对比只能作为保留历史强基线的实际结果比较，不能归因为单独的LR效果或不同clip阈值。

## 有限复制与原结果复用

按stop与成本晋级plateau后，登记6fit：residue_mean和mean_pre各复制fold1/seed1701、fold0/seed3407；charge_deep在fold1/1701新做fresh/plateau一对。对应旧常数3e-4结果经过同角色、同scaler核验并重新计算预测指标，没有重跑旧夜间续训。

| 家庭 / fold / seed | fresh H → plateau H | ΔH |
|---|---|---:|
| residue_mean / 0 / 1701 | 0.04910350 → 0.04804177 | -0.00106173 |
| residue_mean / 1 / 1701 | 0.04904011 → 0.04795070 | -0.00108940 |
| residue_mean / 0 / 3407 | 0.04915787 → 0.04815976 | -0.00099811 |
| mean_pre / 0 / 1701 | 0.05086163 → 0.04970840 | -0.00115323 |
| mean_pre / 1 / 1701 | 0.05041676 → 0.04955751 | -0.00085925 |
| mean_pre / 0 / 3407 | 0.05045040 → 0.05034288 | -0.00010752 |
| charge_deep / 0 / 1701 | 0.05306243 → 0.05256278 | -0.00049965 |
| charge_deep / 1 / 1701 | 0.05245446 → 0.05233937 | -0.00011509 |

residue_mean三配置均约降低H 2.0%–2.2%，配对组描述性区间均支持改善。mean_pre的seed3407及charge的fold1增益较小，区间跨0；不能称各家庭已获得同等稳定增量。各组区间未调整选型、多重比较或seed不确定性；这是部分fold/seed复制，不是完整factorial或正式五折。

与既有三个夜间低LR重置续训结果相比，plateau的H只进一步降低约0.000215、0.000045、0.000154，其中两个区间跨0。并非所有附加指标改善：相对夜间续训，三个配置的首末bin质量MAE分别增加约4.10e-6、1.07e-5、6.80e-6；fold1的CDF MSE增加约1.34e-6。相对fresh，mean_pre部分复制的CDF也略升。因而选择plateau主要依据其可执行性、stop和整体成本，不宣称它全面支配旧续训或已满足尚未确定的尾部用途容忍度。

同一流程下residue_mean仍优于mean_pre与电荷基线，支持继续保留非线性后pooling主线；最终架构仍需新C任务和完整验证决定。

## 数值和科学数据复核

18fit逐份核验了预测概率合法性、独立H/CDF/尾误差复算、模型/预测SHA、config identity、代码与计划hash。train/stop/evaluation的同源组和精确序列不交叉；warm分支与源模型角色表、五项scaler、fold/seed、release/cache身份一致。

每个best checkpoint的AdamW步数与其最佳模型epoch相符；retain分支的初始moments/step digest与真正源best状态一致，reset为空状态。CPU检验中，保存并恢复模型/AdamW/RNG、改变LR后下一步参数及optimizer状态逐位一致。没有把最佳模型与最终epoch优化器错配。

全部18checkpoint在新进程从保存scaler和冻结缓存重放完整评价预测，最大绝对误差均为0（验收容差atol1e-6、rtol1e-5）。已固定的数据、缓存、模型和代码身份不因这些数值检查而自动获得新C科学资格。

## 成本、关闭与下一步

当前18fit内计时合计1,572.75秒；队列记录的训练进程占用1,725.51 GPU秒，即**0.4793 GPU小时**，低于3 GPU小时预算。新进程恢复脚本计时合计122.94秒，另列；它包含脚本内缓存装载，不含进程启动/import。峰值单进程GPU allocated约12.64 GB、RSS约25.53 GB。未重提encoder、未修改旧训练或生产；只用GPU0/1，任务已退出。

本阶段停止新增fit，交付可复用的连续plateau实现、mean_pre对照及未被替代的旧capacity强电荷基线。新C训练时可把plateau作为待验证的encoder读出默认，不直接复用本阶段监督权重；旧/新保护闭包和标签测度的资格仍须单独通过。误差诊断应同时展示H、CDF和尾质量的局部代价，不能只按平均H展示“胜出”。

公开入口：`bridge_optimization_public.json`、`bridge_optimization_public_provenance.json`及本报告，只有aggregate、run identity和源码hash；逐样本预测与角色表留在私有运行目录。训练、模型、队列、独立复算/恢复实现及两份预登记计划可随公开路径映射归档。
