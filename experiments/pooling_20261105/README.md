# Pooling stage — 20 fits completed

The registered 12-fit comparison and eight-fit amendment completed with 20/20 early-stopped fits and no failed fits. Stop-only selection produced 12 checkpoints. This is retrospective development on the same historical T→vertex-p release as the previous Atlas stage: 21,921 rows, 21,915 unique teacher-retained T sequences and 7,999 homology groups. The common evaluation set contains 8,756 rows, 8,753 unique T sequences and 3,199 groups across folds 0 and 1. Seed 3407 repeats the same evaluation observations; it does not enlarge that denominator.

## Candidate comparison

Group-macro Hellinger H is lower-is-better. Each seed-1701 mode/fold selects one of two learning rates using stop H. The second seed uses the inherited family/fold learning rate without rescanning. The registered development-score rule chose residue_mean for this replication.

| Readout | Trainable parameters | Seed 1701 H | Seed 3407 H |
|---|---:|---:|---:|
| Mean before nonlinear transform (`mean_pre`) | 1,736,255 | 0.050639128 | 0.050452125 |
| Max before nonlinear transform (`max_pre`) | 1,736,255 | 0.062559205 | Not registered |
| Shared residue transform then mean (`residue_mean`) | 1,736,255 | **0.049071796** | **0.049053467** |
| Learned residue attention (`residue_attention`) | 1,801,920 | 0.051826453 | Not selected |

The first three modes have exactly matched parameter counts. Attention adds 65,665 scoring parameters (+3.78%) and is not an exact-capacity control. All modes use the same frozen FP16 residue cache with FP32 operations and the same charge features, training-only normalization, split and readout optimization rules. The old stage's FP32 mean-fusion score is not an isolated pooling control because cache precision and batching also differ.

Residue_mean improves H over mean_pre in both folds and both seeds. Relative H reductions are 3.10% and 2.77%. Paired group ΔH is -0.001567332 (descriptive 95% interval [-0.001942276, -0.001208855]) for seed 1701, and -0.001398658 ([-0.001770499, -0.001028817]) for seed 3407. These conditional intervals exclude model-selection, multiplicity and seed uncertainty; two seeds do not fully characterize randomness.

| Seed | Readout | Group-macro CDF MSE | Group-macro first/last mass MAE |
|---|---|---:|---:|
| 1701 | mean_pre | 0.000197031 | 0.002008631 |
| 1701 | residue_mean | 0.000179103 | 0.001970185 |
| 3407 | mean_pre | 0.000200196 | 0.002005763 |
| 3407 | residue_mean | 0.000180602 | 0.001979043 |

CDF MSE averages 62 ordered internal cutpoints; it is not continuous Wasserstein distance. The label measure remains historical vertex probability, distinct from new C area-weighted labels.

## Predefined subgroup limits

The table gives residue_mean minus mean_pre H. Conditional groups can span strata and must not be added across rows.

| Subgroup | Rows / conditional groups | Seed 1701 ΔH [descriptive 95%] | Seed 3407 ΔH [descriptive 95%] |
|---|---:|---|---|
| Strict T=U | 453 / 286 | -0.001640 [-0.003067, -0.000234] | -0.001832 [-0.003369, -0.000392] |
| T length >600 aa | 724 / 297 | -0.001949 [-0.003547, -0.000258] | -0.002904 [-0.004563, -0.001241] |
| Strict T=U and >600 aa | 20 / 16 | +0.007268 [-0.002418, +0.017227] | -0.000096 [-0.007333, +0.007234] |

The small intersection remains uncertain, and its CDF MSE worsens in both seeds: 0.000285117→0.000488651 and 0.000475099→0.000619559. It does not support a full-length long-protein deployment claim. In the broader >600 aa stratum, seed-1701 tail MAE worsens from 0.003628053 to 0.003666959 (Δ +0.000038906, about 1.07%); its CDF MSE is also slightly worse. Thus the overall improvement does not mean every subgroup metric improves. Intervals spanning zero are not equivalence evidence.

## Reproducibility and recorded costs

All 20 prediction sets were independently recomputed with group/exact-T isolation checks. Fresh CPU loads reproduced 16 length-spanning evaluation rows for each selected checkpoint (12×16 checks); maximum absolute difference was 2.682209014892578e-7. This is a checkpoint restoration subset check, not fresh evaluation data. All fits reached early stopping within the registered budgets; early stopping is not proof of a global optimum.

The accepted cache contains 21,915 complete T sequences and 6,358,684 residues. New FP32 encoder means exactly matched the parent cache; the FP16-derived mean maximum quantization difference was 8.4043e-5. Mean and max readouts reduce the same quantized cache as the residue models.

| Recorded quantity | Value | Interpretation |
|---|---:|---|
| Sum of fit timers | 1,525.533 s | Excludes startup/cache loading; elapsed time, not GPU busy time |
| Sum of head queue durations | 1,737.801 s (0.483 h) | Includes startup/loading/cleanup; concurrent sums are not stage elapsed time |
| Head queue span | 1,633.404 s | Includes inter-queue analysis and selection gaps |
| Encoder launch-to-receipt sum | 482.369 s | Includes two workers, pilot, startup and hashing |
| Cache publication/verification | 64.447 s | Separate from encoder and head timing |
| Published cache / including work cache | 12.379 / 24.844 GB | Decimal GB; private arrays are not published |
| Selected mean_pre cached-readout throughput | 87,435–96,570 sequences/s | Short repeated cached-feature prediction pass |
| Selected residue_mean cached-readout throughput | 10,476–10,718 sequences/s | Same timing definition; full residue cache resident |
| Mean_pre / residue_mean peak GPU allocation | 0.149 / 12.638 GB | Residue allocation includes the resident full cache |

These throughput numbers exclude encoder inference, cache loading and deployment I/O; they are not full sequence-to-p throughput. Peak residue-head host RSS was 25.459 GB; publisher peak RSS was 25.609 GB. The registered head, encoder and cache storage caps were met. Full per-fit timing definitions, presentations, selected learning rates, environments and hashes are in [cost_public.json](cost_public.json).

## Practical reference against the historical charge-deep baseline

The saved predictions were also compared with the fixed seed-1701 charge-deep baseline from the previous stage, after checking identical release, labels and fold roles. This comparison has different parameter counts and effective batching/optimization paths; it cannot isolate pooling as the cause. Both new seeds reuse the same historical baseline predictions.

| Subgroup | Seed 1701 residue−charge ΔH [descriptive 95%] | Seed 3407 residue−charge ΔH [descriptive 95%] |
|---|---|---|
| All | -0.003115 [-0.003702, -0.002546] | -0.003133 [-0.003661, -0.002613] |
| Strict T=U | -0.001564 [-0.003396, +0.000206] | -0.000992 [-0.002744, +0.000761] |
| T length >600 aa | -0.000515 [-0.002915, +0.001983] | -0.001865 [-0.004102, +0.000511] |

Overall H improves by about 6% relative to charge-deep. However, both seeds' strict-T=U and >600 aa H intervals include zero, and subgroup CDF/tail directions are mixed. For the strict-U-and-long intersection, both H point differences and CDF/tail differences worsen versus charge-deep, with uncertain H intervals. The matched-mean subgroup gains therefore do not establish that residue_mean fixes the weakness relative to charge features. Keep the strong charge baseline. Full aggregate evidence and caveats are in [pooling_vs_charge_public.json](pooling_vs_charge_public.json).
## Decision / 阶段决定

Retain residue_mean as the stronger pooling candidate alongside the mean_pre reference and the strong charge baseline for future bounded comparison. Close max_pre and the tested attention branch for this stage. Residue_mean remains permutation-invariant; this result supports a nonlinear residue transform before mean pooling under the tested conditions, not sequence-order learning, a novel architecture claim, or final deployment qualification.

本阶段20次训练全部早停，残基共享非线性变换后再mean在两折、两seed均胜过先mean，代价是更高缓存、显存和读出成本。严格T=U与>600 aa的整体H改善，但“全长且长”的20条/16组仍不确定，且CDF两seed退化；不能用总分掩盖这些限制。与旧电荷深头的实用参考比较虽有全体约6%的H增益，但严格全长/长序列两seed区间均跨零，不能说已解决相对电荷基线的弱点，仍保留强电荷基线。本阶段关闭继续搜架构，不冻结最终模型，也不直接放行新C初始化或全U扫描。

The observed outcomes do not prove new-C initialization benefit, final five-fold generalization or full-U validity. The original amendment and timing clarification are both retained. See [reproduction notes](../../docs/pooling_reproduction.md), [initial plan](../../configs/pooling_plan.json), [amendment](../../configs/pooling_amendment.json), [timing note](../../configs/pooling_amendment_timing_note.json), [source provenance](../../docs/pooling_source_provenance.json), [split scope](../../docs/pooling_round_split_reference.md), and [bridge conditions](../../docs/pooling_round_bridge_status.md). The [parent bridge aggregate](bridge_subgroup_aggregate.json) is a post-hoc analysis of the previous stage, not a pooling result.
