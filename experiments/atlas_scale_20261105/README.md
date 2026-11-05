# Atlas retrospective scale stage — completed 2026-11-05

This stage completed 52 fits and selected 26 configurations using the stop split. It tests teacher-retained T sequence → historical vertex-count electrostatic probability p on 21,921 rows, 21,915 unique T sequences and 7,999 homology groups. All labels in this release are historical PDB vertex-p labels. Results are retrospective development evidence from evaluation folds 0 and 1, seed 1701; they are not final five-fold or new-C results.

## Learning curves

Group-macro Hellinger distance H (lower is better), over the same 3,199 evaluation groups. Training groups are nested; the largest scale has 4,799 or 4,800 groups per fold. Each original family/scale/fold has two learning-rate candidates, selected only using the stop split.

| Input / readout | 200 train groups | 1,000 train groups | All train groups |
|---|---:|---:|---:|
| Composition + length / small MLP | 0.065417 | 0.056807 | 0.052732 |
| Composition + charge / small MLP | 0.064247 | 0.055888 | 0.052633 |
| ESMC300 mean / deep MLP | 0.078511 | 0.064674 | 0.057264 |
| ESMC300 mean + charge / deep MLP | 0.067130 | 0.056935 | **0.050140** |
| Composition + charge / deep MLP control | — | — | 0.052187 |

Small heads use hidden widths [256, 128] and dropout 0.1; deep heads use [1024, 512, 256] and dropout 0.3. The extra four-fit charge control was registered after the original 48-fit results were exposed. It matches hidden widths, depth and dropout, not exact total parameter count (753,215 charge-deep versus 1,736,255 fusion).

At the largest scale, fusion improves H by 3.92% relative to charge-deep: paired ΔH = -0.002047125, descriptive group-bootstrap 95% interval [-0.002611766, -0.001497169]. Both evaluation folds improve. These intervals are conditional on selected fits and exclude model-selection, multiplicity and seed uncertainty. Fusion improves 4.74% relative to the smaller charge head. Mean-only remains worse than the charge baseline. Within this common T release, fusion's rank changes with training scale; comparison with the earlier 1,061 full-U cohort cannot isolate a pure scale effect.

| Largest-scale metric | Charge small | Charge deep | ESMC + charge |
|---|---:|---:|---:|
| Group-macro H | 0.052633268 | 0.052186794 | 0.050139669 |
| Cutpoint CDF MSE | 0.000202265 | 0.000198165 | 0.000192363 |
| First/last bin mass MAE | 0.002111526 | 0.002089049 | 0.001992132 |

CDF MSE averages 62 internal ordered cutpoints without coordinate weighting; it is not continuous Wasserstein distance. The historical labels use vertex measure and LPBE without explicit salt, distinct from the new area-weighted physical release.

## Verification and cost

All 52 saved prediction sets were independently recomputed. Of 52 fits, 48 reached early stopping and four reached the epoch ceiling; all 26 selected fits reached early stopping. Fresh CPU checkpoint loads reproduced 128 evaluation rows per selected fit, with maximum absolute difference 3.129243850708008e-7. This is a reproducibility subset check, not a new evaluation. The ESMC cache exactly matched all 1,061 legacy exact-sequence overlaps.

| Recorded cost | Value | Interpretation |
|---|---:|---|
| Sum of individual fit wall times | 592.8915 s | Not end-to-end elapsed time or measured GPU busy time |
| Encoder allocated GPU time, including pilot | 452.0090 GPU-s | Recorded allocated duration; separate from head fitting |
| Encoder peak allocated GPU memory | 1,583,277,568 bytes | 1.475 GiB |
| Encoder peak RSS | 2,315,931,648 bytes | Process host memory |
| Mean embedding cache | 84,153,728 bytes | 80.26 MiB; cache is not published |

Per-family queue times, per-fit head-only inference throughput, memory and checkpoint hashes are in [summary.json](summary.json). Head-only throughput excludes encoder extraction and is not sequence-to-p deployment throughput. Two launch repairs are retained: a missing encoder tools directory before extraction, and a capacity-control launch with Python lacking pandas before training. The latter used the original head runtime on retry; no algorithm changed.

[Stratum aggregates](strata/README.md) cover T/U relation, T length and inherited coverage. They are post-hoc conditional strata with overlapping groups, not independent causal subpopulations. For strict T=U and T length >600 aa, fusion versus charge-deep has no clear advantage (ΔH +0.0004563 and +0.0003969, respectively; both exploratory intervals include zero); this is not equivalence evidence. Other length bins show exploratory improvements. Existing pairwise search files cannot estimate nearest training-T identity; unavailable identity is not zero or evidence of absent remote homology.

## Decision and limitations / 决策与边界

Retain the stronger charge baseline and ESMC+charge fusion for limited follow-up; close this architecture-search stage. This experiment does not establish initialization benefit, new-C transfer permission, final five-fold performance, or full-U/whole-sequence-space deployment validity. A legal new-C initialization requires fold-specific historical-label exclusions against the new protected closure, followed by a random-versus-history initialization comparison on accepted new labels. See [stage decisions](../../docs/stage_decisions_20261105.md).

本阶段完成52次训练与独立结果复算，26个stop选定配置均正常早停。扩大同一T数据集后，融合模型在两折上超过电荷基线；同hidden/dropout的深头对照将相对优势收窄至3.92%，因此保留强电荷基线与融合候选，关闭本阶段继续搜架构。这里是历史T→顶点p、两折一个seed的曝光开发证据；不证明历史权重对新C有正迁移，不放行正式五折或全U扫描。编码、缓存、训练和推理成本须分别核算。

The archive includes aggregate metrics, scripts and artifact hashes only. Private sequences, per-sample/group identities, predictions, embeddings, checkpoints and full logs remain in the execution workspace. [Source provenance](../../docs/source_provenance.json) distinguishes the executed source from portable archive adaptations.
