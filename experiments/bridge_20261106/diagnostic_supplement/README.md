# Diagnostic supplement: existing optimization logs and fixed best states

This archive records completed development measurements. No new fit was run for the supplement. It is separate from the physical-qualification archive and is recorded separately.

The 18 existing fit logs reproduce the original checkpoint selection, learning-rate transitions and early-stop rules. All fits stopped after 30–38 epochs without a new selected best. Online training H² and stop group-macro H differ in model state, reduction and units; their visual separation is not a generalization gap.

Two saved residue-mean checkpoints (fold 0, seed 1701; fresh epoch 68 and plateau epoch 104) were separately evaluated with fixed parameters, dropout off and the same group-macro reduction:

| Partition | Fresh H | Plateau H | Fresh CDF MSE | Plateau CDF MSE | Fresh open-edge MAE | Plateau open-edge MAE |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.03788067 | 0.03257511 | 7.30184074e-05 | 4.3949852e-05 | 0.001538802635 | 0.001344971648 |
| stop | 0.04851890 | 0.04758425 | 0.00017211827 | 0.000161299273 | 0.001915744904 | 0.001900989963 |
| evaluation | 0.04910350 | 0.04804177 | 0.000183222498 | 0.000171543383 | 0.002021698605 | 0.002023031852 |

Train, stop and evaluation H/CDF improve, while evaluation open-edge MAE rises by about 1.33e-6. This is a descriptive change; no significance claim is made. A wider fixed-state train–stop H gap alone does not establish harmful overfitting when stop/evaluation error improves. These two selected states do not reconstruct the missing epoch trajectory or form a final independent test.

“Open-edge MAE” concerns the first and last open histogram bins. It is not neg5/pos5 screening accuracy. The labels remain historical teacher-sequence/vertex-measure distributions; the result does not qualify new-C area targets or full-sequence-space deployment.

The measurement took 49.77 seconds inside the GPU process, including loading/checks and CPU reductions, with zero optimization updates. Both evaluation probability replays matched saved predictions exactly; stop H differed from the original receipts by less than 4e-9. Exact support counts, H², resource definitions and source bindings are in [summary.json](results/summary.json) and [provenance](../../../docs/diagnostic_supplement/source_provenance.json).

中文结论：18份日志的状态机复核通过；两个实际保存best的统一eval补齐了有限的同口径train/stop比较。H差距扩大与stop/evaluation改善同时发生，不能单独判为有害过拟合。首末开放bin质量误差微升，不称显著，也不等同筛选阈值指标。后续机制研究尚未由本补充完成。

The public archive contains source, this explanation and small aggregates only. Private HTML, PNG/SVG, logs, member tables, probabilities and weights are excluded. The programs still produce private detailed outputs and require the separately controlled inputs described in [reproduction.md](../../../docs/diagnostic_supplement/reproduction.md).
