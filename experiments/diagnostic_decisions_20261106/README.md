# Four decision questions: continuation after the completed bridge

This prepared archive follows commit `260d925bf59c30d3594d763faea17f6edf098325`. The read-only evidence and the two-arm loss replacement are complete. The H² control and all 134 paired random/batch paths passed; CE/4 failed the frozen promotion rule and no second fold or rescue tuning is performed. Private gallery pages and individual cases are excluded from public artifacts.

## 1. Does further training improve generalization?

The 18 original optimization logs reproduce all checkpoint/LR/early-stop decisions. All eventually early-stopped, with the selected best 30–38 epochs before the end. Descriptive final-window stop-H trends were 10 rising, 3 near-flat and 5 falling; the falling runs still failed the original minimum-improvement rule. Online dropout-on, parameter-updating training H² is not comparable to fixed-state group-macro stop H.

The two selected best states, measured in a common evaluation mode, improve train/stop/evaluation H and CDF. Evaluation open-edge MAE rises slightly. The wider train–stop gap does not alone establish harmful overfitting. This establishes a bounded past benefit, not a promise that adding epochs will continue to help, and not a recovered arbitrary-epoch trajectory.

## 2. Are special shapes and extreme screening usable?

Absolute strong-Bowley threshold detection remains inadequate. The training partition itself contains 38 strong negative and 11 strong positive T; fresh detects 1/0 and plateau 2/0. Stop and evaluation detect none under those absolute thresholds. Thus the failure is not limited to unseen groups. Fixed-budget ranking retains some signal but must be judged separately from threshold calls.

Exact-count ±5 mass bounds identify 14 certainly negative-concentrated and 135 certainly positive-concentrated T at mass >=0.8; another 22/374 T remain unresolved. These are identification bounds, not confidence intervals. Charge detects 11/10 known negative positives across the two seeds, versus residue 4/7. Residue detects 64/70 known positive positives but still misses 71/65, including deferred calls. Whole-distribution H alone cannot select the best screening method.

The historical source-population percentile requirements and old AF physical operator are unavailable here. Neither the full-support flag nor the conditional-window comparison is a recreation of that complete historical extreme-screening rule. Unknown truths and deferred predictions remain in the aggregate evidence.

## 3. Do checked explanations account for the main failures?

Label/QC/count linkage checks remove several simple mismatch explanations. Six exact model-input pairs with different targets establish a small-support ambiguity, but cannot explain the main global failure or identify structural/noise causes. Target/source/fold/coverage-matched length analysis retained only 8 pairs from 304 long and 1,907 short T, below the preregistered minimum 20 pairs. No matching rule was relaxed and no causal length conclusion is drawn.

Two previously frozen descriptive cases and two prespecified matched normal controls passed the requested identity checks. They establish representation-visible prediction failures without making within-bin reconstruction assumptions. Merely observing teacher/full-sequence differences or lower global embedding proximity does not suffice to explain those examples. The controls were selected partly for low H, so their H difference is not an independent statistical or causal test. Individual identities, target values, distributions, neighbors and private reports are excluded.

## 4. Has the highest-value next change been tested?

The declared next test fixes architecture, initialization, roles, batch/random paths, the existing 134-epoch learning-rate schedule and stop-H model selection, and changes H² to CE/4. It tests a finite loss replacement under an H²-derived workflow, not the best possible CE pipeline or a uniquely identified physical mechanism. The actual result is negative. Stop H improves 0.04758425 → 0.04748576, while evaluation H changes 0.04804177 → 0.04817614. Strong-shape union Bowley AE worsens on stop (0.190056 → 0.193994) and evaluation (0.161356 → 0.166242). The summed two-direction fixed-budget stop true positives fall 9 → 7; evaluation remains 11. Absolute strong-shape threshold TP remains zero on stop/evaluation in both arms. H² is retained.

Negative open-tail bias magnitude improves by 21.35% on stop, but its AE falls only about 3.4e-8. Evaluation bias magnitude improves by 13.59%, while its AE rises about 1.06%. These do not meet both-partition requirements. Clip fraction and CE probability-floor counts are zero. CE selects epoch 134 at the budget endpoint and its final-window stop H/CE still improve, so convergence or global CE inferiority is not established.

The original wording permits two readings of “shape improvement OR tail improvement, on both partitions.” The executed analyzer uses a per-partition OR; the conservative same-route reading requires the same improvement type on both partitions. Both are false here because both evaluation primary predicates are false and the stop ranking guardrail also fails. Original plan/code remain unchanged; future plans must freeze the Boolean predicate explicitly. The successful controlled intervention does not isolate an asymptotic Bayes-center cause.

The H² and CE processes took 264.924917 and 231.866754 seconds, respectively (about 0.138 process-hours on the assigned GPU, including loading and CPU work, not measured GPU-active accounting). No second fold, weight/LR rescue or threshold tuning follows this result.

See [metric rationale](docs/metric_rationale.md), [reproduction boundary](docs/reproduction.md), [aggregate evidence](results/diagnostic_decisions.json), and [actual/archive provenance](docs/source_provenance.json). The three principal views are whole-distribution H, fixed-cut CDF error and scientific task errors/budget recall. Current CDF MSE is neither raw CRPS nor W1.
