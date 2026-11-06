# Completed bounded development studies — 2026-11-07

This stage continues the completed bridge and decision-diagnostics archive. It does not declare a final model, new-Atlas generalization, or a completed full-sequence-space scan. Private biological cases, distributions, figures, meeting slides, predictions and weights remain in the execution workspace.

## Model mechanism: better scalar values did not improve the screening decision

The backbone and its probability distribution were frozen. A training-only affine transformation, four feature sources with raw/binned targets and 24 ridge coefficient fits, followed by one preregistered two-arm nonlinear-head family, were actually executed. The nonlinear partners shared initialization and all 200 batch/dropout paths; each selected epoch 3 by stop error, and checkpoint replay was exact.

| Raw strong-shape readout | Stop Bowley MAE | Evaluation Bowley MAE | Stop / evaluation budget TP |
|---|---:|---:|---:|
| Original PMF-derived score | 0.19006 | 0.16136 | 9 / 11 |
| Positive affine transformation | 0.16877 | 0.13551 | 9 / 11 |
| Frozen hidden + ridge | 0.16273 | 0.13112 | 7 / 9 |
| Frozen hidden + small raw-target MLP | 0.15290 | 0.11869 | 7 / 10 |

The affine transform preserved every original ranking. The nonlinear readout improved rare-value errors but reduced budget hits and failed the frozen promotion rule. No threshold/head rescue follows. These scalar probes leave the PMF unchanged; their failure does not prove that the representation lacks information or establish a physical cause. Supports are only 13 stop T and 16 evaluation T in one exposed development fold/seed. See [mechanism evidence and stopping decision](mechanism/README.md).

## Biological use: distinguish concentration thresholds from review budgets

Historical canonical-T OOF screening contains 135 certified positive-concentrated targets and another 374 whose status is unidentified within the original bins. At a fixed 220/21,915 budget, learned charge hits 124/121 certified positives across two seeds, versus residue-mean 118/117 and simple KR−DE density 46. Unknown targets are retained, and these counts are not full-truth precision/recall or RNA-function accuracy.

The explicit RNA-annotation subset contains 3,094 T, with budget 310 and 109 certified positive-concentrated targets. Its gains support narrowing a later physical-review set, not identifying functional binding or a structural causal mechanism. Individual examples remain private. See [aggregate task evidence](biology/night_20261107_biology_public_report.md).

## Measured cost and conditional scale

On the same 95 length-stratified inputs and historical-frame weights, warm singleton ESMC+residue inference measured 18.831 ms/sequence; the exact charge route measured 0.569 ms, and cached-charge head inference 0.148 ms. The approximately 33.1 route ratio is not matched-accuracy speedup. A million sequences at the historical length mix gives conditional process-time scenarios of 5.231 hours and 0.158 hours respectively, excluding unmeasured sustained input/output and recovery effects.

Current production's 9,935 timed successes average 305.96 CPU seconds and 434.93 worker-wall seconds per object. Recorded failures total 0.636 CPU hours. The one-object AF qualification required 94 PB tasks and 7.89 APBS task-hours; it is a separate method-development cost, not routine per-object cost or measured CPU accounting. For one million inputs with 1% selected, assumed selected costs of 1×/2×/4× the production mean give 849.90/1,699.80/3,399.59 physical CPU hours. CPU and GPU process-time quantities remain separate. Historical LPBE/vertex screening and NPBE/area production pricing are different protocols; this conditional cascade has not established new-label recall or realized wall-time savings. See [cost definitions and limits](cost/REPORT.md).

## New Atlas: a real development input path is executable

An accepted AF development label completed the full prepared-sequence → frozen ESMC residue embedding → charge/residue-mean model → new head → new area-weighted target path on CPU. Random and historical-hidden initialization each completed eight updates with finite gradients and exact checkpoint reload. The three stages took 13.46 process-seconds in total. N=1 is solely an interface/contract check; loss changes do not select initialization or estimate transfer accuracy.

Old vertex-LPBE and new area-NPBE labels differ in physical protocol, measure and bin boundaries despite both having 63 outputs. The output head is reset; historical hidden-layer transfer explicitly rebases scaler coordinates. Historical supervised/stop/model-selection exposure remains development-only. Future independent release admission fails closed until the required group/profile closure and all role gates are complete. Protected labels and predictions were not read. See [actual handoff and fair-comparison boundary](atlas/README.md).

## Reproduction and archive scope

Each subdirectory preserves an explicit reviewed source/aggregate package. [Archive mapping](archive_manifest.json) binds candidate and repository bytes. Portable adaptations are identified separately from actual execution hashes. Private upstream data, caches, checkpoints and production dependencies are not supplied; this is an evidence archive rather than an unrestricted runnable full-data release. No PNG, SVG, HTML, PPTX, sample membership, sequence, prediction array or weight is public in this stage.

The retained engineering candidate is the H² residue-mean distribution model, with learned charge and simple sequence-charge controls kept for task-specific comparison. The next release must establish target/task alignment and rare-shape support before independent model comparison. There is no generic architecture-first or functional-causality claim.
