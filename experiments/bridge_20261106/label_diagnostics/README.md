# Label representation and shape retrieval diagnostics

These completed diagnostics separate label consistency, representation limits and model retrieval performance. They use exposed historical Atlas development data: 21,921 rows, 21,915 distinct teacher sequences (T), and 7,999 homology groups. They add no fits, inference, physical solves or raw-potential reads. This archive does not qualify new area-weighted labels or a protected final test.

The main finding is a remaining retrieval failure: improving whole-distribution Hellinger error did not produce reliable absolute Bowley-shape detection. Neither binning alone nor the six duplicate inputs explains all of the observed gap. The full historical screening rule also required source-population percentiles; those conditions are not applied here.

## What the checks establish

| Check | Aggregate result | Limitation |
|---|---|---|
| Existing labels and QC | All 65,763 Q25/Q50/Q75 values lie within exact-count type-7-compatible support intervals; current FP32 p matches count normalization bit for bit. | Current snapshot consistency does not independently validate original structures or physical truth. |
| Conservative Bowley certificate | Raw absolute conditions identify 61 negative and 17 positive T; sufficient bin bounds certify 1 negative and 0 positive. | The remaining 77 are not proven irrecoverable. |
| Native finite-window major peaks | Native view: 21,853 one-major-peak, 1 two-major-peak, 0 multi-major-peak, 67 uncertain. Three-view consensus: 19,397 one-major-peak, 2,524 uncertain. | 2,452 consensus rejections lack positive interval overlap; unconfirmed is not absent. One visible major peak permits smaller shoulders and is not raw unimodality. |
| Exact duplicate model inputs | Six T pairs have identical charge bytes/NaN masks and cache addresses but different labels; pair H ranges 0.02018–0.04968. | Source file hashes differ, but causal structure/preparation/protocol/grid effects remain unresolved; no same-structure repeat control exists. |

The duplicate pairs touch only 12 rows. Their maximum direct contribution under the existing group-macro H weighting is 0.0003924; the empirical triangle-inequality floor is about 0.00000614. These are limited support bounds, not a global noise estimate, observed model error or a bound on indirect training effects.

## Existing 30-fit OOF predictions

The comparison retains three model families and two seeds separately. All 21,921 rows have finite proxy quartiles and IQR >= 1. All six model/seed combinations have zero true positives under the absolute negative/positive Bowley thresholds; this is not an open-tail or low-IQR abstention effect. Precision remains undefined when no positives are called.

The true-63-bin reference uses a declared uniform-within-bin inverse CDF. It is an assumption reference, not raw type-7 quantiles, an optimal attainable model or an irreducible error bound. At a fixed 220-T candidate budget per direction, it recovers all 61 negative T and 16/17 positive T. Model recovery is lower:

| Family | Seed | Group H | Q50 AE | IQR AE | Bowley AE | Negative TP / 61 | Positive TP / 17 |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle_true63_uniform | reference | 0.000000 | 0.040980 | 0.192485 | 0.012414 | 61 | 16 |
| composition_charge_deep | 1701 | 0.052344 | 0.514108 | 1.038893 | 0.048283 | 21 | 3 |
| mean_pre | 1701 | 0.050601 | 0.536966 | 0.926930 | 0.046494 | 30 | 8 |
| residue_mean | 1701 | 0.049038 | 0.506447 | 0.885147 | 0.046082 | 30 | 6 |
| composition_charge_deep | 3407 | 0.052358 | 0.516211 | 1.050076 | 0.048348 | 21 | 5 |
| mean_pre | 3407 | 0.050433 | 0.536336 | 0.918706 | 0.046407 | 26 | 7 |
| residue_mean | 3407 | 0.049027 | 0.503638 | 0.890743 | 0.046152 | 28 | 6 |

Q50/IQR errors are in kBT/e; Bowley is dimensionless. Candidate ranks use one fixed canonical row per T, and truth is any original view of that T satisfying the same-direction absolute condition. No probability or shape averaging determines the rank. Seeds reuse the same objects and are not independent sample replicates. Positive support is only 17 T, so these counts do not establish significance or a final architecture winner.

## Eight matched comparisons of existing optimization predictions

This reuses 12 fresh/plateau observations plus four necessary older fixed-LR references: 16 prediction files, eight matched family/fold/seed/role pairs. It is not a complete fivefold OOF result. Each fold-specific candidate budget is 44 distinct T.

| Family | Fold | Seed | Reference H → plateau H | Negative TP, reference → plateau / truth | Positive TP, reference → plateau / truth |
|---|---:|---:|---:|---:|---:|
| mean_pre | 0 | 1701 | 0.050862 → 0.049708 | 8 → 9 / 12 | 3 → 3 / 4 |
| charge_deep | 0 | 1701 | 0.053062 → 0.052563 | 5 → 5 / 12 | 1 → 0 / 4 |
| residue_mean | 0 | 1701 | 0.049104 → 0.048042 | 7 → 8 / 12 | 3 → 3 / 4 |
| charge_deep | 1 | 1701 | 0.052454 → 0.052339 | 4 → 4 / 11 | 1 → 0 / 2 |
| mean_pre | 1 | 1701 | 0.050417 → 0.049558 | 6 → 5 / 11 | 0 → 0 / 2 |
| residue_mean | 0 | 3407 | 0.049158 → 0.048160 | 8 → 9 / 12 | 1 → 1 / 4 |
| residue_mean | 1 | 1701 | 0.049040 → 0.047951 | 6 → 5 / 11 | 1 → 0 / 2 |
| mean_pre | 0 | 3407 | 0.050450 → 0.050343 | 6 → 7 / 12 | 2 → 2 / 4 |

All eight pairs improve H, IQR AE and Bowley AE. Negative candidate recovery increases in four pairs, decreases in two and is unchanged in two; positive recovery never increases (three decrease, five unchanged). All 16 predictions have zero absolute-threshold TP and FP, with no quartile/IQR abstentions. CDF MSE increases in three pairs; open-edge MAE and Q50 AE each increase in one. Better H does not mean every feature or retrieval target improves.

Four references come from a different earlier runner. Bound roles and scaler evidence match, but source/control-history differences remain; these are engineering comparisons, not isolated causal learning-rate experiments. The failed v1 aggregation was preserved: a provenance field collided with a count field after prediction checks. Successful v2 only separates those field names.

## Interpretation and reproduction boundary

The evidence supports keeping overall distribution accuracy and scientific retrieval success as separate objectives. The uniform-bin reference retains ranking signal that the existing models fail to recover, while threshold mismatch already occurs for that reference. The next diagnostic or objective change must address the intended retrieval task explicitly; this draft selects no new model, loss or training run.

中文：已有标签/QC 配对内部一致；保守表示证书弱不等于证明信息丢失。模型总体 H 改善没有自动解决偏态筛选，真实 63 箱代理保留的候选排名信号也尚未充分恢复。6 组相同实际输入对应不同目标已证实，但支持极少、成因未分解，不能用来解释主要全局失败。

Only generic source, the frozen coarse-method plan and aggregate JSON are included. No sample/member lists, raw or predicted arrays, selected-candidate tables, model weights, HTML, PNG or SVG are published. The public code still writes private detailed outputs when run with authorized private inputs; those outputs are not publication-authorized by this source archive. See [reproduction](../../../docs/label_diagnostics/reproduction.md) and [source provenance](../../../docs/label_diagnostics/source_provenance.json).

Results: [representation](results/representation.json), [coarse shape](results/coarse_shape.json), [duplicate input/target](results/duplicate_T.json), [30-fit OOF shape](results/shape_OOF.json), [eight optimization pairs](results/shape_optimization.json).
