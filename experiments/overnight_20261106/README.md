# Overnight stage, 2026-11-06

Completed aggregate-only evidence for four lanes. Read the [full report](../../docs/overnight_20261106.md) for model decisions, subgroup failures, measured cost and scientific limits.

| Lane | New fits | Reused fits | Evidence |
|---|---:|---:|---|
| Confirmation: three families, five folds, two seeds | 20 | 10 | [Summary](confirmation/summary.json), [CPU replay](confirmation/checkpoint_replay.json), [cost](confirmation/cost_public.json) |
| Optimization/aggregation exploration | 10 | Warm-start prerequisites retained separately | [Summary and replay](explore/summary_public.json), [source identity](explore/provenance_public.json) |
| Traditional CPU baselines | 14 | 0 | [Summary](traditional/summary.json), [fresh-process replay](traditional/independent_checkpoint_replay.json) |
| AFDB input/engineering diagnostic | 0 | Two existing task heads | [Diagnostic](afdb/summary_public.json), [cold-process pipeline](afdb/e2e_public.json) |

There are 44 new fits; the AFDB panel has no physical-accuracy claim. Five-fold results are exposed historical development OOF. Each seed is reduced separately. Private AFDB sequences and all sample-level arrays are excluded. See [reproduction](../../docs/overnight_reproduction.md) and [execution versus archive hashes](../../docs/overnight_source_provenance.json).
