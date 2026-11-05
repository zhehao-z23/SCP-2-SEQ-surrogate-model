# Pooling stage reproduction

The initial `pooling_20261105` plan registers 12 retrospective development fits: `mean_pre`, `max_pre`, and `residue_mean`, each with evaluation folds 0/1 and learning rates 0.001/0.0003, seed 1701. Each mode has 1,736,255 trainable parameters. `configs/pooling_amendment.json` subsequently registers four attention fits and four fixed-learning-rate second-seed fits, for 20 total. The original three modes remain exactly parameter matched; attention is not.

The three initial modes share the historical Atlas release, charge descriptors, fit-only preprocessing, frozen ESMC300 encoder, FP16 residue cache and FP32 readout operations. Mean and max are computed in FP32 from that same quantized residue cache, excluding padding and special tokens, with no truncation. This comparison changes whether a nonlinear residue transform precedes pooling. It is distinct from the previous stage's FP32 mean-cache fusion score.

## Separate roots and assets

```text
PARENT_ROOT/                 # completed atlas_scale_20261105 execution
  code/train_atlas_scale.py  # original split/metric helper, SHA bound in each fit
  data/                     # immutable historical labels and sample metadata
  data/esmc_300m_mean/       # previous full-T FP32 mean cache
POOLING_ROOT/               # new stage; do not overwrite prior runs
  code/                     # versioned pooling scripts
  records/                  # plans, queue identities, cache acceptance, logs
  cache_work/               # two disjoint encoder workers
  cache/                    # accepted flat/index/mean/max cache
  runs/                     # private per-fit evidence and checkpoint files
  results/                  # private and public aggregate analysis outputs
PROJECT_ROOT/               # existing historical qualification source
  data/qualification/release_v1/sample_manifest.parquet
```

`--parent` points to the completed Atlas stage, not to the repository's package directory. `--project` in the summarizer points to the legacy qualification source used for T/U relation strata. `--source` in the cache tools points to a private JSON sequence snapshot, containing one `sequence_sha256`, `sequence`, and `length` record per unique T. The snapshot must match the parent release and cache order; it is not published. `--backend-src` points to a directory containing `seq2scp_training/hf_backend.py`; its hash and the local checkpoint manifest are bound by each cache worker. The repository includes the minimal user-owned backend source, not encoder weights or model-specific checkpoint assets.

Use the head runtime described in `requirements-stage.txt` for training and summary operations. Use the separately qualified local checkpoint runtime for encoder extraction. This execution code is Linux-oriented (`fcntl`, `/proc`, resource accounting and CUDA) and is not a native Windows training launcher.

## Cache extraction and acceptance

Copy the versioned cache scripts into `POOLING_ROOT/code/`, create the execution directories, and use explicit assigned GPU UUIDs. The public launcher replaces the original machine-specific UUIDs with a required two-value argument; it is otherwise the same two-worker stage launcher. Execution and archive hashes are distinguished in `pooling_source_provenance.json`.

```bash
python src/pooling_cache_launch.py --root "$POOLING_ROOT" \
  --source "$T_SNAPSHOT" --python "$ENCODER_PYTHON" \
  --backend-src "$BACKEND_SRC" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --gpu-uuids "$GPU_UUID_0" "$GPU_UUID_1" --attempt pilot
# Wait for both eight-sequence pilot receipts and inspect their process identities.
python src/pooling_cache_check_pilot.py --root "$POOLING_ROOT" \
  --legacy-cache "$PARENT_ROOT/data/esmc_300m_mean"
python src/pooling_cache_launch.py --root "$POOLING_ROOT" \
  --source "$T_SNAPSHOT" --python "$ENCODER_PYTHON" \
  --backend-src "$BACKEND_SRC" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --gpu-uuids "$GPU_UUID_0" "$GPU_UUID_1" --attempt full
# Wait for both full receipts before publication.
python src/pooling_cache_publish.py --root "$POOLING_ROOT" \
  --source "$T_SNAPSHOT" --legacy-cache "$PARENT_ROOT/data/esmc_300m_mean"
python src/accept_pooling_cache.py --root "$POOLING_ROOT"
```

The publisher hashes all worker artifacts, checks disjoint complete coverage and every residue's finite values, recomputes mean/max, and compares newly extracted FP32 means with the previous cache. Cache acceptance independently hashes the published payload once. Each head verifies the receipt and accepted file size/mtime; all non-flat payloads are rehashed per fit. Do not modify an accepted cache in place. Full hashing is deliberately not repeated for the multi-GB flat cache on every fit.

## Readouts and independent checks

Copy `pooling_model.py` and `train_pooling.py` into `POOLING_ROOT/code/`; place the versioned plan in `POOLING_ROOT/records/`. The queue uses fixed settings corresponding to the plan rather than dynamically loading that JSON. Run at most two queues concurrently and retain queue/PID identities; no automatic retry is provided.

```bash
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" python src/run_pooling_queue.py \
  --root "$POOLING_ROOT" --parent "$PARENT_ROOT" --mode mean_pre
# Repeat for max_pre and residue_mean within the registered concurrency budget.
python src/summarize_pooling.py --root "$POOLING_ROOT" \
  --parent "$PARENT_ROOT" --project "$PROJECT_ROOT" --expected 12
python src/replay_pooling.py --root "$POOLING_ROOT" --parent "$PARENT_ROOT"
```

Evaluation fold i and stop fold (i+1)%5 remain fixed; the other three folds supply training data. Preprocessing fits training data only. Learning rates are selected using stop scores. The initial per-fit budgets are 300 epochs or 900 seconds, with minimum 40 epochs and patience 30 for early stopping. Record wall/epoch-budget stops separately from early stopping; equal time caps can produce unequal optimization opportunities across pooling modes.

The summarizer independently recomputes saved predictions, checks group and exact-T separation, and produces paired group bootstrap intervals and marginal/crossed strata. The intervals are descriptive after selection, without multiplicity or seed correction. The replay reloads selected checkpoints on CPU for 16 evaluation rows spanning input lengths; this is a restoration check, not an independent evaluation.

Retain strict T=U, T length >600 aa and their intersection in the final comparison. Uncertain small-stratum results do not demonstrate equivalence or establish deployment performance. See `pooling_round_split_reference.md` and `pooling_round_bridge_status.md` for the inherited grouping scope and the separate requirements for new-C initialization.

## Publication boundary

Publish source, configurations, aggregate-only outcomes and hashes after verification. Keep sequence snapshots, sample/sequence/group identity tables, predicted distributions, embeddings, checkpoints and full logs private. Stage aggregate results remain historical T→vertex-p development evidence, not final five-fold, new-C, or full-U acceptance.

The archived `bridge_subgroup_aggregate.json` is a completed post-hoc analysis of the parent stage, not a pooling-fit result. Reproduce it using the private parent stratum table:

```bash
python src/analyze_bridge_subgroups.py --root "$PARENT_ROOT" > bridge_subgroup_aggregate.json
```

## Registered extension: attention and seed 3407

The versioned amendment used a last-inspected state of 11 completed original fits and was recorded before attention launch. The appended `pooling_amendment_timing_note.json` establishes that the twelfth fit had already ended by that launch; it corrects the observed-state wording only, without changing settings. It is a post-exposure development decision, not a blind confirmation. Preserve both the original 12-fit plan and the amendment; do not rewrite the initial plan to suggest the extension was decided before seeing results.

| Component | Modes and settings | Fits | Trainable parameters |
|---|---|---:|---:|
| Original seed 1701 comparison | mean_pre, max_pre, residue_mean; folds 0/1; both registered LRs | 12 | 1,736,255 each |
| Attention extension | residue_attention; folds 0/1; both registered LRs; seed 1701 | 4 | 1,801,920 |
| Seed 3407 replication | mean_pre and the chosen alternative; folds 0/1; each family/fold reuses its seed-1701 stop-selected LR | 4 | Depends on chosen alternative |

Attention adds 65,665 scoring parameters (+3.78%). Its comparison tests learned residue weighting with this extra scoring capacity; do not describe it as exact parameter matching. Once all 16 seed-1701 fits finish, choose between residue_mean and residue_attention by lower combined two-fold development group-macro H. The reference remains mean_pre. Run seed 3407 once per selected family/fold without another learning-rate scan. Record the chosen candidate and the four inherited learning rates before those fits start.

This second seed assesses randomness after model selection on the same development folds; it does not create an independent test set. Aggregate and compare models separately by seed, retaining paired folds and predefined subgroups. Do not pool duplicated evaluation rows across seeds as if they were new proteins. The training algorithm remains unchanged. `run_pooling_replicate.py` binds its exact LR-source selection file; the revised summarizer aggregates by (mode, seed). Its optional cross-seed summary averages already reduced paired effects only when evaluation membership matches, and does not generate a pooled seed bootstrap. Executed and archived source hashes are recorded; the multiseed evaluator matches the executed bytes.

The amendment retains the 900-second single-fit limit, two-fit concurrency, four head-GPU-hour total cap and two-repair budget. After the registered 20 fits, report convergence, performance, subgroup uncertainty and costs, then close this stage. Further CNN/TCN/encoder searches require a separately justified stage decision.

After all 16 seed-1701 fits finish, run the explicit selection and replication workflow:

```bash
python src/summarize_pooling.py --root "$POOLING_ROOT" \
  --parent "$PARENT_ROOT" --project "$PROJECT_ROOT" --expected 16
python src/select_pooling_confirmation.py --root "$POOLING_ROOT"
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" python src/run_pooling_replicate.py \
  --root "$POOLING_ROOT" --parent "$PARENT_ROOT" --mode mean_pre
# Use the recorded chosen_mode for the alternative queue, within concurrency 2.
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" python src/run_pooling_replicate.py \
  --root "$POOLING_ROOT" --parent "$PARENT_ROOT" --mode "$CHOSEN_MODE"
python src/summarize_pooling.py --root "$POOLING_ROOT" \
  --parent "$PARENT_ROOT" --project "$PROJECT_ROOT" --expected 20
python src/replay_pooling.py --root "$POOLING_ROOT" --parent "$PARENT_ROOT"
```

Retain the immutable confirmation selection and per-queue LR-source hash. The runner checks the requested mode's two seed-1701 folds; the operator must use the mode in the registered confirmation decision. A later summary rewrite must not be substituted for the snapshot identity bound at launch.

## Final completion and cost export

The stage completed 20 fits, all early-stopped; 12 stop-selected checkpoints passed the 16-row, length-spanning CPU replay. Final prediction, convergence, subgroup and cost evidence is in `../experiments/pooling_20261105/`. `residue_mean` was selected for the fixed-LR seed-3407 replication. No further architecture extension is part of this stage.

```bash
python src/export_pooling_cost.py --root "$POOLING_ROOT" --expected 20
```

The cost exporter validates completed queue/fit identities and stop-only selection, then exports recorded aggregate costs and hashes. It does not rerun predictions or measure deployment throughput. Read the exported timing definitions before interpreting totals or throughput.

The historical charge reference is reconstructed from existing saved predictions, without refitting or tuning:

```bash
python src/compare_pooling_to_charge.py --root "$POOLING_ROOT" \
  --parent "$PARENT_ROOT" --project "$PROJECT_ROOT"
```

This reference uses the same release and fold roles but different parameter counts, optimization/batching paths and one fixed historical baseline seed. It is a practical baseline comparison, not the matched pooling mechanism experiment.
