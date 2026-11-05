# Atlas development: source and data layout

The source modules in `package/src/seq2scp_training/` are a minimal extract of the user's existing Seq2SCP execution project, not a copy of its full training platform. Their original and archived SHA-256 values are recorded in `source_provenance.json`.

`prepare_atlas_scale.py` creates a new retrospective development release from an existing qualified Atlas source. It does not distribute the underlying sequences, labels, or per-protein manifests. Its required source layout is:

```text
SOURCE_ROOT/
  package/src/seq2scp_training/
  data/qualification/release_v1/sample_manifest.parquet
  data/qualification/release_v1/all_sequence_groups.parquet
  data/labels/index.parquet
  data/labels/p.float32.npy
  records/label_integrity.json
  data/labels/release.pending.json
```

Paths are supplied with `--source` and `--output`; no server address or individual user's directory is required by the code. On a new execution machine, construct a source workspace referencing the qualified private data, with its `package/` directory pointing to this repository's `package/`. Do not overwrite or relocate the historical source workspace to do this. Record the actual paths and hashes in the private execution manifest.

```bash
python src/prepare_atlas_scale.py --source "$SOURCE_ROOT" --output "$NEW_RELEASE_DIR"
```

Preparation needs Python, NumPy, pandas and a parquet engine such as PyArrow. The model/loss modules additionally need PyTorch. `requirements-stage.txt` records the relevant versions observed in the execution environment on 2026-11-05, not a claim that every hardware/driver combination has been tested. Encoder extraction uses its own validated environment and is not performed by this script.

The archived `contracts.safe_output` retains the explicit allowed write root and caller-supplied forbidden locations. Its historical machine-specific protected path was removed from the portable copy; callers must pass any protected paths explicitly. This adaptation does not modify the Byte source or the scientific labels. Other imported modules are byte-for-byte copies. Actual training receipts bind the original Byte module hashes; do not substitute the archived `contracts.py` hash for an executed-source hash.

This stage's target is historical vertex-count probability **p** with teacher-retained sequence **T** input. It is not the new area-weighted C target or a full-length U deployment validation. It reuses the existing development-role grouping. Its weights remain development-only until the new C protected closure is mapped and fold-specific supervised initialization is permitted.

## Bounded learning curves

`configs/scale_plan.json` registers four input families, three nested training-group scales, two evaluation folds and two learning rates. The runner implements these fixed settings; it does not dynamically read the plan. A change to either requires a new recorded run/configuration identity.

The execution layout is separate from `SOURCE_ROOT`:

```text
RUN_ROOT/
  code/train_atlas_scale.py
  data/                      # prepared release, features and manifests
  records/                   # queue ownership and full logs
  runs/                      # per-fit artifacts and checkpoints
  results/                   # independently recomputed summaries
```

Copy the versioned `src/train_atlas_scale.py` into `RUN_ROOT/code/` before running a queue and record its SHA-256. Create the listed execution directories without altering an existing run. For heads, `--source` locates `package/src/`; it does not select a different label release. The preparation command uses `--source` for both the qualified data and modules. The separate arguments make the legacy data source, archived package, and new run location explicit, though they use the same legacy source root in the original execution.

```bash
CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" python src/run_scale_queue.py \
  --root "$RUN_ROOT" --source "$SOURCE_ROOT" --family composition_charge
python src/summarize_atlas_scale.py --root "$RUN_ROOT"
```

Start another family only within the declared concurrent-process and memory budget. Existing queue records intentionally prevent duplicate ownership; inspect the live process and artifact receipts before any recovery. The ESMC families additionally need a separately verified encoder cache. The code performs no model download or automatic encoder extraction.

The evaluator recomputes saved per-sample predictions, checks group isolation, chooses learning rates using the stop split, and aggregates evaluation results. Its group bootstrap intervals are descriptive, conditional on the selected fits; they do not include model-selection or seed uncertainty. Final checked metrics will be added separately. Archiving these scripts is not a claim that all planned fits have completed.

## Historical label contract correction

The initial preparation receipt included `source/shared_bins.json` as an extra reference. That file describes a different, new physical bin contract; it was not used to transform target arrays or calculate the reported metrics. The current preparation script binds the actual `data/labels/release.pending.json` instead. For the already prepared release, `bind_label_contract.py` appends a separate `label_contract_addendum.json` binding the immutable original release to the actual historical contract. The arrays and model fits are unchanged.

The historical boundaries span approximately -44.62 to 108.05 kBT/e, with vertex-count measure and an LPBE protocol without explicit salt. This is distinct from the new area-weighted physical release. Hellinger does not depend on bin-coordinate values; the reported CDF score is an unweighted mean over 62 ordered cutpoints, and tail error refers to first/last bin masses. None is an exact continuous Wasserstein distance.

```bash
python src/bind_label_contract.py --source "$SOURCE_ROOT" --root "$RUN_ROOT"
```

This command is a corrective addendum for the initial release, not a general relabeling operation. Preserve the original receipt, the addendum, and their hashes. The checked run's immutable receipt remains the authority for its source code version.

## ESMC mean-cache stage

`cache_atlas_esmc_mean.py` uses an explicit sequence snapshot, local checkpoint manifest, backend source path and assigned GPU UUID. It performs singleton unpadded extraction, excludes special tokens, preserves all residues without truncation, and records the exact backend and checkpoint identities. Its dependencies include the archived `hf_backend.py` and `encoders.py`; model weights and model-specific custom implementation files remain external and must match the checkpoint manifest. No model download is performed.

`publish_atlas_esmc_mean.py` is the specific two-worker stage publisher. It expects the documented `RUN_ROOT/cache/esmc300_T/worker{0,1}` layout and `full01`/`pilot02` receipts; those are stage conventions rather than a generic arbitrary-worker interface. It checks shard hashes, disjoint complete row coverage, and numerical agreement on all exact-sequence overlaps with a specified legacy singleton cache before publishing the new mean cache.

Encoder extraction uses the separately validated checkpoint environment. Its full environment and encoder-cache receipts stay with the private execution record; `requirements-stage.txt` describes the head/data runtime only. Encoder extraction cost must be reported separately from head fitting and inference cost.
