# Reproducing the completed bridge sub-stages

This publication covers completed optimization, anonymous diagnostics and provisional role registration. Physical qualification is active and omitted from numerical success claims. The authoritative execution layout is `${PROJECT_ROOT}/bridge_20261106`; the earlier Atlas, pooling and overnight releases remain immutable sibling directories. Original data and private artifacts are not shipped.

## Source and environment

`bridge_source_provenance.json` maps each local source snapshot, actual execution SHA where recorded, and archived SHA. Portable copies replace machine-specific paths with `PROJECT_ROOT`, `AFDB_PROJECT_ROOT`, and `MMSEQS`. JSON plans contain `${PROJECT_ROOT}` and `${GPU_UUID_0/1}` placeholders: substitute them in a separate machine-local plan before use. Do not change historical executed plan hashes or present the archive as the executed source.

Use the recorded numerical environments in each result/provenance file. Optimization uses the existing PyTorch head environment and frozen residue cache; diagnostics use NumPy/pandas/SciPy/matplotlib on CPU; role finalization requires pandas/parquet and the bounded MMseqs search uses Linux resource limits. No external project source was copied. Existing `package/src/seq2scp_training` and historical trainer dependencies remain part of the repository.

## Optimization

Stage the model, trainer, audit and replay scripts under a fresh optimization workspace `code/`, and the materialized plan as `plan.json`. Queue scripts derive their workspace from their own `code/` parent; do not run them directly from repository `src/`. A new workspace must provide `runs/` and `records/`. Confirmation has its own plan and queue. Preserve failed/started receipts and reconcile an interrupted job before rerunning.

The trainer accepts explicit `--root`, `--parent`, `--cache-root`, `--mode`, `--policy`, `--fold`, `--seed`, `--lr` and time limits. `${PROJECT_ROOT}/atlas_scale_20261105/code` must contain the archived earlier selection code with its immutable data; the cache comes from the accepted pooling release. `bridge_optimization_build.py` is the source-generation recipe; the generated trainer is also archived directly.

The first plan ran 12 fits; stop-based promotion added six. `bridge_optimization_audit.py --root WORKSPACE` recomputes metrics and validates optimizer branches. `bridge_optimization_replay.py --root WORKSPACE` restores every selected checkpoint with its saved transformations in a new process. Published results already passed these checks; no historical training was repeated for this archive.

## Anonymous diagnostics

`bridge_diagnostics_build.py --project PROJECT_ROOT --output PRIVATE_OUTPUT` constructs private diagnostics plus public aggregates. `bridge_tail_diagnostic.py` takes the project/output arguments and adds target-scale analysis. Inspect `--help` before staging output. **The full generated output is private**: only the explicitly archived aggregate files are included in this public commit. Eleven anonymous PNG/SVG pairs were reviewed locally, but image publication was rejected by automatic review and all images are omitted here. Never publish the generated index, cases, private tables or artifact manifest.

The current builder binds the historical label contract and its CDF cutpoints. This revision changed plot axes, not pre-existing aggregate metrics. The tail analysis uses observed target scale retrospectively; that variable is unavailable as a prospective prediction input.

## Provisional reservation and future preflight

The role scripts reproduce a bounded sequence-only audit; private input manifests, identities, source FASTAs and component mappings must be supplied in the authorized workspace. The selector emits IDs/hashes/lengths only. Official fetch limits count attempted requests. The exposure supplement executes on its source server in a temporary directory and returns hashes/relationships without transferring historical sequences. Do not substitute a bulk FASTA transfer.

`bridge_roles_search_v2.py` expects the stage tree, historical `sequence_search_manifest.parquet`, immutable groups and all new-C aliases. It preserves original identities while verifying normalized search hashes. The finalizer accepts `--project-root`, `--search`, `--output`, `--guard` and `--exposure-supplement`; relationships must be complete. It produces a private pinned registry and aggregate summary. Portable new executions need newly pinned contracts; original execution hashes remain unchanged.

Before a future job loads targets/models, invoke the guard using the previously registered contract SHA:

```sh
python bridge_roles_guard.py --contract PRIVATE_REGISTRY/guard_contract.json \
  --contract-sha256 PREVIOUSLY_REGISTERED_SHA \
  --request PRIVATE_REQUEST.json --receipt NEW_GUARD_RECEIPT.json
```

The request declares operation, `stage: development`, actual sequence hashes and optional aliases. Unknown identities and protected components are denied. A pass clears only this bridge preflight; existing release-role and eligibility gates still apply. The current registry denies every `final_release` request. Changed grouping, inputs or exposure require renewed audit and a new pinned registry. The guard is not OS access control, so launchers must enforce its successful receipt.

Sixteen checks passed against the real registry and isolated mutation fixtures, including protected-list tampering under an unchanged contract pin. They did not reveal protected targets or predictions. The exposure audit pins a source snapshot; it does not continuously monitor remote files.


## Completed physical bridge and diagnostic supplements

The earlier partial-stage record above is retained. One fixed-coordinate AF development object has now passed all eight registered compound numerical sensitivity checks and final main/independent/acceptance gates. See [final physical report](physical_bridge/physical_final_report.md), [source and dependency boundary](physical_bridge/physical_reproduction.md), [fixed best-state diagnostics](../experiments/bridge_20261106/diagnostic_supplement/README.md) and [label/shape diagnostics](../experiments/bridge_20261106/label_diagnostics/README.md). This closes this bounded stage, not the final model or full-sequence-space deployment. Protected roles remain provisional and existing release gates are mandatory. Private images and detailed arrays are not published.
