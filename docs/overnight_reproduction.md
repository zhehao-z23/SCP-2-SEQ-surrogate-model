# Overnight stage: reproduction and provenance

This stage reuses the immutable historical T/vertex-probability release and frozen ESMC300 cache. It contains four distinct lanes: five-fold/two-seed development completion, bounded aggregation/continuation experiments, CPU traditional baselines, and a 64-sequence AFDB engineering diagnostic. The final stage README reports what completed and separates reused from new fits.

## Roots and source versions

Set `PROJECT_ROOT` to an existing machine-local Seq2SCP project layout containing `atlas_scale_20261105/`, `pooling_20261105/` and the new `overnight_20261106/`. Parameterized scripts also accept their explicit `--root`, `--parent`, `--atlas`, `--pooling` and `--project` arguments. Do not use a repository source directory as a substitute for the private execution data layout.

Source files and public result records bind actual execution SHA-256 values. `overnight_source_provenance.json` distinguishes executed code from archived copies. Some historical one-off scripts contained fixed project paths and GPU UUIDs; only those literals are replaced by required `PROJECT_ROOT`, `GPU_UUID_0` and `GPU_UUID_1` environment variables in the public copy. No existing execution files or scientific arrays were changed by archiving. Materializing paths on another machine changes configuration/run hashes; it does not reproduce the historical artifact identity.

The configuration JSON files containing `${...}` are portable templates, not directly executable plans. Expand them into a new machine-local run directory before starting queues; fail if any placeholder remains. For example:

```python
import json, os
from pathlib import Path
text = os.path.expandvars(Path("configs/overnight_explore_plan.json").read_text())
assert "${" not in text, "Unresolved execution setting"
value = json.loads(text)
Path(os.environ["EXPLORE_ROOT"], "plan.json").write_text(json.dumps(value, indent=2) + "\n")
```

Assign resources according to the registered lane budgets. Each queue preserves its prior record and has no automatic retry after observation timeout. Confirm PID/start-time identity before recovery. The scripts are Linux-oriented and use `/proc`, resource accounting, CUDA and symlinks.

## Confirmation lane

`overnight_prepare_confirmation.py --root "$CONFIRM_ROOT" --pooling "$POOLING_ROOT" --atlas "$ATLAS_ROOT"` binds ten eligible existing fits and prepares twenty new jobs, without copying data or embeddings. Copy the versioned queue, summary, replay and exporter scripts into `CONFIRM_ROOT/code/`. Preparation copies the original head trainers from their prior execution roots and records their hashes. Both queue indices must run with their separately assigned CUDA visibility:

```bash
python src/overnight_confirmation_queue.py --root "$CONFIRM_ROOT" --project "$PROJECT_ROOT" --queue 0
python src/overnight_confirmation_queue.py --root "$CONFIRM_ROOT" --project "$PROJECT_ROOT" --queue 1
python src/overnight_summarize_confirmation.py --root "$CONFIRM_ROOT" --project "$PROJECT_ROOT"
PYTHONPATH="$CONFIRM_ROOT/code" python src/overnight_replay_confirmation.py --root "$CONFIRM_ROOT" --project "$PROJECT_ROOT"
python src/overnight_export_confirmation.py --root "$CONFIRM_ROOT"
```

The final 30 configurations are three families × five folds × two seeds. Each seed's OOF observations are reduced separately. Completing previously exposed folds is development evidence, not a newly protected test. The new checkpoint replay covers twenty newly trained fits; the ten reused fits retain prior-stage restoration evidence.

## Explore lane

Place the initial `overnight_explore_model.py`, trainer and queue in `EXPLORE_ROOT/code/` and the materialized initial plan at `EXPLORE_ROOT/plan.json`. Queues derive the stage root from their own file location. Thus launch the deployed queue, rather than running it directly from the repository's `src/`. The audits use the `PROJECT_ROOT` layout described above.

The confirmation directory is a separate two-fit extension. There, deploy the archived `overnight_explore_confirmation_train.py` under the executed filename `confirmation/code/overnight_explore_train.py`, together with the model and confirmation queue. The distinct filename in Git preserves both trainer versions; do not overwrite the original eight-fit trainer. Keep the original and promotion plans separate. Warm-start checkpoints, optimizer restart and prerequisite costs remain explicit.

The attention/continuation experiments compare complete registered procedures, not automatically equal-capacity causal interventions. Review the original diagnostic, completed summaries and independent replay alongside the mechanism notes. No attention maps, per-residue outputs or model weights are distributed.

## Traditional CPU lane

Deploy the traditional trainer/queue/summary/replay under the execution `tools/` layout. The queue's CPU-only environment and persisted wall deadline are intentional. It accepts `--phase screen` and `--phase confirm --candidates ...`; the promotion record controls which candidates receive the remaining folds. The source imports scikit-learn, NumPy and pandas; the recorded fit runtime was Python 3.10.21, scikit-learn 1.7.2, NumPy 1.26.4 and pandas 2.3.3. SciPy 1.15.3 and joblib 1.5.3 were read from the same unchanged runtime after completion, rather than separately recorded in original fit receipts. See the archive provenance for this distinction. No third-party library source is vendored.

## AFDB engineering lane

The selector reads a private source FASTA and creates a private panel and sequence snapshot. These files, raw embeddings and predicted arrays are explicitly excluded from this repository. Encoding reuses the already archived `pooling_cache_extract.py` and qualified local checkpoint assets. The diagnostic and separate cold-process end-to-end script take explicit paths; the public plan/report/provenance document selection and timing scope.

The 64 sequences form a deterministic length-stratified sample from a historical annotated/candidate pool, not an AFDB-wide random panel. No qualified compatible physical target was used to score accuracy. Model disagreement, finite probabilities, feature shift and prediction replay establish engineering behavior only. They do not authorize a whole-library scan or identify the more physically accurate model.

## Public archive boundary

Only source, configuration templates, aggregate-only reports and artifact hashes are included. `overnight_afdb_private_panel.json`, `overnight_afdb_sequences.json`, sample/role membership, predictions, raw labels, weights, cache arrays and full logs remain private. Public result paths may be project-relative references to private artifacts; a hash/reference does not distribute the artifact itself.
