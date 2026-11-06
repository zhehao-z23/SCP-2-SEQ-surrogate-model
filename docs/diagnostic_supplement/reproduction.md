# Reproduction boundary and private output handling

These are exact executed, version-specific scripts, not a standalone dataset or unrestricted inference package. Their archived and executed SHA256 values match. No paths, numerical functions, identity checks or acceptance thresholds were changed for this archive.

| Script | Existing operation | Inputs and outputs |
|---|---|---|
| `bridge_optimization_generalization_curve_review.py` | Replays the original sequential min-delta checkpoint selection and LR/early-stop state machine for 18 completed fits. | `--mirror`, `--training-source`, `--output-prefix`; requires 54 hash-bound private log files and mirror receipt. Outputs private per-run JSON/text. |
| `bridge_best_state_diagnostics.py` | Restores only two pinned existing best states and measures train/stop/evaluation in identical eval mode. No optimizer step. | `--optimization-root`, `--roles-root`, `--output`; machine-local plan supplies parent/cache paths and GPU allocation. Writes private probabilities, indices, per-row metrics and an internal summary. |
| `bridge_diagnostics_optimization_section.py` | Builds the private gallery extension from existing logs and the completed fixed-best summary. | `--review`, `--logs`, `--gallery`, optional `--plot-packages`; expects the original 66-artifact gallery, pinned historical summaries and later completed best-state summary. Outputs private HTML, images, small logs, manifests and QA contact sheet. |

Example commands use private environment bindings, not public data bundled here:

```sh
python bridge_optimization_generalization_curve_review.py \
  --mirror "$LOG_MIRROR" --training-source "$FROZEN_TRAINER" \
  --output-prefix "$PRIVATE_REVIEW_PREFIX"

python bridge_best_state_diagnostics.py \
  --optimization-root "$OPTIMIZATION_ROOT" --roles-root "$ROLES_ROOT" \
  --output "$NEW_PRIVATE_OUTPUT"

python bridge_diagnostics_optimization_section.py \
  --review "$PRIVATE_REVIEW_ROOT" --logs "$LOG_MIRROR" \
  --gallery "$PRIVATE_GALLERY_ROOT"
```

The fixed-best recipe deliberately retains its exact checkpoint, model, split/metric, label-release and guard pins. It checks role restrictions before target/checkpoint/prediction arrays. The private plan, cache acceptance receipt, source receipts, complete role-contract dependencies and dataset must remain available and valid; they are not copied here. Relocating or changing those inputs can fail existing exact path/mtime/hash checks and requires a separately validated reproduction setup. Do not bypass a failed guard or reinterpret this archive as authorization for protected labels or another release.

Runtime evidence: fixed-best execution used Linux Python 3.10.21, PyTorch 2.8.0+cu128 and NumPy 1.26.4. Local gallery execution used Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.11.2. The curve-review script uses the Python standard library. The GPU recipe also needs pandas, a parquet engine, the pinned project model/metrics code and accepted ESMC caches. The gallery needs Pillow and Matplotlib dependencies. These notes are not a complete environment lock or a transitive dependency archive.

`results/summary.json` is a reviewed aggregate derivative, not the unfiltered output of any private program. It retains only six group-macro partition rows and support counts, the 18-fit review counts, zero-fit/update verification, replay agreement and scoped resource statistics. It omits sample/sequence/group memberships, run listings, probability arrays, model weights, GPU UUID/affinity, internal absolute paths and all images/HTML. Private source evidence is bound by hash in the provenance record.

The original online train H² trace is dropout-on and changes parameters during the epoch; stop H is epoch-end eval with equal groups. They are not comparable gaps. The separate two-best measurement uses the same fixed-state reduction across partitions and remains a limited exposed-development contrast. Last-ten-minus-previous-ten classifications use a descriptive +/-1e-5 band only; no new convergence, fallback or statistical-significance gate was introduced.
