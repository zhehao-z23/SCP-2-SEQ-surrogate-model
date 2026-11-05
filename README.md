# SCP to sequence surrogate model

Research code and lightweight evidence for predicting protein surface electrostatic-potential distributions from sequence.

## Current scope

Historical Atlas data support development experiments, model comparison, learning curves, and supervised initialization research. The new construct-sequence task maps experimental construct sequence C to the area-weighted potential distribution of its selected experimental structure. These are distinct label releases; their metrics and weights must not be silently combined.

The current research stage has not established a final model or validated whole-sequence-space deployment. Natural full-length sequence deployment requires separate transfer evidence and an explicit database release and sequence manifest.

## Repository layout

- `src/`: reusable code with configurable paths.
- `configs/`: versioned experimental configurations.
- `experiments/`: lightweight run manifests and verified result summaries.
- `docs/`: environment, provenance, evaluation, and reproduction notes.

Large data, embeddings, checkpoints, and complete logs remain in the execution workspace. Repository records reference them by asset identity and manifest; do not commit credentials or restricted materials.

## Development

Work on `codex/` feature branches with focused commits and read `CONTRIBUTING.md`. The bounded Atlas learning-curve scripts are in `src/`; their fixed stage plan is in `configs/scale_plan.json`. See [reproduction notes](docs/reproduction.md) for the private source-data layout, portable module provenance, environment versions, and execution commands. Literature-driven experiment decisions are documented separately from measured model results.
