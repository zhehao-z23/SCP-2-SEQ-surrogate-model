# Contribution and experiment records

Use a `codex/` feature branch and focused commits. Do not rewrite remote history. Changes should include only relevant source, configuration, environment notes, and lightweight evidence.

Machine paths must be arguments, environment variables, or machine-local configuration. Keep credentials, raw datasets, cached embeddings, checkpoints, and full logs in the execution workspace. Before committing, inspect the staged diff and file sizes. Do not copy third-party source without its license and attribution.

Every real experiment needs a unique run ID and references to its dataset and label release, sequence view, split and exposure records, configuration, code commit, seed, resource budget, and artifact locations. Record failed or incomplete runs; launching a process does not prove completion.

Report group-macro Hellinger separately from protein-macro Hellinger and Hellinger squared. Preserve label measure, bin edges, units, structure source, and protocol identity. Historical development results are not independent confirmation of a new label release or full-length deployment.

Validate changes with checks appropriate to their effects. Numerical or identity-sensitive changes require checks that can expose incorrect labels, leakage, reduction, or restoration. Do not regenerate valid scientific results merely to refresh documentation.
