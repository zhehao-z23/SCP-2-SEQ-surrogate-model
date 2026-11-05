# Experiment index

Store only lightweight manifests, aggregate result tables, decision notes, and links to execution artifacts here. Data, per-protein bulk predictions, checkpoint files, and full logs stay outside Git.

A run record should identify:

- run ID, question, status, and decision scope;
- dataset/label release and hash, input sequence view, target measure, bins and units;
- split/group identity, development exposure, training protein and group counts;
- code commit, configuration hash, seed, runtime and environment reference;
- fit count, per-fit and total budgets, convergence or failure status;
- verified metrics, their aggregation and uncertainty convention;
- artifact manifest and execution path, plus the next decision trigger.

Do not list a result as complete until its artifacts and metrics have been checked. Exact duplicate completed fits can be reused; changes in data, split, configuration, code, seed or learning-rate path require a new identity.
