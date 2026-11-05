"""Fail-closed release, identity, partition and output validation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def primary_evaluation(row):
    """Only common full-target rows may select stopping or enter primary OOF."""
    return row.get("purpose", "primary") == "primary" and row.get("primary_evaluation", True) is True


def validate_release(release):
    required = {"release_id", "target", "bin_count", "edges", "units", "measure", "phase", "qualified", "boundary_policy"}
    if not required <= release.keys():
        raise ValueError("missing release fields")
    if release["target"] != "p":
        raise ValueError("only canonical p is accepted; legacy q is forbidden")
    if type(release["qualified"]) is not bool or not isinstance(release["boundary_policy"],str) or not release["boundary_policy"]: raise ValueError("release qualification/boundary metadata")
    k = release["bin_count"]
    if type(k) is not int or k < 2 or len(release["edges"]) != k - 1:
        raise ValueError("K categories with two overflow tails require K-1 finite boundaries")
    edges = np.asarray(release["edges"], dtype=float)
    if not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("invalid boundaries")
    if release["phase"] not in {"historical", "new_physical", "synthetic"}:
        raise ValueError("unknown release phase")
    if release["measure"] not in {"equal_retained_vertices", "surface_area"} or not release["units"]:
        raise ValueError("missing/unknown physical meaning")
    return k


def validate_p(p, release):
    k = validate_release(release)
    a = np.asarray(p, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != k or len(a) == 0:
        raise ValueError("p shape mismatch")
    if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("invalid probability; never silently renormalize labels")
    return a


def validate_records(rows, release, split_id):
    seen = set(); parents = {}; sequences = {}; groups = {}
    for row in rows:
        required = {"sample_id", "parent_id", "sequence", "sequence_sha256", "release_id", "group_id", "split_id", "role", "outer_fold", "purpose", "eligibility", "sequence_view", "source_origin", "source_release", "pairing_receipt_sha256"}
        if not required <= row.keys():
            raise ValueError("incomplete candidate record")
        if not all(isinstance(row[k], str) and row[k] for k in ["sample_id", "parent_id", "group_id"]):
            raise ValueError("empty identity")
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample")
        seen.add(row["sample_id"])
        sequence = row["sequence"]
        if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWYXBZJUO" for c in sequence):
            raise ValueError("noncanonical input string; normalization must be explicit upstream")
        if sequence_hash(sequence) != row["sequence_sha256"] or row["sequence_view"] != "U":
            raise ValueError("parent sequence identity mismatch")
        if row["release_id"] != release["release_id"] or row["split_id"] != split_id:
            raise ValueError("release/split identity mismatch")
        if row["source_origin"] not in {"current_canonical","explicit_isoform","archival","synthetic"} or not row["source_release"] or not row["pairing_receipt_sha256"]: raise ValueError("source provenance missing")
        role, fold = row["role"], row["outer_fold"]
        if role not in {"development", "independent_test", "physical_calibration", "excluded"}:
            raise ValueError("unknown role")
        if (role == "development" and (type(fold) is not int or fold not in range(5))) or (role != "development" and fold is not None):
            raise ValueError("invalid fold")
        location = (role, fold)
        # Parent aliases must be canonicalized before admission. Otherwise one U
        # can receive extra training and metric weight under several source IDs.
        for table, key, value in [(parents,row["parent_id"],(row["sequence_sha256"],row["group_id"],location)),(sequences,row["sequence_sha256"],(row["parent_id"],row["group_id"],location)),(groups,row["group_id"],location)]:
            if key in table and table[key] != value:
                raise ValueError("parent/sequence/homology group role conflict")
            table[key] = value
        if row["purpose"] not in {"primary", "fragment_auxiliary"}:
            raise ValueError("unknown supervision purpose")
        if "primary_evaluation" in row and type(row["primary_evaluation"]) is not bool:
            raise ValueError("primary evaluation eligibility must be explicit boolean")
        if row.get("target_scope") == "partial_teacher_proxy":
            if row["purpose"] != "primary" or row.get("primary_evaluation") is not False or not row.get("proxy_policy_sha256"):
                raise ValueError("partial teacher proxy requires a bound training-only policy")
        if row["purpose"] == "fragment_auxiliary":
            mapping = row.get("fragment_to_U")
            fragment = row.get("fragment_sequence")
            if not isinstance(mapping, list) or not mapping or not isinstance(fragment, str) or len(mapping) != len(fragment):
                raise ValueError("missing source-valid fragment mapping")
            if any(type(i) is not int or i < 0 or i >= len(sequence) for i in mapping) or mapping != sorted(set(mapping)):
                raise ValueError("invalid fragment mapping")
            # A readout mask encodes position, not amino-acid substitutions. An audit flag
            # alone cannot make two different constructs distinguishable to this adapter.
            if ''.join(sequence[i] for i in mapping) != fragment:
                raise ValueError("fragment substitutions require an explicit sequence-context adapter; mask-only supervision cannot represent them")
    if not seen:
        raise ValueError("empty cohort")
    return True


def inner_partition(rows, outer_fold, stop_groups):
    outer = [r for r in rows if r["role"] == "development" and r["outer_fold"] == outer_fold]
    pool = [r for r in rows if r["role"] == "development" and r["outer_fold"] != outer_fold]
    available = {r["group_id"] for r in pool}
    stop_groups = set(stop_groups)
    if not stop_groups or not stop_groups < available:
        raise ValueError("inner stop must be a nonempty proper subset of development training groups")
    fit = [r for r in pool if r["group_id"] not in stop_groups]
    stop = [r for r in pool if r["group_id"] in stop_groups]
    if not outer:
        raise ValueError("empty outer fold")
    return fit, stop, outer


def safe_output(path, allowed_root, forbidden=()):
    """Require an explicit write root and caller-supplied protected locations."""
    root = Path(allowed_root).resolve()
    p = Path(path).resolve()
    if p == root or root not in p.parents:
        raise ValueError("output must be inside explicit write root")
    for item in forbidden:
        bad = Path(item).resolve()
        if p == bad or bad in p.parents:
            raise ValueError("protected output path")
    return p


def real_run_blockers(readiness):
    fields = ["training_enabled", "execution_authorized", "gpu_verified", "cohort_qualified", "full_U_homology_frozen", "split_frozen", "test_isolation_verified", "encoder_verified", "numerics_calibrated", "release_qualified"]
    missing = [f for f in fields if readiness.get(f) is not True]
    budget = readiness.get("fit_budget")
    if type(budget) is not int or budget <= 0:
        missing.append("fit_budget")
    if readiness.get("max_concurrent_gpus", 8) > 7:
        missing.append("resource_allocation")
    return missing
