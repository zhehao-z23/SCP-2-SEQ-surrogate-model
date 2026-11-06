"""Audit copied overlap arrays and captured parent headers on the local computer.

No network, PB calls, or server writes. Parent payload hashes remain the native
field auditor's responsibility; this checks actual captured DX headers against
the frozen plan. Previously audited surface areas are reused by exact identity.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prior-batch", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir
    prior = load(args.prior_batch)
    qual = load(root / "qualification.json")
    assert qual["sample_id"] == prior["case"] == args.case
    record = next(row for row in qual["comparisons"] if row["axis"] == "overlap")
    assert (record["test"], record["reference"]) == ("refined", "overlap12")
    weights = np.load(root / "refined/weights.npy")
    weight_sha = sha(root / "refined/weights.npy")
    assert weight_sha == prior["variants"]["base"]["weights_sha256"]
    assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
    bins_sha = sha(root / "shared_bins.json")
    edges = np.array(load(root / "shared_bins.json")["internal_edges"])
    assert len(edges) == 62 and np.all(np.diff(edges) > 0)
    variants, arrays, manifests, bound = {}, {}, {}, []
    for variant in ["refined", "overlap12"]:
        directory = root / variant
        outcome = load(directory / "outcome.json")
        manifest = load(directory / "manifest.json")
        assert outcome["status"] == "completed" and outcome["sample_id"] == args.case
        assert sha(directory / "manifest.json") == outcome["representation"]["sha256"]
        inputs = manifest["spec"]["inputs"]
        assert manifest["potential_unit"] == "kBT/e" and not manifest["potential_clipping_or_rescaling"]
        assert inputs["area_weights"]["sha256"] == weight_sha
        assert inputs["shared_bins"]["sha256"] == bins_sha == prior["shared_bins_sha256"]
        mesh_key = inputs["vertices"]["sha256"] + ":" + inputs["faces"]["sha256"]
        assert mesh_key == prior["variants"]["base"]["mesh_identity"]
        phi_sha = sha(directory / "phi.npy")
        assert phi_sha == inputs["corrected_phi"]["sha256"]
        phi = np.load(directory / "phi.npy")
        assert phi.shape == weights.shape and np.isfinite(phi).all()
        hp = np.bincount(np.searchsorted(edges, phi, side="right"), weights=weights, minlength=63)
        hp /= weights.sum()
        stored_p = record["metrics"]["p_test" if variant == "refined" else "p_reference"]
        assert np.allclose(hp, stored_p, rtol=0, atol=1e-13)
        variants[variant] = {
            "outcome_sha256": sha(directory / "outcome.json"),
            "representation_sha256": outcome["representation"]["sha256"],
            "phi_sha256": phi_sha, "weights_sha256": weight_sha,
            "mesh_identity": mesh_key, "PQR_sha256": prior["variants"]["base"]["PQR_sha256"],
            "queries": len(phi), "area_A2": float(weights.sum()),
            "potential_min_max": [float(phi.min()), float(phi.max())],
            "probability_sum": float(hp.sum()),
            "registered_probability_max_abs_difference": float(np.abs(hp - stored_p).max()),
        }
        bound += [inputs["corrected_phi"], inputs["area_weights"], inputs["shared_bins"], outcome["representation"]]
        arrays[variant] = (phi, hp)
        manifests[variant] = manifest
    x, px = arrays["refined"]
    y, py = arrays["overlap12"]
    values = np.concatenate([x, y])
    signed_mass = np.concatenate([weights / weights.sum(), -weights / weights.sum()])
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cdf = np.cumsum(signed_mass[order], dtype=np.float64)
    last = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    unique, delta = values[last], cdf[last]
    metrics = {
        "Hellinger": float(np.sqrt(.5 * np.square(np.sqrt(px) - np.sqrt(py)).sum())),
        "W1_kBT_per_e": float(np.sum(np.abs(delta[:-1]) * np.diff(unique))),
        "CDF_sup": float(np.abs(delta).max()),
    }
    differences = {key: value - record["metrics"][key] for key, value in metrics.items()}
    assert max(abs(value) for value in differences.values()) < 1e-11
    passed = all(metrics[key] <= limit for key, limit in {"Hellinger": .02, "W1_kBT_per_e": .1, "CDF_sup": .01}.items())
    assert passed and record["pass"] is True

    capture = load(root / "parent_headers.json")
    plan = load(root / "nested_plan.json")
    execution = capture["execution_records"]
    preflight = execution["further_execution_preflight.json"]
    launch = execution["further_launch.json"]
    assert preflight["record"]["ready"]
    assert launch["record"]["preflight"]["sha256"] == preflight["sha256"]
    assert sha(root / "nested_plan.json") == preflight["record"]["geometry_plan"]["sha256"]
    parent_refs = {item["path"]: item["sha256"] for item in manifests["overlap12"]["spec"]["PB_field_receipts"]}
    parents = {item["key"]: item for item in capture["parents"]}
    assert len(parents) == 7
    margins, parent_checks = [], {}
    for row in plan["rows"]:
        parent = parents[row["parent_source_key"]]
        receipt = parent["receipt"]
        assert parent_refs[parent["receipt_path"]] == parent["receipt_sha256"]
        assert receipt["status"] == parent["ledger_status"] == "completed"
        assert parent["ledger_result_path"] == parent["receipt_path"]
        assert receipt["source"]["source_PQR_sha256"] == prior["variants"]["base"]["PQR_sha256"]
        header, steps = {}, []
        for line in parent["raw_DX_header"]:
            words = line.split()
            if "gridpositions" in words:
                header["dims"] = [int(value) for value in words[-3:]]
            elif words and words[0] == "origin":
                header["origin"] = [float(value) for value in words[1:]]
            elif words and words[0] == "delta":
                steps.append([float(value) for value in words[1:]])
        pp = row["parent_query_plan"]
        expected_origin = np.array(pp["global_origin"]) + .125 * np.array(pp["compute_start"])
        assert header["dims"] == receipt["dims"] == pp["local_dime"]
        assert np.allclose(steps, .125 * np.eye(3), rtol=0, atol=5e-6)
        assert np.allclose(header["origin"], expected_origin, rtol=0, atol=5e-5)
        assert np.allclose(header["origin"], receipt["origin"], rtol=0, atol=5e-5)
        assert np.allclose(receipt["h"], [.125] * 3, rtol=0, atol=5e-6)
        po = np.array(header["origin"])
        pu = po + (np.array(header["dims"]) - 1) * np.diag(steps)
        co = np.array(row["origin"])
        cu = co + np.array(row["length"])
        assert np.allclose(row["length"], .0625 * (np.array(row["local_dime"]) - 1), rtol=0, atol=1e-10)
        margin = np.minimum(co - po, pu - cu)
        assert np.all(margin >= .125 - 1e-5)
        margins.append(float(margin.min()))
        parent_checks[parent["key"]] = {
            "receipt_path": parent["receipt_path"], "receipt_sha256": parent["receipt_sha256"],
            "PB_attempt_id": receipt["PB_attempt_id"], "actual_header_matches_plan": True,
            "ledger_completed_and_bound": True, "DX_sha256_from_native_receipt": receipt["dx_sha256"],
            "full_DX_rehashed_this_batch": False,
        }
    assert len(plan["rows"]) == 68 and plan["full_query_count"] == len(weights)
    result = {
        "schema": "independent_AF_distribution_audit_batch_v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case": args.case, "execution_location": "local_windows",
        "verifier_sha256": sha(__file__), "numpy_version": np.__version__,
        "axes": [{"axis": "overlap", "test": "refined", "reference": "overlap12",
                  "metrics": metrics, "difference_from_recorded": differences, "pass": True}],
        "variants": variants, "shared_bins_sha256": bins_sha,
        "source_qualification_sha256": sha(root / "qualification.json"),
        "previous_area_audit": {"path": str(args.prior_batch.resolve()), "sha256": sha(args.prior_batch),
                                "mesh_and_weights_exact_identity": True, "areas_not_recomputed": True},
        "verified_copied_artifact_bindings": bound,
        "actual_parent_geometry": {
            "pass": True, "parent_fields": 7, "planned_children_checked": 68,
            "minimum_actual_parent_containment_of_planned_child_A": min(margins),
            "actual_child_fields_not_yet_verified": True, "parents": list(parent_checks.values()),
            "header_capture": {"path": str((root / "parent_headers.json").resolve()), "sha256": sha(root / "parent_headers.json")},
            "actual_launch_preflight": {"path": preflight["path"], "sha256": preflight["sha256"]},
        },
        "whole_surface_no_query_subsampling": True, "PB_calls": 0,
        "qualification_scope": "overlap metrics and actual parent headers only; further is not complete",
        "full_DX_and_map_hashes_pending_final_audit": True,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"axes": result["axes"], "parents": len(parent_checks), "minimum_parent_margin_A": min(margins),
                      "source_qualification_sha256": result["source_qualification_sha256"], "verifier_sha256": result["verifier_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
