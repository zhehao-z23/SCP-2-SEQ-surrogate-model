"""Read-only independent AF distribution audit. Prints one cumulative-ready batch.

This verifier does not import project metric, geometry, or sampling functions.
Run with an existing scientific Python. No PB calls or remote files are written.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.dont_write_bytecode = True
import numpy as np

PAIRS = {
    "spacing": ("base", "refined"),
    "phase": ("base", "shifted"),
    "rotation": ("base", "rotation"),
    "expanded_domain": ("base", "expanded"),
    "coarse_parent": ("base", "parent_refined"),
    "surface_resolution": ("base", "surface16"),
    "overlap": ("refined", "overlap12"),
    "further_refinement": ("refined", "further"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--axes", nargs="+", choices=list(PAIRS), required=True)
    args = ap.parse_args()
    root = Path(args.root)
    case = root / "cases" / args.case
    hashes = {}
    records = []
    variants = {}
    arrays = {}
    mesh_audits = {}
    field_checks = {}

    def sha(path):
        key = str(path)
        if key not in hashes:
            h = hashlib.sha256()
            with open(path, "rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                    h.update(chunk)
            hashes[key] = h.hexdigest()
        return hashes[key]

    def load(path):
        return json.loads(Path(path).read_text())

    def check_ref(ref):
        path = Path(ref["path"])
        assert sha(path) == ref["sha256"], str(path)
        if "bytes" in ref:
            assert path.stat().st_size == ref["bytes"], str(path)
        records.append({"path": str(path), "sha256": ref["sha256"]})

    qualification = load(root / "results" / (args.case + "_qualification.json"))
    stored = {row["axis"]: row for row in qualification["comparisons"]}
    ledger = sqlite3.connect(
        "file:" + str(root / "records/budget/PB_attempts.sqlite") + "?mode=ro", uri=True
    )
    needed = sorted({v for axis in args.axes for v in PAIRS[axis]})
    common_bins = None
    for variant in needed:
        path = case / variant
        outcome = load(path / "outcome.json")
        assert outcome["status"] == "completed"
        check_ref(outcome["representation"])
        manifest = load(outcome["representation"]["path"])
        spec = manifest["spec"]
        inputs = spec["inputs"]
        for ref in inputs.values():
            check_ref(ref)
        for product in manifest["products"]:
            check_ref({**product, "path": str(path / "representation" / product["name"])})
        assert manifest["potential_unit"] == "kBT/e"
        assert manifest["area_unit"] == "angstrom^2"
        assert not manifest["potential_clipping_or_rescaling"]
        bins_sha = inputs["shared_bins"]["sha256"]
        common_bins = bins_sha if common_bins is None else common_bins
        assert bins_sha == common_bins
        edges = np.array(load(inputs["shared_bins"]["path"])["internal_edges"])
        assert len(edges) == 62 and np.all(np.diff(edges) > 0)
        physical = load(inputs["protocol_configuration"]["path"])["PB_physical_parameters"]
        source_case = load(inputs["source_case_record"]["path"])
        expected_pqr_sha = sha(source_case["pqr_path"])
        for ref in spec["PB_field_receipts"]:
            check_ref(ref)
            receipt = load(ref["path"])
            assert receipt["status"] == "completed"
            row = ledger.execute(
                "select status,result_path from attempts where id=?", (receipt["PB_attempt_id"],)
            ).fetchone()
            assert row and row[0] == "completed" and Path(row[1]) == Path(ref["path"])
            assert receipt["source"]["source_PQR_sha256"] == expected_pqr_sha
            assert receipt["source"]["physical"] == physical
            input_file = Path(receipt["result"]["cwd"]) / "apbs.in"
            assert sha(input_file) == receipt["source"]["input_sha256"]
            field_checks[ref["path"]] = {
                "receipt_sha256": ref["sha256"], "attempt_id": receipt["PB_attempt_id"],
                "ledger_completed_and_bound": True, "PQR_and_physical_bound": True,
                "APBS_input_actual_hash_matches": True, "DX_rehashed_this_batch": False,
            }
        phi = np.load(inputs["corrected_phi"]["path"])
        weights = np.load(inputs["area_weights"]["path"])
        assert phi.ndim == 1 and phi.shape == weights.shape
        assert np.isfinite(phi).all() and np.isfinite(weights).all()
        assert (weights >= 0).all() and weights.sum() > 0
        hp = np.bincount(np.searchsorted(edges, phi, side="right"), weights=weights, minlength=63)
        hp /= weights.sum()
        exported = np.load(path / "representation/vectors.npz")["corrected_area_p"]
        assert np.allclose(hp, exported, atol=1e-13, rtol=0)
        mesh_key = inputs["vertices"]["sha256"] + ":" + inputs["faces"]["sha256"]
        if mesh_key not in mesh_audits:
            vertices = np.load(inputs["vertices"]["path"])
            faces = np.load(inputs["faces"]["path"])
            area_weights = np.zeros(len(vertices), dtype=np.float64)
            positive = 0
            total_area = 0.0
            for start in range(0, len(faces), 200000):
                ff = faces[start:start + 200000]
                tri = vertices[ff]
                face_area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                assert np.isfinite(face_area).all() and (face_area > 0).all()
                total_area += float(face_area.sum())
                positive += len(face_area)
                area_weights += np.bincount(ff.ravel(), weights=np.repeat(face_area / 3, 3), minlength=len(vertices))
            assert len(weights) == len(vertices)
            assert np.allclose(area_weights, weights, rtol=1e-12, atol=1e-13)
            mesh_audits[mesh_key] = {
                "vertices": len(vertices), "faces": len(faces), "positive_faces": positive,
                "independent_triangle_area_A2": total_area,
                "area_weights_max_abs_difference": float(np.abs(area_weights - weights).max()),
                "weights_sha256": inputs["area_weights"]["sha256"],
                "variants": [variant],
            }
        else:
            assert mesh_audits[mesh_key]["weights_sha256"] == inputs["area_weights"]["sha256"]
            mesh_audits[mesh_key]["variants"].append(variant)
        variants[variant] = {
            "outcome_sha256": sha(path / "outcome.json"),
            "representation_sha256": outcome["representation"]["sha256"],
            "phi_sha256": inputs["corrected_phi"]["sha256"],
            "weights_sha256": inputs["area_weights"]["sha256"],
            "mesh_identity": mesh_key, "PQR_sha256": expected_pqr_sha,
            "queries": len(phi), "area_A2": float(weights.sum()),
            "potential_min_max": [float(phi.min()), float(phi.max())],
            "probability_sum": float(hp.sum()),
            "export_probability_max_abs_difference": float(np.abs(hp - exported).max()),
        }
        arrays[variant] = (phi, weights, hp)
    ledger.close()
    rows = []
    for axis in args.axes:
        a, b = PAIRS[axis]
        assert stored[axis]["test"] == a and stored[axis]["reference"] == b
        x, wx, px = arrays[a]
        y, wy, py = arrays[b]
        support = np.concatenate([x, y])
        signed_mass = np.concatenate([wx / wx.sum(), -wy / wy.sum()])
        order = np.argsort(support, kind="mergesort")
        values = support[order]
        cdf_delta = np.cumsum(signed_mass[order], dtype=np.float64)
        last = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
        unique, delta = values[last], cdf_delta[last]
        metrics = {
            "Hellinger": float(np.sqrt(.5 * np.square(np.sqrt(px) - np.sqrt(py)).sum())),
            "W1_kBT_per_e": float(np.sum(np.abs(delta[:-1]) * np.diff(unique))),
            "CDF_sup": float(np.abs(delta).max()),
        }
        differences = {k: v - stored[axis]["metrics"][k] for k, v in metrics.items()}
        assert max(abs(v) for v in differences.values()) < 1e-11
        passed = metrics["Hellinger"] <= .02 and metrics["W1_kBT_per_e"] <= .1 and metrics["CDF_sup"] <= .01
        assert passed == stored[axis]["pass"]
        rows.append({"axis": axis, "test": a, "reference": b, "metrics": metrics,
                     "difference_from_recorded": differences, "pass": passed})
    rotation = None
    if "rotation" in needed:
        record = load(case / "rotation/rotation.json")
        check_ref(record["source_pqr"])
        check_ref(record["rotated_pqr"])
        def pqr(path):
            parts = [x.split() for x in Path(path).read_text().splitlines() if x.startswith(("ATOM ", "HETATM "))]
            return [tuple(x[:-5]) for x in parts], np.array([[float(v) for v in x[-5:]] for x in parts])
        keys_a, aa = pqr(record["source_pqr"]["path"])
        keys_b, bb = pqr(record["rotated_pqr"]["path"])
        center = np.array(record["center"])
        rot = np.array(record["rotation"])
        assert keys_a == keys_b and np.array_equal(aa[:, 3:], bb[:, 3:])
        assert np.allclose(rot @ rot.T, np.eye(3), rtol=0, atol=1e-14)
        assert abs(np.linalg.det(rot) - 1) < 1e-14
        expected = (aa[:, :3] - center) @ rot.T + center
        error = float(np.abs(expected - bb[:, :3]).max())
        assert error <= 5.1e-10 and not record["reparameterized"]
        axis = np.array([1., 2., 3.]) / np.sqrt(14.)
        assert np.allclose(rot @ axis, axis, rtol=0, atol=1e-14)
        angle = float(np.degrees(np.arccos((np.trace(rot) - 1) / 2)))
        assert abs(angle - 37) < 1e-12
        rotation = {"atoms": len(aa), "identity_charge_radius_exactly_equal": True,
                    "coordinate_max_abs_rounding_error_A": error,
                    "rotation_angle_degrees": angle, "axis_1_2_3_confirmed": True,
                    "PQR_sha256": record["rotated_pqr"]["sha256"]}
    print(json.dumps({
        "schema": "independent_AF_distribution_audit_batch_v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verifier_sha256": globals().get("INDEPENDENT_VERIFIER_SHA256"),
        "case": args.case, "axes": rows, "variants": variants,
        "mesh_audits": mesh_audits, "rotation_identity": rotation,
        "actual_hash_checks": len(records), "unique_files_actually_hashed": len(hashes),
        "all_checked_hashes_match": True, "field_bindings": field_checks,
        "shared_bins_sha256": common_bins, "source_qualification_sha256": sha(root / "results" / (args.case + "_qualification.json")),
        "numeric_implementation": "independent direct bins and full support signed weighted CDF; no project metrics imported",
        "whole_surface_no_query_subsampling": True, "PB_calls": 0,
        "qualification_scope": "requested comparisons only; full eight-axis status is not inferred",
        "full_DX_and_map_hashes_pending_final_audit": True,
    }, indent=2))


if __name__ == "__main__":
    main()
