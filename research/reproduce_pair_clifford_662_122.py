"""Reconstruct a disclosed two-qubit Clifford image and verify its retained witnesses.

Run from a qldpc-challenge checkout, with --definition, --parent, --candidate,
and --out pointing to the committed construction data and the desired new report.
This is deterministic matrix/witness verification, not a distance lower bound.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "verify/gf2.py").is_file())
sys.path.insert(0, str(ROOT / "verify"))
import heuristic_distance as hd
from gf2 import rank
from qldpc_verify import verify


def supports(rows, n):
    result = np.zeros((len(rows), n), dtype=np.int8)
    for i, row in enumerate(rows):
        result[i, row] = 1
    return result


def reconstruct(definition):
    m = definition["m"]
    base_a, base_b = supports([definition["a"]], m)[0], supports([definition["b"]], m)[0]
    ca = np.array([np.roll(base_a, i) for i in range(m)])
    cb = np.array([np.roll(base_b, i) for i in range(m)])
    hx, hz = np.c_[ca, cb], np.c_[cb.T, ca.T]
    parent = np.block([[hx, np.zeros_like(hx)], [np.zeros_like(hz), hz]])
    transformed = parent.copy()
    n = 2 * m
    pairs = definition["pairs"]
    assert len({q for pair in pairs for q in pair}) == 2 * len(pairs)
    assert all(0 <= q < n for pair in pairs for q in pair)
    assert definition["pauli_images"] == {"X1": "Z1Y2", "X2": "Y1Z2", "Z1": "Z1X2", "Z2": "X1Z2"}
    # Simultaneous binary coordinate equations; images are defined up to Pauli phase.
    for left, right in pairs:
        x1, x2, z1, z2 = (parent[:, i] for i in [left, right, n + left, n + right])
        transformed[:, left] = x2 ^ z2
        transformed[:, right] = x1 ^ z1
        transformed[:, n + left] = x1 ^ x2 ^ z1
        transformed[:, n + right] = x1 ^ x2 ^ z2
    return hx, hz, transformed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), "Use a new output path to preserve earlier reports."
    definition = json.loads(args.definition.read_text())
    parent = json.loads(args.parent.read_text())
    candidate = json.loads(args.candidate.read_text())
    hx, hz, s = reconstruct(definition)
    n = 2 * definition["m"]
    assert np.array_equal(hx, supports(parent["checks"]["X"], n))
    assert np.array_equal(hz, supports(parent["checks"]["Z"], n))
    assert candidate["code_type"] == "stabilizer"
    a, b = hd.stabilizer_matrices(candidate)
    assert np.array_equal(np.c_[a, b], s)
    assert not ((a @ b.T + b @ a.T) % 2).any()
    k = n - rank(s)
    assert k == definition["k"] == candidate["k"] == parent["k"]
    check_weight = int((a | b).sum(axis=1).max())
    assert check_weight == definition["max_check_weight"] <= 32
    parent_result = verify(parent)
    candidate_result = verify(candidate)
    report = {
        "scope": "Exact construction and witness checks; no distance lower bound.",
        "construction_validated": True,
        "parent_structural": parent_result,
        "candidate_structural": candidate_result,
        "n": n,
        "k": k,
        "witnessed_upper_bound": candidate["distance"]["d"],
        "check_weight": check_weight,
        "pair_count": len(definition["pairs"]),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "parent_sha256": hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        "definition_sha256": hashlib.sha256(args.definition.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    assert parent_result["ok"] and candidate_result["ok"], "Inspect the preserved structural reports."
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "construction_validated",
                    "n",
                    "k",
                    "witnessed_upper_bound",
                    "check_weight",
                    "pair_count",
                    "candidate_sha256",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
