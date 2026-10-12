"""Transport saved parent logicals under cyclic shifts and CSS duality.

This enumerates a finite orbit of supplied witnesses, not all logical operators.
Every orbit minimum and any improved candidate are preserved before completion.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from reproduce_pair_clifford_662_122 import ROOT, reconstruct

sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import heuristic_distance as hd
from qldpc_verify import verify
from submit import save_submission


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(witness, n):
    v = np.zeros(2 * n, dtype=np.int8)
    v[witness["X"]] = 1
    v[np.asarray(witness["Z"], dtype=int) + n] = 1
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["definition", "bank", "followups", "candidate", "out"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    definition = json.loads(args.definition.read_text())
    bank = json.loads(args.bank.read_text())
    followups = json.loads(args.followups.read_text())
    candidate = json.loads(args.candidate.read_text())
    hx, hz, target_s = reconstruct(definition)
    n, m = candidate["n"], definition["m"]
    parent_a = np.vstack([hx, np.zeros_like(hz)])
    parent_b = np.vstack([np.zeros_like(hx), hz])
    target_a, target_b = hd.stabilizer_matrices(candidate)
    assert np.array_equal(target_s, np.c_[target_a, target_b])
    unique = {}
    for row in bank + followups["variants"]:
        v = vector(row["parent_witness"], n)
        unique.setdefault(v.tobytes(), v)
    permutation = np.r_[m + (-np.arange(m)) % m, (-np.arange(m)) % m]
    report = {
        "status": "running",
        "scope": __doc__,
        "candidate_sha256": digest(args.candidate),
        "definition_sha256": digest(args.definition),
        "bank_sha256": digest(args.bank),
        "followups_sha256": digest(args.followups),
        "script_sha256": digest(Path(__file__)),
        "orbits": [],
        "refutations": [],
    }
    all_weights = []
    current = copy.deepcopy(candidate)

    def persist():
        (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    persist()
    for index, v in enumerate(unique.values()):
        assert hd.valid_pauli_logical(v, parent_a, parent_b)
        dual = np.r_[v[n:][permutation], v[:n][permutation]]
        words = np.asarray(
            [np.roll(base.reshape(2, 2, m), shift, axis=2).reshape(-1) for base in [v, dual] for shift in range(m)],
            dtype=np.int8,
        )
        mapped = words.copy()
        for left, right in definition["pairs"]:
            x1, x2, z1, z2 = (words[:, q] for q in [left, right, n + left, n + right])
            mapped[:, left] = x2 ^ z2
            mapped[:, right] = x1 ^ z1
            mapped[:, n + left] = x1 ^ x2 ^ z1
            mapped[:, n + right] = x1 ^ x2 ^ z2
        weights = (mapped[:, :n] | mapped[:, n:]).sum(axis=1)
        all_weights.append(weights)
        best = int(np.argmin(weights))
        weight = int(weights[best])
        assert hd.valid_pauli_logical(words[best], parent_a, parent_b)
        assert hd.valid_pauli_logical(mapped[best], target_a, target_b)
        report["orbits"].append(
            {
                "source_index": index,
                "images": len(words),
                "source_parent_witness": hd.pauli_witness(v, n),
                "minimum_weight": weight,
                "dual": best >= m,
                "shift": best % m,
                "parent_witness": hd.pauli_witness(words[best], n),
                "target_witness": hd.pauli_witness(mapped[best], n),
            }
        )
        persist()
        if weight < current["distance"]["d"]:
            revised = copy.deepcopy(current)
            revised["distance"] = {
                "d": weight,
                "P": {"value": weight, "confidence": "upper_bound", "witness": hd.pauli_witness(mapped[best], n)},
            }
            revised["name"] = f"[[{n},{current['k']},{weight}]] pair-Clifford image of a cyclic GB code"
            path = args.out / f"{n}-{current['k']}-{weight}-refutation.json"
            assert not save_submission(revised, str(path))
            checked = verify(revised)
            path.with_suffix(".structural.json").write_text(json.dumps(checked, indent=2) + "\n")
            assert checked["ok"], checked
            report["refutations"].append({"weight": weight, "candidate": path.name, "candidate_sha256": digest(path)})
            current = revised
            persist()
    np.savez_compressed(args.out / "orbit-weights.npz", weights=np.asarray(all_weights))
    report.update(
        status="completed",
        unique_parent_words=len(unique),
        images_evaluated=2 * m * len(unique),
        minimum_weight=min(row["minimum_weight"] for row in report["orbits"]),
    )
    persist()
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["status", "unique_parent_words", "images_evaluated", "minimum_weight", "refutations"]
            }
        )
    )


if __name__ == "__main__":
    main()
