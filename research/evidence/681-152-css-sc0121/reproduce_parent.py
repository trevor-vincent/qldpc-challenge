"""Reconstruct cyclic checks and recheck the retained logical operators."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-construction", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    artifact_root = here.parents[2]
    repo = args.repo_root.resolve() if args.repo_root else artifact_root
    sys.path[:0] = [str(repo / "research/kit"), str(repo / "verify")]
    from gf2poly import circulant, cyclic_gb, div, from_support, gcd, mod, reciprocal, to_vector, x_pow_m_minus_1
    from surrogate import validate_logical
    from two_block_equivalence import normalise

    recipe = json.loads((here / "construction.json").read_text())
    history = json.loads((here / "search-history.json").read_text())
    m = recipe["m"]
    modulus = x_pow_m_minus_1(m)
    parent = recipe["parent"]
    g = gcd(gcd(from_support(parent["a"], m), from_support(parent["b"], m)), modulus)
    assert hex(g) == recipe["g"]
    kernel = circulant(to_vector(reciprocal(div(modulus, g)), m))
    assert hashlib.sha256(kernel.tobytes()).hexdigest() == recipe["kernel_matrix_sha256"]
    identity = np.eye(m, dtype=np.int8)
    checks = []
    replays = []
    if args.replay_construction:
        import gf2_fast

        args.replay_construction.mkdir(parents=True, exist_ok=False)
    ancestor = recipe["ancestor"]
    ancestor_g = gcd(gcd(from_support(ancestor["a"], m), from_support(ancestor["b"], m)), modulus)
    assert hex(ancestor_g) == ancestor["g"]
    ancestor_kernel = circulant(to_vector(reciprocal(div(modulus, ancestor_g)), m))
    assert hashlib.sha256(ancestor_kernel.tobytes()).hexdigest() == ancestor["kernel_matrix_sha256"]
    sample = ancestor["sample"]
    assert mod(from_support(sample["support"], m), ancestor_g) == 0
    assert sum(sample["coordinate_weights"][q] for q in sample["support"]) == sample["weighted_objective"]
    ancestor_word = normalise(sample["support"], m)
    derived_parent = [
        list(normalise([u * q % m for q in ancestor_word], m)) for u in recipe["parent_frobenius_multipliers"]
    ]
    assert derived_parent == [parent["a"], parent["b"]]
    assert mod(g, ancestor_g) == 0
    if args.replay_construction:
        parameters = dict(ancestor["kernel_parameters"], seed=sample["seed"])
        weighted, support = gf2_fast.fold_rand_witness(
            ancestor_kernel, identity, sample["coordinate_weights"], **parameters
        )
        retained = {
            "purpose": "Classical ancestor check construction only",
            "parameters": parameters,
            "coordinate_weights": sample["coordinate_weights"],
            "weighted_objective": weighted,
            "support": list(map(int, support)),
        }
        path = args.replay_construction / "ancestor-sample-raw.json"
        path.write_text(json.dumps(retained, indent=2) + "\n")
        assert weighted is not None and retained["support"]
        assert mod(from_support(retained["support"], m), ancestor_g) == 0
        assert sum(sample["coordinate_weights"][q] for q in retained["support"]) == weighted
        replays.append(
            {
                "seed": sample["seed"],
                "weight": len(retained["support"]),
                "same_normalized_support": normalise(retained["support"], m) == ancestor_word,
                "retained_raw": str(path),
            }
        )
    for index, derivation in enumerate(recipe["derivations"]):
        original = derivation["original_support"]
        assert mod(from_support(original, m), g) == 0
        if derivation["kind"] == "saved source polynomial":
            assert original in (parent["a"], parent["b"])
        else:
            assert derivation["kind"] == "native shortened kernel"
            sample = derivation["sample"]
            raw_support = sample["support"]
            assert len(raw_support) == sample["returned_weight"]
            assert not set(raw_support).intersection(sample["zero_coordinates"])
            assert list(normalise(raw_support, m)) == original
            if args.replay_construction:
                parameters = dict(recipe["kernel_parameters"], trials=sample["trials"], seed=sample["seed"])
                shortened = np.concatenate([kernel, identity[sample["zero_coordinates"]]], axis=0)
                weight, support = gf2_fast.fold_rand_witness(shortened, identity, [1] * m, **parameters)
                retained = {
                    "purpose": "Classical parity-check construction only",
                    "parameters": parameters,
                    "zero_coordinates": sample["zero_coordinates"],
                    "returned_weight": weight,
                    "support": list(map(int, support)),
                }
                path = args.replay_construction / f"sample{index}-raw.json"
                path.write_text(json.dumps(retained, indent=2) + "\n")
                assert weight is not None and retained["support"]
                assert len(retained["support"]) == weight
                assert not set(retained["support"]).intersection(sample["zero_coordinates"])
                assert mod(from_support(retained["support"], m), g) == 0
                replays.append(
                    {
                        "seed": sample["seed"],
                        "weight": weight,
                        "same_normalized_support": list(normalise(retained["support"], m)) == original,
                        "retained_raw": str(path),
                    }
                )
        checks.append(list(normalise([derivation["frobenius_multiplier"] * q % m for q in original], m)))
    assert checks == [recipe["a"], recipe["b"]]
    hx, hz = cyclic_gb(m, *checks)
    source = artifact_root / history["code_path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == history["code_sha256"]
    doc = json.loads(source.read_text())
    for side, matrix in (("X", hx), ("Z", hz)):
        assert [list(map(int, np.flatnonzero(row))) for row in matrix] == doc["checks"][side]
    groups = [history["initial"]["witnesses"], *[r["witnesses"] for r in history["records"]]]
    groups.append(
        {s: {"weight": doc["distance"][s]["value"], "support": doc["distance"][s]["witness"]} for s in ("X", "Z")}
    )
    count = 0
    for group in groups:
        for side, hit in group.items():
            valid, reason = validate_logical(hx, hz, side, hit["weight"], hit["support"])
            assert valid, (side, hit["weight"], reason)
            count += 1
    if args.output:
        np.savez(args.output, hx=hx, hz=hz)
    print(
        json.dumps(
            {
                "matrices_match_submission": True,
                "retained_witnesses_validated": count,
                "distance_claim": f"d <= {doc['distance']['d']}; no exact-distance certificate",
                "construction_replay": replays,
            }
        )
    )


if __name__ == "__main__":
    main()
