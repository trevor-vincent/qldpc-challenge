"""Rebuild the X-shortened CSS code and validate its retained operators."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
from gf2 import rank
from qldpc_verify import verify
from surrogate import validate_logical


def matrix(doc, side):
    result = np.zeros((len(doc["checks"][side]), doc["n"]), dtype=np.int8)
    for i, support in enumerate(doc["checks"][side]):
        result[i, support] = 1
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_witness(hx, hz, side, hit):
    ok, detail = validate_logical(hx, hz, side, hit["weight"], hit["support"])
    assert ok, (side, hit["weight"], detail)


def main():
    recipe = json.loads((HERE / "construction.json").read_text())
    parent_path = ROOT / recipe["parent_path"]
    child_path = ROOT / "codes/681-172-76.json"
    assert digest(parent_path) == recipe["parent_sha256"]
    assert digest(child_path) == recipe["submitted_code_sha256"]
    for group in ("source_screen_raw_sha256", "independent_local_raw_sha256"):
        for name, expected in recipe[group].items():
            assert digest(HERE / name) == expected
    parent = json.loads(parent_path.read_text())
    child = json.loads(child_path.read_text())
    assert verify(parent)["ok"] and verify(child)["ok"]

    parent_x, parent_z = matrix(parent, "X"), matrix(parent, "Z")
    q = recipe["deleted_parent_qubit"]
    assert recipe["measurement_axis"] == "X"
    child_x = np.delete(parent_x, q, axis=1)
    child_z = np.delete(parent_z[parent_z[:, q] == 0], q, axis=1)
    for side, computed in (("X", child_x), ("Z", child_z)):
        assert [list(map(int, np.flatnonzero(row))) for row in computed] == child["checks"][side]
    assert [rank(parent_x), rank(parent_z)] == [255, 255]
    assert [rank(child_x), rank(child_z)] == [255, 254]
    assert child["n"] - rank(child_x) - rank(child_z) == child["k"] == 172

    count = 0
    for side in ("X", "Z"):
        hit = {"weight": parent["distance"][side]["value"],
               "support": parent["distance"][side]["witness"]}
        check_witness(parent_x, parent_z, side, hit)
        count += 1
        for doc in (child, json.loads((HERE / "initial-child.json").read_text())):
            hit = {"weight": doc["distance"][side]["value"],
                   "support": doc["distance"][side]["witness"]}
            check_witness(child_x, child_z, side, hit)
            count += 1
    projected = json.loads((HERE / "projected-witnesses.json").read_text())
    for side, hit in projected.items():
        check_witness(child_x, child_z, side, hit)
        count += 1
    for name in ("fold-witness.json", "general0-witness.json", "general1-witness.json",
                 "general2-witness.json", "general3-witness.json"):
        raw = json.loads((HERE / name).read_text())
        for side, hit in raw.get("witnesses", {}).items():
            check_witness(child_x, child_z, side, hit)
            count += 1
    for name in ("local-100000-raw.json", "local-1000000-raw.json"):
        raw = json.loads((HERE / name).read_text())
        check_witness(child_x, child_z, raw["side"], raw)
        count += 1
    print(json.dumps({"matrices_match_submission": True,
                      "parent_ranks": [255, 255], "child_ranks": [255, 254],
                      "witnesses_validated": count,
                      "distance": "d <= 76; no exact-distance certificate"}))


if __name__ == "__main__":
    main()
