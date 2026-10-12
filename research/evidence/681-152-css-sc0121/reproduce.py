"""Reconstruct a CSS shortening and validate its retained logical operators."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
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
    from gf2 import rank
    from qldpc_verify import verify
    from surrogate import validate_logical

    recipe = json.loads((here / "shortening.json").read_text())
    history = json.loads((here / "child-search-history.json").read_text())
    with tempfile.TemporaryDirectory(prefix="qldpc-shortening-reproduction-") as temp:
        output = Path(temp) / "parent.npz"
        command = [sys.executable, str(here / "reproduce_parent.py"), "--repo-root", str(repo), "--output", str(output)]
        if args.replay_construction:
            command += ["--replay-construction", str(args.replay_construction.resolve())]
        parent_result = json.loads(subprocess.check_output(command, text=True))
        with np.load(output) as matrices:
            parent_x, parent_z = matrices["hx"], matrices["hz"]
    parent_path = artifact_root / recipe["parent_path"]
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == recipe["parent_sha256"]
    assert verify(json.loads(parent_path.read_text()))["ok"]
    q, axis = recipe["deleted_qubit"], recipe["measurement_axis"]
    keep_x = np.flatnonzero(parent_x[:, q] == 0) if axis == "Z" else np.arange(parent_x.shape[0])
    keep_z = np.flatnonzero(parent_z[:, q] == 0) if axis == "X" else np.arange(parent_z.shape[0])
    assert list(map(int, keep_x)) == recipe["kept_parent_X_rows"]
    assert list(map(int, keep_z)) == recipe["kept_parent_Z_rows"]
    hx, hz = np.delete(parent_x[keep_x], q, axis=1), np.delete(parent_z[keep_z], q, axis=1)
    assert rank(hx) == rank(parent_x) - (axis == "Z")
    assert rank(hz) == rank(parent_z) - (axis == "X")
    source = artifact_root / history["code_path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == history["code_sha256"]
    doc = json.loads(source.read_text())
    for side, matrix in (("X", hx), ("Z", hz)):
        assert [list(map(int, np.flatnonzero(row))) for row in matrix] == doc["checks"][side]
    assert verify(doc)["ok"]
    count = 0
    for group in [
        history["initial"]["witnesses"],
        history["projected"]["witnesses"],
        *[r["witnesses"] for r in history["records"]],
        {s: {"weight": doc["distance"][s]["value"], "support": doc["distance"][s]["witness"]} for s in ("X", "Z")},
    ]:
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
                "parent_reproduction": parent_result,
                "child_retained_witnesses_validated": count,
                "deleted_parent_qubit": q,
                "measurement_axis": axis,
                "child_ranks": {"X": int(rank(hx)), "Z": int(rank(hz))},
                "distance_claim": f"d <= {doc['distance']['d']}; no exact-distance certificate",
            }
        )
    )


if __name__ == "__main__":
    main()
