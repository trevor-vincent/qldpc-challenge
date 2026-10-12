"""Solve a saved finite binary proposal problem, without any distance claim."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--seconds", type=float, default=45)
args = parser.parse_args()
data = np.load(args.input)
start = time.monotonic()
solution = milp(
    data["c"],
    integrality=np.ones(len(data["c"])),
    bounds=Bounds(0, 1),
    constraints=LinearConstraint(csc_array(data["A"].astype(float)), data["lower"], data["upper"]),
    options={"time_limit": args.seconds, "mip_rel_gap": 0.1},
)
result = {
    "status": int(solution.status),
    "message": str(solution.message),
    "seconds": time.monotonic() - start,
    "scipy_version": scipy.__version__,
    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "x": None if solution.x is None else solution.x.tolist(),
    "objective": None if solution.fun is None else float(solution.fun),
    "mip_gap": None if solution.get("mip_gap") is None else float(solution.mip_gap),
    "authority": "Finite saved-operator constraints only; no quantum distance certificate.",
}
assert not args.out.exists()
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "x"}), flush=True)
