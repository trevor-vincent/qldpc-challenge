"""Upper-bound a CSS code's distance by folding it onto a symmetric subspace.

For any partition of the n qubits into equal-size orbits, the vectors constant
on orbits are u = O v, and H_Z u = 0 collapses to (H_Z O) v = 0 on n/r columns.
Such a u is a logical exactly when it also escapes the rowspace of H_X, and its
weight is r wt(v). So each partition bounds the distance on a problem a factor
r smaller, with no assumption about the family: an automorphism is what makes a
partition fruitful, never what makes the bound valid.

The fold does not do the searching. It reduces the code and hands the small one
to the trusted distance routine, because the interesting folded logicals are not
sparse in any orbit basis: [[420,20,50]] is refuted by an x^35-invariant logical
of folded weight 6 over 6 orbits, which an enumeration of small basis
combinations cannot reach however many partitions it tries.

Random information-set decoding on the full code is close to blind to these
logicals rather than merely slow: the invariant subspace is not a
positive-measure slice of the information sets it samples. Measured on a cyclic
generalized-bicycle code whose true distance is 40, 32 seeds at 2,000,000 trials
each reported 80.

    python foldbound.py codes/<entry>.json
    python foldbound.py --gb <m> <a-exponents> <b-exponents> [claimed]
"""
import json, os, sys
import numpy as np

MAXFOLD = 160          # folded columns; the search below is cheap at this size
TRIALS = 200_000
SEEDS = (5, 131, 211)


def masks_for(n, b, t):
    s = n // b
    r = s // t
    out = []
    for i in range(b):
        for p in range(t):
            out.append([i * s + p + j * t for j in range(r)])
    return out, r


def partitions(n):
    for b in (1, 2, 3, 4):
        if n % b:
            continue
        s = n // b
        for t in range(1, s + 1):
            if s % t or s // t < 2 or b * t > MAXFOLD:
                continue
            yield b, t


def fold(H, orbits):
    """Column-sum H onto orbits over GF(2)."""
    F = np.zeros((H.shape[0], len(orbits)), dtype=np.int8)
    for j, o in enumerate(orbits):
        F[:, j] = H[:, o].sum(1) % 2
    return F


def bound(n, HX, HZ, repo=None):
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "verify"))
    import gf2_fast
    best = None
    for b, t in partitions(n):
        orbits, r = masks_for(n, b, t)
        FX, FZ = fold(HX, orbits), fold(HZ, orbits)
        if not FX.any() or not FZ.any():
            continue
        for seed in SEEDS:
            try:
                d, side, sup = gf2_fast.distance_rand_witness(
                    FX, FZ, TRIALS, seed, 4, 1)
            except Exception:
                continue
            if not d:
                continue
            w = int(d) * r
            if best is None or w < best[0]:
                # Lift the folded support back and keep only a real logical.
                u = np.zeros(n, dtype=np.int8)
                for j in sup:
                    u[orbits[j]] = 1
                chk = HZ if side == "X" else HX
                if (chk @ u % 2).any():
                    continue
                best = (w, b, t, r, str(side),
                        [int(c) for c in np.nonzero(u)[0]])
    return best


def gb(m, ae, be):
    n = 2 * m
    HX = np.zeros((m, n), dtype=np.int8)
    HZ = np.zeros((m, n), dtype=np.int8)
    for i in range(m):
        for e in ae:
            HX[i, (i + e) % m] ^= 1
            HZ[i, m + ((i - e) % m)] ^= 1
        for e in be:
            HX[i, m + ((i + e) % m)] ^= 1
            HZ[i, (i - e) % m] ^= 1
    return n, HX, HZ


if __name__ == "__main__":
    if sys.argv[1] == "--gb":
        m = int(sys.argv[2])
        ae = [int(x) for x in sys.argv[3].split(",")]
        be = [int(x) for x in sys.argv[4].split(",")]
        n, HX, HZ = gb(m, ae, be)
        claimed = int(sys.argv[5]) if len(sys.argv) > 5 else None
        name = f"GB m={m}"
    else:
        doc = json.load(open(sys.argv[1]))
        n, claimed, name = doc["n"], doc["distance"]["d"], doc["name"]
        def M(rows):
            A = np.zeros((len(rows), n), dtype=np.int8)
            for i, rr in enumerate(rows):
                A[i, rr] = 1
            return A
        HX, HZ = M(doc["checks"]["X"]), M(doc["checks"]["Z"])
    r = bound(n, HX, HZ)
    if r is None:
        print(f"{name}: no folded logical")
        sys.exit(0)
    w, b, t, orb, side, sup = r
    verdict = "REFUTES" if claimed and w < claimed else "clears"
    print(f"{name}: claims {claimed}, fold gives d<={w} ({b} blocks, t={t}, "
          f"orbit {orb}, {side} side) -> {verdict}")
    print(f"  witness weight {len(sup)}: {sup[:14]}")
    sys.exit(1 if claimed and w < claimed else 0)
