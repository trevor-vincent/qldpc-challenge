"""Upper-bound a CSS code's distance by folding it onto a symmetric subspace.

For any partition of the n qubits into equal-size orbits, the vectors that are
constant on orbits are u = O v, and the condition H_Z u = 0 collapses to
(H_Z O) v = 0 on n/r columns. Such a u is a logical exactly when it also escapes
the rowspace of H_X, and its weight is r wt(v). So every partition yields an
upper bound computed on a problem a factor r smaller, with no assumption about
the family: an automorphism is what makes a partition fruitful, never what makes
the bound valid.

Random information-set decoding is close to blind to these logicals. The
invariant subspace is not a positive-measure slice of the information sets RIS
samples, so adding seeds or trials does not reach it. Measured on a cyclic
generalized-bicycle code whose true distance is 40: 32 seeds at 2,000,000 trials
each reported 80, while the fold settles it in 0.1s.

    python foldbound.py codes/<entry>.json        one entry
    python foldbound.py --board [shard] [nshard]  every CSS entry on origin/main
"""
import io, json, os, subprocess, sys, tarfile
from itertools import combinations

SHARD = int(sys.argv[2]) if len(sys.argv) > 3 else 0
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MAXFOLD = 48          # folded dimension; the nullspace work is cubic in this
SMAX = 3              # combinations of nullspace basis vectors


def rref(rows):
    """Bit-packed GF(2) row reduction. Returns (pivot -> row) in echelon form."""
    piv = {}
    for r in rows:
        while r:
            b = r.bit_length() - 1
            if b not in piv:
                piv[b] = r
                break
            r ^= piv[b]
    return piv


def in_span(piv, v):
    while v:
        b = v.bit_length() - 1
        if b not in piv:
            return False
        v ^= piv[b]
    return True


def nullspace(rows, ncols):
    """Basis of {v : M v = 0} with M given as bit-packed rows over ncols cols.

    The pivot rows have to be FULLY reduced before back-substitution: with mere
    echelon form the free-variable read-off below is wrong, and it produced 258
    over-claims against the board whose witnesses did not commute.
    """
    piv = {}
    for r in rows:
        rr = r
        while rr:
            b = rr.bit_length() - 1
            if b not in piv:
                piv[b] = rr
                break
            rr ^= piv[b]
    for b in sorted(piv):
        for c in sorted(piv):
            if c != b and piv[c] >> b & 1:
                piv[c] ^= piv[b]
    pcols = set(piv)
    basis = []
    for f in range(ncols):
        if f in pcols:
            continue
        v = 1 << f
        for b in sorted(piv):
            if piv[b] >> f & 1:
                v |= 1 << b
        basis.append(v)
    return basis


def divisors(x):
    return [t for t in range(1, x + 1) if x % t == 0]


def partitions(n):
    """(orbit masks, orbit size) for each equal-block, equal-residue split."""
    for b in (1, 2, 3, 4):
        if n % b:
            continue
        s = n // b
        for t in divisors(s):
            r = s // t
            if r < 2 or b * t > MAXFOLD:
                continue
            masks = []
            for i in range(b):
                for p in range(t):
                    m = 0
                    for j in range(r):
                        m |= 1 << (i * s + p + j * t)
                    masks.append(m)
            yield masks, r, b, t


def fold_side(ker_rows, stab_piv, masks, r, nrows_cols):
    """Least r wt(v) over invariant v in ker, outside the stabilizer group."""
    folded = []
    for row in ker_rows:
        f = 0
        for j, m in enumerate(masks):
            if bin(row & m).count("1") & 1:
                f |= 1 << j
        folded.append(f)
    basis = nullspace(folded, len(masks))
    if not basis:
        return None, None
    best = None
    for s in range(1, SMAX + 1):
        hit = None
        for comb in combinations(range(len(basis)), s):
            v = 0
            for i in comb:
                v ^= basis[i]
            if not v:
                continue
            w = bin(v).count("1") * r
            if best is not None and w >= best:
                continue
            u = 0
            for j in range(len(masks)):
                if v >> j & 1:
                    u |= masks[j]
            if in_span(stab_piv, u):
                continue
            best, hit = w, (u, s)
        if hit:
            break
    return (best, hit[0]) if hit else (best, None)


def commutes(rows, u):
    return all(bin(r & u).count("1") % 2 == 0 for r in rows)


def audit(n, X, Z):
    """Least fold bound over both sides and every partition tried."""
    XR = [sum(1 << c for c in row) for row in X]
    ZR = [sum(1 << c for c in row) for row in Z]
    pX, pZ = rref(XR), rref(ZR)
    best = None
    for masks, r, b, t in partitions(n):
        for ker_rows, stab in ((ZR, pX), (XR, pZ)):
            w, u = fold_side(ker_rows, stab, masks, r, n)
            if w and (best is None or w < best[0]):
                if not commutes(ker_rows, u):
                    raise AssertionError("witness does not commute")
                side = "X" if ker_rows is ZR else "Z"
                best = (w, r, b, t, side,
                        [c for c in range(n) if u >> c & 1])
    return best


def report(path):
    b = json.load(open(path))
    ch = b["checks"]
    r = audit(b["n"], ch["X"], ch["Z"])
    claimed = b["distance"]["d"]
    if r is None:
        print(f'{b["name"]}: no folded logical found'); return 0
    w, orb, blocks, t, side, sup = r
    print(f'{b["name"]}: claims d={claimed}, fold gives d<={w} '
          f'({blocks} blocks, t={t}, orbit {orb}, {side} side)')
    print(f'  witness weight {len(sup)}: {sup[:12]}{" ..." if len(sup) > 12 else ""}')
    return 1 if w < claimed else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--board":
        sys.exit(report(sys.argv[1]))
    OUT = os.environ.get("FOLDBOUND_OUT", "foldbound.json")
    blob = subprocess.run(["git", "archive", "origin/main", "codes/"],
                          capture_output=True).stdout
    ents = []
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for mem in tf.getmembers():
            if not mem.name.endswith(".json"):
                continue
            b = json.load(tf.extractfile(mem))
            ch = b.get("checks", {})
            if not ch.get("X") or not ch.get("Z"):
                continue
            ents.append((os.path.basename(mem.name)[:-5], b["n"], b["k"],
                         b["distance"]["d"], ch["X"], ch["Z"],
                         (b.get("provenance", {}).get("authors") or [None])[0]))
    ents.sort()
    mine = ents[SHARD::NSHARD]
    print(f"{len(mine)} of {len(ents)} CSS entries in shard {SHARD}/{NSHARD}",
          flush=True)
    bad = []
    for i, (name, n, k, d, X, Z, who) in enumerate(mine):
        try:
            fb = audit(n, X, Z)
        except Exception as e:
            print(f"  ERR {name}: {e}", flush=True)
            continue
        if fb and fb[0] < d:
            bad.append({"name": name, "n": n, "k": k, "d_claimed": d,
                        "d_fold": fb[0], "r": fb[1], "blocks": fb[2],
                        "t": fb[3], "side": fb[4], "witness": fb[5],
                        "author": who})
            print(f"  OVER-CLAIM {name:<16} claims {d:>4}  fold "
                  f"b={fb[2]} t={fb[3]} r={fb[1]} gives d<={fb[0]}  {who}",
                  flush=True)
            json.dump(bad, open(OUT, "w"))
        if i % 50 == 49:
            print(f"  .. {i+1}/{len(mine)}  {len(bad)} over-claims", flush=True)
    json.dump(bad, open(OUT, "w"))
    print(f"\n{len(bad)} of {len(mine)} over-claim against the fold bound")
