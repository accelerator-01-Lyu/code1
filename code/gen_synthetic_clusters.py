"""gen_synthetic_clusters.py — E1 synthetic cluster-instance generator.

Generates 72 controlled Gaussian-mixture instances for the "synthetic
clustering ensemble (E1)" experiment.  Paper §5.3 cites the overlap definition
implemented here.

Overlap definition (FROZEN, cited in paper §5.3)
------------------------------------------------
For two clusters with centres m_1, m_2, centre distance D = ||m_1 - m_2|| and
within-cluster covariance matrices Sigma_1, Sigma_2:

    overlap = 1 - D / ( sqrt(tr Sigma_1) + sqrt(tr Sigma_2) ).

* overlap = 0  -> the two centres sit exactly at the just-touching boundary
  (D = sqrt(tr Sigma_1) + sqrt(tr Sigma_2)): fully separated.
* overlap = 0.9 -> D = 0.1 * (sqrt(tr Sigma_1)+sqrt(tr Sigma_2)): highly
  overlapping.

Multi-class control
-------------------
For c > 2 clusters we place c centres on a unit d-sphere with a deterministic
repulsive ("greedy placement") relaxation so that pairwise angular distances are
as even as possible, then trace-normalise every per-cluster covariance so that
sqrt(tr Sigma_i) = 1 for every i (covariances remain RANDOM in shape/orientation:
Sigma_i = A_i A_i^T with A_i ~ N(0,1), then trace-normalised).  Because every
pair has the same sqrt(tr Sigma_i)+sqrt(tr Sigma_j) = 2, the design reduces to a
single scale factor lambda on all centre distances:

    overlap_ij = 1 - lambda * D0_ij / 2,   D0_ij = ||m_i - m_j|| (on sphere).

We set lambda = (1 - overlap_target) / max_{i<j} (D0_ij / 2), i.e. the WORST
(normally farthest) pair sits EXACTLY at the design overlap; all other pairs are
at least as overlapping as the target.  Because the repulsive placement makes
D0_ij nearly equal, the realised pairwise overlaps concentrate tightly around the
target (deviation statistics are printed and stored in meta.json).  This is the
"iterative centre shrinkage / greedy placement" route specified for E1.

Block factor
------------
* balanced   : equal sample sizes n_i ~ N/c (remainder spread round-robin).
* imbalanced : n_i proportional to Zipf(1.2) weights w_i = 1/i^1.2 (i=1..c),
               rounded to sum exactly N.

Instance numbering (documented rule)
------------------------------------
instance_id is 1-based, 1..72.  Overlap level k in {0,1,2,3} maps to
overlap_levels[k] = [0.0, 0.3, 0.6, 0.9][k] and owns ids 18k+1 .. 18k+18.
Within each overlap level the 18 instances follow the fixed SCHEDULE table below
(same (block, c, d, N) grid at every overlap, so overlap effects are clean).
Per-instance data seed = 1000 + instance_id (numpy RandomState, not
default_rng, for bit-level reproducibility).

The three downstream-E2 anchor combos (c=2,d=5), (c=5,d=10), (c=8,d=20) appear
in every overlap level with N=2000 (both block settings).
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "synthetic")

OVERLAP_LEVELS = [0.0, 0.3, 0.6, 0.9]

# 18 rows per overlap: (block, c, d, N). Deterministic design table.
# Rows 1..6 are the three E2 anchor (c,d) combos, both blocks, N=2000.
SCHEDULE = [
    ("balanced",   2,  5,  2000),   # row  1  anchor (c=2,d=5)
    ("imbalanced", 2,  5,  2000),   # row  2  anchor (c=2,d=5)
    ("balanced",   5,  10, 2000),   # row  3  anchor (c=5,d=10)
    ("imbalanced", 5,  10, 2000),   # row  4  anchor (c=5,d=10)
    ("balanced",   8,  20, 2000),   # row  5  anchor (c=8,d=20)
    ("imbalanced", 8,  20, 2000),   # row  6  anchor (c=8,d=20)
    ("balanced",   3,  2,  500),    # row  7
    ("imbalanced", 3,  2,  500),    # row  8
    ("balanced",   3,  5,  2000),   # row  9
    ("imbalanced", 3,  5,  2000),   # row 10
    ("balanced",   2,  10, 5000),   # row 11
    ("imbalanced", 2,  10, 5000),   # row 12
    ("balanced",   5,  2,  500),    # row 13
    ("imbalanced", 5,  2,  500),    # row 14
    ("balanced",   8,  5,  5000),   # row 15
    ("imbalanced", 8,  5,  5000),   # row 16
    ("balanced",   3,  20, 5000),   # row 17
    ("imbalanced", 3,  20, 5000),   # row 18
]


def place_centers_evenly(c, d, rng, n_iter=400, step=0.15):
    """c points nearly evenly spread on the unit d-sphere via repulsive
    relaxation (greedy placement).  Falls back gracefully when c > d+1
    (points live on a (d-1)-sphere, not a regular simplex)."""
    Z = rng.randn(c, d)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    for _ in range(n_iter):
        F = np.zeros((c, d))
        for i in range(c):
            diff = Z[i] - Z
            dist2 = (diff ** 2).sum(1) + 1e-8
            dist2[i] = 1.0
            F[i] = (diff / dist2[:, None]).sum(0)
        Z += step * F
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return Z


def make_random_psd(d, rng, trace=1.0):
    """Random full PSD covariance Sigma = A A^T (random orientation/eigenspectrum),
    rescaled so tr(Sigma) = trace (=> sqrt(tr Sigma) = sqrt(trace))."""
    A = rng.randn(d, d)
    S = A @ A.T
    S = S / np.trace(S) * trace
    return S


def class_sizes(c, N, block, rng):
    if block == "balanced":
        base, rem = divmod(N, c)
        n = np.full(c, base, dtype=int)
        n[:rem] += 1
        return n
    w = 1.0 / (np.arange(1, c + 1) ** 1.2)          # Zipf(1.2)
    w = w / w.sum()
    raw = w * N
    n = np.floor(raw).astype(int)
    rem = N - n.sum()
    # distribute remaining samples to the largest fractional parts
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    n[order[:rem]] += 1
    return n


def realized_overlaps(centers, sigmas):
    """Population-level pairwise overlap values.  sigmas: list of Sigma matrices.
    Returns upper-triangle overlap array."""
    c = len(sigmas)
    s = np.array([np.sqrt(np.trace(S)) for S in sigmas])
    ov = np.zeros((c, c))
    for i in range(c):
        for j in range(i + 1, c):
            D = np.linalg.norm(centers[i] - centers[j])
            ov[i, j] = 1.0 - D / (s[i] + s[j])
    return ov[np.triu_indices(c, 1)]


def gen_instance(instance_id, overlap_target, block, c, d, N):
    seed = 1000 + instance_id
    rng = np.random.RandomState(seed)

    # --- per-cluster covariances: random PSD, trace-normalised (s_i = 1) --- #
    sigmas = [make_random_psd(d, rng, trace=1.0) for _ in range(c)]
    chols = [np.linalg.cholesky(S + 1e-10 * np.eye(d)) for S in sigmas]

    # --- centres: even on sphere, then scaled to hit the design overlap ---- #
    Z = place_centers_evenly(c, d, rng)
    s = np.array([np.sqrt(np.trace(S)) for S in sigmas])     # all == 1.0
    # pair normalised distances on the unit sphere
    D0 = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    denom = s[:, None] + s[None, :]
    with np.errstate(invalid="ignore"):
        rmat = D0 / denom
    iu = np.triu_indices(c, 1)
    max_r = float(rmat[iu].max())
    lam = (1.0 - overlap_target) / max_r
    centers = lam * Z

    ov_real = realized_overlaps(centers, sigmas)

    # --- sample class counts and points ------------------------------------ #
    n_per = class_sizes(c, N, block, rng)
    Xs, ys = [], []
    for k in range(c):
        Xk = centers[k] + rng.standard_normal((n_per[k], d)) @ chols[k].T
        Xs.append(Xk.astype(np.float64))
        ys.append(np.full(n_per[k], k, dtype=int))
    X = np.vstack(Xs).astype(np.float64)
    y = np.concatenate(ys).astype(int)
    # deterministic row shuffle so class blocks are not contiguous
    perm = rng.permutation(N)
    X = X[perm]; y = y[perm]

    meta = {
        "instance_id": instance_id,
        "overlap_target": overlap_target,
        "overlap_realized_min": float(ov_real.min()),
        "overlap_realized_mean": float(ov_real.mean()),
        "overlap_realized_max": float(ov_real.max()),
        "overlap_dev_abs_max": float(np.abs(ov_real - overlap_target).max()),
        "block": block, "c": c, "d": d, "N": N,
        "n_per_class": n_per.tolist(),
        "seed": seed, "scale_lambda": lam,
        "schedule_row": ((instance_id - 1) % 18) + 1,
    }
    return X, y, meta


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows_log = []
    for k, ov in enumerate(OVERLAP_LEVELS):
        for row_idx, (block, c, d, N) in enumerate(SCHEDULE):
            iid = k * 18 + row_idx + 1
            X, y, meta = gen_instance(iid, ov, block, c, d, N)
            dpath = os.path.join(DATA_DIR, f"instance_{iid:03d}")
            os.makedirs(dpath, exist_ok=True)
            np.save(os.path.join(dpath, "X.npy"), X)
            np.save(os.path.join(dpath, "y_true.npy"), y)
            with open(os.path.join(dpath, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            rows_log.append(meta)
            print(f"[gen] id={iid:3d} ov={ov:.2f} {block:10s} c={c} d={d} "
                  f"N={N:4d} | realized ov min={meta['overlap_realized_min']:+.3f} "
                  f"mean={meta['overlap_realized_mean']:+.3f} "
                  f"max={meta['overlap_realized_max']:+.3f} "
                  f"| n_per={meta['n_per_class']}", flush=True)

    # --- self-check: deviation of realised population overlap vs target --- #
    print("\n=== self-check: realised-overlap deviation vs target ===")
    for ov in OVERLAP_LEVELS:
        sub = [m for m in rows_log if abs(m["overlap_target"] - ov) < 1e-12]
        devs = np.array([m["overlap_dev_abs_max"] for m in sub])
        mins = np.array([m["overlap_realized_min"] for m in sub])
        print(f"target ov={ov:.2f}: n={len(sub)}  "
              f"|max|dev| mean={devs.mean():.4f} max={devs.max():.4f} | "
              f"realized-min mean={mins.mean():+.4f} range=[{mins.min():+.4f},{mins.max():+.4f}]",
              flush=True)
    print(f"\n[gen] done: {len(rows_log)} instances -> {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
