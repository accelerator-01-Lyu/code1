# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""make_synthetic_3d.py — generate isolated 3-D synthetic benchmarks for the
post-hoc validation of Theorem 2 (double counting) and Proposition 2
(Fisher-Rao boundary detection).

ISOLATION:
  * All pkl + manipulation_check.csv live in results/synthetic_3d/.
  * Data-gen seed is FIXED at 20261010, independent of the AL seeds
    (20261010..20261019).  Each (dataset, sigma) pkl is generated ONCE and
    reused by all 10 AL seeds.
  * pkl schema mirrors reference_repo/data/ecoli.pkl:
        {'X': float64(N,d), 'y': int(N,)}
    with y in {0..K-1} normal clusters and y == -1 eccentric decoy points.

GEOMETRIES (K=6, 200 normal points / cluster = 1200 normal, +10 eccentric):
  A. Spiral Skewer  : mu_k = (R cos th_k, R sin th_k, alpha th_k),
                      th_k = k*2pi/6, R=2, alpha=1; cluster points = mu_k + sigma*N(0,I).
  B. Swiss Roll     : (t cos t, t sin t, t) + sigma*N(0,I), t in [0,4pi],
                      split into 6 contiguous t-bands of 200 points.
  C. Hourglass      : cone r = |z|, z in [-3,3]; upper z in [0,3] and lower
                      z in [-3,0] each split into 3 bands of 200 points.

ECCENTRIC DECOYS (10 per pkl, y=-1, coordinates fixed across AL seeds):
  Spiral   : cluster-0 centroid mu0, pushed radially away from the z-axis
             along mu0/||mu0||.  mu0 = (2,0,0) so the outward direction is
             (+x).  Target = mu0 + 3 sigma * (1,0,0).
  SwissRoll: t-min band (cluster 0) centroid; pushed along the LOCAL OUTWARD
             RADIAL direction of the roll.  At small t the roll's tangent in
             the xy-plane points azimuthally, while "away from the manifold"
             means radially outward from the z-axis: dir = (cos tbar, sin tbar, 0),
             tbar = mean t of cluster 0.  Target = centroid + 3 sigma * dir.
  Hourglass: upper cone rim at z ~= 3, pushed radially outward in xy:
             dir = (cos phi0, sin phi0, 0) with phi0 = 0  => (+x,0).
             Target = (3,0,3) + 3 sigma * (1,0,0).

  Jitter (deterministic, from the data-gen RNG) so the 10 decoys never
  coincide: radial scatter in [2.5 sigma, 3.5 sigma] along the outward dir,
  orthogonal jitter <= 0.2 sigma in the plane perpendicular to dir.

USAGE
-----
    python make_synthetic_3d.py            # generate all 9 pkl + manipulation check
    python make_synthetic_3d.py --quick    # only spiral sigma=0.6 (first-report probe)
"""
import os
import argparse
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "synthetic_3d")
DATA_GEN_SEED = 20261010
K = 6
PER = 200            # normal points per cluster
N_ECC = 10           # eccentric decoys per dataset

# palettes / geometry constants (documented)
R_SPIRAL = 2.0
ALPHA_SPIRAL = 1.0
T_MAX = 4.0 * np.pi
Z_HOUR = 3.0


# --------------------------------------------------------------------------- #
# Geometry builders.  Each returns X_norm (1200,3), y_norm (1200,).
# --------------------------------------------------------------------------- #
def build_spiral(rng, sigma):
    X, y = [], []
    for k in range(K):
        th = k * 2.0 * np.pi / K
        mu = np.array([R_SPIRAL * np.cos(th), R_SPIRAL * np.sin(th),
                        ALPHA_SPIRAL * th])
        X.append(mu + sigma * rng.randn(PER, 3))
        y.append(np.full(PER, k, dtype=int))
    return np.vstack(X), np.concatenate(y)


def build_swissroll(rng, sigma):
    X, y = [], []
    band = T_MAX / K
    for k in range(K):
        t = k * band + rng.uniform(0.0, band, size=PER)
        pts = np.column_stack([t * np.cos(t), t * np.sin(t), t])
        X.append(pts + sigma * rng.randn(PER, 3))
        y.append(np.full(PER, k, dtype=int))
    return np.vstack(X), np.concatenate(y)


def build_hourglass(rng, sigma):
    X, y = [], []
    half = Z_HOUR / 3.0           # 3 bands per cone
    # upper cone z in [0,3]: bands [0,1],[1,2],[2,3] -> clusters 0,1,2
    for k in range(3):
        z = k * half + rng.uniform(0.0, half, size=PER)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=PER)
        pts = np.column_stack([z * np.cos(phi), z * np.sin(phi), z])
        X.append(pts + sigma * rng.randn(PER, 3))
        y.append(np.full(PER, k, dtype=int))
    # lower cone z in [-3,0]: bands [-1,0],[-2,-1],[-3,-2] -> clusters 3,4,5
    for k in range(3):
        z = -(k * half + rng.uniform(0.0, half, size=PER))
        r = -z
        phi = rng.uniform(0.0, 2.0 * np.pi, size=PER)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        X.append(pts + sigma * rng.randn(PER, 3))
        y.append(np.full(PER, k + 3, dtype=int))
    return np.vstack(X), np.concatenate(y)


# --- D. helical_cone: spiral up a cone, radius shrinks 3 -> 0.5, z in [0,6].
# 6 centroids evenly spaced along the conical spiral (1.5 turns, fixed).
H_CONE = 6.0
R_CONE_WIDE = 3.0
R_CONE_NARROW = 0.5
CONE_TURNS = 1.5


def build_helical_cone(rng, sigma):
    X, y = [], []
    for k in range(K):
        z = H_CONE * k / K
        r = R_CONE_WIDE + (R_CONE_NARROW - R_CONE_WIDE) * (z / H_CONE)
        th = k * (CONE_TURNS * 2.0 * np.pi / K)
        mu = np.array([r * np.cos(th), r * np.sin(th), z])
        X.append(mu + sigma * rng.randn(PER, 3))
        y.append(np.full(PER, k, dtype=int))
    return np.vstack(X), np.concatenate(y)


# --- E. spiral_saddle: centroids on saddle z = x^2 - y^2, planar spiral r 0.5->3.
R_SADDLE_MIN = 0.5
R_SADDLE_MAX = 3.0


def build_spiral_saddle(rng, sigma):
    X, y = [], []
    for k in range(K):
        r = R_SADDLE_MIN + (R_SADDLE_MAX - R_SADDLE_MIN) * k / (K - 1)
        th = k * 2.0 * np.pi / K
        x = r * np.cos(th); y2 = r * np.sin(th)
        z = x * x - y2 * y2
        mu = np.array([x, y2, z])
        X.append(mu + sigma * rng.randn(PER, 3))
        y.append(np.full(PER, k, dtype=int))
    return np.vstack(X), np.concatenate(y)


# --------------------------------------------------------------------------- #
# Eccentric decoys.  Given the normal geometry, return X_ecc (10,3).
# outward_dir is a UNIT vector pointing away from the manifold.
# --------------------------------------------------------------------------- #
def _scatter_decoy(rng, base_point, outward_dir, sigma):
    """One decoy near base_point + ~3 sigma * outward_dir.
    radial offset along outward_dir ~ U(2.5 sigma, 3.5 sigma);
    orthogonal jitter in the perpendicular plane, |.| <= 0.2 sigma.
    """
    radial = rng.uniform(2.5 * sigma, 3.5 * sigma)
    # random orthogonal direction
    v = rng.randn(3)
    v = v - np.dot(v, outward_dir) * outward_dir
    nrm = np.linalg.norm(v) + 1e-12
    v = v / nrm
    orth = rng.uniform(0.0, 0.2 * sigma)
    return base_point + radial * outward_dir + orth * v


def eccentric_points(kind, X_norm, y_norm, rng, sigma):
    if kind == "spiral":
        # cluster 0 centroid = mu0 = (R,0,0); outward = (+x)
        c0 = X_norm[y_norm == 0].mean(axis=0)
        d = np.array([1.0, 0.0, 0.0])
        base = c0
    elif kind == "swissroll":
        # cluster 0 = t-min band.  local outward radial in xy = (cos tbar, sin tbar, 0)
        c0 = X_norm[y_norm == 0].mean(axis=0)
        # recover tbar ~= mean z of this band (z = t on the noiseless roll)
        tbar = c0[2]
        d = np.array([np.cos(tbar), np.sin(tbar), 0.0])
        d = d / (np.linalg.norm(d) + 1e-12)
        base = c0
    elif kind == "hourglass":
        # The hourglass bands each span the FULL azimuth phi in [0,2pi], so
        # every cluster centroid collapses onto the z-axis at z = band
        # midpoint: upper bands 0,1,2 -> z~=+0.5,+1.5,+2.5; lower bands
        # 3,4,5 -> z~=-0.5,-1.5,-2.5.  Pushing a decoy radially outward at the
        # rim (old geometry) adds the SAME large r^2 distance to ALL z-axis
        # centroids -> nearly equidistant -> near-uniform membership (high H,
        # low umax), which failed the manipulation check.
        # Theorem-2 target instead: pick the TOP upper band (cluster 2,
        # z in [2,3], centroid ~ (0,0,2.5)) and push the decoy FURTHER ALONG +z
        # beyond z=3.  This direction is simultaneously (i) away from the z=0
        # junction where upper/lower cones meet, and (ii) away from the
        # adjacent band cluster 1 below it.  The z-axis separation between
        # centroids (1.0 per band) then makes cluster 2 the clear nearest
        # centroid -> peaked, low-entropy, high-eccentricity membership.
        c2 = X_norm[y_norm == 2].mean(axis=0)
        base = c2
        d = np.array([0.0, 0.0, 1.0])     # +z: away from z=0 junction
    elif kind == "helical_cone":
        # Wide end R=3 at z=0 = cluster 0 centroid.  Push radially outward in
        # xy at theta_0=0 => (+x,0), away from the z-axis and away from the
        # narrowing cone above.
        c0 = X_norm[y_norm == 0].mean(axis=0)
        base = c0
        d = np.array([1.0, 0.0, 0.0])
    elif kind == "spiral_saddle":
        # Outermost ring cluster (r=3.0, k=5).  Push radially outward in xy
        # along its azimuth (theta_5 = 5*2pi/6), away from the saddle bowl.
        c5 = X_norm[y_norm == 5].mean(axis=0)
        base = c5
        th = 5 * 2.0 * np.pi / K
        d = np.array([np.cos(th), np.sin(th), 0.0])
        d = d / (np.linalg.norm(d) + 1e-12)
    else:
        raise ValueError(kind)
    pts = np.array([_scatter_decoy(rng, base, d, sigma) for _ in range(N_ECC)])
    return pts


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def make_one(kind, sigma, rng):
    if kind == "spiral":
        Xn, yn = build_spiral(rng, sigma)
    elif kind == "swissroll":
        Xn, yn = build_swissroll(rng, sigma)
    elif kind == "hourglass":
        Xn, yn = build_hourglass(rng, sigma)
    elif kind == "helical_cone":
        Xn, yn = build_helical_cone(rng, sigma)
    elif kind == "spiral_saddle":
        Xn, yn = build_spiral_saddle(rng, sigma)
    Xe = eccentric_points(kind, Xn, yn, rng, sigma)
    ye = np.full(N_ECC, -1, dtype=int)
    X = np.vstack([Xn.astype(np.float64), Xe.astype(np.float64)])
    y = np.concatenate([yn.astype(int), ye])
    fname = f"{kind}_sigma{sigma}.pkl"
    path = os.path.join(OUT, fname)
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y,
                     "posthoc": "Post-hoc Mechanism Validation — not part of "
                                "pre-registered P1-P4 verdicts",
                     "data_gen_seed": DATA_GEN_SEED,
                     "sigma": sigma, "kind": kind}, f)
    return path, X, y


def manipulation_check(X, y, kind, sigma):
    """Unsupervised FCM (c=K) memberships over ALL points; report per eccentric
    decoy u, H(u), h_i = ||u - ubar||_1 vs the top-10 entropy NORMAL boundary
    points as control.  Returns rows for CSV."""
    from fcm_update import plcfcm_fit
    U, V, it = plcfcm_fit(X, c=K, labeled_idx=None, labels=None,
                          seed=DATA_GEN_SEED)
    p = np.clip(U, 1e-12, 1.0)
    H = -(p * np.log(p)).sum(axis=1)
    u_bar = U.mean(axis=0)
    h = np.abs(U - u_bar).sum(axis=1)
    umax = U.max(axis=1)
    ecc_mask = y == -1
    norm_mask = ~ecc_mask
    # control: 10 normal points with HIGHEST entropy (cluster boundary)
    h_idx = np.where(norm_mask)[0]
    order = np.argsort(-H[h_idx])[:N_ECC]
    ctrl_idx = h_idx[order]

    rows = []
    for i in np.where(ecc_mask)[0]:
        rows.append({"kind": kind, "sigma": sigma, "point_type": "eccentric",
                     "idx": int(i),
                     "H": float(H[i]), "h_i": float(h[i]),
                     "umax": float(umax[i]),
                     "argmax_cluster": int(U[i].argmax())})
    for i in ctrl_idx:
        rows.append({"kind": kind, "sigma": sigma, "point_type": "boundary_ctrl",
                     "idx": int(i),
                     "H": float(H[i]), "h_i": float(h[i]),
                     "umax": float(umax[i]),
                     "argmax_cluster": int(U[i].argmax())})
    return rows, U, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="only spiral sigma=0.6 (first-report probe)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.quick:
        jobs = [("spiral", 0.6)]
    else:
        jobs = [("spiral", 0.4), ("spiral", 0.6), ("spiral", 0.8),
                ("swissroll", 0.4), ("swissroll", 0.6), ("swissroll", 0.8),
                ("hourglass", 0.3), ("hourglass", 0.4), ("hourglass", 0.6),
                ("helical_cone", 0.4), ("helical_cone", 0.6),
                ("helical_cone", 0.8),
                ("spiral_saddle", 0.4), ("spiral_saddle", 0.6),
                ("spiral_saddle", 0.8)]

    rng = np.random.RandomState(DATA_GEN_SEED)   # single fixed stream, fixed order
    all_rows = []
    for kind, sigma in jobs:
        path, X, y = make_one(kind, sigma, rng)
        rows, _, _ = manipulation_check(X, y, kind, sigma)
        all_rows.extend(rows)
        print(f"[gen] {path}  X={X.shape}  n_ecc={(y==-1).sum()}")

    import csv
    csv_path = os.path.join(OUT, "manipulation_check.csv")
    write_header = not os.path.exists(csv_path) or args.quick
    mode = "w" if args.quick else "a"
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "sigma", "point_type", "idx",
                                          "H", "h_i", "umax", "argmax_cluster"])
        if write_header:
            w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[mc] wrote {csv_path} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
