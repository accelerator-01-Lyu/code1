"""gen_controlled_overlap.py — E4 controlled-overlap synthetic instances (FINAL).

[E4 revision 3, 2026-10-22, organizer-approved] c=2 with EXTREME imbalance.

History of the structural fix (see check_work/changelog_e4.md):
  - c=2 balanced: overlap class EMPTY at every level (corr(H,h) ~ -0.95;
    on a balanced simplex the pool centroid u_bar IS the max-entropy point,
    so h=0 exactly where H is max).
  - c=3 balanced / c=3 1200/500/300: probe still gives overlap quadrant = 0
    at all levels (corr -0.89..-0.97); the anti-correlation is structural.
  - The ecoli/glass overlap classes are non-empty because of UNBALANCED
    priors: the max-entropy region then sits AWAY from the centroid, so it
    carries a non-zero h, while the large class compresses the h distribution
    and pushes its Q75 below the max-entropy point's h.

Final design (faithful to the construction plan except balanced=True):
  c=2, priors 0.9/0.1 -> sizes (1800, 200), N=2000, d=5,
  overlap_levels = [0.0,0.2,0.4,0.6,0.8], seeds 3000+level_id.
  Isotropic Sigma=I; centers on the first axis separated by
      D_target = (1 - overlap) * 2 * sqrt(d)
  overlap_ij = 1 - D / (sqrt(tr(Sigma_i)) + sqrt(tr(Sigma_j)))  (two-class).

Outputs under v2_unified/data/controlled_overlap_c3/overlap_{level}/:
    X.npy, y_true.npy, meta.json
    (directory name kept for pipeline continuity; meta records c=2,
     priors=[0.9,0.1], balanced=false)
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_ROOT = os.path.join(ROOT, "data", "controlled_overlap_c3")

OVERLAP_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
# Probe: 1800/200 made FCM collapse the rare centroid (u_bar~0.55 not 0.9).
# Trying shallower 1700/300 (priors 0.85/0.15) -> more rare anchors.
CLASS_SIZES = [1700, 300]
N_TOTAL = 2000
D_DIM = 5
C = 2
BASE_SEED = 3000


def make_instance(target_overlap, seed):
    rng = np.random.RandomState(seed)
    d = D_DIM
    s0, s1 = CLASS_SIZES[0], CLASS_SIZES[1]
    D_target = (1.0 - target_overlap) * 2.0 * np.sqrt(d)
    mu0 = np.zeros(d); mu0[0] = -D_target / 2.0
    mu1 = np.zeros(d); mu1[0] = +D_target / 2.0
    X0 = rng.randn(s0, d) + mu0
    X1 = rng.randn(s1, d) + mu1
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.concatenate([np.zeros(s0, np.int64),
                        np.ones(s1, np.int64)])
    # empirical overlap (two-class plug-in)
    mu0e, mu1e = X0.mean(0), X1.mean(0)
    D_emp = float(np.linalg.norm(mu0e - mu1e))
    tr0 = np.trace(np.cov(X0, rowvar=False))
    tr1 = np.trace(np.cov(X1, rowvar=False))
    ov_emp = 1.0 - D_emp / (np.sqrt(tr0) + np.sqrt(tr1))
    return X, y, {"D_target": D_target, "D_empirical": D_emp,
                  "overlap_empirical": float(ov_emp),
                  "deviation": float(ov_emp - target_overlap)}


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"{'level':>6} {'D_target':>9} {'D_emp':>9} {'ov_emp':>8} {'dev':>8}")
    for level_id, ov in enumerate(OVERLAP_LEVELS):
        seed = BASE_SEED + level_id
        X, y, diag = make_instance(ov, seed)
        folder = os.path.join(OUT_ROOT, f"overlap_{ov:.1f}")
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "X.npy"), X)
        np.save(os.path.join(folder, "y_true.npy"), y)
        meta = {"overlap": ov, "N": int(X.shape[0]), "d": D_DIM, "c": C,
                "seed": seed, "priors": [0.85, 0.15], "balanced": False,
                "class_sizes": CLASS_SIZES, **diag}
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"{ov:6.1f} {diag['D_target']:9.4f} {diag['D_empirical']:9.4f} "
              f"{diag['overlap_empirical']:8.4f} {diag['deviation']:+8.4f}")
    print("[gen_controlled_overlap c2-imbalanced] done ->", OUT_ROOT)


if __name__ == "__main__":
    main()
