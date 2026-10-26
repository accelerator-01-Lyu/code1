"""_probe_e4c3.py — deeper probe of the overlap quadrant for c=3.

Replicates the initial FCM fit (seed=0) and reports per level:
  corr(H,h), quadrant counts (quartile rule as in exploratory.run_x2_diag),
  and NEAR-MISS overlap counts: H within 0.05 of qH_hi AND h within 0.10 of
  qh_hi (how close the high-H population is to qualifying as high-h).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_experiments import plcfcm_fit, INIT_FRAC

DATA = os.path.join(ROOT, "data", "controlled_overlap_c3")
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]

for lv in LEVELS:
    f = os.path.join(DATA, f"overlap_{lv:.1f}")
    X = np.load(os.path.join(f, "X.npy"))
    y = np.load(os.path.join(f, "y_true.npy"))
    n = X.shape[0]
    rng = np.random.RandomState(0)
    n_init = max(3, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c=2, labeled_idx=np.where(labeled)[0],
                         labels=y[labeled], seed=0)
    unlab = ~labeled
    Uu = U[unlab]
    u_bar = U.mean(0)
    H = -(np.clip(Uu, 1e-12, 1) * np.log(np.clip(Uu, 1e-12, 1))).sum(1)
    h = np.abs(Uu - u_bar).sum(1)
    qH_hi, qH_lo = np.quantile(H, [0.75, 0.25])
    qh_hi, qh_lo = np.quantile(h, [0.75, 0.25])
    hiH = H >= qH_hi; loH = H <= qH_lo
    hih = h >= qh_hi; loh = h <= qh_lo
    n_boundary = int((hiH & loh).sum())
    n_eccentric = int((loH & hih).sum())
    n_overlap = int((hiH & hih).sum())
    n_other = len(H) - n_boundary - n_eccentric - n_overlap
    rho = np.corrcoef(H, h)[0, 1]
    near = int(((H >= qH_hi - 0.05) & (h >= qh_hi - 0.10)).sum())
    # h distribution among top-H quartile
    topH_h = h[hiH]
    print(f"level={lv:.1f} corr={rho:+.3f} qH_hi={qH_hi:.3f} qh_hi={qh_hi:.3f} "
          f"| boundary={n_boundary} eccentric={n_eccentric} overlap={n_overlap} "
          f"other={n_other} near-miss={near}")
    print(f"           top-H pts h: median={np.median(topH_h):.3f} "
          f"max={topH_h.max():.3f} | u_bar={np.round(u_bar,3)}")
