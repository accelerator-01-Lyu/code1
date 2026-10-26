"""_debug_e4c3.py — V positions at each level for 1700/300."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_experiments import plcfcm_fit, INIT_FRAC

for lv in (0.0, 0.2, 0.4, 0.6, 0.8):
    f = os.path.join(ROOT, "data", "controlled_overlap_c3", f"overlap_{lv:.1f}")
    X = np.load(os.path.join(f, "X.npy"))
    y = np.load(os.path.join(f, "y_true.npy"))
    n = X.shape[0]
    rng = np.random.RandomState(0)
    n_init = max(2, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    U, V, it = plcfcm_fit(X, c=2, labeled_idx=np.where(labeled)[0],
                          labels=y[labeled], seed=0)
    # true centers: majority at x=-D/2, rare at x=+D/2
    D = (1.0 - lv) * 2.0 * np.sqrt(5)
    print(f"level={lv}: true maj~{-D/2:+.2f} rare~{+D/2:+.2f} | "
          f"V0[0]={V[0,0]:+.2f} V1[0]={V[1,0]:+.2f} u_bar={U.mean(0)[0]:.3f}")
