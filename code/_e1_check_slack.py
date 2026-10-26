import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_experiments import run_traj
from run_synthetic_sweep import run_traj_slackuniform

dpath = os.path.join(os.path.dirname(HERE), "data", "synthetic", "instance_001")
X = np.load(os.path.join(dpath, "X.npy")); y = np.load(os.path.join(dpath, "y_true.npy"))
meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
c = meta["c"]
for seed in (0, 1):
    rows_ne, *_ = run_traj(X, y, c, "syn001", "P2_NoExploration", seed,
                           l0=0, eta=0.0, alpha0=50.0, lam_mult=1.0, run_x2=False)
    rows_su = run_traj_slackuniform(X, y, c, seed)
    ne = [r["ACC"] for r in rows_ne]; su = [r["ACC"] for r in rows_su]
    print(f"seed={seed} budgets=[0.05,0.10,0.15,0.20]", flush=True)
    print(f"  NoExploration ACC: {[round(v,4) for v in ne]}", flush=True)
    print(f"  SlackUniform   ACC: {[round(v,4) for v in su]}", flush=True)
    print(f"  equal? {np.allclose(ne, su)}", flush=True)
