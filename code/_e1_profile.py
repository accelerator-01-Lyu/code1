import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_experiments import run_traj

for iid in (1, 15, 69):
    dpath = os.path.join(os.path.dirname(HERE), "data", "synthetic", f"instance_{iid:03d}")
    X = np.load(os.path.join(dpath, "X.npy")); y = np.load(os.path.join(dpath, "y_true.npy"))
    meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
    t0 = time.time()
    run_traj(X, y, meta["c"], f"syn{iid:03d}", "P2_NoExploration", 0,
             l0=0, eta=0.0, alpha0=50.0, lam_mult=1.0, run_x2=False)
    t1 = time.time()
    print(f"iid={iid} N={meta['N']} c={meta['c']} d={meta['d']}: NoExploration traj = {t1-t0:.1f}s", flush=True)
    t0 = time.time()
    run_traj(X, y, meta["c"], f"syn{iid:03d}", "Random", 0,
             l0=0, eta=0.0, alpha0=50.0, lam_mult=1.0, run_x2=False)
    t1 = time.time()
    print(f"iid={iid}: Random traj = {t1-t0:.1f}s", flush=True)
