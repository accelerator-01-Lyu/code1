import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _factor_lib as F

for iid in (1, 37, 59):
    X, y, meta = F.load_instance(iid)
    for seed in (0, 3):
        t1 = time.time()
        m = F.run_traj_factor(X, y, meta["c"], seed=seed, rho=0.1,
                              kappa=1.0, alpha0=50.0, lam_mult=1.0)
        dt = time.time() - t1
        print(f"iid={iid} c={meta['c']} d={meta['d']} seed={seed}: "
              f"{dt:.2f}s ACC={m['ACC']:.4f}", flush=True)
