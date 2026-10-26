"""Bridge test: slow vs fast gain engine, end-to-end shard outputs."""
import numpy as np, pandas as pd
import run_experiments as R
import acquisition as A
from gain_fast import calculate_gain_batch
R.run_x2_diag = lambda *a, **k: None

orig_calc = A.calculate_gain

def patched_calc(u, alpha_0, epsilon_q, M_MC, kappa=1.0, order=2, seed=0):
    w = calculate_gain_batch(u[None, :], alpha_0, epsilon_q, M_MC,
                              kappa=kappa, order=order, seed=int(seed))
    return {"w_robust": float(w[0]), "gain": float(w[0])}

jobs = []
for ds in ["ecoli", "aggregation"]:
    for method in ["P2_Full", "P2_NoExploration", "P2_NoRobust"]:
        for seed in [1, 2, 3]:
            jobs.append((ds, method, seed))

rows = []
for ds, method, seed in jobs:
    X, y = R.load_pkl(ds); c = R.NCLASSES[ds]
    R.calculate_gain = orig_calc
    rs, *_ = R.run_traj(X, y, c, ds, method, seed, 0, 0.0, 50.0, 1.0, budgets=R.BUDGETS)
    A.calculate_gain = patched_calc
    rf, *_ = R.run_traj(X, y, c, ds, method, seed, 0, 0.0, 50.0, 1.0, budgets=R.BUDGETS)
    A.calculate_gain = orig_calc
    d = pd.DataFrame(rs); f = pd.DataFrame(rf)
    dm = (d[["NMI", "ARI", "ACC"]] - f[["NMI", "ARI", "ACC"]]).abs().max()
    rows.append({"ds": ds, "method": method, "seed": seed,
                 "maxd_NMI": dm.NMI, "maxd_ARI": dm.ARI, "maxd_ACC": dm.ACC})
    print(ds, method, seed, "maxd=", round(dm.max(), 4), flush=True)
df = pd.DataFrame(rows)
print("worst ACC diff:", df.maxd_ACC.max())
print("worst NMI diff:", df.maxd_NMI.max())
