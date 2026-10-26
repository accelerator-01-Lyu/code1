# POST-HOC EXPLORATORY DIAGNOSTIC. NOT PART OF PRE-REGISTERED P1-P4 VERDICTS.
"""p2c_eccentricity.py — parallel, incremental."""
import os, csv, numpy as np, pandas as pd
from multiprocessing import Pool
from fcm_update import plcfcm_fit
from acquisition import select_batch, exploration_reward
from run_experiments import load_pkl, NCLASSES, INIT_FRAC, M_MC, RHO, _lambda2_over_n

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV_PATH = os.path.join(OUT, "p2c_eccentricity.csv")
COLS = ["dataset", "seed", "budget", "method", "entropy_q25", "h_i_q75",
        "rate_low_entropy_high_h"]


def one(job):
    ds, method, seed = job
    X, y = load_pkl(ds); c = NCLASSES[ds]
    rng = np.random.RandomState(seed)
    n = len(X); ni = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool); labeled[rng.choice(n, ni, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c, labeled_idx=np.where(labeled)[0],
                          labels=y[labeled], seed=seed)
    unlab0 = ~labeled
    H0 = -(np.clip(U, 1e-12, 1) * np.log(np.clip(U, 1e-12, 1))).sum(1)
    h0 = exploration_reward(U, u_all=U)
    qH = float(np.quantile(H0[unlab0], 0.25))
    qh = float(np.quantile(h0[unlab0], 0.75))
    rows = []; ce, cs = 0, 0
    for bud in (0.05, 0.10, 0.15, 0.20):
        target = int(round(bud * n))
        while labeled.sum() < target:
            unlab = np.where(~labeled)[0]; b = target - int(labeled.sum())
            beta = _lambda2_over_n(U)
            sel = select_batch(U[unlab], X[unlab], V, method, b, alpha_0=50.,
                                M_MC=M_MC, rho_expl=RHO, lam_red=beta,
                                seed=seed, u_all=U)
            sel = unlab[sel]
            Us = U[sel]
            Hs = -(np.clip(Us, 1e-12, 1) * np.log(np.clip(Us, 1e-12, 1))).sum(1)
            hs = exploration_reward(Us, u_all=U)
            ce += int(((Hs <= qH) & (hs >= qh)).sum()); cs += len(sel)
            labeled[sel] = True
            U, V, _ = plcfcm_fit(X, c, labeled_idx=np.where(labeled)[0],
                                  labels=y[labeled], seed=seed)
        rows.append({"dataset": ds, "seed": seed, "budget": bud, "method": method,
                     "entropy_q25": qH, "h_i_q75": qh,
                     "rate_low_entropy_high_h": ce / max(cs, 1)})
    return rows


if __name__ == "__main__":
    done = set()
    if os.path.exists(CSV_PATH):
        for r in csv.DictReader(open(CSV_PATH)):
            done.add((r["dataset"], r["method"], int(r["seed"])))
    jobs = []
    for ds in ("ecoli", "glass", "iris", "wine", "aggregation", "compound"):
        for method in ("P2_Full", "P2_NoExploration"):
            for seed in range(30):
                if (ds, method, seed) not in done:
                    jobs.append((ds, method, seed))
    print("pending", len(jobs), flush=True)
    new = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if new: w.writeheader()
    with Pool(1) as p:
        for rows in p.imap_unordered(one, jobs):
            for r in rows: w.writerow(r)
            f.flush()
    f.close()
    print("done", flush=True)
