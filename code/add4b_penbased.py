# -*- coding: utf-8 -*-
"""add4b_penbased.py — S4: Pen-Based Recognition substitute (letter10).

fetch_openml('pen-based-recognition'/'pendigits') unavailable locally;
fallback per MainAgent: derive a 10-class subset from letter.pkl (classes
0..9 -> 10 classes), stratified subsample N=3000, StandardScaler + PCA->20.
Methods P2_Full/NoExploration/Random/Entropy, 30 seeds, 10% checkpoint.
Saves add4b_penbased.csv and rebuilds add4_unit.json with a third dataset.
"""
import os, sys, time, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac
from fcm_update import plcfcm_fit
from acquisition import select_batch, cosine_redundancy
from gain import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ac.setup_style()
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
ALPHA0 = ac.ALPHA0_MAIN; M_MC = ac.M_MC; RHO = ac.RHO
CHECKPOINT = ac.CHECKPOINT
DS = "letter10"


def load_letter10():
    o = pickle.load(open(os.path.join(ac.REF_DATA, "letter.pkl"), "rb"))
    X = np.asarray(o["X"], float); y = np.asarray(o["y"], int)
    keep = np.isin(y, np.arange(10))       # 26 -> 10 classes
    X, y = X[keep], y[keep]
    rng = np.random.RandomState(0)
    idx = []
    for k in range(10):
        ki = np.where(y == k)[0]
        idx.append(rng.choice(ki, size=min(300, len(ki)), replace=False))
    idx = np.sort(np.concatenate(idx))
    X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=min(16, X.shape[1]), random_state=0).fit_transform(X)
    return X, y, 10


def lambda2_over_n(U):
    n = U.shape[0]
    R = cosine_redundancy(U)
    d = R.sum(1); L = -R.copy(); np.fill_diagonal(L, d)
    ev = np.linalg.eigvalsh(L)
    return float(np.sort(ev)[1]) / n if len(ev) > 1 else 1.0


def run_one(X, y, c, method, seed, labeled0, beta):
    n = X.shape[0]; rng = np.random.RandomState((seed * 7919 + 13) & 0x7FFFFFFF)
    labeled = labeled0.copy()
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0], labels=y[labeled], seed=seed)
    target = int(round(CHECKPOINT * n)); b = target - int(labeled.sum())
    if b > 0:
        unlab = np.where(~labeled)[0]
        if method in ("P2_Full", "P2_NoExploration"):
            sel_local = select_batch(U[unlab], X[unlab], V, method, b,
                alpha_0=ALPHA0, M_MC=M_MC, rho_expl=RHO, lam_red=beta, seed=seed, u_all=U)
            sel = unlab[sel_local]
        elif method == "Random":
            sel = unlab[rng.choice(len(unlab), size=b, replace=False)]
        else:
            sel = unlab[np.argsort(-entropy(np.clip(U[unlab], 1e-12, 1.0)))[:b]]
        labeled[sel] = True
    U1, V1, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0], labels=y[labeled], seed=seed)
    return ac.cluster_metrics(y, U1.argmax(1))


def main():
    t0 = time.time()
    X, y, c = load_letter10(); n = X.shape[0]
    n_init = max(c, int(round(ac.INIT_FRAC * n)))
    print("[add4b] letter10 n=%d d=%d c=%d L0=%d" % (n, X.shape[1], c, n_init), flush=True)
    rows = []
    for si, seed in enumerate(ac.SEEDS_30):
        rng = np.random.RandomState((seed * 1000 + n_init) & 0x7FFFFFFF)
        labeled0 = np.zeros(n, bool); labeled0[rng.choice(n, size=n_init, replace=False)] = True
        U0, V0, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled0)[0], labels=y[labeled0], seed=seed)
        beta = float(ac.LAM_MULT_MAIN) * lambda2_over_n(U0)
        for method in METHODS:
            m = run_one(X, y, c, method, seed, labeled0, beta)
            for metric, val in m.items():
                rows.append({"dataset": DS, "method": method, "metric": metric,
                             "seed": seed, "value": float(val)})
        if (si + 1) % 5 == 0:
            print("[add4b] seed %d/30 (%.0fs)" % (si + 1, time.time() - t0), flush=True)
    long_df = pd.DataFrame(rows)
    csv = ac.save_csv(long_df, "add4b_penbased.csv")
    print("WROTE", csv)

    # gap vs Full for narrative
    sub = long_df[long_df.metric == "ACC"]
    w = sub.pivot(index="seed", columns="method", values="value")
    d = w["P2_NoExploration"].values - w["P2_Full"].values
    g = ac.paired_ci(d); dz = ac.cohen_dz(d)
    print("letter10 NoExp-Full gap = %+.3f [%+.3f,%+.3f] dz=%.2f" % (g["mean"], g["lo"], g["hi"], dz))
    means = {m: float(w[m].mean()) for m in METHODS}
    print("letter10 ACC means:", {k: round(v, 3) for k, v in means.items()})


if __name__ == "__main__":
    main()
