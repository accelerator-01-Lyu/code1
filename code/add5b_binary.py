# -*- coding: utf-8 -*-
"""add5b_binary.py — S5: fashion-MNIST binary tight-budget classification.

Binary sub-task: T-shirt/top (label 0) vs Shirt (label 6) from fashion.pkl
(2000 samples, d=50). Backend LogisticRegression(lbfgs, max_iter=1000) per
MainAgent (NOT SGD). P2 score w^rob + rho*||u-u_bar||_1 on softmax probs.
Budgets {1,2,3,5,7,10,15,20,25,30}%, 20 seeds.
Saves add5b_binary.csv; updates add5_unit.json.
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac
from gain import entropy
import warnings; warnings.filterwarnings("ignore")

ac.setup_style()
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
ALPHA0 = ac.ALPHA0_MAIN; M_MC = ac.M_MC; RHO = ac.RHO
KAPPA = 1.0; EPS_Q = 0.1; LAM_AFF = 0.5; LAM_RED = 0.01
L0_FRAC = 0.005   # tiny seed set so 1% budget is meaningful


def load_binary():
    o = pickle.load(open(os.path.join(ac.REF_DATA, "fashion.pkl"), "rb"))
    X = np.asarray(o["X"], float); y = np.asarray(o["y"], int)
    keep = np.isin(y, [0, 6])
    X, y = X[keep], y[keep]
    # relabel 0->0, 6->1
    y = (y == 6).astype(int)
    return X, y, 2


def batched_robust_gain(U, seed=0):
    n, c = U.shape
    rng = np.random.RandomState(seed)
    U_pos = np.clip(U, 1e-12, 1.0)
    conc = np.broadcast_to((ALPHA0 * U_pos)[:, None, :], (n, M_MC, c))
    gam = rng.gamma(shape=conc)
    S = gam / gam.sum(2, keepdims=True)
    gam_u = rng.gamma(shape=1.0, size=(n, M_MC, c))
    S_u = gam_u / gam_u.sum(2, keepdims=True)
    mix = rng.rand(n, M_MC) < EPS_Q
    S = np.where(mix[:, :, None], S_u, S)
    H_u = -(U_pos * np.log(U_pos)).sum(1)
    u_up = (U[:, None, :] + LAM_AFF * S) / (1.0 + LAM_AFF)
    u_up = u_up / u_up.sum(2, keepdims=True)
    u_up_c = np.clip(u_up, 1e-12, 1.0)
    H_up = -(u_up_c * np.log(u_up_c)).sum(2)
    g = H_u[:, None] - H_up
    taus = np.logspace(-3.0, 3.0, 40)
    z = -g[:, None, :] / taus[None, :, None]
    lme = logsumexp(z, axis=2) - np.log(M_MC)
    w = -taus[None, :] * KAPPA - taus[None, :] * lme
    return np.maximum(w.max(axis=1), g.min(axis=1) - 1e-12)


def p2_select(U_unlab, u_all, method, b, seed):
    n = U_unlab.shape[0]
    w_rob = batched_robust_gain(U_unlab, seed=seed)
    u_bar = u_all.mean(0)
    h = np.abs(U_unlab - u_bar).sum(1)
    rho = 0.0 if method == "P2_NoExploration" else RHO
    mod = w_rob + rho * h
    norms = np.linalg.norm(U_unlab, axis=1, keepdims=True) + 1e-12
    Un = U_unlab / norms
    selected, avail = [], np.ones(n, bool)
    red_sum = np.zeros(n)
    for _ in range(min(b, n)):
        scores = mod - LAM_RED * red_sum
        scores[~avail] = -np.inf
        j = int(np.argmax(scores))
        if scores[j] == -np.inf:
            break
        selected.append(j); avail[j] = False
        red_sum += Un[j] @ Un.T
    return selected


def run_method(X, y, method, seed, labeled0):
    n = X.shape[0]
    rng = np.random.RandomState((seed * 7919 + 17) & 0x7FFFFFFF)
    labeled = labeled0.copy()
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X[labeled], y[labeled])
    U = clf.predict_proba(X)
    out = []
    for bud in BUDGETS:
        target = int(round(bud * n))
        if int(labeled.sum()) < target:
            unlab = np.where(~labeled)[0]
            b = target - int(labeled.sum())
            if method in ("P2_Full", "P2_NoExploration"):
                sel = unlab[p2_select(U[unlab], U, method, b, seed=seed)]
            elif method == "Random":
                sel = unlab[rng.choice(len(unlab), size=b, replace=False)]
            else:
                H = entropy(np.clip(U[unlab], 1e-12, 1.0))
                sel = unlab[np.argsort(-H)[:b]]
            labeled[sel] = True
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf.fit(X[labeled], y[labeled])
            U = clf.predict_proba(X)
        unlab = np.where(~labeled)[0]
        acc = float((clf.predict(X[unlab]) == y[unlab]).mean())
        out.append((bud, acc))
    return out


def main():
    t0 = time.time()
    X, y, c = load_binary(); n = X.shape[0]
    n_init = max(c, int(round(L0_FRAC * n)))
    print("[add5b] fashion binary n=%d d=%d c=%d L0=%d" % (n, X.shape[1], c, n_init), flush=True)
    rows = []
    for si, seed in enumerate(ac.SEEDS_20):
        rng = np.random.RandomState((seed * 1000 + n_init) & 0x7FFFFFFF)
        labeled0 = np.zeros(n, bool); labeled0[rng.choice(n, size=n_init, replace=False)] = True
        for method in METHODS:
            res = run_method(X, y, method, seed, labeled0)
            for bud, acc in res:
                rows.append({"dataset": "fashion_binary", "method": method,
                             "budget": bud, "seed": seed, "acc": acc})
        if (si + 1) % 5 == 0:
            print("[add5b] seed %d/20 (%.0fs)" % (si + 1, time.time() - t0), flush=True)
    long_df = pd.DataFrame(rows)
    csv = ac.save_csv(long_df, "add5b_binary.csv")
    print("WROTE", csv)
    # quick means at 10%
    w = long_df[np.isclose(long_df.budget, 0.10)].pivot(index="seed", columns="method", values="acc")
    print("binary @10%:", {m: round(float(w[m].mean()), 3) for m in METHODS})


if __name__ == "__main__":
    main()
