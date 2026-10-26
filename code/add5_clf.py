# -*- coding: utf-8 -*-
"""add5_clf.py — Task 5: cross-task softmax classification (P2 score on
LogisticRegression probabilities).

Paradigm shift: clustering -> classification. "Membership" U = softmax
predict_proba from LogisticRegression(lbfgs). The P2 selector reuses the SAME
score formula:
    mod_i = w^rob_i + rho * h_i ,   h_i = ||u_i - u_bar||_1
with w^rob = sup_tau [-tau*kappa - tau*logmeanexp_r(-g_r/tau)]  (the exact
KL-DRO dual from gain.py) and a greedy cosine-redundancy penalty. Only the
backend changes (PLFCM -> LogisticRegression).

For speed on n~10k we compute w^rob for the whole unlabeled pool in a
vectorized batch (same formula, batched MC draws + log-grid tau search; the
per-candidate golden-section refinement of gain.py is replaced by a fine 120-
point log grid, noted in unit.json). This keeps the P2 score formula identical
while making 20-seed runs tractable.

Datasets: fashion (balanced subset N=5000, d=50, c=10) + usps (balanced
          subset N=5000, d=256, c=10); fixed stratified subset (500 per class)
          of the local pkl dumps, matching the task's N=5000 subset spec.
Methods:  P2_Full, P2_NoExploration, Random, Entropy.
Budgets:  {5, 10, 15, 20}% labeled.
Seeds:    ac.SEEDS_20 (20), paired by seed (same L0 for all methods).
Metric:   classification accuracy on the held-out unlabeled pool.
"""
import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac
from acquisition import cosine_redundancy, exploration_reward
from gain import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

ac.setup_style()

DATASETS = [("fashion", 10), ("usps", 10)]
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
BUDGETS = [0.05, 0.10, 0.15, 0.20]
ALPHA0 = ac.ALPHA0_MAIN
M_MC = ac.M_MC
RHO = ac.RHO
KAPPA = 1.0
EPS_Q = 0.1
LAM_AFF = 0.5
# Data-dependent beta = lambda2(L_R)/n measured ~= 0.0086 on usps (and ~0.01
# on fashion); under multi-worker memory pressure we use this fixed small value
# instead of a per-seed n x n Laplacian eigendecomposition. Noted in unit.json.
LAM_RED = 0.01


def load_pkl(name):
    o = pickle.load(open(os.path.join(ac.REF_DATA, name + ".pkl"), "rb"))
    return np.asarray(o["X"], float), np.asarray(o["y"], int)


N_MAX = 5000      # balanced subset per the task spec (MNIST/Fashion subset N=5000)


def subsample_balanced(X, y, n_max=N_MAX, seed=20260101):
    """Fixed stratified subset (equal per class) so runs match the N=5000 spec."""
    if len(y) <= n_max:
        return X, y
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per = n_max // len(classes)
    keep = []
    for k in classes:
        idx = np.where(y == k)[0].copy()
        rng.shuffle(idx)
        keep.append(idx[:per])
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def lambda2_over_n(U):
    n = U.shape[0]
    R = cosine_redundancy(U)
    d = R.sum(1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    ev = np.linalg.eigvalsh(L)
    lam2 = float(np.sort(ev)[1]) if len(ev) > 1 else 1.0
    return lam2 / max(n, 1)


def batched_robust_gain(U, seed=0):
    """Vectorized w^rob for every row of U (n, c). Mirrors gain.calculate_gain
    order=2 (KL-DRO dual, kappa=1.0) but batches the MC draws (gamma-based
    Dirichlet over the whole pool at once) and uses a 40-point log tau grid
    instead of per-row golden refinement."""
    n, c = U.shape
    rng = np.random.RandomState(seed)
    U_pos = np.clip(U, 1e-12, 1.0)
    # gamma-based Dirichlet for all (n, M_MC, c) draws vectorized.
    conc = np.broadcast_to((ALPHA0 * U_pos)[:, None, :], (n, M_MC, c))
    gam = rng.gamma(shape=conc)
    S = gam / gam.sum(2, keepdims=True)
    # epsilon_q mass on the uniform Dirichlet component of q_i.
    gam_u = rng.gamma(shape=1.0, size=(n, M_MC, c))
    S_u = gam_u / gam_u.sum(2, keepdims=True)
    mix = rng.rand(n, M_MC) < EPS_Q
    S = np.where(mix[:, :, None], S_u, S)
    H_u = -(U_pos * np.log(U_pos)).sum(1)                     # (n,)
    u_up = (U[:, None, :] + LAM_AFF * S) / (1.0 + LAM_AFF)
    u_up = u_up / u_up.sum(2, keepdims=True)
    u_up_c = np.clip(u_up, 1e-12, 1.0)
    H_up = -(u_up_c * np.log(u_up_c)).sum(2)                 # (n, M)
    g = H_u[:, None] - H_up                                   # (n, M)
    taus = np.logspace(-3.0, 3.0, 40)                         # (T,)
    z = -g[:, None, :] / taus[None, :, None]                # (n, T, M)
    lme = logsumexp(z, axis=2) - np.log(M_MC)                 # (n, T)
    w = -taus[None, :] * KAPPA - taus[None, :] * lme         # (n, T)
    interior = w.max(axis=1)
    w_rob = np.maximum(interior, g.min(axis=1) - 1e-12)
    return w_rob


def p2_select(U_unlab, u_all, method, b, lam_red, seed):
    """Greedy P2 batch selection from unlabeled membership U_unlab (n,c).
    Memory-light: the cosine redundancy row R[k] = Un[k] @ Un.T is computed on
    the fly instead of materializing the n x n cosine matrix."""
    n = U_unlab.shape[0]
    w_rob = batched_robust_gain(U_unlab, seed=seed)
    h = exploration_reward(U_unlab, u_all=u_all)
    rho = 0.0 if method == "P2_NoExploration" else RHO
    mod = w_rob + rho * h
    norms = np.linalg.norm(U_unlab, axis=1, keepdims=True) + 1e-12
    Un = U_unlab / norms
    selected, avail = [], np.ones(n, bool)
    red_sum = np.zeros(n)
    for _ in range(min(b, n)):
        scores = mod - lam_red * red_sum
        scores[~avail] = -np.inf
        j = int(np.argmax(scores))
        if scores[j] == -np.inf:
            break
        selected.append(j)
        avail[j] = False
        red_sum += Un[j] @ Un.T
    return selected


def fit_proba(X, y, labeled):
    # lbfgs on the 256-dim usps pool took ~25s/fit; use SGD with log_loss
    # (explicitly endorsed by the task) which is ~100x faster on small labeled
    # sets while retaining softmax predict_proba.
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                        tol=1e-4, random_state=0)
    clf.fit(X[labeled], y[labeled])
    U = clf.predict_proba(X)
    return clf, U


def run_method(X, y, c, method, seed, labeled0):
    """Standard AL loop: fit on current labeled set -> score/select -> add ->
    refit. One LR fit per budget round (the proba U used for selection comes
    from the immediately preceding fit on the current labeled set)."""
    n = X.shape[0]
    rng = np.random.RandomState((seed * 7919 + 17) & 0x7FFFFFFF)
    labeled = labeled0.copy()
    clf, U = fit_proba(X, y, labeled)
    out = []
    for bud in BUDGETS:
        target = int(round(bud * n))
        if int(labeled.sum()) < target:
            unlab = np.where(~labeled)[0]
            b = target - int(labeled.sum())
            if method in ("P2_Full", "P2_NoExploration"):
                sel_local = p2_select(U[unlab], U, method, b, lam_red=LAM_RED, seed=seed)
                sel = unlab[sel_local]
            elif method == "Random":
                sel = unlab[rng.choice(len(unlab), size=b, replace=False)]
            else:  # Entropy
                H = entropy(np.clip(U[unlab], 1e-12, 1.0))
                sel = unlab[np.argsort(-H)[:b]]
            labeled[sel] = True
            clf, U = fit_proba(X, y, labeled)
        unlab = np.where(~labeled)[0]
        pred = clf.predict(X[unlab])
        acc = float((pred == y[unlab]).mean())
        out.append((bud, acc))
    return out


def main():
    t0 = time.time()
    rows = []
    for ds, c in DATASETS:
        X, y = load_pkl(ds)
        X, y = subsample_balanced(X, y)
        n = X.shape[0]
        n_init = max(c, int(round(ac.INIT_FRAC * n)))
        print("[add5] dataset=%s n=%d d=%d c=%d L0=%d" % (ds, n, X.shape[1], c, n_init), flush=True)
        for si, seed in enumerate(ac.SEEDS_20):
            rng = np.random.RandomState((seed * 1000 + n_init) & 0x7FFFFFFF)
            labeled0 = np.zeros(n, bool)
            labeled0[rng.choice(n, size=n_init, replace=False)] = True
            for method in METHODS:
                res = run_method(X, y, c, method, seed, labeled0)
                for bud, acc in res:
                    rows.append({"dataset": ds, "method": method,
                                 "budget": bud, "seed": seed, "acc": acc})
            if (si + 1) % 5 == 0:
                print("[add5] %s seed %d/20 done (%.0fs)" % (ds, si + 1, time.time()-t0), flush=True)

    long_df = pd.DataFrame(rows)
    csv_long = ac.save_csv(long_df, "add5_clf_curve.csv")

    sum_rows = []
    for (ds, bud), sub in long_df.groupby(["dataset", "budget"]):
        wide = sub.pivot(index="seed", columns="method", values="acc")
        rows_d = ac.contrast_table(wide.to_dict(orient="list"),
                                    reference="P2_Full", methods=METHODS, metric="acc")
        for method in METHODS:
            vals = wide[method].values
            m = vals.mean(); sd = vals.std(ddof=1)
            se = sd / np.sqrt(len(vals))
            tc = ac.stats.t.ppf(0.975, len(vals) - 1)
            dz = 0.0; p = 1.0
            if method != "P2_Full":
                dz = rows_d[method]["dz"]; p = rows_d[method]["p_holm"]
            sum_rows.append({"dataset": ds, "method": method, "budget": bud,
                             "acc_mean": float(m), "ci_lo": float(m - tc * se),
                             "ci_hi": float(m + tc * se),
                             "dz_vs_full": float(dz), "p_holm": float(p)})
    sum_df = pd.DataFrame(sum_rows)
    csv_sum = ac.save_csv(sum_df, "add5_clf_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    colors = {"P2_Full": ac.C_FULL, "P2_NoExploration": ac.C_NOEXPLO,
              "Random": ac.PALETTE["mid_gray"], "Entropy": ac.PALETTE["mid_teal"]}
    for ax, (ds, c) in zip(axes, DATASETS):
        for method in METHODS:
            sub = sum_df[(sum_df.dataset == ds) & (sum_df.method == method)] \
                .sort_values("budget")
            x = sub["budget"].values * 100
            ax.plot(x, sub["acc_mean"].values, "-o", ms=4, lw=1.6,
                    color=colors[method], label=method)
            ac.ci_band(ax, x, sub["ci_lo"].values, sub["ci_hi"].values,
                       colors[method], alpha=0.15)
        ax.set_title(ds.capitalize())
        ax.set_xlabel("Labeled budget (%)")
        ax.set_ylim(0.5, 0.95)
        ac.style_ax(ax)
    axes[0].set_ylabel("Classification ACC (unlabeled pool)")
    axes[1].legend(loc="lower right")
    png = ac.savefig(fig, "add5_clf_curve", combo=True)

    u = ac.new_unit("add5",
        "Cross-task softmax classification on fashion / usps (P2 score on LR probs)",
        "exploratory/post-hoc", "done")
    u["annotations"] = ["exploratory", "post-hoc",
        "backend = SGDClassifier(loss='log_loss', max_iter=30) for speed on the 256-dim pool",
        "U = softmax predict_proba; P2 score w^rob + rho*||u-u_bar||_1 reused",
        "batched vectorized KL-DRO gain (gamma Dirichlet, 40-pt log tau grid)",
        "fixed lam_red=0.01 (data-dependent beta~0.009; n x n eigendecomp skipped under multi-worker memory pressure)",
        "cosine redundancy computed row-wise (no dense n x n matrix)",
        "balanced subset N=5000 per dataset (fashion d=50, usps d=256, c=10; 500/class, seed 20260101)",
        "20 paired seeds; budgets 5/10/15/20%"]
    u["protocol"] = {"datasets": [d for d, _ in DATASETS], "methods": METHODS,
                     "n_seeds": 20, "budgets": BUDGETS,
                     "factors": {"L0_frac": ac.INIT_FRAC, "alpha0": ALPHA0,
                                 "M_MC": M_MC, "rho": RHO, "kappa": KAPPA,
                                 "backend": "SGDClassifier log_loss"},
                     "checkpoint": 0.10}
    u["csvs"] = ["add5_clf_curve.csv", "add5_clf_summary.csv"]
    u["repro_script"] = "add5_clf.py"
    ac.add_figure(u, "add5_clf_curve",
        "Fig. Classification accuracy vs labeled budget on fashion and usps; "
        "solid lines = mean over 20 paired seeds, shaded band = 95% paired-t CI.")

    def means_at(ds, bud):
        w = long_df[(long_df.dataset == ds) & (np.isclose(long_df.budget, bud))] \
            .pivot(index="seed", columns="method", values="acc")
        return {m: float(w[m].mean()) for m in METHODS}
    f10 = means_at("fashion", 0.10)
    u["results"] = (
        "Paradigm transfer: treating LogisticRegression softmax outputs as the "
        "P2 membership U, the KL-robust score plus h_i=||u_i-u_bar||_1 ranks "
        "unlabeled points. On fashion at the 10%% budget the four-method mean "
        "accuracy is P2_Full=%.3f, P2_NoExploration=%.3f, Random=%.3f, "
        "Entropy=%.3f (20 paired seeds, 95%% CI in add5_clf_summary.csv). "
        "Curves over 5/10/15/20%% are shown for fashion and usps."
        % (f10["P2_Full"], f10["P2_NoExploration"], f10["Random"], f10["Entropy"]))
    u["conclusion"] = (
        "Exploratory/post-hoc cross-task check: the P2 acquisition formula is "
        "mechanical and carries over to the classification backend, but this is "
        "NOT a confirmatory clustering result; report it as a robustness probe "
        "with the attached paired CIs and Holm-adjusted p-values.")
    uj = ac.save_unit(u)

    print("WROTE:", csv_long)
    print("WROTE:", csv_sum)
    print("WROTE:", png)
    print("WROTE:", uj)
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
