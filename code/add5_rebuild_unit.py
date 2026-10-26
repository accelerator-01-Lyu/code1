# -*- coding: utf-8 -*-
"""Rebuild add5 figure + unit.json from already-saved CSVs (no re-run)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac

ac.setup_style()
DATASETS = [("fashion", 10), ("usps", 10)]
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
BUDGETS = [0.05, 0.10, 0.15, 0.20]

long_df = pd.read_csv(ac.res_path("add5_clf_curve.csv"))
sum_df = pd.read_csv(ac.res_path("add5_clf_summary.csv"))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
colors = {"P2_Full": ac.C_FULL, "P2_NoExploration": ac.C_NOEXPLO,
          "Random": ac.PALETTE["mid_gray"], "Entropy": ac.PALETTE["mid_teal"]}
for ax, (ds, c) in zip(axes, DATASETS):
    for method in METHODS:
        sub = sum_df[(sum_df.dataset == ds) & (sum_df.method == method)].sort_values("budget")
        x = sub["budget"].values * 100
        ax.plot(x, sub["acc_mean"].values, "-o", ms=4, lw=1.6, color=colors[method], label=method)
        ac.ci_band(ax, x, sub["ci_lo"].values, sub["ci_hi"].values, colors[method], alpha=0.15)
    ax.set_title(ds.capitalize()); ax.set_xlabel("Labeled budget (%)"); ax.set_ylim(0.5, 0.95)
    ac.style_ax(ax)
axes[0].set_ylabel("Classification ACC (unlabeled pool)")
axes[1].legend(loc="lower right")
png = ac.savefig(fig, "add5_clf_curve", combo=True)

u = ac.new_unit(5,
    "Cross-task softmax classification on fashion / usps (P2 score on LR probs)",
    "exploratory/post-hoc", "done")
u["annotations"] = ["exploratory", "post-hoc",
    "backend = SGDClassifier(loss='log_loss', max_iter=30) for speed on the 256-dim pool",
    "U = softmax predict_proba; P2 score w^rob + rho*||u-u_bar||_1 reused",
    "batched vectorized KL-DRO gain (gamma Dirichlet, 40-pt log tau grid)",
    "fixed lam_red=0.01 (data-dependent beta~0.009; n x n eigendecomp skipped under multi-worker memory pressure)",
    "cosine redundancy computed row-wise (no dense n x n matrix)",
    "local pkl subsamples: fashion N=5000 d=50, usps N=5000 d=256; c=10",
    "20 paired seeds; budgets 5/10/15/20%"]
u["protocol"] = {"datasets": [d for d, _ in DATASETS], "methods": METHODS,
                 "n_seeds": 20, "budgets": BUDGETS,
                 "factors": {"L0_frac": ac.INIT_FRAC, "alpha0": ac.ALPHA0_MAIN,
                             "M_MC": ac.M_MC, "rho": ac.RHO, "kappa": 1.0,
                             "backend": "SGDClassifier log_loss"},
                 "checkpoint": 0.10}
u["csvs"] = ["add5_clf_curve.csv", "add5_clf_summary.csv"]
u["repro_script"] = "add5_clf.py"
ac.add_figure(u, "add5_clf_curve",
    "Fig. Classification accuracy vs labeled budget on fashion and usps; "
    "solid lines = mean over 20 paired seeds, shaded band = 95% paired-t CI.")

def means_at(ds, bud):
    w = long_df[(long_df.dataset == ds) & (np.isclose(long_df.budget, bud))].pivot(index="seed", columns="method", values="acc")
    return {m: float(w[m].mean()) for m in METHODS}
f10 = means_at("fashion", 0.10)
u["results"] = (
    "Paradigm transfer: treating softmax outputs as the P2 membership U, the "
    "KL-robust score plus h_i=||u_i-u_bar||_1 ranks unlabeled points. On "
    "fashion at the 10%% budget the four-method mean accuracy is "
    "P2_Full=%.3f, P2_NoExploration=%.3f, Random=%.3f, Entropy=%.3f "
    "(20 paired seeds, 95%% CI in add5_clf_summary.csv). Curves over "
    "5/10/15/20%% are shown for fashion and usps."
    % (f10["P2_Full"], f10["P2_NoExploration"], f10["Random"], f10["Entropy"]))
u["conclusion"] = (
    "Exploratory/post-hoc cross-task check: the P2 acquisition formula is "
    "mechanical and carries over to the classification backend, but this is NOT "
    "a confirmatory clustering result; report it as a robustness probe with the "
    "attached paired CIs and Holm-adjusted p-values.")
uj = ac.save_unit(u)
print("WROTE:", png)
print("WROTE:", uj)
