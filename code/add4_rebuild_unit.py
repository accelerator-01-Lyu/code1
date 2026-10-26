# -*- coding: utf-8 -*-
"""Rebuild add4 figure + unit.json from already-saved CSVs (no re-run)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac

ac.setup_style()
DATASETS = ["segment", "letter"]
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]

long_df = pd.read_csv(ac.res_path("add4_large_paired.csv"))
sum_df = pd.read_csv(ac.res_path("add4_large_summary.csv"))

acc = sum_df[sum_df.metric == "ACC"].copy()
fig, ax = plt.subplots(figsize=(6.4, 4.0))
x = np.arange(len(DATASETS)); width = 0.18
colors = {"P2_Full": ac.C_FULL, "P2_NoExploration": ac.C_NOEXPLO,
          "Random": ac.PALETTE["mid_gray"], "Entropy": ac.PALETTE["mid_teal"]}
for k, method in enumerate(METHODS):
    means = [acc[(acc.dataset == ds) & (acc.method == method)]["mean"].iloc[0] for ds in DATASETS]
    lo = [acc[(acc.dataset == ds) & (acc.method == method)]["ci_lo"].iloc[0] for ds in DATASETS]
    hi = [acc[(acc.dataset == ds) & (acc.method == method)]["ci_hi"].iloc[0] for ds in DATASETS]
    err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    ax.bar(x + (k - 1.5) * width, means, width, label=method, color=colors[method],
           yerr=err, capsize=3, error_kw=dict(ecolor=ac.PALETTE["ink"], lw=0.8))
ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DATASETS])
ax.set_ylabel("Clustering ACC @10% (Hungarian)"); ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", ncol=2); ac.style_ax(ax)
png = ac.savefig(fig, "add4_large_bar", combo=True)

u = ac.new_unit(4,
    "Large-scale clustering: letter (subsampled 26 classes) + segment",
    "supportive/post-hoc/underpowered", "done")
u["annotations"] = ["supportive", "post-hoc", "underpowered",
    "satellite unavailable locally, substituted by segment",
    "letter stratified-subsampled to N=2000 (~77/class); full-N=7990 result right-censored",
    "segment N=2310 d=19 c=7; letter N=1976 d=16 c=26",
    "30 paired seeds; frozen alpha0=50, rho=0.1, M_MC=50"]
u["protocol"] = {"datasets": DATASETS, "methods": METHODS, "n_seeds": 30,
                 "budgets": [0.10],
                 "factors": {"L0_frac": ac.INIT_FRAC, "alpha0": ac.ALPHA0_MAIN,
                             "M_MC": ac.M_MC, "rho": ac.RHO, "letter_subsample": 2000},
                 "checkpoint": 0.10}
u["csvs"] = ["add4_large_paired.csv", "add4_large_summary.csv"]
u["repro_script"] = "add4_large.py"
ac.add_figure(u, "add4_large_bar",
    "Fig. Large-scale clustering ACC @10% (mean over 30 paired seeds, 95% paired-t CI).")

def gap(ds, a="P2_NoExploration"):
    sub = long_df[(long_df.dataset == ds) & (long_df.metric == "ACC")]
    w = sub.pivot(index="seed", columns="method", values="value")
    d = w[a].values - w["P2_Full"].values
    r = ac.paired_ci(d)
    return r["mean"], r["lo"], r["hi"], ac.cohen_dz(d)
g_letter = gap("letter"); g_seg = gap("segment")
u["results"] = (
    "At the 10%% checkpoint, P2_NoExploration minus P2_Full ACC = "
    "%+.3f (95%% CI [%+.3f, %+.3f], d_z=%.2f) on the stratified letter "
    "subsample (N~2000, 26 classes), and %+.3f [%+.3f, %+.3f], d_z=%.2f on "
    "segment (N=2310, 7 classes). Bars show group means with paired-t 95%% "
    "CIs; all 30 seeds share the identical L0. NMI/ARI accompany ACC."
    % (g_letter[0], g_letter[1], g_letter[2], g_letter[3],
       g_seg[0], g_seg[1], g_seg[2], g_seg[3]))
u["conclusion"] = (
    "Supportive/post-hoc, underpowered stress test on a high-class letter "
    "subsample and segment; no confirmatory claim. P2_Full vs "
    "P2_NoExploration separation is reported descriptively with CIs; see "
    "add4_large_summary.csv for Holm-adjusted p-values.")
uj = ac.save_unit(u)
print("WROTE:", png); print("WROTE:", uj)
print("letter gap NoExp-Full = %+.3f [%+.3f, %+.3f]" % (g_letter[0], g_letter[1], g_letter[2]))
print("segment gap NoExp-Full = %+.3f [%+.3f, %+.3f]" % (g_seg[0], g_seg[1], g_seg[2]))
