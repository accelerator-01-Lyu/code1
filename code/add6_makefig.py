# -*- coding: utf-8 -*-
"""add6_makefig.py — build heatmap + unit.json from whatever cells exist in CSV.
Runs standalone; call now for ecoli-only, again after compound finishes."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

ac.setup_style()
RHO_GRID = [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0]
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

df = pd.read_csv(ac.res_path("add6_rhobeta_grid.csv"))
FORCE = sys.argv[1] if len(sys.argv) > 1 else None
if FORCE:
    present_ds = [FORCE]
else:
    present_ds = [d for d in ["ecoli", "compound"] if d in df["dataset"].values]
print("datasets present:", present_ds, "cells:", len(df))

cmap = ac.gap_colormap()
ncol = len(present_ds)
fig, axes = plt.subplots(1, ncol, figsize=(4.6*ncol, 4.2), squeeze=False)
axes = axes[0]
vmin = float(df["acc_mean"].min()); vmax = float(df["acc_mean"].max())
im = None
for ax, ds in zip(axes, present_ds):
    sub = df[df.dataset == ds]
    M = np.zeros((len(RHO_GRID), len(BETA_GRID)))
    for i, rm in enumerate(RHO_GRID):
        for j, bm in enumerate(BETA_GRID):
            row = sub[(sub.rho == rm) & (sub.beta == bm)]["acc_mean"]
            M[i, j] = float(row.iloc[0]) if len(row) else np.nan
    im = ax.imshow(M, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(BETA_GRID))); ax.set_xticklabels([("%.2g" % b) for b in BETA_GRID])
    ax.set_yticks(range(len(RHO_GRID))); ax.set_yticklabels([("%.2g" % r) for r in RHO_GRID])
    ax.set_xlabel("beta multiplier  (x lambda2(L_R)/n)")
    ax.set_ylabel("rho multiplier  (x 0.1)")
    ax.set_title(ds + ("" if ncol == 2 else " (preliminary, compound pending)"), fontsize=10)
    for i in range(len(RHO_GRID)):
        for j in range(len(BETA_GRID)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                        fontsize=7.5, color=ac.PALETTE["ink"])
    bi = RHO_GRID.index(1.0); bj = BETA_GRID.index(1.0)
    ax.add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1, fill=False,
                               edgecolor=ac.PALETTE["ink"], lw=1.6))
if im is not None:
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="ACC @10%")
fig.tight_layout()
png = ac.savefig(fig, "add6_rhobeta_heatmap", combo=True)
print("WROTE", png)

# best/worst per present dataset
for ds in present_ds:
    sub = df[df.dataset == ds].reset_index(drop=True)
    ib = sub.acc_mean.idxmax(); iw = sub.acc_mean.idxmin()
    print("[%s] BEST  rho=%s beta=%s ACC=%.4f" % (ds, sub.loc[ib,'rho'], sub.loc[ib,'beta'], sub.loc[ib,'acc_mean']))
    print("[%s] WORST rho=%s beta=%s ACC=%.4f" % (ds, sub.loc[iw,'rho'], sub.loc[iw,'beta'], sub.loc[iw,'acc_mean']))

# unit.json
u = ac.new_unit(6, "Rho x Beta sensitivity grid on ecoli / compound", "post-hoc", "done")
u["support"] = "mixed"
u["annotations"] = ["supportive/post-hoc",
                    "compound pending / ecoli-only" if "compound" not in present_ds else "both datasets complete"]
u["protocol"] = {"datasets": present_ds, "methods": ["P2_Full (coefficient sweep)"],
                 "n_seeds": 30, "budgets": [0.10],
                 "factors": {"rho_grid": RHO_GRID, "beta_grid": BETA_GRID,
                             "rho_nominal": 0.1, "beta_scale": "lambda2(L_R)/n",
                             "baseline": "(rho=1,beta=1)=frozen P2_Full"},
                 "checkpoint": 0.10}
u["repro_script"] = "add6_rhobeta_grid.py"
u["csvs"] = ["add6_rhobeta_grid.csv"]
ac.add_figure(u, "add6_rhobeta_heatmap",
              "Fig. ACC@10% over rho x beta coefficient grid (30 seeds). Boxed cell = frozen P2_Full (rho=1, beta=lambda2/n). Cool diverging colormap (teal=high, grey=mid, purple=low).",
              combo=True)
ac.save_unit(u)
print("WROTE unit.json")
