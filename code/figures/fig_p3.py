"""fig_p3_lambda_grid.py — final P3 lambda2/n multiplier grid (real P3 shards)."""
import os, glob, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)              # repository code/ root
sys.path.insert(0, CODE)
from figures.palette import COLORS, INK

RES = os.path.join(CODE, "results"); FIG = os.path.join(CODE, "figures", "final")
DATASETS = ["iris", "wine", "glass", "ecoli", "aggregation", "compound"]
LAMS = [0, .25, .5, .75, 1, 1.5, 2, 3]
VERDICT = {"iris": ("REJECTED", .25), "wine": ("REJECTED", .25),
           "glass": ("REJECTED", 0.0), "ecoli": ("CONFIRMED", 2.0),
           "aggregation": ("REJECTED", .75), "compound": ("REJECTED", 3.0)}
frames = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, "shards", "*.csv"))
          if "diagnostics" not in f]
long = pd.concat(frames, ignore_index=True)
p3 = long[long.seed_set.astype(str) == "P3"]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ECECF1",
                     "grid.linewidth": .6})
fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0), sharex=True)
for ax, ds in zip(axes.flat, DATASETS):
    means, los, his, ns = [], [], [], []
    for lm in LAMS:
        v = p3[(p3.dataset == ds) & (p3.lam_mult == lm) & (p3.budget == .10)].ACC.values
        n = len(v); m = v.mean()
        se = v.std(ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(.975, n - 1)
        means.append(m); los.append(m - tc * se); his.append(m + tc * se); ns.append(n)
    assert min(ns) >= 30, f"{ds} incomplete P3 grid (min n={min(ns)})"
    ax.axvspan(.5, 2, color="#DDE6F3", alpha=.55, zorder=0)
    ax.plot(LAMS, means, color=COLORS["P2_Full"], lw=1.5, marker="o",
            ms=3, zorder=3)
    ax.fill_between(LAMS, los, his, color=COLORS["P2_Full"], alpha=.12,
                    zorder=2)
    vd, best = VERDICT[ds]
    bi = LAMS.index(best)
    bc = "#D08AB6" if vd == "CONFIRMED" else "#B8C4D4"
    ax.scatter([LAMS[bi]], [means[bi]], s=46, facecolor=bc,
               edgecolor=INK, linewidths=.7, zorder=4)
    ax.set_title(f"{ds}: {vd} (best={best:g})", fontsize=7.5,
                 color=INK)
    ax.tick_params(labelsize=6.5)
for ax in axes[1, :]:
    ax.set_xlabel(r"$\lambda_2/n$ multiplier", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("ACC at 10% budget", fontsize=8)
fig.text(.01, .006,
         "30 seeds, P2_NoExploration, 10% budget; shaded band: pre-registered "
         "(0.5, 2) neighborhood; pink = neighborhood confirmed, grey = rejected.",
         fontsize=6, color=INK, style="italic")
plt.tight_layout(rect=(0, .03, 1, 1))
plt.savefig(os.path.join(FIG, "fig_p3_lambda_grid.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG, "fig_p3_lambda_grid.png"), dpi=300,
            bbox_inches="tight")
print("p3 fig done ->", os.path.join(FIG, "fig_p3_lambda_grid.png"))
