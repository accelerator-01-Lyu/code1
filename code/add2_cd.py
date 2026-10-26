# -*- coding: utf-8 -*-
"""add2_cd.py — Experiment 2: Friedman + Nemenyi critical-difference diagram.
Aggregates EXISTING main shards (no re-running) at budget=0.10, 6 datasets,
8 formally-ranked methods (+ OriginalP2 as an unranked archived-bug control).
Outputs: results/additional/add2_cd.csv, add2_cd.png/.pdf, add2_unit.json.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

ac.setup_style()
SHARDS = os.path.join(ac.HERE, "results", "shards")

# ---- 1. aggregate existing shards at budget = 0.10 -------------------------- #
frames = []
for fp in glob.glob(os.path.join(SHARDS, "*.csv")):
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if "budget" not in df.columns or "ACC" not in df.columns:
        continue
    sub = df[np.isclose(df["budget"], 0.10)]
    if len(sub):
        frames.append(sub)
big = pd.concat(frames, ignore_index=True)
print("rows at budget=0.10:", len(big))
print("datasets:", sorted(big["dataset"].unique()))
print("methods in shards:", sorted(big["method"].unique()))

RANKED = ac.RANKED_METHODS  # 8 methods, OriginalP2 excluded from ranking
CONTROL = "OriginalP2"

# keep only frozen main shard (alpha0=50, lam_mult=1, no outlier injection)
big = big[(np.isclose(big["alpha0"], 50.0)) & (np.isclose(big["lam_mult"], 1.0))]

datasets = sorted(big["dataset"].unique())

# ---- 2. per-dataset method ACC (mean over seeds) -------------------------- #
rows = []
score_matrix = np.zeros((len(datasets), len(RANKED)))
seed_counts = {}
for i, ds in enumerate(datasets):
    for j, m in enumerate(RANKED):
        vals = big[(big["dataset"] == ds) & (big["method"] == m)]["ACC"].values
        score_matrix[i, j] = vals.mean()
        seed_counts[(ds, m)] = len(vals)
        rows.append({"dataset": ds, "method": m, "n_seeds": len(vals),
                     "ACC_mean": float(vals.mean()),
                     "ACC_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0})

agg = pd.DataFrame(rows)
csv_path = ac.save_csv(agg, "add2_cd_dataset_method_acc.csv")

# control (unranked) for the table
ctrl_rows = []
for ds in datasets:
    vals = big[(big["dataset"] == ds) & (big["method"] == CONTROL)]["ACC"].values
    ctrl_rows.append({"dataset": ds, "method": CONTROL, "n_seeds": len(vals),
                      "ACC_mean": float(vals.mean()) if len(vals) else np.nan})
ctrl = pd.DataFrame(ctrl_rows)

# ---- 3. Friedman + Nemenyi ------------------------------------------------ #
res = ac.friedman_nemenyi(score_matrix, RANKED)
print("avg_rank:", json.dumps(res["avg_rank"], indent=2))
print("Friedman chi2=%.3f p=%.4g  CD=%.3f" % (res["chi2"], res["p"], res["CD"]))

# ---- 4. draw CD diagram --------------------------------------------------- #
order = sorted(RANKED, key=lambda m: res["avg_rank"][m])
ranks = [res["avg_rank"][m] for m in order]
k = len(RANKED)
cd = res["CD"]

fig, ax = plt.subplots(figsize=(8.2, 3.4))
lo, hi = 1.0, float(k)
ax.set_xlim(lo - 0.6, hi + 0.6)
ax.set_ylim(-2.1, 2.3)
ax.axis("off")
# baseline
ax.hlines(0, lo, hi, color=ac.PALETTE["ink"], lw=1.0)
for mval in np.arange(lo, hi + 0.001, 1.0):
    ax.vlines(mval, -0.06, 0.06, color=ac.PALETTE["ink"], lw=0.8)
# CD scale bar at top
ax.hlines(1.35, lo, lo + cd, color=ac.PALETTE["mid_gray"], lw=2.2)
ax.text(lo + cd / 2, 1.55, "CD = %.2f" % cd, ha="center",
        color=ac.PALETTE["mid_gray"], fontsize=9)

# method points: greedy vertical-layer assignment so nearby ranks do not overlap
YCHOICES = [0.55, 1.15, -0.55, -1.15]
levels = []
for m, r in zip(order, ranks):
    y = YCHOICES[0]
    for cand in YCHOICES:
        if not any(abs(r - r0) < 0.75 and cand == y0 for (r0, y0) in levels):
            y = cand
            break
    levels.append((r, y))
    ax.vlines(r, 0, y * 0.78, color=ac.method_color(m), lw=1.0)
    ax.plot(r, 0, "o", color=ac.method_color(m), ms=6, zorder=3)
    ax.text(r, y, m, ha="center", va="bottom" if y > 0 else "top",
            color=ac.PALETTE["ink"], fontsize=8.5)

# cluster bars: connect methods whose rank gap <= CD (no significant diff)
ybar = -1.15
for a in range(len(order)):
    for b in range(a + 1, len(order)):
        if abs(ranks[a] - ranks[b]) <= cd + 1e-9:
            ax.hlines(ybar, ranks[a], ranks[b],
                      color=ac.PALETTE["mid_gray"], lw=3.0)
            ybar -= 0.18
ax.invert_xaxis()  # best (rank 1) on the left
fig.tight_layout()
png = ac.savefig(fig, "add2_cd_diagram", combo=True)

# ---- 5. unit.json --------------------------------------------------------- #
u = ac.new_unit(2,
    "Friedman + Nemenyi Critical-Difference Comparison over Six Benchmarks",
    "supportive", "complete")
u["support"] = "mixed"
u["annotations"] = ["supportive (post-hoc ranking over existing 10% checkpoint)"]
u["motivation"] = ("To assess overall robustness beyond pairwise E. coli/Compound "
    "contrasts, we rank the eight acquisition strategies jointly across the six "
    "benchmarks at the frozen 10% checkpoint using a Friedman test followed by a "
    "Nemenyi post-hoc critical-difference (CD) diagram.")
u["design"] = ("Six datasets (Iris, Wine, Glass, E.coli, Aggregation, Compound); "
    "eight formally-ranked methods (P2_Full, P2_NoExploration, P2_NoRobust, "
    "P2_NoRedundancy, Random, Entropy, CoreSet, BADGE); ACC@10%, mean over the "
    "30-seed confirmatory set per dataset. OriginalP2 is shown as an unranked "
    "archived-bug control.")
u["protocol"] = {"datasets": datasets, "methods": RANKED + [CONTROL],
                 "n_seeds": 30, "budgets": [0.10],
                 "factors": {"friedman_alpha": 0.05, "nemenyi_cd": cd,
                             "friedman_chi2": res["chi2"], "friedman_p": res["p"]},
                 "checkpoint": 0.10}
# table: per-dataset ACC means
hdr = ["Dataset"] + RANKED + ["OriginalP2 (unranked)"]
trows = []
for ds in datasets:
    row = [ds]
    for m in RANKED:
        v = agg[(agg.dataset == ds) & (agg.method == m)]["ACC_mean"]
        row.append("%.3f" % float(v.iloc[0]) if len(v) else "-")
    cv = ctrl[ctrl.dataset == ds]["ACC_mean"]
    row.append("%.3f" % float(cv.iloc[0]) if len(cv) and not np.isnan(cv.iloc[0]) else "-")
    trows.append(row)
# mean-rank summary row
rank_row = ["Mean rank"] + ["%.2f" % res["avg_rank"][m] for m in RANKED] + ["-"]
trows.append(rank_row)
ac.add_table(u, "Table. ACC@10% (mean over 30 seeds) and mean ranks across benchmarks.",
             hdr, trows,
             notes=["Friedman chi2=%.3f, p=%.4g (alpha=0.05); Nemenyi CD=%.2f." % (res["chi2"], res["p"], cd),
                    "OriginalP2 carries five archived implementation bugs and is excluded from the formal ranking.",
                    "ACC after Hungarian matching; frozen alpha0=50, lambda multiplier=1."])
ac.add_figure(u, "add2_cd_diagram",
              "Fig. Critical-difference diagram. Methods are positioned by mean rank "
              "(best on the left); methods joined by a grey bar do not differ "
              "significantly under Nemenyi (rank gap <= CD).", combo=True)
u["results"] = ("The global Friedman test is NOT significant at alpha=0.05 "
    "(chi2=%.3f, p=%.4g); under the standard decision rule we treat the Nemenyi "
    "CD diagram as descriptive only and declare no pairwise contrast significant. "
    "Mean ranks (best=1): CoreSet=%.2f, P2_NoRedundancy=%.2f, P2_NoExploration=%.2f, "
    "P2_NoRobust=%.2f, P2_Full=%.2f, Entropy=%.2f, Random=%.2f, BADGE=%.2f; "
    "Nemenyi CD=%.2f."
    % (res["chi2"], res["p"],
       res["avg_rank"]["CoreSet"], res["avg_rank"]["P2_NoRedundancy"],
       res["avg_rank"]["P2_NoExploration"], res["avg_rank"]["P2_NoRobust"],
       res["avg_rank"]["P2_Full"], res["avg_rank"]["Entropy"],
       res["avg_rank"]["Random"], res["avg_rank"]["BADGE"], cd))
u["conclusion"] = ("Because the Friedman test fails to reach significance (p=%.3f > "
    "0.05) and no method pair exceeds CD=%.2f, the apparent ordering (CoreSet on "
    "top, the three P2 variants in the middle, BADGE last) is an underpowered, "
    "descriptive ranking over only six benchmarks, not a confirmatory advantage "
    "of any acquisition strategy. P2_NoExploration does not significantly "
    "underperform or outperform P2_Full on this aggregate view."
    % (res["p"], cd))
u["annotations"] = ["supportive", "post-hoc descriptive", "underpowered",
                    "Friedman not significant (p=%.3f)" % res["p"]]
u["repro_script"] = "add2_cd.py"
uj = ac.save_unit(u)

print("WROTE:", csv_path)
print("WROTE:", png)
print("WROTE:", uj)
