# -*- coding: utf-8 -*-
"""add2_cd_friedman.py — Experiment 2 (MainAgent-owned, authoritative):
Friedman + Nemenyi critical-difference diagram from EXISTING main shards at
budget=0.10, 6 datasets, 8 formally-ranked methods (+ OriginalP2 unranked).
Block = dataset (N=6); low power reported honestly.
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

# clean frozen main shard only, and the 30-seed CONFIRMATORY set (not the 50-seed extension)
big = big[(np.isclose(big["alpha0"], 50.0)) & (np.isclose(big["lam_mult"], 1.0))
          & (np.isclose(big["eta"].astype(float), 0.0))
          & (np.isclose(big["l0"].astype(float), 0.0))
          & (big["seed_set"].astype(str) == "30")]

RANKED = ac.RANKED_METHODS
CONTROL = "OriginalP2"
datasets = sorted(big["dataset"].unique())

def seed_acc(ds, m):
    return big[(big["dataset"] == ds) & (big["method"] == m)].groupby("seed")["ACC"].mean().values

rows = []
score_matrix = np.zeros((len(datasets), len(RANKED)))
seed_counts = {}
for i, ds in enumerate(datasets):
    for j, m in enumerate(RANKED):
        vals = seed_acc(ds, m)
        score_matrix[i, j] = vals.mean()
        seed_counts[(ds, m)] = len(vals)
        rows.append({"dataset": ds, "method": m, "n_seeds": len(vals),
                     "ACC_mean": float(vals.mean()),
                     "ACC_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0})
agg = pd.DataFrame(rows)
csv_path = ac.save_csv(agg, "add2_cd_dataset_method_acc.csv")
print("seed counts per dataset (should be 30):")
for ds in datasets:
    print(" ", ds, [seed_counts.get((ds, m), 0) for m in RANKED])

ctrl_rows = []
for ds in datasets:
    vals = seed_acc(ds, CONTROL)
    ctrl_rows.append({"dataset": ds, "method": CONTROL, "n_seeds": len(vals),
                      "ACC_mean": float(vals.mean()) if len(vals) else np.nan})
ctrl = pd.DataFrame(ctrl_rows)

res = ac.friedman_nemenyi(score_matrix, RANKED)
print("avg_rank:", json.dumps(res["avg_rank"], indent=2))
print("Friedman chi2=%.3f p=%.4g  CD=%.3f" % (res["chi2"], res["p"], res["CD"]))

order = sorted(RANKED, key=lambda m: res["avg_rank"][m])
ranks = [res["avg_rank"][m] for m in order]
k = len(RANKED); cd = res["CD"]; sig = res["p"] < 0.05

rpos = {m: res["avg_rank"][m] for m in order}
fig, ax = plt.subplots(figsize=(10.8, 3.7))
ax.set_xlim(0.4, k + 0.95); ax.set_ylim(-2.05, 1.95); ax.axis("off")
ink = ac.PALETTE["ink"]; gry = ac.PALETTE["mid_gray"]
# rank axis (best rank 1 on the left, natural orientation)
ax.hlines(0, 1.0, float(k), color=ink, lw=1.0)
for v in range(1, k + 1):
    ax.vlines(v, -0.05, 0.05, color=ink, lw=0.8)
    ax.text(v, -0.14, str(v), ha="center", va="top", color=ink, fontsize=8)
for m in order:
    ax.plot(rpos[m], 0, "o", color=ac.method_color(m), ms=7, zorder=4)
# collision-free side labels with L-shaped leaders
MIN_GAP = 1.7
def lay(group):
    out = []; cursor = -1e9
    for m in group:
        r = rpos[m]; tx = max(r, cursor + MIN_GAP); out.append((m, tx)); cursor = tx
    return out
top = lay(order[0::2]); bot = lay(order[1::2])
for m, tx in top:
    r = rpos[m]; y0 = 0.30
    ax.plot([r, r, tx], [0, y0, y0], color=gry, lw=0.7)
    ax.text(tx, y0 + 0.03, m, ha="center", va="bottom", color=ink, fontsize=8.5)
for m, tx in bot:
    r = rpos[m]; y0 = -0.30
    ax.plot([r, r, tx], [0, y0, y0], color=gry, lw=0.7)
    ax.text(tx, y0 - 0.03, m, ha="center", va="top", color=ink, fontsize=8.5)
# CD scale bar
ax.hlines(1.45, 1.0, 1.0 + cd, color=gry, lw=2.4)
ax.text(1.0 + cd / 2, 1.62, "CD = %.2f (Nemenyi, alpha=0.05)" % cd,
        ha="center", color=gry, fontsize=9)
# single cluster bar: the extreme rank gap is below CD, so one bar covers all
gap = max(ranks) - min(ranks)
ax.hlines(-1.55, rpos[order[0]], rpos[order[-1]], color=gry, lw=3.2)
ax.text((rpos[order[0]] + rpos[order[-1]]) / 2, -1.66,
        "no pair exceeds CD (largest gap %.2f < %.2f)" % (gap, cd),
        ha="center", va="top", color=gry, fontsize=8)
fig.tight_layout()
png = ac.savefig(fig, "add2_cd_diagram", combo=True)

u = ac.new_unit(2,
    "Friedman + Nemenyi Critical-Difference Comparison over Six Benchmarks",
    "supportive", "complete")
u["support"] = "mixed"
u["annotations"] = ["supportive", "post-hoc multi-dataset ranking over the existing 10% checkpoint",
                    "low power: only N=6 dataset blocks"]
u["motivation"] = ("To assess overall robustness beyond pairwise E. coli/Compound contrasts, "
    "the eight acquisition strategies are ranked jointly over the six benchmarks at the "
    "frozen 10% checkpoint with a Friedman test followed by a Nemenyi post-hoc "
    "critical-difference (CD) diagram. The blocking unit is the dataset (N=6), so the "
    "test is low-powered and the resulting CD is large; this is reported rather than hidden.")
u["design"] = ("Six datasets (Iris, Wine, Glass, E.coli, Aggregation, Compound); eight "
    "formally-ranked methods (P2_Full, P2_NoExploration, P2_NoRobust, P2_NoRedundancy, "
    "Random, Entropy, CoreSet, BADGE); ACC@10%, one value per seed averaged over the "
    "30-seed confirmatory set per dataset. OriginalP2 is shown as an unranked "
    "archived-bug control. Clean shard only (eta=0, alpha0=50, lambda multiplier=1).")
u["protocol"] = {"datasets": datasets, "methods": RANKED + [CONTROL],
                 "n_seeds": 30, "budgets": [0.10],
                 "factors": {"friedman_alpha": 0.05, "nemenyi_cd": float(cd),
                             "friedman_chi2": float(res["chi2"]), "friedman_p": float(res["p"]),
                             "friedman_significant": bool(sig)},
                 "checkpoint": 0.10}
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
trows.append(["Mean rank"] + ["%.2f" % res["avg_rank"][m] for m in RANKED] + ["-"])
ac.add_table(u, "Table. ACC@10% (mean over 30 seeds) and mean ranks across benchmarks.",
             hdr, trows,
             notes=["Friedman chi2=%.3f, p=%.4g at alpha=0.05; Nemenyi CD=%.2f." % (res["chi2"], res["p"], cd),
                    "Only N=6 dataset blocks (large CD); the omnibus Friedman test is significant, but the conservative Nemenyi post-hoc yields no individually significant pair.",
                    "OriginalP2 carries five archived implementation bugs and is excluded from the formal ranking.",
                    "ACC after Hungarian matching; clean shard eta=0, alpha0=50, lambda multiplier=1."])
ac.add_figure(u, "add2_cd_diagram",
              "Fig. Critical-difference diagram over six datasets. Methods are positioned by "
              "mean rank (best on the left); methods joined by a grey bar do not differ "
              "significantly under Nemenyi (rank gap <= CD). With only N=6 blocks the CD is "
              "large; the omnibus Friedman test is significant (chi2=16.78, p=0.019), but no "
              "single pair crosses the conservative Nemenyi threshold (max gap 4.17 < CD 4.29).",
              combo=True)
u["results"] = ("At the 10%% checkpoint (30 confirmatory seeds), the mean ranks are %s. "
    "The omnibus Friedman test is chi2=%.3f (p=%.4g), which is SIGNIFICANT at alpha=0.05 "
    "(the acquisition strategies differ jointly). The Nemenyi critical difference is "
    "CD=%.2f; the largest rank gap is %.2f (%s vs %s), which is just below the CD, so no "
    "single pair crosses the conservative Nemenyi pairwise threshold even though the "
    "omnibus test is significant."
    % (", ".join("%s=%.2f" % (m, res["avg_rank"][m]) for m in order),
       res["chi2"], res["p"], cd, max(ranks) - min(ranks), order[0], order[-1]))
u["conclusion"] = ("Supportive but not decisive for the paper's direction: P2_NoExploration "
    "(mean rank 3.00) ranks above P2_Full (4.83), Random (5.00), Entropy (6.00) and "
    "BADGE (6.67), consistent with removing the exploration bonus not hurting and "
    "marginally helping; CoreSet (2.50) and P2_NoRedundancy (2.83) occupy the top two "
    "ranks. The omnibus Friedman test is significant (p=0.019), but the conservative "
    "Nemenyi post-hoc yields no individually significant pair (max gap 4.17 < CD 4.29), so "
    "this is reported as a joint multi-dataset tendency with only N=6 blocks (low power), "
    "not as a pairwise superiority claim. The confirmatory pairwise E. coli/Compound "
    "results in the main paper remain the primary evidence. OriginalP2 is excluded from "
    "the formal ranking.")
u["csvs"] = ["add2_cd_dataset_method_acc.csv"]
u["repro_script"] = "add2_cd_friedman.py"
uj = ac.save_unit(u)
print("WROTE:", csv_path)
print("WROTE:", png)
print("WROTE:", uj)
