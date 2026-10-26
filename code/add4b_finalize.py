# -*- coding: utf-8 -*-
"""Rebuild add4 grouped bar with 3 datasets (segment, letter, letter10) and
update add4_unit.json to include the Pen-Based substitute (letter10)."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac

ac.setup_style()
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
DS_ORDER = ["segment", "letter", "letter10"]

d4 = pd.read_csv(ac.res_path("add4_large_paired.csv"))
d4b = pd.read_csv(ac.res_path("add4b_penbased.csv"))
long_df = pd.concat([d4, d4b], ignore_index=True)

# summary
sum_rows = []
for (ds, metric), sub in long_df.groupby(["dataset", "metric"]):
    wide = sub.pivot(index="seed", columns="method", values="value")
    rows = ac.contrast_table(wide.to_dict(orient="list"), reference="P2_Full",
                             methods=METHODS, metric=metric)
    for method in METHODS:
        vals = wide[method].values
        m = vals.mean(); sd = vals.std(ddof=1); se = sd / np.sqrt(len(vals))
        tc = ac.stats.t.ppf(0.975, len(vals) - 1)
        dz = 0.0; p = 1.0
        if method != "P2_Full":
            dz = rows[method]["dz"]; p = rows[method]["p_holm"]
        sum_rows.append({"dataset": ds, "method": method, "metric": metric,
                         "mean": m, "ci_lo": m - tc*se, "ci_hi": m + tc*se,
                         "dz_vs_full": dz, "p_holm": p})
sum_df = pd.DataFrame(sum_rows)
ac.save_csv(sum_df, "add4_large_summary.csv")

acc = sum_df[sum_df.metric == "ACC"]
fig, ax = plt.subplots(figsize=(7.0, 4.0))
x = np.arange(len(DS_ORDER)); width = 0.18
colors = {"P2_Full": ac.C_FULL, "P2_NoExploration": ac.C_NOEXPLO,
          "Random": ac.PALETTE["mid_gray"], "Entropy": ac.PALETTE["mid_teal"]}
for k, method in enumerate(METHODS):
    means = [acc[(acc.dataset == ds) & (acc.method == method)]["mean"].iloc[0] for ds in DS_ORDER]
    lo = [acc[(acc.dataset == ds) & (acc.method == method)]["ci_lo"].iloc[0] for ds in DS_ORDER]
    hi = [acc[(acc.dataset == ds) & (acc.method == method)]["ci_hi"].iloc[0] for ds in DS_ORDER]
    err = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    ax.bar(x + (k - 1.5)*width, means, width, label=method, color=colors[method],
           yerr=err, capsize=3, error_kw=dict(ecolor=ac.PALETTE["ink"], lw=0.8))
ax.set_xticks(x); ax.set_xticklabels(["Segment", "Letter (26cls, subsamp)", "Letter10 (10cls)"])
ax.set_ylabel("Clustering ACC @10% (Hungarian)"); ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", ncol=2); ac.style_ax(ax)
png = ac.savefig(fig, "add4_large_bar", combo=True)

# update unit.json
uj = os.path.join(ac.RES_ADD, "add4_unit.json")
u = json.load(open(uj, encoding="utf-8"))
u["annotations"] = u.get("annotations", []) + [
    "S4 add-on: Pen-Based Recognition unavailable on OpenML; substituted by letter10 (10-class subset of letter.pkl, N=2985, PCA->16)",
    "letter10 gap NoExploration-Full = +0.001 [-0.001,+0.002], dz=0.16 (no difference)"]
u["csvs"] = list(set(u.get("csvs", []) + ["add4b_penbased.csv"]))
u["protocol"]["datasets"] = DS_ORDER
u["protocol"]["n_seeds"] = 30
json.dump(u, open(uj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# narrative numbers
def gap(ds):
    sub = long_df[(long_df.dataset == ds) & (long_df.metric == "ACC")]
    w = sub.pivot(index="seed", columns="method", values="value")
    d = w["P2_NoExploration"].values - w["P2_Full"].values
    r = ac.paired_ci(d)
    return r["mean"], r["lo"], r["hi"], ac.cohen_dz(d)
g10 = gap("letter10")
open("_chk4.txt", "w").write("letter10 gap %+.3f [%+.3f,%+.3f] dz=%.2f\n" % g10 + "WROTE " + png + "\nWROTE " + uj)
