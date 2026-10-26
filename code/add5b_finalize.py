# -*- coding: utf-8 -*-
"""Build add5b binary figure and update add5_unit.json (binary tight-budget)."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import additional_common as ac

ac.setup_style()
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

long_df = pd.read_csv(ac.res_path("add5b_binary.csv"))
rows = []
for (method, bud), sub in long_df.groupby(["method", "budget"]):
    vals = sub["acc"].values
    m = vals.mean(); sd = vals.std(ddof=1)
    se = sd / np.sqrt(len(vals)); tc = ac.stats.t.ppf(0.975, len(vals) - 1)
    rows.append({"method": method, "budget": bud, "acc_mean": m,
                 "ci_lo": m - tc * se, "ci_hi": m + tc * se})
sum_df = pd.DataFrame(rows)
ac.save_csv(sum_df, "add5b_binary_summary.csv")

fig, ax = plt.subplots(figsize=(6.4, 4.0))
colors = {"P2_Full": ac.C_FULL, "P2_NoExploration": ac.C_NOEXPLO,
          "Random": ac.PALETTE["mid_gray"], "Entropy": ac.PALETTE["mid_teal"]}
for method in METHODS:
    sub = sum_df[sum_df.method == method].sort_values("budget")
    x = sub["budget"].values * 100
    ax.plot(x, sub["acc_mean"].values, "-o", ms=4, lw=1.6, color=colors[method], label=method)
    ac.ci_band(ax, x, sub["ci_lo"].values, sub["ci_hi"].values, colors[method], alpha=0.15)
ax.set_xlabel("Labeled budget (%)"); ax.set_ylabel("Binary ACC (T-shirt vs Shirt)")
ax.set_title("Fashion binary, tight budget"); ax.legend(loc="lower right")
ac.style_ax(ax)
png = ac.savefig(fig, "add5b_binary_curve", combo=True)

# update add5_unit.json
uj = os.path.join(ac.RES_ADD, "add5_unit.json")
import json
u = json.load(open(uj, encoding="utf-8"))
u["annotations"] = u.get("annotations", []) + [
    "S5 add-on: fashion binary T-shirt/top(0) vs Shirt(6), N=2000, 2-class",
    "backend LogisticRegression(lbfgs, max_iter=1000) for the binary task",
    "tight budgets 1/2/3/5/7/10/15/20/25/30%, L0=0.5%",
    "10-class fashion/usps results are exploratory; 2-class tight-budget is the primary S5 result"]
u["csvs"] = list(set(u.get("csvs", []) + ["add5b_binary.csv", "add5b_binary_summary.csv"]))
u.setdefault("fig_files", [])
u["figures"].append({"file": "add5b_binary_curve.png",
    "caption": "Fig. Binary fashion T-shirt vs Shirt accuracy vs tight labeled budget; "
               "mean over 20 paired seeds, shaded 95% paired-t CI."})
# narrative
w = long_df[np.isclose(long_df.budget, 0.10)].pivot(index="seed", columns="method", values="acc")
m10 = {m: float(w[m].mean()) for m in METHODS}
u["results"] = u.get("results", "") + (
    " S5 tight-budget binary add-on (T-shirt vs Shirt, N=2000): at 10%% budget the "
    "mean accuracy is P2_Full=%.3f, P2_NoExploration=%.3f, Random=%.3f, Entropy=%.3f "
    "(20 paired seeds, 10 budget points from 1%% to 30%%)."
    % (m10["P2_Full"], m10["P2_NoExploration"], m10["Random"], m10["Entropy"]))
json.dump(u, open(uj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("WROTE", png)
print("WROTE", uj)
print("binary @10%:", {k: round(v, 3) for k, v in m10.items()})
