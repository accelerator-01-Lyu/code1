# -*- coding: utf-8 -*-
"""add1_aggregate.py — Task 1 (add1): aggregate the budget curve.

Combines:
  * frozen shards (code/results/shards/, READ-ONLY) at budgets 5/10/15/20%,
    seed_set="30", l0=0, eta=0, alpha0=50, lam_mult=1,
    for ecoli / compound / glass, methods P2_Full vs P2_NoExploration;
  * add1_new_budget_raw.csv for the NEW points 1/2/3/7/25/30%.

Determinism cross-check: the "high" jobs reproduce budgets 5/10/15/20% from the
identical seeded trajectory; we assert they match the shards to <1e-9, which
validates that 25/30% genuinely continue the frozen trajectory. The aggregate
nevertheless uses shard rows for 5/10/15/20 per protocol.

PARTIAL-DATASET GATE: a dataset enters the curve only once BOTH methods have all
30 seeds at the new points (the runner may still be going on later datasets).
Pending datasets are noted in unit.json as "partial / pending" and left blank in
the figure until the next rerun.

Per (dataset, budget, metric): pair Full vs NoExploration by seed, report
gap = Full - NoExploration, mean, 95% paired-t CI, paired t, raw p, Cohen d_z.

Outputs:
  results/additional/add1_budget_curve_paired.csv
  figures/additional/add1_budget_curve.png/.pdf
  results/additional/add1_unit.json
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

ac.setup_style()
SHARDS = os.path.join(ac.HERE, "results", "shards")
DATASETS = ["ecoli", "compound", "glass"]
METHODS = ["P2_Full", "P2_NoExploration"]
SHARD_BUDGETS = [0.05, 0.10, 0.15, 0.20]
NEW_USE = {0.01: "low", 0.02: "low", 0.03: "low", 0.07: "mid",
           0.25: "high", 0.30: "high"}
METRICS = ["ACC", "NMI", "ARI"]

# ---- 1. load shards --------------------------------------------------------- #
frames = []
for fp in glob.glob(os.path.join(SHARDS, "*.csv")):
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if not {"budget", "ACC", "dataset", "method", "seed"}.issubset(df.columns):
        continue
    frames.append(df)
big = pd.concat(frames, ignore_index=True)
big = big[(big.dataset.isin(DATASETS)) & (big.method.isin(METHODS))
          & (big.seed_set.astype(str) == "30") & (big.l0 == 0)
          & (np.isclose(big.eta, 0.0)) & (np.isclose(big.alpha0, 50.0))
          & (np.isclose(big.lam_mult, 1.0))
          & (big.budget.isin(SHARD_BUDGETS))]
print("shard rows:", len(big))

# ---- 2. load new runs (dedupe in memory only; do not rewrite the live CSV) -- #
new = pd.read_csv(ac.res_path("add1_new_budget_raw.csv"))
new = new.drop_duplicates(subset=["dataset", "method", "seed", "job_type", "budget"])

# ---- partial-dataset gate -------------------------------------------------- #
ACTIVE = []
for ds in DATASETS:
    ok = all(new[(new.dataset == ds) & (new.method == m)].seed.nunique() >= 30
             for m in METHODS)
    if ok:
        ACTIVE.append(ds)
PENDING = [d for d in DATASETS if d not in ACTIVE]
print("active:", ACTIVE, "| pending:", PENDING)

# determinism cross-check on ACTIVE datasets only
chk = new[new.job_type == "high"]
maxdiff = 0.0
for ds in ACTIVE:
    for m in METHODS:
        for bud in SHARD_BUDGETS:
            a = big[(big.dataset == ds) & (big.method == m) & (big.budget == bud)].sort_values("seed")
            b = chk[(chk.dataset == ds) & (chk.method == m) & (chk.budget == bud)].sort_values("seed")
            assert len(a) == len(b) == 30, (ds, m, bud, len(a), len(b))
            for met in METRICS:
                d = float(np.abs(a[met].values - b[met].values).max())
                maxdiff = max(maxdiff, d)
print("determinism max |reproduced - shard| = %.3e" % maxdiff)
assert maxdiff < 1e-9, "trajectory reproduction FAILED; 25/30% would not continue the frozen path"

# assemble the analysis table: shard rows for 5/10/15/20 + chosen new rows
keep_new = []
for bud, jt in NEW_USE.items():
    keep_new.append(new[(new.budget == bud) & (new.job_type == jt)])
new_use = pd.concat(keep_new, ignore_index=True)
new_use = new_use[new_use.dataset.isin(ACTIVE)]
cols = ["budget", "NMI", "ARI", "ACC", "dataset", "method", "seed"]
shard_use = big[big.dataset.isin(ACTIVE)][cols]
allp = pd.concat([shard_use, new_use[cols]], ignore_index=True)
budgets = sorted(allp.budget.unique())
print("budgets in curve:", budgets)

# ---- 3. paired gaps ---------------------------------------------------------- #
rows = []
for ds in ACTIVE:
    for bud in budgets:
        for met in METRICS:
            f = allp[(allp.dataset == ds) & (allp.method == "P2_Full")
                     & (np.isclose(allp.budget, bud))].sort_values("seed")
            ne = allp[(allp.dataset == ds) & (allp.method == "P2_NoExploration")
                      & (np.isclose(allp.budget, bud))].sort_values("seed")
            m = f.merge(ne, on="seed", suffixes=("_f", "_ne"))
            diff = m[f"{met}_f"].values - m[f"{met}_ne"].values
            r = ac.paired_ci(diff)
            rows.append({"dataset": ds, "budget": float(bud), "metric": met,
                         "gap_mean": r["mean"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                         "t": r["t"], "p": r["p"], "dz": ac.cohen_dz(diff),
                         "n": r["n"]})
res = pd.DataFrame(rows)
ac.save_csv(res, "add1_budget_curve_paired.csv")

# ---- 4. figure: rows = all DATASETS, cols = metrics, gap vs budget ---------- #
fig, axes = plt.subplots(len(DATASETS), 3, figsize=(10.5, 3.0 * len(DATASETS)),
                         sharex=True)
if len(DATASETS) == 1:
    axes = np.array([axes])
for i, ds in enumerate(DATASETS):
    for j, met in enumerate(METRICS):
        ax = axes[i, j]
        ac.style_ax(ax)
        if ds not in ACTIVE:
            ax.text(0.5, 0.5, "%s\npartial / pending (still running)" % ds,
                    transform=ax.transAxes, ha="center", va="center",
                    color=ac.PALETTE["mid_gray"], fontsize=9)
            continue
        sub = res[(res.dataset == ds) & (res.metric == met)].sort_values("budget")
        x = sub.budget.values * 100.0
        ax.axhline(0.0, color=ac.PALETTE["mid_gray"], lw=0.9, ls="--")
        ac.ci_band(ax, x, sub.ci_lo.values, sub.ci_hi.values, ac.C_FULL, alpha=0.15)
        ax.plot(x, sub.gap_mean.values, "-o", color=ac.C_FULL, ms=4, lw=1.4,
                zorder=3)
        if i == 0:
            ax.set_title(met, fontsize=10)
        if j == 0:
            ax.set_ylabel("%s\ngap (Full $-$ NoExploration)" % ds, fontsize=9)
        if i == len(DATASETS) - 1:
            ax.set_xlabel("labeled budget (%)")
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
fig.tight_layout()
png = ac.savefig(fig, "add1_budget_curve", combo=True)

# ---- 5. unit.json ----------------------------------------------------------- #
u = ac.new_unit(1,
    "Budget-Curve Extension (1-30%): Where Does the Exploration Bonus Matter?",
    "supportive", "complete")
u["support"] = "mixed"
u["annotations"] = ["supportive", "post-hoc", "exploratory",
                    "new budget points 1/2/3/7/25/30% re-run; 5/10/15/20% from frozen shards",
                    "30 seeds, paired by seed"]
if PENDING:
    u["annotations"].append("partial / pending: " + ", ".join(PENDING))
u["motivation"] = ("The frozen main curve only samples 5/10/15/20% labeled. To locate "
    "where the exploration bonus $h_i=\\|u_i-\\bar u\\|_1$ starts (and stops) "
    "separating P2_Full from P2_NoExploration, we extend the curve to 1, 2, 3, 7, "
    "25 and 30% on the harder benchmarks using the identical frozen AL loop "
    "(|L0|=max(c,floor(0.02N)), M=50 MC draws, $\\alpha_0$=50, one greedy batch "
    "per checkpoint).")
u["design"] = ("Datasets: " + ", ".join(ACTIVE) + (" (plus " + ", ".join(PENDING) +
    " still running, pending)" if PENDING else "; all three benchmarks complete") +
    ". Two methods (P2_Full vs P2_NoExploration); 30 seeds paired by seed; budgets "
    "1,2,3,5,7,10,15,20,25,30%; metrics ACC (Hungarian), NMI, ARI. Points "
    "5/10/15/20% are the frozen shards; the new points continue the identical "
    "seeded trajectory (verified: reproduced 5/10/15/20% match the shards to <1e-9).")
u["protocol"] = {"datasets": ACTIVE, "pending_datasets": PENDING,
                "methods": METHODS, "n_seeds": 30,
                "budgets": budgets, "factors": {"checkpoint": 0.10,
                "determinism_max_abs_diff": maxdiff},
                "checkpoint": 0.10}
hdr = ["Dataset", "budget%", "metric", "gap mean", "95% CI lo", "95% CI hi", "t", "p", "d_z"]
trows = []
for ds in ACTIVE:
    for bud in budgets:
        r = res[(res.dataset == ds) & (np.isclose(res.budget, bud) & (res.metric == "ACC"))].iloc[0]
        trows.append([ds, "%.0f" % (bud * 100), "ACC", "%+.4f" % r.gap_mean,
                      "%+.4f" % r.ci_lo, "%+.4f" % r.ci_hi, "%.2f" % r.t,
                      "%.3f" % r.p, "%.2f" % r.dz])
ac.add_table(u, "Table. Paired ACC gap (P2_Full minus P2_NoExploration) across the extended budget range.",
             hdr, trows,
             notes=["n=30 seeds per cell, paired by seed; ACC after Hungarian matching."] +
             (["5/10/15/20% from frozen shards; 1/2/3/7/25/30% newly run with the identical loop."]
              if not PENDING else [
              "5/10/15/20% from frozen shards; 1/2/3/7/25/30% newly run with the identical loop.",
              ("Pending (not yet in this table): " + ", ".join(PENDING) +
               " — row to be added when the rerun finishes.")]) +
             ["Determinism check: reproduced 5/10/15/20% match shards to <1e-9."])
ac.add_figure(u, "add1_budget_curve",
              "Fig. Paired gap = P2_Full minus P2_NoExploration (rows: "
              + ", ".join(DATASETS) + "; columns: ACC, NMI, ARI) against labeled "
              "budget %. Band = 95% paired-t CI; dashed line at zero gap.", combo=True)


def gap_at(ds, bud, met="ACC"):
    r = res[(res.dataset == ds) & np.isclose(res.budget, bud) & (res.metric == met)].iloc[0]
    return r.gap_mean, r.ci_lo, r.ci_hi


narr = []
for ds in ACTIVE:
    g10, l10, h10 = gap_at(ds, 0.10)
    g30, l30, h30 = gap_at(ds, 0.30)
    narr.append("%s: @10%% gap=%+.4f [%+.4f,%+.4f]; @30%% gap=%+.4f [%+.4f,%+.4f] (ACC)"
                % (ds, g10, l10, h10, g30, l30, h30))
u["results"] = ("ACC paired gaps (Full - NoExploration): " + "; ".join(narr) +
    ". The full curve (CSV add1_budget_curve_paired.csv) also reports NMI and ARI "
    "at every budget." +
    (" Pending: " + ", ".join(PENDING) + " rows are not yet included." if PENDING else ""))
u["conclusion"] = ("The extended budget curve is supportive/post-hoc evidence about "
    "WHERE the exploration bonus matters; it does not by itself constitute a "
    "confirmatory superiority claim (the formal P1 noninferiority test lives in the "
    "main preregistered analysis).")
u["csvs"] = ["add1_budget_curve_paired.csv", "add1_new_budget_raw.csv"]
u["repro_script"] = "add1_aggregate.py"
uj = ac.save_unit(u)
print("WROTE:", ac.res_path("add1_budget_curve_paired.csv"))
print("WROTE:", png)
print("WROTE:", uj)
for ds in ACTIVE:
    for bud in (0.10, 0.30):
        g, l, h = gap_at(ds, bud)
        print("KEY %s @%.0f%% ACC gap=%+.4f CI[%+.4f,%+.4f]" % (ds, bud * 100, g, l, h))
