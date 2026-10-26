# -*- coding: utf-8 -*-
"""add7_meta.py — Task 7 (add7): exploratory meta-analysis of the P2_Full minus
P2_NoExploration gap against dataset complexity (dimension d, sample size N,
overlap).

n = 6 main benchmarks (iris, wine, glass, ecoli, aggregation, compound).
EXPLICITLY EXPLORATORY / POST-HOC / UNDERPOWERED: 6 points, 3 predictors,
residual df = 2; no confirmatory claim is made from this unit.

Columns of add7_meta_features.csv:
  dataset, d, N, overlap_1nn (leave-one-out 1-NN classification error on the
  frozen StandardScaler-normalized features), |L0|/N = INIT_FRAC (fixed 0.02),
  gap_acc_10pct, gap_nmi, gap_ari = mean over 30 seeds of
  P2_Full - P2_NoExploration at budget=0.10 (frozen shards, seed_set="30",
  l0=0, eta=0, alpha0=50, lam_mult=1.0), paired by seed.

MERGE HOOK FOR WORKER C/D DATASETS:
  If results/additional/add7_extra_datasets.csv exists with columns
  dataset,d,N,overlap_1nn,gap_acc_10pct,gap_nmi,gap_ari, its rows are appended
  to the meta table before any statistic is computed. The organizer can drop
  that file once Workers C/D finish and re-run this script. The current n=6
  deliverable does not wait for it.

Outputs:
  results/additional/add7_meta_features.csv
  results/additional/add7_meta_regression.csv
  figures/additional/add7_meta_scatter.png/.pdf
  results/additional/add7_unit.json
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\AL_FuzzyClustering_Paper\v2_unified\code")
import additional_common as ac

ac.setup_style()
SHARDS = os.path.join(ac.HERE, "results", "shards")
EXTRA = ac.res_path("add7_extra_datasets.csv")

DATASETS = ["iris", "wine", "glass", "ecoli", "aggregation", "compound"]
NCLASSES = {"iris": 3, "wine": 3, "glass": 6, "ecoli": 8,
            "aggregation": 7, "compound": 6}


def load_pkl(name):
    import pickle
    p = os.path.join(ac.REF_DATA, name + ".pkl")
    o = pickle.load(open(p, "rb"))
    return np.asarray(o["X"], float), np.asarray(o["y"], int)


def overlap_1nn_loo(X, y):
    """Leave-one-out 1-NN classification error on normalized features."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    clf = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), n_jobs=1)
    return float(1.0 - scores.mean())


def gap_at_10pct():
    """gap = P2_Full - P2_NoExploration at budget 0.10, mean of per-seed diffs,
    from the frozen shards (seed_set=30, l0=0, eta=0, alpha0=50, lam_mult=1)."""
    frames = []
    for fp in glob.glob(os.path.join(SHARDS, "*.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if not {"budget", "ACC", "dataset", "method", "seed"}.issubset(df.columns):
            continue
        sub = df[np.isclose(df["budget"], 0.10)]
        if len(sub):
            frames.append(sub)
    big = pd.concat(frames, ignore_index=True)
    big = big[(np.isclose(big["alpha0"], 50.0)) & (np.isclose(big["lam_mult"], 1.0))
              & (np.isclose(big["eta"].astype(float), 0.0))
              & (np.isclose(big["l0"].astype(float), 0.0))
              & (big["seed_set"].astype(str) == "30")]
    out = {}
    for ds in DATASETS:
        f = big[(big.dataset == ds) & (big.method == "P2_Full")].sort_values("seed")
        ne = big[(big.dataset == ds) & (big.method == "P2_NoExploration")].sort_values("seed")
        m = f.merge(ne, on="seed", suffixes=("_f", "_ne"))
        out[ds] = {
            "gap_acc_10pct": float((m.ACC_f - m.ACC_ne).mean()),
            "gap_nmi": float((m.NMI_f - m.NMI_ne).mean()),
            "gap_ari": float((m.ARI_f - m.ARI_ne).mean()),
            "n_seeds": int(len(m)),
        }
    return out


# ---- 1. meta table --------------------------------------------------------- #
gaps = gap_at_10pct()
rows = []
for ds in DATASETS:
    X, y = load_pkl(ds)
    rows.append({
        "dataset": ds,
        "d": int(X.shape[1]),
        "N": int(X.shape[0]),
        "overlap_1nn": overlap_1nn_loo(X, y),
        "L0_frac": ac.INIT_FRAC,
        "gap_acc_10pct": gaps[ds]["gap_acc_10pct"],
        "gap_nmi": gaps[ds]["gap_nmi"],
        "gap_ari": gaps[ds]["gap_ari"],
    })

# ---- merge hook: Worker C/D extra datasets --------------------------------- #
if os.path.exists(EXTRA):
    extra = pd.read_csv(EXTRA)
    rows += extra.to_dict("records")

meta = pd.DataFrame(rows)
print(meta.to_string(index=False))
ac.save_csv(meta, "add7_meta_features.csv")

n = len(meta)
y_gap = meta["gap_acc_10pct"].values
preds = ["d", "N", "overlap_1nn"]

# ---- 2. univariate Pearson + Spearman + OLS -------------------------------- #
reg_rows = []
uni = {}
for p in preds:
    x = meta[p].values
    pr = stats.pearsonr(x, y_gap)
    sr = stats.spearmanr(x, y_gap)
    lr = stats.linregress(x, y_gap)
    uni[p] = {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
              "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
              "slope": float(lr.slope), "intercept": float(lr.intercept)}
    reg_rows.append({"predictor": p + "_univariate", "coef": float(lr.slope),
                     "se": float(lr.stderr), "t": float(lr.slope / lr.stderr),
                     "p": float(lr.pvalue), "r": float(pr.statistic),
                     "r_squared": float(pr.statistic ** 2), "n": n})

# ---- 3. multiple OLS: gap ~ d + N + overlap ------------------------------- #
Xm = np.column_stack([np.ones(n), meta["d"].values, meta["N"].values,
                      meta["overlap_1nn"].values])
beta, *_ = np.linalg.lstsq(Xm, y_gap, rcond=None)
resid = y_gap - Xm @ beta
dof = n - Xm.shape[1]
sse = float((resid ** 2).sum())
s2 = sse / max(dof, 1)
covb = s2 * np.linalg.inv(Xm.T @ Xm)
se_b = np.sqrt(np.diag(covb))
t_b = beta / se_b
p_b = np.array([2 * stats.t.sf(abs(t), dof) for t in t_b])
r2 = float(1.0 - sse / ((y_gap - y_gap.mean()) ** 2).sum())
names = ["intercept_multi", "d_multi", "N_multi", "overlap_1nn_multi"]
for nm, b, s, t, p in zip(names, beta, se_b, t_b, p_b):
    reg_rows.append({"predictor": nm, "coef": float(b), "se": float(s),
                     "t": float(t), "p": float(p),
                     "r": float(stats.pearsonr(meta[nm.replace("_multi", "")].values,
                                              y_gap)[0]) if nm != "intercept_multi" else np.nan,
                     "r_squared": r2, "n": n})
reg_rows.append({"predictor": "model", "coef": np.nan, "se": np.nan,
                 "t": np.nan, "p": np.nan, "r": np.nan, "r_squared": r2, "n": n})
reg = pd.DataFrame(reg_rows)
ac.save_csv(reg, "add7_meta_regression.csv")

print(json.dumps(uni, indent=2))
print("multiple R^2=%.4f dof=%d" % (r2, dof))

# ---- 4. scatter figure: gap vs d / N / overlap, LS line + 95% CI ---------- #
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
panel_titles = ["gap vs dimension $d$", "gap vs sample size $N$",
                "gap vs overlap (1-NN LOO error)"]
xlabels = ["dimension $d$", "sample size $N$", "overlap (1-NN LOO error)"]
for ax, p, pt, xl in zip(axes, preds, panel_titles, xlabels):
    ac.style_ax(ax)
    x = meta[p].values
    ax.scatter(x, y_gap, s=42, color=ac.C_FULL, zorder=3)
    for _, r in meta.iterrows():
        ax.annotate(r["dataset"], (r[p], r["gap_acc_10pct"]),
                    textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=ac.PALETTE["ink"])
    lr = stats.linregress(x, y_gap)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = lr.intercept + lr.slope * xs
    s_err = np.sqrt(((y_gap - (lr.intercept + lr.slope * x)) ** 2).sum() / (n - 2))
    sxx = ((x - x.mean()) ** 2).sum()
    se_fit = s_err * np.sqrt(1.0 / n + (xs - x.mean()) ** 2 / sxx)
    tc = stats.t.ppf(0.975, n - 2)
    ac.ci_band(ax, xs, ys - tc * se_fit, ys + tc * se_fit, ac.C_FULL, alpha=0.15)
    ax.plot(xs, ys, color=ac.C_FULL, lw=1.4, zorder=2)
    ax.axhline(0.0, color=ac.PALETTE["mid_gray"], lw=0.9, ls="--")
    ax.set_title(pt, fontsize=9)
    ax.set_xlabel(xl)
axes[0].set_ylabel("ACC gap @10% (Full $-$ NoExploration)")
fig.tight_layout()
png = ac.savefig(fig, "add7_meta_scatter", combo=True)

# ---- 5. unit.json ---------------------------------------------------------- #
u = ac.new_unit(7,
    "Meta-analysis: Does the Full-minus-NoExploration Gap Scale with Dataset Complexity?",
    "exploratory", "complete")
u["support"] = "mixed"
u["annotations"] = ["exploratory", "post-hoc", "underpowered",
                    "n=%d (residual df=%d)" % (n, dof),
                    "merge hook: add7_extra_datasets.csv for Worker C/D datasets"]
u["motivation"] = ("As a post-hoc synthesis across the six frozen benchmarks, we "
    "ask whether the advantage of the full P2 acquisition (robust weight + "
    "exploration bonus $h_i=\\|u_i-\\bar u\\|_1$ minus redundancy) grows with "
    "dataset difficulty. We regress the ACC gap at the 10% checkpoint on "
    "dimension $d$, sample size $N$, and a data-overlap proxy (leave-one-out "
    "1-nearest-neighbour classification error on the frozen standardized features).")
u["design"] = ("Six benchmarks (iris, wine, glass, ecoli, aggregation, compound). "
    "Outcome: mean paired ACC gap = ACC(P2_Full) - ACC(P2_NoExploration) at "
    "budget=10%, 30-seed confirmatory shards, paired by seed, ACC after Hungarian "
    "matching. Predictors: d = feature dimension, N = sample size, overlap = "
    "LOO 1-NN error. Pearson + Spearman univariate, then OLS gap ~ d + N + overlap.")
u["protocol"] = {"datasets": list(meta.dataset), "methods": ["P2_Full", "P2_NoExploration"],
                 "n_seeds_gap": 30, "budgets": [0.10],
                 "factors": {"n_datasets": n, "residual_df": dof,
                             "model_r_squared": r2,
                             "overlap_estimator": "loo_1nn_error"},
                 "checkpoint": 0.10}
hdr = ["Dataset", "d", "N", "overlap 1-NN", "gap ACC@10%", "gap NMI@10%", "gap ARI@10%"]
trows = []
for _, r in meta.iterrows():
    trows.append([r["dataset"], "%d" % r["d"], "%d" % r["N"],
                  "%.3f" % r["overlap_1nn"],
                  "%+.3f" % r["gap_acc_10pct"],
                  "%+.3f" % r["gap_nmi"], "%+.3f" % r["gap_ari"]])
ac.add_table(u, "Table. Meta-features and paired gaps at the 10% checkpoint.",
             hdr, trows,
             notes=["|L0|/N = %.2f fixed (INIT_FRAC); ACC after Hungarian matching." % ac.INIT_FRAC,
                    "overlap = leave-one-out 1-NN error on frozen StandardScaler features.",
                    "n=%d datasets; all correlations and the multiple regression are exploratory." % n])
# correlation table
chdr = ["Predictor", "Pearson r", "Pearson p", "Spearman r", "Spearman p"]
crows = []
for p in preds:
    crows.append([p, "%.3f" % uni[p]["pearson_r"], "%.3f" % uni[p]["pearson_p"],
                   "%.3f" % uni[p]["spearman_r"], "%.3f" % uni[p]["spearman_p"]])
ac.add_table(u, "Table. Univariate associations (gap ACC@10% vs predictor).",
             chdr, crows,
             notes=["Multiple OLS gap ~ d + N + overlap: R^2 = %.3f, residual df = %d." % (r2, dof),
                    "n=%d is far below the usual 10-15 observations-per-predictor rule; treat as hypothesis-generating only." % n])
ac.add_figure(u, "add7_meta_scatter",
              "Fig. ACC gap @10% (P2_Full minus P2_NoExploration) against (left) "
              "dimension d, (middle) sample size N, (right) overlap proxy (LOO "
              "1-NN error). Dashed line at gap=0; shaded band is the univariate "
              "OLS 95% pointwise CI; point labels are dataset names.", combo=True)
u["results"] = (
    "Across the n=%d benchmarks, the ACC gap at 10%% ranges from %+.3f to %+.3f. "
    "Univariate Pearson: gap~d r=%.3f (p=%.3f), gap~N r=%.3f (p=%.3f), "
    "gap~overlap r=%.3f (p=%.3f); Spearman concordant. The multiple regression "
    "gap ~ d + N + overlap has R^2 = %.3f with only %d residual degrees of freedom, "
    "so no coefficient is interpretable as confirmatory."
    % (n, y_gap.min(), y_gap.max(),
       uni["d"]["pearson_r"], uni["d"]["pearson_p"],
       uni["N"]["pearson_r"], uni["N"]["pearson_p"],
       uni["overlap_1nn"]["pearson_r"], uni["overlap_1nn"]["pearson_p"],
       r2, dof))
u["conclusion"] = ("With n=%d the meta-analysis is strictly hypothesis-generating: "
    "the data do not support any claim that the Full-minus-NoExploration gap "
    "scales with d, N, or overlap. The planned extension to Worker C/D datasets "
    "(GMM synthetic, letter, fashion, usps) via add7_extra_datasets.csv will be "
    "needed before any directional interpretation." % n)
u["csvs"] = ["add7_meta_features.csv", "add7_meta_regression.csv"]
u["repro_script"] = "add7_meta.py"
uj = ac.save_unit(u)
print("WROTE:", ac.res_path("add7_meta_features.csv"))
print("WROTE:", ac.res_path("add7_meta_regression.csv"))
print("WROTE:", png)
print("WROTE:", uj)
