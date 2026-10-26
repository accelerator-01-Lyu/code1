# -*- coding: utf-8 -*-
"""analyze_cross_task.py — E3 cross-task extension analysis.

Reads results/cross_task/summary.csv and produces:
  paired_contrasts.csv : task, n_classes, budget, mean_diff, ci_low, ci_high, d_z
                         (effect = ACC(P2_NoExploration) - ACC(P2_Full), n=20
                         paired seeds; 95% Student-t CI with df=n-1;
                         d_z = mean_diff / SD(diff); se = SD(diff)/sqrt(n)).
  table11.csv          : Table 11 construction data (per-task + Meta pooled row).
  meta_analysis.csv    : per-task effects at 10%/20% + DerSimonian-Laird pooled
                         random-effect rows (Q, I^2, tau^2) + WLS meta-regression
                         rows (effect ~ n_classes).
  fig12_data.csv       : effect points vs n_classes + WLS fitted line data.

DerSimonian-Laird random-effects formulas (per checkpoint):
  fixed effect      m_f = sum(w_i y_i)/sum(w_i),  w_i = 1/v_i, v_i = se_i^2
  heterogeneity     Q   = sum w_i (y_i - m_f)^2,            df = k - 1
                    p   = chi2.sf(Q, df)
                    C   = sum w_i - sum(w_i^2)/sum w_i
                    tau2 = max(0, (Q - df)/C)
  random effect     m_re = sum w_re_i y_i / sum w_re_i,  w_re_i = 1/(v_i + tau2)
                    se_re = sqrt(1/sum w_re_i), 95% CI = m_re +/- 1.96 se_re
                    I2  = max(0, (Q - df)/Q) * 100%
WLS meta-regression effect ~ a + b*n_classes, weights w_i = 1/v_i:
  b = (W*Sxy - Sx*Sy)/(W*Sxx - Sx^2), with W=sum w, Sx=sum wx, Sy=sum wy,
      Sxx=sum w x^2, Sxy=sum w x y;  residual s2 = sum w(y-a-bx)^2/(k-2);
      Var(b) = s2*W/(W*Sxx - Sx^2);  p = 2 Phi(-|b/se|).

All numbers come from summary.csv (real runs). No fabrication.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES_DIR = os.path.join(ROOT, "results", "cross_task")
SUMMARY = os.path.join(RES_DIR, "summary.csv")
META_JSON = os.path.join(ROOT, "data", "cross_task", "tasks_meta.json")

CHECKPOINTS = [0.10, 0.20]
BOOT_B = 10000
BOOT_SEED = 20261022


# --------------------------------------------------------------------------- #
# Paired contrast helpers
# --------------------------------------------------------------------------- #
def paired_contrast(ne, full, boot_seed=0):
    """ne, full: per-seed ACC arrays (same seed order, n=20 paired seeds).
    diff = ne - full. Returns mean_diff, ci_low, ci_high (Student-t 95% CI with
    df=n-1, matching additional_common.paired_ci), d_z = mean(SD(diff)), se =
    SD(diff)/sqrt(n) whose square feeds the meta-analytic weights."""
    d = np.asarray(ne, float) - np.asarray(full, float)
    n = len(d)
    mean_diff = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    dz = float(mean_diff / sd) if sd > 0 else 0.0
    tc = stats.t.ppf(0.975, n - 1)
    ci_low, ci_high = mean_diff - tc * se, mean_diff + tc * se
    return {"mean_diff": mean_diff, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "d_z": dz, "se": se,
            "sd_diff": sd, "n": n}


def direction_of(mean_diff, ci_low, ci_high, margin=0.0):
    """Effect = NoExploration - Full. positive => NoExploration better
    (exploration bonus hurts); negative => Full better.
    'ns' when the 95% CI covers 0."""
    if ci_low > margin:
        return "positive"
    if ci_high < -margin:
        return "negative"
    return "ns"


# --------------------------------------------------------------------------- #
# DerSimonian-Laird random effects + WLS meta-regression
# --------------------------------------------------------------------------- #
def dersimonian_laird(y, v):
    """y: per-study effect, v: per-study variance (SE^2).
    Returns pooled random-effect estimate with 95% z-CI, Q, df, I^2, tau^2."""
    y = np.asarray(y, float); v = np.asarray(v, float)
    k = len(y)
    w = 1.0 / v
    m_fixed = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - m_fixed) ** 2))
    df = max(k - 1, 1)
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    m_re = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    ci_low, ci_high = m_re - 1.96 * se_re, m_re + 1.96 * se_re
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    p = float(stats.chi2.sf(Q, df))
    return {"pooled_effect": m_re, "pooled_ci_low": ci_low,
            "pooled_ci_high": ci_high, "Q": Q, "Q_df": df, "Q_p": p,
            "I2": I2, "tau2": tau2, "fixed_effect": m_fixed}


def wls_metaregress(x, y, v):
    """Weighted least squares: y ~ a + b*x, weights w = 1/v.
    Returns slope, se_slope, ci, p (z-based, large-sample)."""
    x = np.asarray(x, float); y = np.asarray(y, float); w = 1.0 / np.asarray(v, float)
    W = w.sum()
    Sx = (w * x).sum()
    Sy = (w * y).sum()
    Sxx = (w * x * x).sum()
    Sxy = (w * x * y).sum()
    denom = W * Sxx - Sx * Sx
    slope = (W * Sxy - Sx * Sy) / denom
    intercept = (Sy - slope * Sx) / W
    # residual variance estimator (Harvey-like, does not re-bias much with k=6)
    resid = y - (intercept + slope * x)
    s2 = float((w * resid ** 2).sum() / max(len(y) - 2, 1))
    var_slope = s2 * W / denom
    se = float(np.sqrt(var_slope))
    z = slope / se
    p = float(2 * stats.norm.sf(abs(z)))
    return {"slope": float(slope), "slope_se": se,
            "slope_ci_low": float(slope - 1.96 * se),
            "slope_ci_high": float(slope + 1.96 * se),
            "slope_p": p, "intercept": float(intercept)}


# --------------------------------------------------------------------------- #
def main():
    df = pd.read_csv(SUMMARY)
    tasks = list(dict.fromkeys(df["task"].tolist()))
    n_classes_map = df.drop_duplicates("task").set_index("task")["n_classes"].to_dict()
    print("tasks:", tasks, "rows:", len(df))

    # ---- 1. paired contrasts at every (task, budget) ---- #
    contr_rows = []
    for t in tasks:
        sub = df[df.task == t]
        budgets = sorted(sub.budget.unique())
        for bud in budgets:
            ss = sub[np.isclose(sub.budget, bud)]
            wide = ss.pivot(index="seed", columns="method", values="ACC")
            if "P2_NoExploration" not in wide or "P2_Full" not in wide:
                continue
            r = paired_contrast(wide["P2_NoExploration"].values,
                                 wide["P2_Full"].values,
                                 boot_seed=BOOT_SEED + int(round(bud * 1000)))
            contr_rows.append({"task": t, "n_classes": int(n_classes_map[t]),
                               "budget": float(bud), **r})
    contr = pd.DataFrame(contr_rows)
    contr.to_csv(os.path.join(RES_DIR, "paired_contrasts.csv"),
                 index=False, encoding="utf-8")
    print(contr.round(4).to_string(index=False))

    # ---- 2. per-task effects at 10% / 20% ---- #
    eff = {}   # checkpoint -> DataFrame task rows
    meta_rows = []
    for cp in CHECKPOINTS:
        rows = []
        for t in tasks:
            r = contr[(contr.task == t) & np.isclose(contr.budget, cp)].iloc[0]
            rows.append({"task": t, "n_classes": int(r.n_classes),
                         "effect": r.mean_diff, "se": r.se,
                         "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "d_z": r.d_z,
                         "direction": direction_of(r.mean_diff, r.ci_low, r.ci_high)})
        eff[cp] = pd.DataFrame(rows)
        # DerSimonian-Laird pooled
        dl = dersimonian_laird(eff[cp]["effect"].values, eff[cp]["se"].values ** 2)
        # WLS meta-regression effect ~ n_classes
        mr = wls_metaregress(eff[cp]["n_classes"].values,
                             eff[cp]["effect"].values,
                             eff[cp]["se"].values ** 2)
        eff[cp].attrs["dl"] = dl
        eff[cp].attrs["mr"] = mr
        print("\n== checkpoint %.2f ==" % cp)
        print(eff[cp].round(4).to_string(index=False))
        print("DL pooled: %+.4f [%+.4f, %+.4f]  Q=%.2f (df=%d, p=%.3f)  "
              "I2=%.1f%%  tau2=%.5f"
              % (dl["pooled_effect"], dl["pooled_ci_low"], dl["pooled_ci_high"],
                 dl["Q"], dl["Q_df"], dl["Q_p"], dl["I2"], dl["tau2"]))
        print("WLS slope: %+.5f per class [%+.5f, %+.5f]  p=%.4f"
              % (mr["slope"], mr["slope_ci_low"], mr["slope_ci_high"], mr["slope_p"]))

    # ---- 3. meta_analysis.csv ---- #
    out_rows = []
    for cp in CHECKPOINTS:
        e = eff[cp]
        for _, r in e.iterrows():
            out_rows.append({"row_type": "task", "task": r.task,
                             "n_classes": int(r.n_classes),
                             "checkpoint": cp,
                             "effect_10pct": e.set_index("task").loc[r.task, "effect"] if cp == 0.10 else np.nan,
                             "ci_low_10pct": e.set_index("task").loc[r.task, "ci_low"] if cp == 0.10 else np.nan,
                             "ci_high_10pct": e.set_index("task").loc[r.task, "ci_high"] if cp == 0.10 else np.nan,
                             "effect_20pct": e.set_index("task").loc[r.task, "effect"] if cp == 0.20 else np.nan,
                             "ci_low_20pct": e.set_index("task").loc[r.task, "ci_low"] if cp == 0.20 else np.nan,
                             "ci_high_20pct": e.set_index("task").loc[r.task, "ci_high"] if cp == 0.20 else np.nan,
                             "direction": r.direction})
    # unique task rows (merge both checkpoints)
    task_block = {}
    for cp in CHECKPOINTS:
        for _, r in eff[cp].iterrows():
            b = task_block.setdefault(r.task, {"row_type": "task", "task": r.task,
                                               "n_classes": int(r.n_classes),
                                               "direction": r.direction})
            if cp == 0.10:
                b["effect_10pct"], b["ci_low_10pct"], b["ci_high_10pct"] = \
                    r.effect, r.ci_low, r.ci_high
            else:
                b["effect_20pct"], b["ci_low_20pct"], b["ci_high_20pct"] = \
                    r.effect, r.ci_low, r.ci_high
    out_rows = list(task_block.values())
    for cp in CHECKPOINTS:
        dl = eff[cp].attrs["dl"]; mr = eff[cp].attrs["mr"]
        tag = "pooled_10pct" if cp == 0.10 else "pooled_20pct"
        out_rows.append({"row_type": tag, "task": "POOLED",
                         "pooled_effect": dl["pooled_effect"],
                         "pooled_ci_low": dl["pooled_ci_low"],
                         "pooled_ci_high": dl["pooled_ci_high"],
                         "Q": dl["Q"], "Q_df": dl["Q_df"], "Q_p": dl["Q_p"],
                         "I2": dl["I2"], "tau2": dl["tau2"]})
        out_rows.append({"row_type": "metareg_10pct" if cp == 0.10 else "metareg_20pct",
                         "task": "METAREG", "slope": mr["slope"],
                         "slope_se": mr["slope_se"],
                         "slope_ci_low": mr["slope_ci_low"],
                         "slope_ci_high": mr["slope_ci_high"],
                         "slope_p": mr["slope_p"],
                         "intercept": mr["intercept"]})
    meta = pd.DataFrame(out_rows)
    meta.to_csv(os.path.join(RES_DIR, "meta_analysis.csv"),
                index=False, encoding="utf-8")

    # ---- 4. table11.csv ---- #
    t11 = []
    for t in tasks:
        e10 = eff[0.10].set_index("task").loc[t]
        e20 = eff[0.20].set_index("task").loc[t]
        t11.append({
            "Task": t, "Classes": int(n_classes_map[t]),
            "effect_10pct": e10.effect,
            "ci_10pct": "[%+.4f, %+.4f]" % (e10.ci_low, e10.ci_high),
            "effect_20pct": e20.effect,
            "ci_20pct": "[%+.4f, %+.4f]" % (e20.ci_low, e20.ci_high),
            "Direction": e20.direction})
    # Meta pooled row (DerSimonian-Laird random effects), 10% and 20%
    dl10 = eff[0.10].attrs["dl"]; dl20 = eff[0.20].attrs["dl"]
    t11.append({
        "Task": "Meta pooled (DL)", "Classes": "RE",
        "effect_10pct": dl10["pooled_effect"],
        "ci_10pct": "[%+.4f, %+.4f]" % (dl10["pooled_ci_low"], dl10["pooled_ci_high"]),
        "effect_20pct": dl20["pooled_effect"],
        "ci_20pct": "[%+.4f, %+.4f]" % (dl20["pooled_ci_low"], dl20["pooled_ci_high"]),
        "Direction": direction_of(dl20["pooled_effect"], dl20["pooled_ci_low"],
                                 dl20["pooled_ci_high"])})
    pd.DataFrame(t11).to_csv(os.path.join(RES_DIR, "table11.csv"),
                             index=False, encoding="utf-8")

    # ---- 5. fig12_data.csv: points + WLS fit lines ---- #
    fig_rows = []
    for t in tasks:
        e10 = eff[0.10].set_index("task").loc[t]
        e20 = eff[0.20].set_index("task").loc[t]
        fig_rows.append({"row_type": "point", "task": t,
                         "n_classes": int(n_classes_map[t]),
                         "effect_10pct": e10.effect, "se_10pct": e10.se,
                         "effect_20pct": e20.effect, "se_20pct": e20.se})
    xs = np.linspace(2, 10, 40)
    mr10 = eff[0.10].attrs["mr"]; mr20 = eff[0.20].attrs["mr"]
    for x in xs:
        fig_rows.append({"row_type": "fit", "task": "", "n_classes": float(x),
                         "fit_10pct": mr10["intercept"] + mr10["slope"] * x,
                         "fit_20pct": mr20["intercept"] + mr20["slope"] * x})
    pd.DataFrame(fig_rows).to_csv(os.path.join(RES_DIR, "fig12_data.csv"),
                                  index=False, encoding="utf-8")

    # ---- summary printout for changelog ---- #
    print("\n=== KEY NUMBERS ===")
    for cp in CHECKPOINTS:
        dl = eff[cp].attrs["dl"]; mr = eff[cp].attrs["mr"]
        print("budget %.0f%%: pooled=%+.4f [%+.4f,%+.4f] Q=%.2f I2=%.1f%% "
              "tau2=%.5f | slope=%+.5f/ci p=%.4f"
              % (cp * 100, dl["pooled_effect"], dl["pooled_ci_low"],
                 dl["pooled_ci_high"], dl["Q"], dl["I2"], dl["tau2"],
                 mr["slope"], mr["slope_p"]))
    print("WROTE paired_contrasts.csv / table11.csv / meta_analysis.csv / fig12_data.csv")


if __name__ == "__main__":
    main()
