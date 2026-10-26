"""analyze_theorem2_final.py — E4 final analysis on c=2 balanced recorded data.

Source of truth: results/theorem2/c2_record/ (diagnostic.csv, acc_0p10.csv).
No new runs. Produces:
  summary.csv      per (level, class): n, frac_s_neg + bootstrap 95% CI,
                   mean_s + CI  (recomputed, replaces the backed-up copy)
  table12.csv     one row per (level, class) incl. pooled row and the
                   structural n_overlap=0 row
  fig13_data.csv  Panel a: pooled frac_s_neg + CI (boundary/eccentric split);
                   Panel b: effect size + CI
  trend_tests.csv Spearman + Jonckheere-Terpstra for mean_s vs overlap,
                   effect size vs overlap, pooled frac_s_neg
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REC = os.path.join(ROOT, "results", "theorem2", "c2_record")
RES = os.path.join(ROOT, "results", "theorem2")

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
CLASSES = ["overlap", "boundary", "eccentric", "other"]
B = 10000
RNG = np.random.default_rng(20261022)


def boot_prop(x, B=B, rng=RNG):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    bd = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))


def boot_mean(x, B=B, rng=RNG):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    bd = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))


def jonckheere_j(groups):
    """Vectorized JT statistic: sum_{i<j} sum_{x in g_i} [#{y in g_j: y>x} + 0.5*#{y in g_j: y==x}]."""
    J = 0.0
    arrs = [np.sort(np.asarray(g, float)) for g in groups]
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            a, b = arrs[i], arrs[j]
            # for each x in a: #(b>x) = len(b) - searchsorted(b,x,'right'); #(b==x) = searchsorted right-left
            gt = len(b) - np.searchsorted(b, a, side="right")
            eq = np.searchsorted(b, a, side="right") - np.searchsorted(b, a, side="left")
            J += float(np.sum(gt + 0.5 * eq))
    return J


def main():
    dg = pd.read_csv(os.path.join(REC, "diagnostic.csv"), encoding="utf-8")
    acc = pd.read_csv(os.path.join(REC, "acc_0p10.csv"), encoding="utf-8")
    dg = dg.drop_duplicates(subset=["overlap_level", "seed", "method", "point_idx"])
    print(f"[final] diag rows={len(dg)} acc rows={len(acc)}")

    # ---- 1. summary per (level, class) -----------------------------------#
    summ = []
    for lv in LEVELS:
        for cls in CLASSES:
            s = dg[(dg.overlap_level == lv) & (dg["class"] == cls)]["s_i"].values
            if len(s) == 0:
                summ.append({"overlap_level": lv, "class": cls, "n": 0,
                             "frac_s_neg": np.nan, "frac_lo": np.nan,
                             "frac_hi": np.nan, "mean_s": np.nan,
                             "mean_lo": np.nan, "mean_hi": np.nan})
                continue
            f, flo, fhi = boot_prop((s < 0).astype(float))
            m, mlo, mhi = boot_mean(s)
            summ.append({"overlap_level": lv, "class": cls, "n": int(len(s)),
                         "frac_s_neg": f, "frac_lo": flo, "frac_hi": fhi,
                         "mean_s": m, "mean_lo": mlo, "mean_hi": mhi})
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(RES, "summary.csv"), index=False, encoding="utf-8")
    print("\n[summary] per (level,class):")
    print(summary.round(4).to_string(index=False))

    # ---- 2. effect size per level (paired ACC diff, 20 seeds) --------------#
    eff_rows = []
    for lv in LEVELS:
        a = acc[acc.overlap_level == lv].pivot_table(
            index="seed", columns="method", values="ACC").dropna()
        d = (a["P2_NoExploration"] - a["P2_Full"]) * 100.0
        t = stats.ttest_1samp(d, 0.0)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=stats.sem(d))
        dz = d.mean() / (d.std(ddof=1) + 1e-12)
        eff_rows.append({"overlap_level": lv, "n_seeds": int(len(d)),
                         "effect_pp": float(d.mean()),
                         "eff_lo": float(ci[0]), "eff_hi": float(ci[1]),
                         "effect_p": float(t.pvalue), "d_z": float(dz)})
    eff = pd.DataFrame(eff_rows)
    print("\n[effect] NoExploration - Full (pp):")
    print(eff.round(4).to_string(index=False))

    # ---- 3. conflict-trigger rate: mean of run-level fractions ------------#
    dg["neg"] = (dg.s_i < 0).astype(float)
    run_rate = dg.groupby(["overlap_level", "seed", "method"])["neg"].mean().reset_index()
    trig = run_rate.groupby("overlap_level")["neg"].mean().to_dict()
    print("\n[trigger] mean run-level s_i<0 fraction:")
    for lv in LEVELS:
        print(f"  level={lv:.1f}  trigger={trig[lv]:.4f}")

    # ---- 4. trend tests ---------------------------------------------------#
    # 4a. mean_s vs overlap (pooled selected points)
    ms_means, ms_groups = [], []
    for lv in LEVELS:
        s = dg[dg.overlap_level == lv]["s_i"].values
        ms_means.append(float(s.mean()))
        ms_groups.append(s)
    rho_ms, rho_ms_p = stats.spearmanr(LEVELS, ms_means)
    J_ms = jonckheere_j(ms_groups)
    allx = np.concatenate(ms_groups)
    sizes = [len(g) for g in ms_groups]
    J_perm = [jonckheere_j(np.split(RNG.permutation(allx), np.cumsum(sizes)[:-1]))
              for _ in range(5000)]
    jt_ms_p = float(np.mean(np.array(J_perm) >= J_ms))

    # 4b. effect size vs overlap
    rho_eff, rho_eff_p = stats.spearmanr(LEVELS, eff["effect_pp"].values)
    # JT on effect size groups (each level's 20 seed diffs)
    eff_groups = [
        (acc[acc.overlap_level == lv].pivot_table(index="seed", columns="method", values="ACC").dropna()
         .pipe(lambda a: (a["P2_NoExploration"] - a["P2_Full"]).values * 100.0))
        for lv in LEVELS
    ]
    J_eff = jonckheere_j(eff_groups)
    allx_e = np.concatenate(eff_groups)
    sz_e = [len(g) for g in eff_groups]
    Jp_e = [jonckheere_j(np.split(RNG.permutation(allx_e), np.cumsum(sz_e)[:-1]))
            for _ in range(5000)]
    jt_eff_p = float(np.mean(np.array(Jp_e) >= J_eff))

    # 4c. pooled frac_s_neg (saturated ~0.998)
    pooled_fracs = []
    for lv in LEVELS:
        pooled_fracs.append(float((dg[dg.overlap_level == lv]["s_i"] < 0).mean()))
    rho_pool, rho_pool_p = stats.spearmanr(LEVELS, pooled_fracs)

    trend = pd.DataFrame({
        "test": ["mean_s_vs_overlap", "effect_vs_overlap", "pooled_frac_vs_overlap"],
        "spearman_rho": [rho_ms, rho_eff, rho_pool],
        "spearman_p": [rho_ms_p, rho_eff_p, rho_pool_p],
        "JT_stat": [J_ms, J_eff, np.nan],
        "JT_perm_p": [jt_ms_p, jt_eff_p, np.nan],
    })
    trend.to_csv(os.path.join(RES, "trend_tests.csv"), index=False, encoding="utf-8")
    print("\n[trend]")
    print(trend.round(4).to_string(index=False))

    # ---- 5. overlap=0.8 pooled one-sample bootstrap > 0.5 ------------------#
    s8 = dg[dg.overlap_level == 0.8]["s_i"].values
    x8 = (s8 < 0).astype(float)
    n8 = len(x8)
    idx = RNG.integers(0, n8, size=(B, n8))
    bd = x8[idx].mean(1)
    obs8 = float(x8.mean())
    p8 = float(2 * min(np.mean(bd <= 0.5), np.mean(bd >= 0.5)))
    p8 = min(p8, 1.0)
    print(f"\n[ov0.8] pooled n={n8} frac={obs8:.4f} bootstrap two-sided p vs 0.5 = {p8:.4f}")

    # ---- 6. table12 + fig13 ----------------------------------------------#
    t12 = []
    pa, pb = [], []
    for lv in LEVELS:
        s_lv = dg[dg.overlap_level == lv]
        # pooled row
        sp = s_lv["s_i"].values
        fp, fp_lo, fp_hi = boot_prop((sp < 0).astype(float))
        mp, mp_lo, mp_hi = boot_mean(sp)
        e = eff[eff.overlap_level == lv].iloc[0]
        t12.append({"overlap_level": lv, "class": "pooled", "n": int(len(sp)),
                    "frac_s_neg": fp, "ci_low": fp_lo, "ci_high": fp_hi,
                    "mean_s": mp, "effect_pp": e.effect_pp,
                    "trigger_rate": trig[lv]})
        pa.append({"panel": "a_pooled_frac", "overlap": lv, "n": int(len(sp)),
                   "value": fp, "ci_low": fp_lo, "ci_high": fp_hi})
        # class rows
        for cls in CLASSES:
            sc = s_lv[s_lv["class"] == cls]["s_i"].values
            if len(sc):
                fc, fc_lo, fc_hi = boot_prop((sc < 0).astype(float))
                mc, mc_lo, mc_hi = boot_mean(sc)
            else:
                fc = fc_lo = fc_hi = mc = mc_lo = mc_hi = np.nan
            t12.append({"overlap_level": lv, "class": cls, "n": int(len(sc)),
                        "frac_s_neg": fc, "ci_low": fc_lo, "ci_high": fc_hi,
                        "mean_s": mc, "effect_pp": e.effect_pp,
                        "trigger_rate": trig[lv]})
            if cls in ("boundary", "eccentric"):
                pa.append({"panel": f"a_{cls}_frac", "overlap": lv,
                           "n": int(len(sc)), "value": fc,
                           "ci_low": fc_lo, "ci_high": fc_hi})
        pb.append({"panel": "b_effect_pp", "overlap": lv,
                   "value": e.effect_pp, "ci_low": e.eff_lo, "ci_high": e.eff_hi})
    pd.DataFrame(t12).to_csv(os.path.join(RES, "table12.csv"), index=False, encoding="utf-8")
    pd.DataFrame(pa + pb).to_csv(os.path.join(RES, "fig13_data.csv"), index=False, encoding="utf-8")
    print("\n[done] wrote summary.csv / table12.csv / fig13_data.csv / trend_tests.csv")
    print(f"[ov0.8-test] n={n8} obs={obs8:.4f} p={p8:.4f}")


if __name__ == "__main__":
    main()
