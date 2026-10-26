"""analyze_theorem2.py — E4 post-run statistics (c=3 controlled overlap).

Inputs (v2_unified/results/theorem2/):
  diagnostic.csv  : overlap_level, seed, method, point_idx, class, s_i, ...
  acc_0p10.csv    : overlap_level, seed, method, ACC

Primary object: the "overlap" point class (high H AND high h, Q75/Q75 on the
selected set, same rule as exploratory.run_x2_diag).  With c=3 the
high-H & high-h quadrant is non-degenerate and populated.

Statistics (all from the real run; B=10000 bootstrap draws, fixed seed):
  1. Per overlap level, overlap-class pooled points (all seeds/methods):
     frac s_i<0 + bootstrap 95% CI; mean s_i + CI.
  2. Trend: Spearman rho(level, frac s<0) and rho(level, mean s) with
     bootstrap CI; permutation Jonckheere-Terpstra trend test.
  3. Effect size per level: paired (ACC_NoExploration - ACC_Full) over the
     20 seeds, paired-t 95% CI, in percentage points.
  4. If overlap=0.8 overlap-class pooled n >= 30: one-sample bootstrap test
     of frac s_i<0 vs 0.5 (H0=0.5, two-sided p).
  5. Conflict-trigger rate: per-run fraction of selected points with s_i<0,
     averaged over runs, per level.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "theorem2")

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
B = 10000
RNG = np.random.default_rng(20261022)


def boot_mean(x, B=B, rng=RNG):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    bd = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(bd, 0.025)), \
        float(np.quantile(bd, 0.975)), bd


def jonckheere_j(groups):
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            for x in a:
                J += np.sum(b > x) + 0.5 * np.sum(b == x)
    return J


def main():
    dg = pd.read_csv(os.path.join(RES, "diagnostic.csv"), encoding="utf-8")
    acc = pd.read_csv(os.path.join(RES, "acc_0p10.csv"), encoding="utf-8")
    dg = dg.drop_duplicates(
        subset=["overlap_level", "seed", "method", "point_idx"]).reset_index(drop=True)
    dg["neg"] = (dg.s_i < 0).astype(float)
    print(f"[analyze] diagnostic rows={len(dg)}  acc rows={len(acc)}")

    # ---- 1. summary per (level, class) -----------------------------------#
    summ_rows = []
    for lv in LEVELS:
        for cls in ["overlap", "boundary", "eccentric", "other"]:
            s = dg[(dg.overlap_level == lv) & (dg["class"] == cls)]["s_i"].values
            if len(s) == 0:
                summ_rows.append({"overlap_level": lv, "class": cls, "n": 0,
                                  "frac_s_neg": np.nan, "mean_s": np.nan,
                                  "ci_low": np.nan, "ci_high": np.nan})
                continue
            est, lo, hi, _ = boot_mean((s < 0).astype(float))
            summ_rows.append({"overlap_level": lv, "class": cls, "n": int(len(s)),
                              "frac_s_neg": est, "mean_s": float(s.mean()),
                              "ci_low": lo, "ci_high": hi})
    summary = pd.DataFrame(summ_rows)
    summary.to_csv(os.path.join(RES, "summary.csv"), index=False, encoding="utf-8")
    print(summary.round(4).to_string(index=False))

    # ---- PRIMARY: overlap class per level --------------------------------#
    pooled_frac, pooled_mean, groups = [], [], []
    for lv in LEVELS:
        s = dg[(dg.overlap_level == lv) & (dg["class"] == "overlap")]["s_i"].values
        if len(s) == 0:
            pooled_frac.append((np.nan, np.nan, np.nan, 0))
            pooled_mean.append((np.nan, np.nan, np.nan))
            groups.append(np.array([0.0]))
            print(f"[primary] overlap={lv}: overlap-class n=0 (EMPTY)")
            continue
        x = (s < 0).astype(float)
        fe, flo, fhi, _ = boot_mean(x)
        me, mlo, mhi, _ = boot_mean(s)
        pooled_frac.append((fe, flo, fhi, len(x)))
        pooled_mean.append((me, mlo, mhi))
        groups.append(s)
        print(f"[primary] overlap={lv}: n={len(x)} frac_s<0={fe:.4f} "
              f"[{flo:.4f},{fhi:.4f}] mean_s={me:.4f} [{mlo:.4f},{mhi:.4f}]")

    # ---- 2. trend tests ---------------------------------------------------#
    fracs = np.array([p[0] for p in pooled_frac])
    means = np.array([p[0] for p in pooled_mean])
    ok = ~np.isnan(fracs)
    rho_f = p_f = rho_m = p_m = rho_m_lo = rho_m_hi = np.nan
    if ok.sum() >= 3:
        rho_f, p_f = stats.spearmanr(np.array(LEVELS)[ok], fracs[ok])
        rho_m, p_m = stats.spearmanr(np.array(LEVELS)[ok], means[ok])
        rhos_m = []
        for _ in range(2000):
            g2 = [s[RNG.integers(0, len(s), len(s))] for s in groups]
            rhos_m.append(stats.spearmanr(np.array(LEVELS)[ok],
                                          np.array([g.mean() for g in g2])[ok])[0])
        rho_m_lo, rho_m_hi = np.quantile(rhos_m, [0.025, 0.975])
    J_obs = jonckheere_j([g for g in groups if len(g) > 1])
    jt_p = np.nan
    valid = [g for g in groups if len(g) > 1]
    if len(valid) >= 2:
        allx = np.concatenate(valid); sizes = [len(g) for g in valid]
        Jp = []
        for _ in range(5000):
            perm = RNG.permutation(allx)
            Jp.append(jonckheere_j(np.split(perm, np.cumsum(sizes)[:-1])))
        jt_p = float(np.mean(np.array(Jp) >= J_obs))
    print(f"\n[trend] Spearman(level, frac s<0)={rho_f:.3f} p={p_f}")
    print(f"[trend] Spearman(level, mean s)={rho_m:.3f} "
          f"[CI {rho_m_lo:.3f},{rho_m_hi:.3f}] p={p_m}")
    print(f"[trend] JT J={J_obs:.0f} permutation one-sided p={jt_p}")

    # ---- 3. effect size: paired ACC difference per level -----------------#
    eff_rows = []
    for lv in LEVELS:
        a = acc[acc.overlap_level == lv].pivot_table(
            index="seed", columns="method", values="ACC").dropna()
        d = (a["P2_NoExploration"] - a["P2_Full"]) * 100.0
        t = stats.ttest_1samp(d, 0.0)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=stats.sem(d))
        eff_rows.append({"overlap_level": lv, "n_seeds": int(len(d)),
                         "effect_pp": float(d.mean()),
                         "eff_ci_low": float(ci[0]), "eff_ci_high": float(ci[1]),
                         "effect_p": float(t.pvalue)})
    eff = pd.DataFrame(eff_rows)
    print("\n[effect] NoExploration - Full (pp):")
    print(eff.round(3).to_string(index=False))

    # ---- 5. conflict-trigger rate: mean of run-level fractions ----------#
    run_rate = dg.groupby(["overlap_level", "seed", "method"])["neg"].mean().reset_index()
    trig = run_rate.groupby("overlap_level")["neg"].mean().to_dict()

    # ---- 4. overlap=0.8 one-sample bootstrap test vs 0.5 -----------------#
    s8 = dg[(dg.overlap_level == 0.8) & (dg["class"] == "overlap")]["s_i"].values
    p8 = np.nan
    if len(s8) >= 30:
        x8 = (s8 < 0).astype(float)
        idx = RNG.integers(0, len(x8), size=(B, len(x8)))
        bd = x8[idx].mean(1)
        p8 = float(2 * min(np.mean(bd <= 0.5), np.mean(bd >= 0.5)))
        print(f"\n[ov0.8] overlap-class n={len(s8)} frac={x8.mean():.4f} "
              f"bootstrap two-sided p vs 0.5 = {p8:.4g}")
    else:
        print(f"\n[ov0.8] overlap-class n={len(s8)} <30, skip >0.5 test")

    # ---- outputs ----------------------------------------------------------#
    t12 = []
    for i, lv in enumerate(LEVELS):
        e = eff[eff.overlap_level == lv].iloc[0]
        t12.append({
            "Overlap": lv,
            "overlap_class_n": pooled_frac[i][3],
            "frac_s_neg": pooled_frac[i][0],
            "frac_ci_low": pooled_frac[i][1], "frac_ci_high": pooled_frac[i][2],
            "mean_s": pooled_mean[i][0],
            "mean_s_ci_low": pooled_mean[i][1],
            "mean_s_ci_high": pooled_mean[i][2],
            "effect_size_pp": e.effect_pp,
            "effect_ci_low": e.eff_ci_low, "effect_ci_high": e.eff_ci_high,
            "conflict_trigger_rate": trig.get(lv, np.nan),
        })
    pd.DataFrame(t12).to_csv(os.path.join(RES, "table12.csv"),
                             index=False, encoding="utf-8")

    pa = pd.DataFrame([{
        "panel": "a_frac_s_neg", "overlap": lv, "n": pooled_frac[i][3],
        "value": pooled_frac[i][0], "ci_low": pooled_frac[i][1],
        "ci_high": pooled_frac[i][2]} for i, lv in enumerate(LEVELS)])
    pb = pd.DataFrame([{
        "panel": "b_effect_pp", "overlap": lv, "value": eff.iloc[i].effect_pp,
        "ci_low": eff.iloc[i].eff_ci_low,
        "ci_high": eff.iloc[i].eff_ci_high} for i, lv in enumerate(LEVELS)])
    pm = pd.DataFrame([{
        "panel": "c_mean_s", "overlap": lv, "value": pooled_mean[i][0],
        "ci_low": pooled_mean[i][1],
        "ci_high": pooled_mean[i][2]} for i, lv in enumerate(LEVELS)])
    pd.DataFrame(pd.concat([pa, pb, pm], ignore_index=True)).to_csv(
        os.path.join(RES, "fig13_data.csv"), index=False, encoding="utf-8")

    pd.DataFrame({
        "spearman_level_frac_rho": [rho_f], "spearman_level_frac_p": [p_f],
        "spearman_level_means_rho": [rho_m],
        "spearman_level_means_ci_low": [rho_m_lo],
        "spearman_level_means_ci_high": [rho_m_hi],
        "spearman_level_means_p": [p_m],
        "JT_stat": [J_obs], "JT_perm_p": [jt_p],
        "ov08_n": [len(s8)], "ov08_frac": [float((s8 < 0).mean()) if len(s8) else np.nan],
        "ov08_vs0.5_boot_p": [p8],
    }).to_csv(os.path.join(RES, "trend_tests.csv"), index=False, encoding="utf-8")
    print("\n[done] wrote summary.csv / table12.csv / fig13_data.csv / "
          "trend_tests.csv")


if __name__ == "__main__":
    main()
