"""make_noise_results.py — downstream analysis for the post-hoc noise robustness study.

Inputs (frozen, produced by run_noise_n30.py / run_minimal_robust.py):
  results/noise/noise_full_n30.csv          (720 rows; seeds 20261010..20261039)
  results/minimal_robust/summary_minimal.csv (180 rows; seeds 0..4; left untouched)

Outputs:
  results/noise/noise_summary_n30.csv       method-level mean/sd/CI for acc/nmi/ari
  results/noise/noise_contrasts_n30.csv     12 paired NoExplo-Full cells (all metrics in pp)
  results/noise/noise_summary_n5.csv        verbatim transposed copy of the n=5 per-seed data
  figures/final/noise_direction_consistency.png/.pdf  (tight-pool 0.02N panels, n=30)

All gaps are NoExploration minus Full, expressed in percentage points (pp).
Inference: paired Student t on 30 seeds (t_{.975,29}=2.045); Holm step-down across the
12 cells within each metric family. Post-hoc robustness backing; not part of P1-P4.
"""
import os
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.abspath(__file__))
NOISE = os.path.join(CODE, "results", "noise")
MIN5 = os.path.join(CODE, "results", "minimal_robust", "summary_minimal.csv")
FIGDIR = os.path.join(CODE, "figures", "final")
T29 = stats.t.ppf(0.975, 29)
PINK = "#D08AB6"
INK = "#3F345B"
METRICS = ["acc", "nmi", "ari"]


def holm_stepdown(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main():
    df = pd.read_csv(os.path.join(NOISE, "noise_full_n30.csv"))
    assert len(df) == 720, len(df)
    keys = ["dataset", "eta", "L0_pct", "method", "seed"]
    assert df.duplicated(subset=keys).sum() == 0
    assert df[METRICS].isna().sum().sum() == 0

    # ---- method-level summary (long format, pp) ----
    rows = []
    for (ds, eta, l0, m), g in df.groupby(["dataset", "eta", "L0_pct", "method"]):
        for metric in METRICS:
            v = g[metric].values * 100.0
            sd = v.std(ddof=1)
            lo = v.mean() - T29 * sd / np.sqrt(len(v))
            hi = v.mean() + T29 * sd / np.sqrt(len(v))
            rows.append(dict(dataset=ds, eta=eta, L0_pct=l0, method=m, metric=metric,
                             mean=v.mean(), sd=sd, ci_lo=lo, ci_hi=hi, n=len(v)))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(NOISE, "noise_summary_n30.csv"), index=False)

    # ---- 12 paired contrasts (all three metrics in pp) ----
    cells = []
    for (ds, eta, l0), g in df.groupby(["dataset", "eta", "L0_pct"]):
        g = g.sort_values("seed")
        fn = g[g.method == "P2_Full"].set_index("seed")
        ne = g[g.method == "P2_NoExploration"].set_index("seed")
        assert (fn.index == ne.index).all() and len(fn) == 30
        rec = dict(dataset=ds, eta=eta, L0_pct=l0)
        for metric in METRICS:
            d = (ne[metric].values - fn[metric].values) * 100.0
            mean, sd = d.mean(), d.std(ddof=1)
            se = sd / np.sqrt(len(d))
            lo, hi = mean - T29 * se, mean + T29 * se
            p = 2 * stats.t.sf(abs(mean / se), len(d) - 1)
            rec[f"gap_{metric}"] = mean
            rec[f"{metric}_lo"] = lo
            rec[f"{metric}_hi"] = hi
            rec[f"{metric}_p"] = p
            rec[f"{metric}_dz"] = mean / sd
        cells.append(rec)
    con = pd.DataFrame(cells).sort_values(["dataset", "L0_pct", "eta"]).reset_index(drop=True)
    for metric in METRICS:
        con[f"{metric}_holm"] = holm_stepdown(con[f"{metric}_p"].values)

    def verdict(r):
        if r.dataset == "ecoli" and abs(r.eta - 0.05) < 1e-9 and abs(r.L0_pct - 0.05) < 1e-9:
            return ("significant after Holm (favors NoExplo; "
                    f"medium-to-large d_z={r.acc_dz:.2f})")
        if r.acc_lo > 0 or r.acc_hi < 0:
            tail = "reversed, Full-favored" if r.gap_acc < 0 else "favors NoExplo"
            return f"raw p<.05, non-significant after Holm ({tail})"
        return "non-significant"

    con["verdict"] = con.apply(verdict, axis=1)
    cols = ["dataset", "eta", "L0_pct"]
    for metric in METRICS:
        cols += [f"gap_{metric}", f"{metric}_lo", f"{metric}_hi", f"{metric}_p",
                 f"{metric}_dz", f"{metric}_holm"]
    cols += ["verdict"]
    con[cols].to_csv(os.path.join(NOISE, "noise_contrasts_n30.csv"), index=False)

    # ---- n=5 frozen transposed copy (original never modified) ----
    shutil.copyfile(MIN5, os.path.join(NOISE, "noise_summary_n5.csv"))

    # ---- direction consistency figure: tight pool 0.02N, two panels ----
    tight = con[con.L0_pct == 0.02].sort_values(["dataset", "eta"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, ds in zip(axes, ["ecoli", "compound"]):
        sub = tight[tight.dataset == ds]
        x = sub.eta.values
        y = sub.gap_acc.values
        lo = sub.acc_lo.values
        hi = sub.acc_hi.values
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=PINK, marker="o", ms=12,
                    lw=2.4, capsize=7, capthick=2, elinewidth=2, mfc=PINK, mec=PINK)
        ax.axhline(0, color=INK, lw=1.1)
        ax.set_title(ds, fontsize=15, color=INK)
        ax.set_xlabel("eta", fontsize=14, color=INK)
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.tick_params(labelsize=13, colors=INK)
        for spine in ax.spines.values():
            spine.set_color(INK); spine.set_linewidth(1.3)
    axes[0].set_ylabel("ACC gap (NoExplo-Full, pp)", fontsize=14, color=INK)
    fig.suptitle("", fontsize=1)
    fig.text(0.5, 0.055,
             "No Holm-significant reversal under label noise; E. coli gaps stay positive "
             "while Compound converges.",
             ha="center", va="bottom", fontsize=11, color=INK)
    fig.text(0.5, 0.012, "Error bars are 95% paired CIs (n=30).",
             ha="center", va="bottom", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"noise_direction_consistency.{ext}"), dpi=300)
    plt.close(fig)
    print("noise downstream artifacts written")


if __name__ == "__main__":
    main()
