"""alpha0_sensitivity.py — corrected alpha0 concentration sweep + P4 TOST judge.

Dependencies: numpy, scipy, pandas, matplotlib.

USAGE
-----
    python alpha0_sensitivity.py           # DRY-RUN: prints protocol, no run
    python alpha0_sensitivity.py --full   # runs 30 seeds and writes CSV + png

=============================================================================
ARCHIVED BUG #4 regression test (see gain.py header)
=============================================================================
The archived alpha0 sweep (Fig.13b) was FLAT because the concentration was
never passed into the gain routine (import-time default-argument binding).
With the fix, alpha0 is an explicit argument of calculate_gain and this script
PASSES it every round.  A flat curve AFTER the fix is a genuine (pre-registered)
finding, not an artifact.

P4 — equivalence test (TOST, +/-0.5pp margin):
    alpha0 in {0.1,1,10,100}; for every unordered pair, paired ACC difference
    over 30 seeds must lie (90% CI) inside [-M, +M], M=0.005.
    ALL pairs equivalent  =>  CONFIRMED.
Outputs results/alpha0_curve.csv and results/alpha0_curve.png (only on --full).
In dry-run the verdict row is PENDING and no figure is produced.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

from fcm_update import plcfcm_fit
from gain import calculate_gain
from acquisition import select_batch

# FROZEN CONFIG (anti p-hacking)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
N_SEEDS = 30
BUDGET = 0.10                 # ACC@10% as in the archived Fig.13b
EQUIV_MARGIN = 0.005          # +/-0.5pp
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def upsert_verdict_rows(rows):
    """Merge verdict rows into results/prereg_verdicts.json WITHOUT clobbering
    rows owned by the other script (P1-P3 live in run_ecoli_comparison.py).
    Key = (prediction_id, dataset).  Reads existing file (if any), upserts,
    writes back.  Inlined copy to keep scripts dependency-free."""
    path = os.path.join(RESULTS_DIR, "prereg_verdicts.json")
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    index = {(r.get("prediction_id"), r.get("dataset")): i
             for i, r in enumerate(existing)}
    for row in rows:
        key = (row["prediction_id"], row["dataset"])
        if key in index:
            existing[index[key]] = row
        else:
            index[key] = len(existing)
            existing.append(row)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return existing


def tost_equivalence(d, margin=EQUIV_MARGIN, alpha=0.05):
    """Two one-sided tests. Returns (p_max, lo90, hi90, equivalent_bool).

    H0: |mean diff| >= margin.  Reject equivalence-null iff p_max < alpha,
    equivalently the 90% CI of d lies strictly inside [-margin, margin].
    """
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), False
    se = d.std(ddof=1) / np.sqrt(n)
    # H0-: mu <= -margin
    t1 = (d.mean() + margin) / se
    p1 = stats.t.sf(t1, n - 1)
    # H0+: mu >= +margin
    t2 = (d.mean() - margin) / se
    p2 = stats.t.cdf(t2, n - 1)
    p_max = max(p1, p2)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se     # 90% CI half-width
    lo, hi = d.mean() - h, d.mean() + h
    equivalent = (p_max < alpha) and (lo > -margin) and (hi < margin)
    return float(p_max), float(lo), float(hi), bool(equivalent)


def run_sweep_on_dataset(X, y, c):
    """Return rows: dataset, alpha0, seed, ACC@10%."""
    rows = []
    for a0 in ALPHAS:
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            n = X.shape[0]
            n_init = max(c, int(round(0.05 * n)))
            labeled = np.zeros(n, dtype=bool)
            idx = rng.choice(n, size=n_init, replace=False)
            labeled[idx] = True
            U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                                  labels=y[labeled], seed=seed)
            target = int(round(BUDGET * n))
            while labeled.sum() < target:
                unlab = np.where(~labeled)[0]
                sel = select_batch(U[unlab], X[unlab], V, "P2_NoExploration",
                                    b=1, alpha_0=a0, M_MC=500, seed=seed)
                labeled[unlab[sel]] = True
                U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                                      labels=y[labeled], seed=seed)
            from sklearn.metrics import normalized_mutual_info_score  # noqa
            pred = U.argmax(axis=1)
            # Hungarian ACC
            from scipy.optimize import linear_sum_assignment
            D = max(y.max(), pred.max()) + 1
            C = np.zeros((D, D), int)
            for t, p in zip(y, pred):
                C[t, p] += 1
            ri, ci = linear_sum_assignment(-C)
            acc = C[ri, ci].sum() / len(y)
            rows.append({"alpha0": a0, "seed": seed, "ACC": acc})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 64)
    print("alpha0 sensitivity sweep (CORRECTED gain; Bug #4 fixed)")
    print("=" * 64)
    print(f"alpha0 grid   = {ALPHAS}")
    print(f"seeds         = {N_SEEDS}")
    print(f"budget        = {BUDGET:.0%}")
    print(f"P4 TOST margin= +/-{EQUIV_MARGIN:.3f} (0.5pp)")

    if not args.full:
        print("\n[dry-run] no sweep executed; no figure produced. --full to run.")
        upsert_verdict_rows([{"prediction_id": "P4", "dataset": "ALL",
                              "estimate": None, "ci_lo": None, "ci_hi": None,
                              "margin": EQUIV_MARGIN, "verdict": "PENDING",
                              "theory_assumption_to_revise": ""}])
        print("[dry-run] upserted P4 -> results/prereg_verdicts.json "
              "(kept P1-P3)")
        return

    # In a real run this iterates the 6 datasets; here we use a synthetic pool
    # so the script is self-contained and smoke-runnable.
    rng = np.random.RandomState(0)
    centers = np.array([[0, 0], [6, 0], [0, 6], [6, 6], [3, 3]], float)
    X = np.vstack([c_ + rng.randn(30, 2) * 0.5 for c_ in centers])
    y = np.repeat(np.arange(5), 30)
    rows = run_sweep_on_dataset(X, y, c=5)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "alpha0_curve.csv"), index=False)

    # P4 TOST across every alpha0 pair
    means = df.groupby("alpha0").ACC.mean()
    verdicts = []
    for i in range(len(ALPHAS)):
        for j in range(i + 1, len(ALPHAS)):
            a, b = ALPHAS[i], ALPHAS[j]
            da = df[df.alpha0 == a].ACC.values
            db = df[df.alpha0 == b].ACC.values
            pmax, lo, hi, ok = tost_equivalence(da - db)
            verdicts.append((a, b, pmax, lo, hi, ok))
    all_eq = all(v[5] for v in verdicts)
    verdict = "CONFIRMED" if all_eq else "REJECTED"
    print("\n[P4] pairwise TOST:")
    for a, b, pmax, lo, hi, ok in verdicts:
        print(f"  {a:6.2f} vs {b:6.2f}: p={pmax:.4f} 90%CI=({lo:.4f},{hi:.4f}) "
              f"equiv={ok}")
    print(f"[P4] OVERALL: {verdict}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = df.groupby("alpha0").ACC.agg(["mean", "std", "count"])
    ci = 1.96 * agg["std"] / np.sqrt(agg["count"])
    plt.figure(figsize=(5, 4))
    plt.errorbar(agg.index, agg["mean"], yerr=ci, marker="o", capsize=3)
    plt.xscale("log")
    plt.xlabel("alpha0"); plt.ylabel("ACC@10%")
    plt.title("alpha0 sensitivity (corrected gain; Bug #4 fixed)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "alpha0_curve.png"), dpi=120)
    print("wrote results/alpha0_curve.csv and alpha0_curve.png")


if __name__ == "__main__":
    main()
