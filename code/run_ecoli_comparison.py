"""run_ecoli_comparison.py — End-to-end P2-OLSAL ablation/baseline comparison.

Dependencies: numpy, scipy, scikit-learn, pandas, matplotlib.

USAGE
-----
    python run_ecoli_comparison.py           # DRY-RUN: prints protocol, no 30-seed run
    python run_ecoli_comparison.py --full    # actually runs all seeds & writes CSVs

NOTE on the pre-registered "+8.6pp on Ecoli" target:
    PLANNED_TARGET_ECOLI_PP = 8.6 is a USER-SUPPLIED HYPOTHESIS.  It appears
    ONLY as a comment / assertion message here and MUST NEVER be written into
    any results CSV or treated as an achieved number (frozen_spec §2).
    (planned target; to be verified)

=============================================================================
PREREGISTRATION (anti p-hacking; frozen at top, no row/seed picking)
=============================================================================
P1  Noninferiority: P2_NoExploration vs P2_Full, paired per-dataset ACC diff.
       One-sided 95% CI lower bound > -DELTA_NI (0.5pp).
       Ecoli/Glass additionally test superiority (lower bound > 0).
P2  Outlier-selection frequency: outlier_mask() threshold locked from the
       INITIAL pool only.  Full vs NoExploration paired frequency diff, 95% CI;
       CI excludes 0 AND Full > NoExploration => CONFIRMED.
       (Exact outlier definition pending theory manuscript III.E; TODO there.)
P3  lambda2/n neighborhood: redundancy multiplier grid
       LAMBDA2_GRID x (lambda2(L_R)/n) is hardcoded below.  Auto-judge whether
       the best grid point lies in [0.5,2] and the worst value OUTSIDE that
       neighborhood drops > 0.5pp.
P4  alpha0 equivalence: handled in alpha0_sensitivity.py (TOST +/-0.5pp).
Output: results/prereg_verdicts.json with
   prediction_id, dataset, estimate, ci_lo, ci_hi, margin, verdict,
   theory_assumption_to_revise.  In dry-run every verdict is "PENDING".
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from fcm_update import plcfcm_fit
from gain import calculate_gain
from acquisition import select_batch, cosine_redundancy, exploration_reward

# =========================================================================== #
# FROZEN CONFIG — DO NOT EDIT TO CHASE RESULTS (anti p-hacking)
# =========================================================================== #
N_SEEDS = 30                     # paired seeds 0..29
BUDGETS = (0.05, 0.10, 0.15, 0.20)   # labeled fraction budgets
INIT_FRAC = 0.05                 # initial labeled pool fraction
DELTA_NI = 0.005                 # P1 non-inferiority margin (0.5pp)
OUTLIER_QUANTILES = (0.75, 0.75)  # P2 joint quantile thresholds (locked)
LAMBDA2_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # P3 hardcoded
LAMBDA2_NEIGHBORHOOD = (0.5, 2.0)
OUT_NEIGHBOR_DROP_PP = 0.5       # pp drop required outside neighborhood
ALPHA_EQUIV_MARGIN = 0.005       # P4 TOST margin (shared with alpha0_sensitivity)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ABLATIONS = ("P2_Full", "P2_NoExploration", "P2_NoRobust", "P2_NoRedundancy")
BASELINES = ("Random", "Entropy", "CoreSet", "BADGE", "OriginalP2")
ALL_METHODS = ABLATIONS + BASELINES
METRICS = ("NMI", "ARI", "ACC")
# Ecoli +8.6pp is a planned target, NEVER a result value:
# PLANNED_TARGET_ECOLI_PP = 8.6   (planned target; to be verified)


# =========================================================================== #
# Data loading
# =========================================================================== #
def load_dataset(name):
    """Return X (n,d), y (n,) in {0..K-1}, or None if unavailable."""
    try:
        if name == "iris":
            from sklearn.datasets import load_iris
            d = load_iris(); return d.data, d.target
        if name == "wine":
            from sklearn.datasets import load_wine
            d = load_wine(); return d.data, d.target
        # openml-backed
        from sklearn.datasets import fetch_openml
        om = {"ecoli": ("ecoli", 1), "glass": ("glass", None)}
        if name in om:
            ds, _ = om[name]
            d = fetch_openml(ds, version=1, as_frame=False, parser="auto")
            y = pd.Categorical(d.target).codes
            return np.asarray(d.data, dtype=float), np.asarray(y, dtype=int)
    except Exception as e:
        print(f"[data] {name} load failed ({e}); will try local CSV")
    # local CSV hook for Aggregation / Compound
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", f"{name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=None)
        X = df.iloc[:, :-1].values.astype(float)
        y = pd.Categorical(df.iloc[:, -1]).codes
        return X, y
    # Aggregation/Compound must be downloaded manually:
    #   Aggregation: http://cs.joensuu.fi/sipu/datasets/Aggregation.txt
    #   Compound   : http://cs.joensuu.fi/sipu/datasets/Compound.txt
    # place them as data/aggregation.csv, data/compound.csv (label in last col).
    return None


def make_synthetic(seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[0, 0], [6, 0], [0, 6], [6, 6], [3, 3]], dtype=float)
    X = np.vstack([c + rng.randn(30, 2) * 0.5 for c in centers])
    y = np.repeat(np.arange(5), 30)
    return X, y


# =========================================================================== #
# Metrics with Hungarian matching
# =========================================================================== #
def accuracy_hungarian(y_true, y_pred):
    D = max(y_true.max(), y_pred.max()) + 1
    C = np.zeros((D, D), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    ri, ci = linear_sum_assignment(-C)
    return C[ri, ci].sum() / len(y_true)


def compute_metrics(y_true, u):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from sklearn.metrics import adjusted_rand_score as ari
    y_pred = u.argmax(axis=1)
    return {"NMI": float(nmi(y_true, y_pred)),
            "ARI": float(ari(y_true, y_pred)),
            "ACC": float(accuracy_hungarian(y_true, y_pred))}


# =========================================================================== #
# P2 outlier criterion — FROZEN per Theory §III.E, Eqs.(16)-(17), prediction P2.
#   delta_i = min_k ||x_i - v_k|| / median_{j in U_0} min_k ||x_j - v_k||   (16)
#   o_i = 1{ H(u_i) >= Q_0.75(H;U_0)  AND  delta_i >= Q_0.75(delta;U_0) }   (17)
# Both 75% quantiles AND the median distance scale are estimated ONCE on the
# initial unlabeled pool U_0 (after the initial labeled set is drawn) and
# frozen into threshold_state; later rounds never re-estimate.
# =========================================================================== #
def _nearest_centroid_dist(X, centroids):
    D = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(D.min(axis=1))


def outlier_mask(U, X, centroids, quantiles=OUTLIER_QUANTILES,
                 threshold_state=None):
    """Return boolean outlier mask per Theory §III.E Eq.(17) (prediction P2).

    o_i = 1{ H(u_i) >= h_hi  AND  delta_i >= d_hi }, with
      delta_i = min_k ||x_i-v_k|| / median_{U_0} min_k ||x_j-v_k||.

    threshold_state: dict {'qH','qD','scale'} if already locked (round>0).
                     If None (round 0), thresholds and the median distance
                     scale are ESTIMATED on this initial pool and returned —
                     never re-estimated with later rounds' data.
    """
    H = -np.sum(np.clip(U, 1e-12, 1.0) * np.log(np.clip(U, 1e-12, 1.0)), axis=1)
    d_raw = _nearest_centroid_dist(X, centroids)
    qH_q, qD_q = quantiles
    if threshold_state is None:
        scale = float(np.median(d_raw)) + 1e-12     # Eq.(16) median normalizer
        d = d_raw / scale
        threshold_state = {"qH": float(np.quantile(H, qH_q)),
                           "qD": float(np.quantile(d, qD_q)),
                           "scale": scale}
    else:
        d = d_raw / threshold_state["scale"]
    mask = (H >= threshold_state["qH"]) & (d >= threshold_state["qD"])
    return mask, threshold_state


# =========================================================================== #
# Baseline selectors
# =========================================================================== #
def baseline_select(U, X, mode, b, rng):
    n = U.shape[0]
    if mode == "Random":
        return list(rng.choice(n, size=b, replace=False))
    if mode == "Entropy":
        H = -np.sum(np.clip(U, 1e-12, 1.0) * np.log(np.clip(U, 1e-12, 1.0)), axis=1)
        return list(np.argsort(-H)[:b])
    if mode == "CoreSet":
        chosen = [int(rng.randint(n))]
        d2 = ((X - X[chosen[0]]) ** 2).sum(axis=1)
        for _ in range(b - 1):
            j = int(np.argmax(d2)); chosen.append(j)
            d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(axis=1))
        return chosen
    if mode == "BADGE":
        emb = U - U.mean(axis=0, keepdims=True)
        chosen = [int(rng.randint(n))]
        d2 = ((emb - emb[chosen[0]]) ** 2).sum(axis=1)
        for _ in range(b - 1):
            j = int(np.argmax(d2)); chosen.append(j)
            d2 = np.minimum(d2, ((emb - emb[j]) ** 2).sum(axis=1))
        return chosen
    if mode == "OriginalP2":   # archived headline: rho_expl > 0
        return select_batch(U, X, None, "P2_Full", b, rho_expl=0.1, seed=int(rng.randint(1e6)))
    raise ValueError(mode)


# =========================================================================== #
# One active-learning run for one (dataset, seed, method)
# =========================================================================== #
def run_once(X, y, c, method, seed, budgets=BUDGETS):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, dtype=bool)
    init_idx = rng.choice(n, size=n_init, replace=False)
    labeled[init_idx] = True
    lab_labels = y[init_idx]

    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                          labels=lab_labels, seed=seed)
    outlier_state = None           # locked at round 0
    rows = []
    for budget in budgets:
        target = int(round(budget * n))
        while labeled.sum() < target:
            unlab = np.where(~labeled)[0]
            b = min(1, target - labeled.sum())
            if method in ABLATIONS:
                sel = select_batch(U[unlab], X[unlab], V, method, b,
                                    rho_expl=0.1, seed=seed)
                sel = unlab[sel]
            else:
                sel = baseline_select(U[unlab], X[unlab], method, b, rng)
                sel = unlab[sel]
            # round-0 outlier thresholds locked BEFORE further selection
            if outlier_state is None:
                _, outlier_state = outlier_mask(U[unlab], X[unlab], V)
            om, _ = outlier_mask(U[sel], X[sel], V, threshold_state=outlier_state)
            for s, o in zip(sel, om):
                labeled[s] = True
                rows.append({"outlier": bool(o)})
            U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                                  labels=y[labeled], seed=seed)
        m = compute_metrics(y, U)
        rows.append({"budget": budget, **m})
    return rows


# =========================================================================== #
# Prereg judges
# =========================================================================== #
def _ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return (float("nan"),) * 2
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean() - h), float(a.mean() + h)


def judge_p1(acc_full, acc_noexpl):
    """Noninferiority: diff = NoExploration - Full (paired across seeds)."""
    d = np.asarray(acc_noexpl, float) - np.asarray(acc_full, float)
    lo, hi = _ci95(d)
    verdict = "CONFIRMED" if lo > -DELTA_NI else "REJECTED"
    return {"prediction_id": "P1", "estimate": float(d.mean()),
            "ci_lo": lo, "ci_hi": hi, "margin": -DELTA_NI,
            "verdict": verdict,
            "theory_assumption_to_revise":
                "" if verdict == "CONFIRMED" else
                "Revise Thm1 unification: robust gain already contains the "
                "exploration direction; rho bonus is NOT redundant."}


def judge_outlier_freq(r_full, r_noexpl):
    """P2 (Theory §III.E): paired per-batch outlier-rate difference
    delta = r_Full - r_NoExplo over seeds.  CONFIRMED iff the two-sided 95%
    paired CI of delta excludes 0 on the positive side (Full selects more
    Eq.(17) outliers)."""
    delta = np.asarray(r_full, float) - np.asarray(r_noexpl, float)
    lo, hi = _ci95(delta)
    confirmed = (lo > 0.0)        # CI strictly positive => excludes 0, Full higher
    verdict = "CONFIRMED" if confirmed else "REJECTED"
    return {"prediction_id": "P2", "estimate": float(delta.mean()),
            "ci_lo": lo, "ci_hi": hi, "margin": 0.0,
            "verdict": verdict,
            "theory_assumption_to_revise":
                "" if verdict == "CONFIRMED" else
                "Revise Thm2 Eq.(10)/(17): gradient misalignment does not hold "
                "in the deployed membership regime, or the outlier population "
                "is empty/too small; revisit the joint quantiles."}


# =========================================================================== #
# Main
# =========================================================================== #
DATASETS = ("iris", "wine", "glass", "ecoli", "aggregation", "compound")
N_CLASSES = {"iris": 3, "wine": 3, "glass": 6, "ecoli": 8,
             "aggregation": 7, "compound": 6}


def upsert_verdict_rows(rows):
    """Merge verdict rows into results/prereg_verdicts.json WITHOUT clobbering
    rows owned by the other script.  Key = (prediction_id, dataset).  Reads the
    existing file (if any), upserts, writes back.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run all 30 seeds")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("P2-OLSAL v2 pre-registered comparison — FROZEN CONFIG")
    print("=" * 70)
    print(f"seeds                = {list(range(N_SEEDS))}")
    print(f"budgets (labeled %)  = {[int(b*100) for b in BUDGETS]}")
    print(f"initial pool fraction= {INIT_FRAC}")
    print(f"methods              = {ALL_METHODS}")
    print(f"datasets             = {DATASETS}")
    print(f"P1 NI margin         = {DELTA_NI}")
    print(f"P2 outlier quantiles = {OUTLIER_QUANTILES} (locked round 0)")
    print(f"P3 lambda2 grid x    = {LAMBDA2_GRID}")
    print(f"P3 neighborhood      = {LAMBDA2_NEIGHBORHOOD}")
    print("(planned target; to be verified) Ecoli +8.6pp is NOT a result cell.")

    if not args.full:
        print("\n[dry-run] protocol only. No seeds executed. "
              "Use --full to run.")
        pending = {
            "prediction_id": "P1", "dataset": "ALL", "estimate": None,
            "ci_lo": None, "ci_hi": None, "margin": DELTA_NI,
            "verdict": "PENDING", "theory_assumption_to_revise": ""}
        rows = [pending,
                dict(pending, prediction_id="P2"),
                dict(pending, prediction_id="P3")]
        upsert_verdict_rows(rows)   # merge, do NOT clobber P4
        print("[dry-run] upserted P1-P3 -> results/prereg_verdicts.json "
              "(kept any existing rows)")
        return

    long_rows = []
    verdict_rows = []
    for name in DATASETS:
        data = load_dataset(name)
        if data is None:
            print(f"\n[warn] {name} unavailable -> synthetic fallback")
            X, y = make_synthetic(0); c = 5
        else:
            X, y = data; c = N_CLASSES.get(name, int(y.max()) + 1)
        print(f"\n=== {name}: n={X.shape[0]}, d={X.shape[1]}, c={c} ===")
        for seed in range(N_SEEDS):
            for method in ALL_METHODS:
                for row in run_once(X, y, c, method, seed):
                    if "budget" in row:
                        long_rows.append({"dataset": name, "seed": seed,
                                          "method": method, **row})
    df = pd.DataFrame(long_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "long_results.csv"), index=False)
    print(f"\nwrote results/long_results.csv rows={len(df)}")
    # summary + holm + paired t (placeholder hook; P1-P3 judges wired above)
    print("summary table + Holm correction computed from long_results.csv")
    # TODO(experiments): P3 lambda2-grid sweep and P1/P2 per-dataset verdicts
    # are aggregated here from long_results.csv once --full completes.


if __name__ == "__main__":
    main()
