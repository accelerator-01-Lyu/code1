# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""run_synthetic_3d.py — thin driver for the isolated 3-D post-hoc validation.

ISOLATION:
  * Reuses the EXACT confirmatory machinery by import-only (NO fork/mod):
        from fcm_update import plcfcm_fit
        from acquisition import select_batch
        from gain import entropy
    and mirrors run_experiments.run_traj's acquisition/refit rhythm
    (M_MC=50, alpha0=50, u_all=U, lam_red=lambda2/n, batch-per-checkpoint,
    rng = RandomState(seed*1000+l0)).  rho_expl=RHO=0.1 for P2_Full (code
    default, untouched); P2_NoExploration forces rho=0 INSIDE select_batch.

DEVIATIONS FOR THE SYNTHETIC PROTOCOL (documented, isolated to this driver):
  * |L0| = max(c, floor(0.02 * N_normal)); L0 drawn from NORMAL points only
    (eccentric y=-1 decoys always start unlabeled).
  * Budget checkpoint = 10% of N_normal (=120 labeled total incl. L0).
  * Eccentric decoys (y=-1) ENTER FCM fitting and the unlabeled pool and MAY
    be selected; if selected they are hard-pinned to their CURRENT fuzzy
    winner cluster column (argmax of the membership row BEFORE the refit) so
    c=6 stays valid.  They are EXCLUDED from ACC/NMI/ARI; Hungarian matching
    is done on the 6 normal clusters only.
  * Writes trajectory_*.csv and summary_*.csv under results/synthetic_3d/.

USAGE
-----
    python run_synthetic_3d.py --smoke         # ecoli pkl vs main shards, max diff
    python run_synthetic_3d.py --run           # full 9 pkl x 2 methods x 10 seeds
"""
import os
import sys
import csv
import pickle
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

from fcm_update import plcfcm_fit
from acquisition import select_batch, cosine_redundancy
from gain import entropy

HERE = os.path.dirname(os.path.abspath(__file__))
SYN_DIR = os.path.join(HERE, "results", "synthetic_3d")
ECOLI_PKL = os.path.join(HERE, "reference_repo", "data", "ecoli.pkl")
SHARD_DIR = os.path.join(HERE, "results", "shards")

# confirmatory constants copied (read-only) from run_experiments.py (FROZEN):
INIT_FRAC = 0.02
M_MC = 50
RHO = 0.1
ALPHA0 = 50.0
LAM_MULT = 1.0
AL_SEEDS = list(range(20261010, 20261020))     # 10 locked seeds
METHODS = ("P2_Full", "P2_NoExploration")
K = 6


def lambda2_over_n(U):
    """Local identical copy of run_experiments._lambda2_over_n (pure math):
    beta = lambda2(L_R)/n, R = cosine on normalized membership rows."""
    n = U.shape[0]
    R = cosine_redundancy(U)
    d = R.sum(1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    ev = np.linalg.eigvalsh(L)
    lam2 = float(np.sort(ev)[1]) if len(ev) > 1 else 1.0
    return lam2 / max(n, 1)


def load_pkl(path):
    o = pickle.load(open(path, "rb"))
    return np.asarray(o["X"], dtype=np.float64), np.asarray(o["y"], dtype=int)


def acc_hungarian(yt, yp):
    D = max(yt.max(), yp.max()) + 1
    C = np.zeros((D, D), dtype=int)
    for t, p in zip(yt, yp):
        C[t, p] += 1
    ri, ci = linear_sum_assignment(-C)
    return C[ri, ci].sum() / len(yt)


def metrics_masked(y, U):
    """ACC(Hungarian)/NMI/ARI over NORMAL points only (y != -1)."""
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from sklearn.metrics import adjusted_rand_score as ari
    m = y >= 0
    yt = y[m]
    yp = U.argmax(1)[m]
    return {"NMI": float(nmi(yt, yp)), "ARI": float(ari(yt, yp)),
            "ACC": float(acc_hungarian(yt, yp))}


def run_traj_thin(X, y, c, method, seed, budgets, n_budget,
                  l0=0, record=False):
    """Faithful port of run_experiments.run_traj (ablation path only).
    Returns rows[{budget,NMI,ARI,ACC}] and (if record) the greedy selection
    order of global indices."""
    rng = np.random.RandomState((seed * 1000 + l0) & 0x7FFFFFFF)
    n = X.shape[0]
    fit_labels = y.copy()                      # ecc selected -> pinned at argmax
    n_init = max(c, int(round(INIT_FRAC * n_budget)))
    labeled = np.zeros(n, dtype=bool)
    # L0 drawn from NORMAL points only (decoys y=-1 start unlabeled).
    pool0 = np.where(y >= 0)[0] if (y == -1).any() else np.arange(n)
    labeled[rng.choice(pool0, size=n_init, replace=False)] = True

    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=fit_labels[labeled], seed=seed)
    rows = []
    selected_order = []
    for bud in budgets:
        target = int(round(bud * n_budget))
        if labeled.sum() >= target:
            m = metrics_masked(y, U)
            rows.append({"budget": bud, **m})
            continue
        b = target - int(labeled.sum())
        unlab = np.where(~labeled)[0]
        beta = LAM_MULT * lambda2_over_n(U)
        sel = select_batch(U[unlab], X[unlab], V, method, b,
                           alpha_0=ALPHA0, M_MC=M_MC, rho_expl=RHO,
                           lam_red=beta, seed=seed, u_all=U)
        sel = unlab[sel]
        # eccentric decoys absorbed into current fuzzy winner cluster
        for s in sel:
            if y[s] < 0:
                fit_labels[s] = int(U[s].argmax())
        selected_order.extend([int(s) for s in sel])
        labeled[sel] = True
        U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                             labels=fit_labels[labeled], seed=seed)
        m = metrics_masked(y, U)
        rows.append({"budget": bud, **m})
    return rows, selected_order


# --------------------------------------------------------------------------- #
# Smoke: ecoli pkl vs already-written main-experiment shards
# --------------------------------------------------------------------------- #
def smoke():
    X, y = load_pkl(ECOLI_PKL)
    c = 8
    print(f"[smoke] ecoli X={X.shape} classes={np.unique(y)}")
    worst = 0.0
    for seed in (0, 1, 2):
        for method in METHODS:
            rows, _ = run_traj_thin(X, y, c, method, seed,
                                    budgets=(0.05, 0.10, 0.15, 0.20),
                                    n_budget=X.shape[0])
            # read the main-experiment shard
            sf = os.path.join(SHARD_DIR,
                              f"ecoli_{method}_50_{seed}_0_0.0_50.0_1.0.csv")
            ref = {float(r["budget"]): r for r in csv.DictReader(open(sf))}
            for r in rows:
                rb = ref[round(r["budget"], 2)]
                for met in ("NMI", "ARI", "ACC"):
                    d = abs(r[met] - float(rb[met]))
                    worst = max(worst, d)
                    if d > 1e-9:
                        print(f"  DIFF seed={seed} {method} bud={r['budget']} "
                              f"{met}: thin={r[met]:.10f} ref={rb[met]}")
    print(f"[smoke] MAX ABS DIFF vs main-experiment ecoli shards = {worst:.3e}")
    return worst


# --------------------------------------------------------------------------- #
# Synthetic run
# --------------------------------------------------------------------------- #
DATASETS = {
    "spiral": [0.4, 0.6, 0.8],
    "swissroll": [0.4, 0.6, 0.8],
    "hourglass": [0.3, 0.4, 0.6],
}


def _governor_cap():
    """Read .governor (4/3/2/0). 0 => paused. Default 4 if absent."""
    gp = os.path.join(SYN_DIR, ".governor")
    try:
        if os.path.exists(gp):
            v = int(open(gp).read().strip() or "4")
            return max(0, min(4, v))
    except Exception:
        pass
    return 4


def _run_one_job(job):
    """Top-level picklable worker. Loads its own pkl, returns result dict."""
    kind, sigma, method, seed = job
    X, y = load_pkl(os.path.join(SYN_DIR, f"{kind}_sigma{sigma}.pkl"))
    n_normal = int((y >= 0).sum())
    try:
        rows, order = run_traj_thin(X, y, K, method, seed, budgets=(0.10,),
                                    n_budget=n_normal, record=True)
        m = rows[-1]
        return {"ok": True, "kind": kind, "sigma": sigma, "method": method,
                "seed": seed, "acc": m["ACC"], "nmi": m["NMI"],
                "ari": m["ARI"], "order": order}
    except Exception as e:
        return {"ok": False, "kind": kind, "sigma": sigma, "method": method,
                "seed": seed, "err": str(e)}


def run_synthetic():
    from multiprocessing import Pool
    import time as _time
    os.makedirs(SYN_DIR, exist_ok=True)
    traj_path = os.path.join(SYN_DIR, "trajectory_all.csv")
    summ_path = os.path.join(SYN_DIR, "summary_all.csv")
    t_cols = ["dataset", "sigma", "seed", "method", "round", "selected_idx",
              "acc", "nmi", "ari"]
    s_cols = ["dataset", "sigma", "seed", "method", "acc", "nmi", "ari"]
    # resume: skip (kind,sigma,method,seed) already in summary
    done = set()
    if os.path.exists(summ_path):
        import csv as _csv
        for r in _csv.DictReader(open(summ_path)):
            done.add((r["dataset"], str(r["sigma"]), r["method"], r["seed"]))
    have_summary = os.path.exists(summ_path)
    tf = open(traj_path, "a", newline="")
    w_t = csv.DictWriter(tf, fieldnames=t_cols)
    if not have_summary:
        w_t.writeheader()
    sf = open(summ_path, "a", newline="")
    w_s = csv.DictWriter(sf, fieldnames=s_cols)
    if not have_summary:
        w_s.writeheader()

    jobs = []
    for kind, sigmas in DATASETS.items():
        for sigma in sigmas:
            for method in METHODS:
                for seed in AL_SEEDS:
                    if (kind, str(sigma), method, str(seed)) not in done:
                        jobs.append((kind, sigma, method, seed))
    print(f"[run] {len(jobs)} pending shards; Pool=8, governor-aware",
          flush=True)
    sys.stdout.flush()

    NWORKERS = 8
    t0 = _time.time()
    n_done = 0
    with Pool(NWORKERS) as pool:
        for res in pool.imap_unordered(_run_one_job, jobs, chunksize=1):
            # governor pause: 0 => hold off harvesting for 15s
            while _governor_cap() == 0:
                _time.sleep(15)
            if res["ok"]:
                order = res["order"]
                for step, sidx in enumerate(order, start=1):
                    w_t.writerow({"dataset": res["kind"], "sigma": res["sigma"],
                                  "seed": res["seed"], "method": res["method"],
                                  "round": step, "selected_idx": sidx,
                                  "acc": (res["acc"] if step == len(order) else ""),
                                  "nmi": (res["nmi"] if step == len(order) else ""),
                                  "ari": (res["ari"] if step == len(order) else "")})
                w_s.writerow({"dataset": res["kind"], "sigma": res["sigma"],
                              "seed": res["seed"], "method": res["method"],
                              "acc": res["acc"], "nmi": res["nmi"],
                              "ari": res["ari"]})
            else:
                w_s.writerow({"dataset": res["kind"], "sigma": res["sigma"],
                              "seed": res["seed"], "method": res["method"],
                              "acc": "NaN", "nmi": "NaN", "ari": "NaN"})
                print(f"[fail] {res['kind']} {res['sigma']} {res['method']} "
                      f"{res['seed']}: {res['err']}", flush=True)
            n_done += 1
            if n_done % 10 == 0:
                el = _time.time() - t0
                print(f"[progress] {n_done}/{len(jobs)}  {el:.0f}s "
                      f"({n_done/el*60:.1f}/min)", flush=True)
            tf.flush(); sf.flush()
    tf.close(); sf.close()
    el = _time.time() - t0
    print(f"[wrote] {summ_path}  completed={n_done}  {el:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    if args.run:
        run_synthetic()


if __name__ == "__main__":
    main()
