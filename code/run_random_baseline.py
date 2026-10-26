"""run_random_baseline.py — E4 Theorem-2 conflict diagnostic RANDOM BASELINE.

User-mandated control for the saturation issue in E4: the conflict condition
(s_i < 0) is activated at 99.6-99.9% of QUERIED points; a reviewer will ask
whether a RANDOM pair of pool points also yields a negative inner product at
~99%.  This script samples, for every (overlap_level, seed, method) of the
c=2 balanced design (results/theorem2/c2_record/), a random subset of the
unlabeled pool and computes the SAME diagnostic as the queried points:

    s_i = < tan(grad_h_i), grad_w_i^rob >      (exploratory.run_x2_diag,
          M_DIAG=500 CRN finite difference, delta grid 1e-2/1e-3/1e-4)

same membership matrix U (init-fit, pre-refit at the 0.10 checkpoint, exact
mirror of replicate_selection in run_theorem2_diagnostic.py), same
diag_seed = seed + level_id*10000.  The only difference: points are drawn
UNIFORMLY AT RANDOM from the unlabeled pool (excluding the queried ones)
instead of being selected by the acquisition function.

Outputs under v2_unified/results/theorem2/:
  random_baseline.csv   overlap_level, seed, method, point_idx, s_i, norm_h, norm_w
  baseline_done.txt     one key per completed run
  baseline_run.log      progress log

Idempotent: completed (level, seed, method) keys are skipped on re-run.
Per-run sample size N_RAND=32 (200 runs x 32 = 6400 points total; enough to
pin the s<0 fraction to ~+-0.01).  Workers: 4.
"""
import os
import sys
import csv
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from run_experiments import (plcfcm_fit, select_batch, _lambda2_over_n,
                             INIT_FRAC, M_MC, RHO)
import exploratory
exploratory.M_DIAG = 500                      # E4 override (same as E4)
from exploratory import run_x2_diag, grad_h, _tan   # noqa: E402

DATA_ROOT = os.path.join(ROOT, "data", "controlled_overlap")
RES_ROOT = os.path.join(ROOT, "results", "theorem2")
os.makedirs(RES_ROOT, exist_ok=True)

OVERLAP_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
METHODS = ["P2_Full", "P2_NoExploration"]
SEEDS = list(range(20))
BUDGET = 0.10
C_NOW = 2
N_RAND = 32

CSV_PATH = os.path.join(RES_ROOT, "random_baseline.csv")
DONE_TXT = os.path.join(RES_ROOT, "baseline_done.txt")
LOG_TXT = os.path.join(RES_ROOT, "baseline_run.log")

FIELDS = ["overlap_level", "seed", "method", "point_idx",
          "s_i", "norm_h", "norm_w"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_instance(level):
    folder = os.path.join(DATA_ROOT, f"overlap_{level:.1f}")
    X = np.load(os.path.join(folder, "X.npy"))
    y = np.load(os.path.join(folder, "y_true.npy"))
    return X, y


def replicate_selection(X, y, seed):
    """Mirror run_traj (l0=0, eta=0, budgets=(0.10,)) to capture selected
    indices and the pre-refit U at the selection moment (c=2 design)."""
    n = X.shape[0]
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n_init = max(2, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c=C_NOW, labeled_idx=np.where(labeled)[0],
                          labels=y[labeled], seed=seed)
    target = int(round(BUDGET * n))
    b = target - int(labeled.sum())
    unlab = np.where(~labeled)[0]
    beta = 1.0 * _lambda2_over_n(U)
    sel_rel = select_batch(U[unlab], X[unlab], V, "P2_Full", b,
                           alpha_0=50.0, M_MC=M_MC, rho_expl=RHO,
                           lam_red=beta, seed=seed, u_all=U)
    sel = unlab[sel_rel]
    return sel, U


def run_one(job):
    """Top-level worker. job = (level_id, level, seed, method)."""
    level_id, level, seed, method = job
    X, y = load_instance(level)
    sel, U = replicate_selection(X, y, seed)
    # random pool: unlabeled points NOT queried by the acquisition function
    n = X.shape[0]
    labeled0 = np.zeros(n, bool)
    rng0 = np.random.RandomState(0)
    rng0.choice(n, size=max(2, int(round(INIT_FRAC * n))), replace=False)  # same init
    # build the unlabeled mask identically to replicate_selection
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n_init = max(2, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    unlab = np.where(~labeled)[0]
    queried = set(int(i) for i in sel)
    cand = np.array([i for i in unlab if int(i) not in queried])
    rs = np.random.RandomState(777 + level_id * 1000 + seed * 37
                               + METHODS.index(method) * 3)
    rp = cand[rs.choice(len(cand), size=min(N_RAND, len(cand)), replace=False)]
    # same diagnostic call as E4 on the random subset
    diag_seed = seed + level_id * 10000
    di = run_x2_diag(U[rp], U, alpha0=50.0, kappa=1.0, seed=diag_seed)
    gh = grad_h(U[rp], U.mean(0)[None, :], n_pool=U.shape[0])
    gh_tan = _tan(gh)
    norm_h = np.linalg.norm(gh_tan, axis=1)
    rows = []
    for k in range(len(rp)):
        rows.append({
            "overlap_level": level, "seed": seed, "method": method,
            "point_idx": int(rp[k]), "s_i": float(di["s"][k]),
            "norm_h": float(norm_h[k]), "norm_w": float(di["normw"][k]),
        })
    return {"key": f"{level}|{seed}|{method}", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    done = set()
    if os.path.exists(DONE_TXT):
        with open(DONE_TXT, encoding="utf-8-sig", errors="ignore") as f:
            done = set(f.read().split())
    jobs = [(li, lv, sd, md)
            for li, lv in enumerate(OVERLAP_LEVELS)
            for sd in SEEDS
            for md in METHODS
            if f"{lv}|{sd}|{md}" not in done]
    log(f"[start] {len(jobs)} pending runs (of {len(OVERLAP_LEVELS) * len(SEEDS) * len(METHODS)}) "
        f"N_RAND={N_RAND} workers={args.workers} M_DIAG={exploratory.M_DIAG}")
    if not jobs:
        log("[start] nothing to do"); return

    new_csv = not os.path.exists(CSV_PATH)
    f_csv = open(CSV_PATH, "a", encoding="utf-8", newline="")
    w_csv = csv.DictWriter(f_csv, fieldnames=FIELDS)
    if new_csv:
        w_csv.writeheader()

    from multiprocessing import Pool
    t0 = time.time()
    n_done = 0
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(run_one, jobs, chunksize=1):
            for r in res["rows"]:
                w_csv.writerow(r)
            f_csv.flush()
            with open(DONE_TXT, "a", encoding="utf-8") as f:
                f.write(res["key"] + "\n")
            n_done += 1
            if n_done % 20 == 0 or n_done == len(jobs):
                rate = n_done / max(time.time() - t0, 1e-9)
                log(f"progress {n_done}/{len(jobs)} ({rate:.2f} run/s) "
                    f"last={res['key']} pts={len(res['rows'])}")
    f_csv.close()
    log(f"[done] {n_done} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
