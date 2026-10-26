"""run_theorem2_diagnostic.py — E4 Theorem-2 conflict inner-product diagnostic (c=3).

[E4 revision 2026-10-22] Re-run on the c=3 controlled-overlap data
(data/controlled_overlap_c3/) to fix the structural defect of the c=2 design,
where the overlap class (high-H & high-h) was empty at every level (see
results/theorem2/c2_record/ and check_work/changelog_e4.md).  Outputs go to
diagnostic_c3.csv / acc_c3.csv / done_c3.txt / theorem2_c3_run.log; c=2
records are preserved untouched in c2_record/.

For each controlled-overlap instance (overlap in {0.0,0.2,0.4,0.6,0.8}),
method in {P2_Full, P2_NoExploration}, seed in 0..19:

  1. Run the OFFICIAL trajectory via run_experiments.run_traj at the 10%
     budget checkpoint (frozen constants INIT_FRAC=0.02, M_MC=50,
     KAPPA=1.0, RHO=0.1, one FCM refit per checkpoint) and record its ACC.
  2. Replicate the selection loop (identical RNG sequence, identical calls)
     to capture the ~160 points selected at the 0.10 checkpoint together
     with the pre-refit membership matrix U, because run_traj does not
     return selected indices.
  3. On the selected points compute the Theorem-2 conflict inner product
         s_i = < tan(grad_h_i), grad_w_i^rob >
     reusing exploratory.run_x2_diag:
       grad_h_i = (1 - 1/n_pool) sign(u_i - u_bar)   (closed form, Thm 2(i))
       grad_w_i^rob : CRN central finite difference of w^rob(u),
                      robust_gain_from_samples re-evaluated on each
                      perturbed membership vector.
     Point classes (boundary / eccentric / overlap / other) use the SAME
     quartile thresholds as exploratory.run_x2_diag (Table 6 / S.D.2).

Engineering trade-off (documented in check_work/changelog_e4.md):
  exploratory.M_DIAG = 2000 in the original mechanism check is too heavy for
  200 runs x ~160 points; we override to M_DIAG = 500 (CRN finite-difference
  plateau estimate; 500 common-random draws are enough for a stable plateau
  gradient; the delta grid (1e-2,1e-3,1e-4) is unchanged).  Only selected
  points are scored (no full-pool scan).  Parallelism: multiprocessing.Pool
  with 4 workers (machine already runs other jobs).  Idempotent: each
  completed (overlap, seed, method) appends rows to diagnostic.csv and a done
  key to results/theorem2/done.txt.

Outputs under v2_unified/results/theorem2/ (c=3 run):
  diagnostic_c3.csv   overlap_level, seed, method, point_idx, class, s_i,
                      norm_h, norm_w
  acc_c3.csv          overlap_level, seed, method, ACC      (auxiliary)
  done_c3.txt         one key per completed run
  theorem2_c3_run.log progress log
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

from run_experiments import (run_traj, plcfcm_fit, select_batch,
                             _lambda2_over_n, INIT_FRAC, M_MC, RHO)
import exploratory
exploratory.M_DIAG = 500                      # E4 override (see module docstring)
from exploratory import run_x2_diag           # noqa: E402

DATA_ROOT = os.path.join(ROOT, "data", "controlled_overlap_c3")
RES_ROOT = os.path.join(ROOT, "results", "theorem2")
os.makedirs(RES_ROOT, exist_ok=True)

OVERLAP_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
METHODS = ["P2_Full", "P2_NoExploration"]
SEEDS = list(range(20))
BUDGET = 0.10
C_NOW = 3

DIAG_CSV = os.path.join(RES_ROOT, "diagnostic_c3.csv")
ACC_CSV = os.path.join(RES_ROOT, "acc_c3.csv")
DONE_TXT = os.path.join(RES_ROOT, "done_c3.txt")
LOG_TXT = os.path.join(RES_ROOT, "theorem2_c3_run.log")

DIAG_FIELDS = ["overlap_level", "seed", "method", "point_idx",
               "class", "s_i", "norm_h", "norm_w"]


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


def done_keys():
    if not os.path.exists(DONE_TXT):
        return set()
    with open(DONE_TXT, encoding="utf-8-sig", errors="ignore") as f:
        return set(f.read().split())


def replicate_selection(X, y, method, seed):
    """Mirror run_traj (l0=0, eta=0, budgets=(0.10,)) to capture selected
    indices and the pre-refit U at the selection moment."""
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
    sel_rel = select_batch(U[unlab], X[unlab], V, method, b,
                           alpha_0=50.0, M_MC=M_MC, rho_expl=RHO,
                           lam_red=beta, seed=seed, u_all=U)
    sel = unlab[sel_rel]
    return sel, U


def run_one(job):
    """Top-level worker. job = (level_id, level, seed, method)."""
    level_id, level, seed, method = job
    X, y = load_instance(level)
    # 1) official trajectory -> ACC at 0.10
    out = run_traj(X, y, c=C_NOW, ds=f"ctl3{level_id}", method=method,
                   seed=seed, l0=0, eta=0.0, alpha0=50.0, lam_mult=1.0,
                   budgets=(BUDGET,), run_x2=False)
    rows_metrics = out[0]
    acc = float(rows_metrics[0]["ACC"])
    # 2) replicate selection to capture points + pre-refit U
    sel, U = replicate_selection(X, y, method, seed)
    # 3) gradient diagnostic on selected points (CRN, M_DIAG=500)
    diag_seed = seed + level_id * 10000
    di = run_x2_diag(U[sel], U, alpha0=50.0, kappa=1.0, seed=diag_seed)
    # norm_h = ||tan(grad_h)||, same projection run_x2_diag uses internally
    from exploratory import grad_h, _tan
    gh = grad_h(U[sel], U.mean(0)[None, :], n_pool=U.shape[0])
    gh_tan = _tan(gh)
    norm_h = np.linalg.norm(gh_tan, axis=1)
    rows = []
    for k in range(len(sel)):
        rows.append({
            "overlap_level": level,
            "seed": seed,
            "method": method,
            "point_idx": int(sel[k]),
            "class": str(di["label"][k]),
            "s_i": float(di["s"][k]),
            "norm_h": float(norm_h[k]),
            "norm_w": float(di["normw"][k]),
        })
    return {"key": f"{level}|{seed}|{method}", "rows": rows,
            "acc": acc, "n_sel": int(len(sel)),
            "m_diag": int(di["M_diag"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="run one job (overlap 0.8, seed 0, P2_Full) and exit")
    args = ap.parse_args()

    if args.smoke:
        t0 = time.time()
        res = run_one((4, OVERLAP_LEVELS[4], 0, "P2_Full"))
        dt = time.time() - t0
        print(f"[smoke] key={res['key']} n_sel={res['n_sel']} "
              f"ACC={res['acc']:.4f} M_DIAG={res['m_diag']} "
              f"elapsed={dt:.1f}s")
        import pandas as pd
        df = pd.DataFrame(res["rows"])
        print(df.groupby("class").agg(n=("s_i", "size"),
                                      frac_neg=("s_i", lambda s: float((s < 0).mean())),
                                      mean_s=("s_i", "mean")).round(4))
        return

    done = done_keys()
    jobs = [(li, lv, sd, md)
            for li, lv in enumerate(OVERLAP_LEVELS)
            for sd in SEEDS
            for md in METHODS
            if f"{lv}|{sd}|{md}" not in done]
    log(f"[start] {len(jobs)} pending jobs "
        f"({len(done)} already done); workers={args.workers} "
        f"M_DIAG={exploratory.M_DIAG}")
    if not jobs:
        log("[start] nothing to do"); return

    new_diag = not os.path.exists(DIAG_CSV)
    f_diag = open(DIAG_CSV, "a", encoding="utf-8", newline="")
    w_diag = csv.DictWriter(f_diag, fieldnames=DIAG_FIELDS)
    if new_diag:
        w_diag.writeheader()
    f_acc = open(ACC_CSV, "a", encoding="utf-8", newline="")
    w_acc = csv.writer(f_acc)
    acc_new = os.path.getsize(ACC_CSV) == 0 if os.path.exists(ACC_CSV) else True
    if acc_new:
        w_acc.writerow(["overlap_level", "seed", "method", "ACC"])

    from multiprocessing import Pool
    t0 = time.time()
    n_done = 0
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(run_one, jobs, chunksize=1):
            for r in res["rows"]:
                w_diag.writerow(r)
            w_acc.writerow([res["rows"][0]["overlap_level"],
                            res["rows"][0]["seed"],
                            res["rows"][0]["method"],
                            f"{res['acc']:.6f}"])
            f_diag.flush(); f_acc.flush()
            with open(DONE_TXT, "a", encoding="utf-8") as f:
                f.write(res["key"] + "\n")
            n_done += 1
            if n_done % 20 == 0 or n_done == len(jobs):
                rate = n_done / max(time.time() - t0, 1e-9)
                log(f"progress {n_done}/{len(jobs)} ({rate:.2f} run/s) "
                    f"last={res['key']} n_sel={res['n_sel']}")
    f_diag.close(); f_acc.close()
    log(f"[done] {n_done} jobs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
