"""run_synthetic_sweep.py — E1 synthetic sweep driver (resume v2).

Runs the full 72-instance x 4 methods x 20 paired seeds x 4 budgets grid on the
FROZEN AL pipeline (run_experiments.run_traj for Full / NoExploration / Random)
plus a self-contained SlackUniform trajectory implemented below.

SlackUniform implementation (interpretation of the E1 design note; written here
because acquisition.py is FROZEN and must not be edited)
---------------------------------------------------------------------
The codebase has no standalone "capacity budget" term.  We therefore map the
"capacity buffer" onto the per-point KL ambiguity radius kappa:

  * Full (P2_Full): modular score = w_rob(kappa=KAPPA) + RHO * h_i, where the
    rho*h_i term is the trust/exploration placement reward.
  * SlackUniform: drop the exploration bonus entirely (rho = 0) and instead
    uniformly inflate every candidate's KL radius to
        kappa_i = KAPPA * (1 + beta * Tbar),
    where beta = lambda_2(L_R)/n on the CURRENT membership matrix U
    (run_experiments._lambda2_over_n(U), lam_mult=1), and Tbar = mean over the
    unlabeled pool of h_i = exploration_reward(U_unlab, u_all=U).  The redundancy
    penalty stays on at weight beta (identical pairwise-cosine greedy loop as
    acquisition.select_batch: scores = mod - beta * red_sum).

So SlackUniform modular score = w_rob(kappa_i) + 0 * h_i; the KL-robust gain is
shared with Full, only the capacity buffer is spread uniformly and the directed
exploration placement is removed.  Because kappa_i > KAPPA generally, the
robust gain w_rob(kappa_i) < w_rob(KAPPA), so scores differ from NoExploration
(which uses kappa=KAPPA, rho=0) pointwise -> different selected batches.

Init-FCM cache (bit-level compatible)
-------------------------------------
run_traj's loop starts with a full PLCFCM fit on the L0 labeled set.  That fit
depends ONLY on (instance_id, seed) — not on method — because the L0 rng is
seeded by (seed*1000 + l0) with l0=0, eta=0, and plcfcm_fit itself creates an
independent RandomState(seed) for k-means++ init.  We therefore wrap
run_experiments.plcfcm_fit with a disk cache keyed by (iid, seed): the first
call for a (iid, seed) runs the real fit and pickles (U, V); subsequent calls
from the other 3 methods load the pickle.  Atomic write (temp + os.replace).
Validation: see _e1_validate_cache.py.

Resumability: every finished (instance, method, seed) appends its 4 budget rows
to summary.csv and records a key in done.txt; reruns skip done keys.
Workers = 6 (hard cap; see agent-hint).  Progress every 300 tasks with ETA,
stdout -> results/synthetic/sweep2.log (do NOT overwrite sweep.log).
"""
import os
import sys
import json
import time
import pickle
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_experiments
from run_experiments import (run_traj, metrics, _lambda2_over_n,
                              INIT_FRAC, M_MC, KAPPA, RHO, DELTA_NI)
from fcm_update import plcfcm_fit as _real_plcfcm_fit
from acquisition import exploration_reward, cosine_redundancy
from gain import calculate_gain

ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "synthetic")
RES_DIR = os.path.join(ROOT, "results", "synthetic")
CACHE_DIR = os.path.join(RES_DIR, "init_cache")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

METHODS = ["P2_Full", "P2_NoExploration", "SlackUniform", "Random"]
SEEDS = list(range(20))
BUDGETS = (0.05, 0.10, 0.15, 0.20)
ALPHA0 = 50.0
LAM_MULT = 1.0
N_INST = 72
PROGRESS_EVERY = 300
LARGE_N_THRESHOLD = 5000     # N>=5000 = "large job" (peak RSS ~600-700MB)
LARGE_CONCURRENCY = 16       # large-job concurrency cap (ramped 6→14→16 per user)

SUMMARY_COLS = ["instance_id", "overlap", "block", "c", "d", "N", "method",
                "seed", "budget", "ACC", "NMI", "ARI"]


# --------------------------------------------------------------------------- #
# Init-FCM cache wrapper (bit-level compatible)
# --------------------------------------------------------------------------- #
# Per-worker context: _worker sets iid/seed/init_done before calling run_traj.
_JOB_CTX = {"iid": None, "seed": None, "init_done": False}

# Cross-process semaphore for large jobs (N>=5000). Set by pool initializer.
_LARGE_SEM = None
try:
    import psutil as _psutil
except Exception:
    _psutil = None


def _pool_init(sem):
    global _LARGE_SEM
    _LARGE_SEM = sem


def _cached_plcfcm_fit(X, c, labeled_idx=None, labels=None, seed=0, **kw):
    """Drop-in for fcm_update.plcfcm_fit. Intercepts the FIRST (L0) call per
    (iid, seed) and serves a pickled (U, V) from disk; all later refits pass
    through to the real implementation. Bit-level compatible because
    plcfcm_fit is deterministic given (X, c, labeled_idx, labels, seed)."""
    if _JOB_CTX["iid"] is not None and not _JOB_CTX["init_done"]:
        iid = _JOB_CTX["iid"]
        cache_path = os.path.join(CACHE_DIR, f"init_{iid:03d}_seed{seed}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                U0, V0 = pickle.load(f)
            _JOB_CTX["init_done"] = True
            return U0.copy(), V0.copy(), 0
        U, V, it = _real_plcfcm_fit(X, c=c, labeled_idx=labeled_idx,
                                     labels=labels, seed=seed, **kw)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump((U.copy(), V.copy()), f)
        os.replace(tmp, cache_path)
        _JOB_CTX["init_done"] = True
        return U, V, it
    return _real_plcfcm_fit(X, c=c, labeled_idx=labeled_idx, labels=labels,
                             seed=seed, **kw)


# Install the wrapper into BOTH namespaces that call plcfcm_fit:
#   - run_experiments.run_traj  references module-level run_experiments.plcfcm_fit
#   - run_synthetic_sweep.run_traj_slackuniform references this module's plcfcm_fit
run_experiments.plcfcm_fit = _cached_plcfcm_fit
plcfcm_fit = _cached_plcfcm_fit


# --------------------------------------------------------------------------- #
# SlackUniform trajectory (mirrors run_traj exactly except selection)
# --------------------------------------------------------------------------- #
def run_traj_slackuniform(X, y, c, seed, alpha0=ALPHA0, lam_mult=LAM_MULT,
                          budgets=BUDGETS):
    """Self-contained trajectory identical to run_experiments.run_traj except
    the greedy batch uses the SlackUniform score (uniformly inflated kappa,
    rho=0, redundancy penalty beta)."""
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n = X.shape[0]
    yt = y.copy()
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True

    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=yt[labeled], seed=seed)
    rows = []
    for bud in budgets:
        target = int(round(bud * n))
        if labeled.sum() >= target:
            rows.append({"budget": bud, **metrics(yt, U)})
            continue
        b = target - int(labeled.sum())
        unlab = np.where(~labeled)[0]
        beta = lam_mult * _lambda2_over_n(U)

        # --- SlackUniform: rho=0; uniformly inflated KL radius -------------- #
        h_pool = exploration_reward(U[unlab], u_all=U)
        t_bar = float(h_pool.mean())
        kappa_s = KAPPA * (1.0 + beta * t_bar)
        mod = np.empty(len(unlab))
        for i, u in enumerate(U[unlab]):
            g = calculate_gain(u, alpha_0=alpha0, epsilon_q=0.1, M_MC=M_MC,
                               kappa=kappa_s, order=2, seed=seed + i)
            mod[i] = g["w_robust"]          # rho_expl = 0
        R = cosine_redundancy(U[unlab])
        avail = np.ones(len(unlab), dtype=bool)
        red_sum = np.zeros(len(unlab))
        sel_local = []
        for _ in range(min(b, len(unlab))):
            scores = mod - beta * red_sum
            scores[~avail] = -np.inf
            j = int(np.argmax(scores))
            if scores[j] == -np.inf:
                break
            sel_local.append(j)
            avail[j] = False
            red_sum += R[j]
        sel = unlab[sel_local]

        labeled[sel] = True
        U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                             labels=yt[labeled], seed=seed)
        rows.append({"budget": bud, **metrics(yt, U)})
    return rows


# --------------------------------------------------------------------------- #
# Worker (top-level, picklable)
# --------------------------------------------------------------------------- #
def _worker(job):
    iid, method, seed = job
    _JOB_CTX["iid"] = iid
    _JOB_CTX["seed"] = seed
    _JOB_CTX["init_done"] = False
    dpath = os.path.join(DATA_DIR, f"instance_{iid:03d}")
    X = np.load(os.path.join(dpath, "X.npy"))
    y = np.load(os.path.join(dpath, "y_true.npy"))
    meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
    c = meta["c"]
    n_size = int(meta["N"])
    is_large = n_size >= LARGE_N_THRESHOLD

    # Acquire large-job semaphore BEFORE loading X if large; otherwise just run.
    # (X already loaded above; for large jobs we still hold the semaphore during
    # the trajectory. The load itself is small vs the FCM/MC work.)
    if is_large and _LARGE_SEM is not None:
        _LARGE_SEM.acquire()

    rss_before = _psutil.Process().memory_info().rss if _psutil else 0
    t_job = time.time()
    try:
        ds = f"syn{iid:03d}"
        if method == "SlackUniform":
            rows = run_traj_slackuniform(X, y, c, seed)
        else:
            rows, *_ = run_traj(X, y, c, ds, method, seed, l0=0, eta=0.0,
                                alpha0=ALPHA0, lam_mult=LAM_MULT,
                                budgets=BUDGETS, run_x2=False)
    finally:
        rss_after = _psutil.Process().memory_info().rss if _psutil else 0
        if is_large and _LARGE_SEM is not None:
            _LARGE_SEM.release()

    out = []
    for r in rows:
        out.append({
            "instance_id": iid, "overlap": meta["overlap_target"],
            "block": meta["block"], "c": c, "d": meta["d"], "N": n_size,
            "method": method, "seed": seed, "budget": r["budget"],
            "ACC": r["ACC"], "NMI": r["NMI"], "ARI": r["ARI"],
        })
    import gc
    del X, y
    gc.collect()
    return {"job": job, "rows": out, "large": is_large,
            "rss_peak_mb": max(rss_before, rss_after) / 1e6,
            "elapsed_s": time.time() - t_job}


def done_keys():
    p = os.path.join(RES_DIR, "done.txt")
    if not os.path.exists(p):
        return set()
    return set(open(p, encoding="utf-8-sig", errors="ignore").read().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16,
                    help="target 16; large jobs (N>=5000) capped at 6 via semaphore")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only run first N pending jobs")
    args = ap.parse_args()

    summary_path = os.path.join(RES_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        pd.DataFrame(columns=SUMMARY_COLS).to_csv(summary_path, index=False,
                                                   encoding="utf-8")

    jobs = [(iid, m, s) for iid in range(1, N_INST + 1)
            for m in METHODS for s in SEEDS]
    done = done_keys()
    pending = [j for j in jobs
               if f"{j[0]}/{j[1]}/{j[2]}" not in done]
    if args.limit:
        pending = pending[:args.limit]
    # Reorder: small N first (fast accumulation), large N last (semaphore-limited).
    # Load N for each instance once.
    print(f"[sweep2] sorting {len(pending)} pending jobs by N ascending...",
          flush=True)
    n_by_iid = {}
    for iid in range(1, N_INST + 1):
        mp_ = json.load(open(os.path.join(DATA_DIR, f"instance_{iid:03d}",
                                          "meta.json"), encoding="utf-8"))
        n_by_iid[iid] = int(mp_["N"])
    pending.sort(key=lambda j: (n_by_iid[j[0]], j[0], j[1], j[2]))
    total = len(pending)
    print(f"[sweep2] total grid={len(jobs)} done={len(done)} pending={total} "
          f"workers={args.workers} cache_dir={CACHE_DIR}", flush=True)
    if total == 0:
        print("[sweep2] nothing to do", flush=True)
        return

    t0 = time.time()
    n_done = 0
    n_large = 0
    large_rss_max = 0.0
    large_elapsed_sum = 0.0
    done_f = open(os.path.join(RES_DIR, "done.txt"), "a", encoding="utf-8")
    lock = __import__("threading").Lock()

    def handle(res):
        nonlocal n_done, n_large, large_rss_max, large_elapsed_sum
        iid, method, seed = res["job"]
        df = pd.DataFrame(res["rows"], columns=SUMMARY_COLS)
        with lock:
            df.to_csv(summary_path, mode="a", header=False, index=False,
                      encoding="utf-8")
            done_f.write(f"{iid}/{method}/{seed}\n")
            done_f.flush()
            n_done += 1
            large_note = ""
            if res.get("large"):
                n_large += 1
                rss_pk = res.get("rss_peak_mb", 0.0)
                large_rss_max = max(large_rss_max, rss_pk)
                large_elapsed_sum += res.get("elapsed_s", 0.0)
                if n_large % 10 == 0:
                    large_note = (f"  [large:{n_large} rss_peak_max="
                                  f"{large_rss_max:.0f}MB avg_t="
                                  f"{large_elapsed_sum/n_large:.1f}s]")
            if n_done % PROGRESS_EVERY == 0 or n_done == total:
                el = time.time() - t0
                rate = n_done / max(el, 1)
                eta = (total - n_done) / max(rate, 1e-9)
                print(f"[sweep2] {n_done}/{total}  {rate:.3f} jobs/s  "
                      f"elapsed={el/60:.1f}min  ETA={eta/60:.1f}min"
                      f"{large_note}", flush=True)
            elif large_note:
                print(f"[sweep2] progress={n_done}/{total}{large_note}",
                      flush=True)

    sem = mp.Semaphore(LARGE_CONCURRENCY)
    pool = mp.Pool(args.workers, initializer=_pool_init, initargs=(sem,))
    try:
        for res in pool.imap_unordered(_worker, pending):
            handle(res)
    finally:
        pool.close()
        pool.join()
        done_f.close()
    el = time.time()
    # Write cache key manifest
    keys_path = os.path.join(CACHE_DIR, "keys.txt")
    pkls = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl"))
    with open(keys_path, "w", encoding="utf-8") as f:
        for k in pkls:
            f.write(k + "\n")
    with open(os.path.join(RES_DIR, "COMPLETED.flag"), "w", encoding="utf-8") as f:
        f.write(f"done at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"jobs={total} elapsed_sec={el:.1f}\n")
    print(f"[sweep2] ALL DONE: {total} jobs in {el/60:.1f} min "
          f"({el/3600:.2f}h); cache entries={len(pkls)}", flush=True)


if __name__ == "__main__":
    main()
