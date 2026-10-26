"""run_experiments.py — hardened, resumable P2-OLSAL confirmatory harness.

Dependencies (use the project venv): numpy scipy scikit-learn pandas matplotlib.

Run phases (frozen protocol; no row/seed picking):
  --phase smoke      : 3 seeds, ecoli, P2_Full vs P2_NoExploration, 10% budget
  --phase main       : 6 datasets x 9 methods x 30 seeds x budgets {5,10,15,20}%
  --phase thicken    : extends to 50 seeds, 3 L0 pools, eta noise, P3/P4 grids
  --phase aggregate  : recompute stats + verdicts + figures from shard CSVs

Resumability: every completed shard
(dataset,method,seed_set,seed,l0,eta,alpha0,lam_mult) appends one file
shards/<key>.csv and records the key in shards/done.txt.  Re-running skips
keys already in done.txt (crash-safe idempotent).

Data: code/reference_repo/data/*.pkl (already StandardScaler-normalized;
n/d/c verified against frozen spec: Iris150/4/3, Wine178/13/3, Glass214/9/6,
Ecoli336/7/8, Aggregation788/2/7, Compound399/2/6).
"""
import os, sys, json, time, pickle, argparse, itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from fcm_update import plcfcm_fit
from acquisition import select_batch, MODES, exploration_reward, cosine_redundancy
from run_ecoli_comparison import outlier_mask, load_dataset
from gain import entropy

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DATA = os.path.join(HERE, "reference_repo", "data")
OUT = os.path.join(HERE, "results")
SHARD_DIR = os.path.join(OUT, "shards")
ASSETS = r"D:\AL_FuzzyClustering_Paper\v2_unified\assets"
for p in (OUT, SHARD_DIR, ASSETS):
    os.makedirs(p, exist_ok=True)

# ---- FROZEN CONFIG (do not edit to chase results) -------------------------#
DATASETS = ["iris", "wine", "glass", "ecoli", "aggregation", "compound"]
NCLASSES = {"iris": 3, "wine": 3, "glass": 6, "ecoli": 8,
            "aggregation": 7, "compound": 6}
BUDGETS = (0.05, 0.10, 0.15, 0.20)
ABL = ("P2_Full", "P2_NoExploration", "P2_NoRobust", "P2_NoRedundancy")
BASE = ("Random", "Entropy", "CoreSet", "BADGE", "OriginalP2")
METHODS = ABL + BASE
# FROZEN protocol constants (v2 spec V.4) — DO NOT edit to chase results:
INIT_FRAC = 0.02            # |L0|=max(c, floor(0.02N)) per V.4  [was 0.05, fixed]
M_MC = 50                   # M = 50 MC reference draws per V.4   [was 100, fixed]
KAPPA = 1.0                 # KL design radius (A3); fixed archived headline value;
                            # NOT adapted by misfit (v2 positions kappa as design
                            # radius). Frozen before seeing results; no post-hoc tuning.
RHO = 0.1
DELTA_NI = 0.005
ALPHAS = (0.1, 1.0, 10.0, 100.0)
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
NEIGHBOR = (0.5, 2.0)
ETA_GRID = (0.0, 0.1, 0.2, 0.3)
DRY_SCHEDULE = False


# ---- data -----------------------------------------------------------------#
def load_pkl(name):
    o = pickle.load(open(os.path.join(REF_DATA, name + ".pkl"), "rb"))
    return np.asarray(o["X"], float), np.asarray(o["y"], int)


def acc_hungarian(yt, yp):
    D = max(yt.max(), yp.max()) + 1
    C = np.zeros((D, D), int)
    for t, p in zip(yt, yp):
        C[t, p] += 1
    ri, ci = linear_sum_assignment(-C)
    return C[ri, ci].sum() / len(yt)


def metrics(yt, U):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from sklearn.metrics import adjusted_rand_score as ari
    yp = U.argmax(1)
    return {"NMI": nmi(yt, yp), "ARI": ari(yt, yp), "ACC": acc_hungarian(yt, yp)}


# ---- O1/O2: outlier thresholds locked on the INITIAL LABELED SET L0 -------#
def _lock_outlier_L0(U, X, V, labeled_mask):
    """Theory III.E Eq.(16)-(17), O1/O2: estimate qH, qD, scale ONCE on L0.

    H(u_i) on L0; delta_i = min_k||x_i-v_k|| / median_{j in L0} min_k||x_j-v_k||.
    Lock h_hi=Q75(H;L0), d_hi=Q75(delta;L0), scale=median raw dist. Frozen after.
    """
    from run_ecoli_comparison import _nearest_centroid_dist
    idx = np.where(labeled_mask)[0]
    UL = U[idx]; XL = X[idx]
    H = -(np.clip(UL, 1e-12, 1) * np.log(np.clip(UL, 1e-12, 1))).sum(1)
    d_raw = _nearest_centroid_dist(XL, V)
    scale = float(np.median(d_raw)) + 1e-12
    d = d_raw / scale
    return {"qH": float(np.quantile(H, 0.75)),
            "qD": float(np.quantile(d, 0.75)),
            "scale": scale}


def _apply_outlier_mask(U, X, V, state):
    from run_ecoli_comparison import _nearest_centroid_dist
    H = -(np.clip(U, 1e-12, 1) * np.log(np.clip(U, 1e-12, 1))).sum(1)
    d = _nearest_centroid_dist(X, V) / state["scale"]
    return (H >= state["qH"]) & (d >= state["qD"])


def _lambda2_over_n(U):
    """beta scale = lambda2(L_R)/n, R = cosine on normalized membership rows."""
    from acquisition import cosine_redundancy
    n = U.shape[0]
    R = cosine_redundancy(U)
    d = R.sum(1)
    L = -R.copy(); np.fill_diagonal(L, d)     # combinatorial Laplacian
    ev = np.linalg.eigvalsh(L)
    lam2 = float(np.sort(ev)[1]) if len(ev) > 1 else 1.0
    return lam2 / max(n, 1)


# ---- legacy (archived-bug) baseline, reproduced for comparison -------------#
def original_p2_select(U, X, V, unlab_idx, b, seed=0, rng=None):
    """ORIGINAL P2 baseline reproducing the ARCHIVED deployment behavior.
    Archived bugs reproduced here for baseline ONLY (code_audit_memo):
      - archived dual degenerated to max(0, .) clip (lambda->0 collapse),
      - alpha0 bound at import (fixed here, no scanning),
      - FMIS flag stored but never wired off (order=1, plain entropy proxy),
      - kappa=1 fixed, no misfit adaptation.
    Greedy batch w_i = max(0, H(u_i) - const) + rho*h_i, pairwise cosine penalty.
    """
    U = np.asarray(U, float)[unlab_idx]
    n = U.shape[0]
    H = entropy(np.clip(U, 1e-12, 1.0))
    u_bar = U.mean(0, keepdims=True)
    h = np.abs(U - u_bar).sum(1)
    w_arch = np.maximum(H - 1.0, 0.0)          # archived degenerate dual
    mod = w_arch + 0.1 * h
    R = cosine_redundancy(U)
    sel, avail = [], np.ones(n, bool)
    red = np.zeros(n)
    for _ in range(min(b, n)):
        sc = mod - 0.1 * red
        sc[~avail] = -np.inf
        j = int(np.argmax(sc))
        if sc[j] == -np.inf:
            break
        sel.append(j); avail[j] = False; red += R[j]
    return list(unlab_idx[sel])


# ---- baseline selectors (vectorized) --------------------------------------#
def baseline_select(U, X, mode, b, rng):
    n = U.shape[0]
    if mode == "Random":
        return list(rng.choice(n, size=b, replace=False))
    if mode == "Entropy":
        H = -(np.clip(U, 1e-12, 1) * np.log(np.clip(U, 1e-12, 1))).sum(1)
        return list(np.argsort(-H)[:b])
    # CoreSet / BADGE: greedy farthest from selected
    emb = X if mode == "CoreSet" else U - U.mean(0, keepdims=True)
    chosen = [int(rng.randint(n))]
    d2 = ((emb - emb[chosen[0]]) ** 2).sum(1)
    for _ in range(b - 1):
        j = int(np.argmax(d2)); chosen.append(j)
        d2 = np.minimum(d2, ((emb - emb[j]) ** 2).sum(1))
    return chosen


# ---- one trajectory -------------------------------------------------------#
def run_traj(X, y, c, ds, method, seed, l0, eta, alpha0, lam_mult, budgets=BUDGETS, l0_frac=None, return_U=False, run_x2=True):
    rng = np.random.RandomState((seed * 1000 + l0) & 0x7FFFFFFF)
    n = X.shape[0]
    # noise: flip eta fraction of labels
    yt = y.copy()
    if eta > 0:
        flip = rng.choice(n, size=int(round(eta * n)), replace=False)
        yt[flip] = rng.randint(0, c, size=len(flip))
    frac = l0_frac if l0_frac is not None else INIT_FRAC
    n_init = max(c, int(round(frac * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True

    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                          labels=yt[labeled], seed=seed)
    # P2 (primary, FROZEN III.E): lock thresholds on L0 (hard-labeled set).
    out_state = _lock_outlier_L0(U, X, V, labeled)
    # P2b (pre-registered amendment 2026-10-09): lock on initial unlabeled pool
    # U\L0 once at round 0, since L0 memberships are one-hot (H~0) and the
    # frozen (17) entropy half degenerates. Same joint rule, pool-calibrated.
    out_state_pool = _lock_outlier_L0(U, X, V, ~labeled)
    pool_out_frac = float(_apply_outlier_mask(U[~labeled], X[~labeled], V,
                                              out_state).mean())
    pool_out_frac_pool = float(_apply_outlier_mask(U[~labeled], X[~labeled], V,
                                                    out_state_pool).mean())
    # P2c (pre-registered diagnostic 2026-10-09): eccentric = low H AND high h_i
    # (confident membership-eccentric points, Thm2 mechanism). Lock on U\L0.
    Hu = -(np.clip(U, 1e-12, 1) * np.log(np.clip(U, 1e-12, 1))).sum(1)
    hu = exploration_reward(U, u_all=U)
    unlab0 = ~labeled
    qHlo = float(np.quantile(Hu[unlab0], 0.25))
    qhhi = float(np.quantile(hu[unlab0], 0.75))

    # C) X2 gradient inner-product diagnostic (EXPLORATORY, ecoli/glass only)
    x2_rec = None
    if run_x2 and ds in ("ecoli", "glass") and method in ("P2_Full", "P2_NoExploration"):
        from exploratory import run_x2_diag
        x2 = run_x2_diag(U[unlab0], U, alpha0=alpha0, kappa=1.0, seed=seed)
        x2_rec = {"dataset": ds, "method": method, "seed": seed,
                  "frac_s_neg": x2["frac_s_neg"], "mean_s": x2["mean_s"],
                  "jaccard": x2["jaccard"]}
        for lab in ("boundary", "eccentric", "overlap", "other"):
            m = x2["label"] == lab
            x2_rec[f"n_{lab}"] = int(m.sum())
            if m.sum():
                x2_rec[f"frac_sneg_{lab}"] = float((x2["s"][m] < 0).mean())

    diag_beta = np.nan
    out_full, out_tot = 0, 0
    out_full_pool, out_tot_pool = 0, 0
    ecc_full, ecc_tot = 0, 0
    rows = []
    for bud in budgets:
        target = int(round(bud * n))
        if labeled.sum() >= target:
            m = metrics(yt, U)
            rows.append({"budget": bud, **m})
            continue
        # (2a) FROZEN BATCH SIZE amendment 2026-10-09: one greedy batch per
        # checkpoint of size target-labeled.sum(), then one FCM refit. This
        # makes the Eq.13 pairwise penalty active within the batch (b=1 had
        # made P2_NoRedundancy identical to P2_Full).
        b = target - int(labeled.sum())
        unlab = np.where(~labeled)[0]
        beta = lam_mult * _lambda2_over_n(U)
        diag_beta = beta
        if method in ABL or method == "P2_ScoreOnly":
            sel = select_batch(U[unlab], X[unlab], V, method, b,
                                alpha_0=alpha0, M_MC=M_MC, rho_expl=RHO,
                                lam_red=beta, seed=seed, u_all=U)
            sel = unlab[sel]
        elif method == "OriginalP2":
            sel = original_p2_select(U, X, V, unlab, b, seed=seed,
                                       rng=rng)
        else:
            sel = unlab[baseline_select(U[unlab], X[unlab], method, b, rng)]
        om = _apply_outlier_mask(U[sel], X[sel], V, out_state)
        om_pool = _apply_outlier_mask(U[sel], X[sel], V, out_state_pool)
        # P2c eccentric mask for selected points
        Hu_s = -(np.clip(U[sel], 1e-12, 1) * np.log(np.clip(U[sel], 1e-12, 1))).sum(1)
        hu_s = exploration_reward(U[sel], u_all=U)
        ecc = (Hu_s <= qHlo) & (hu_s >= qhhi)
        out_full += int(om.sum()); out_tot += len(sel)
        out_full_pool += int(om_pool.sum()); out_tot_pool += len(sel)
        ecc_full += int(ecc.sum()); ecc_tot += len(sel)
        labeled[sel] = True
        U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                              labels=yt[labeled], seed=seed)
        m = metrics(yt, U)
        rows.append({"budget": bud, **m})
    diag = {"qH": out_state["qH"], "qD": out_state["qD"],
            "scale": out_state["scale"], "pool_out_frac": pool_out_frac,
            "beta_lambda2_n": diag_beta,
            "qH_pool": out_state_pool["qH"], "qD_pool": out_state_pool["qD"],
            "scale_pool": out_state_pool["scale"],
            "pool_out_frac_pool": pool_out_frac_pool,
            "qHlo": qHlo, "qhhi": qhhi}
    base=(rows, out_full, out_tot, out_full_pool, out_tot_pool, diag, ecc_full, ecc_tot, x2_rec)
    if return_U: base=base+(U,)
    return base


# ---- shard bookkeeping ----------------------------------------------------#
def done_keys():
    p = os.path.join(SHARD_DIR, "done.txt")
    if not os.path.exists(p):
        return set()
    return set(open(p, encoding="utf-8-sig", errors="ignore").read().split())


def record_done(key):
    with open(os.path.join(SHARD_DIR, "done.txt"), "a", encoding="utf-8") as f:
        f.write(key + "\n")


def shard_csv_path(key):
    return os.path.join(SHARD_DIR, key.replace("/", "_") + ".csv")


def _worker(job):
    """Top-level picklable shard worker. Pure compute; returns a result dict.
    All RNG derives from (seed,l0) inside run_traj (already does)."""
    ds, method, seed_set, seed, l0, eta, alpha0, lam_mult = job
    X, y = load_pkl(ds); c = NCLASSES[ds]
    buds = (0.10,) if seed_set in ("P3", "P4", "noise") else BUDGETS
    rows, of, ot, ofp, otp, diag, ecc, ecct, x2_rec = run_traj(
        X, y, c, ds, method, seed, l0, eta, alpha0, lam_mult, budgets=buds)
    import gc; del X, y; gc.collect()
    return {"job": job, "rows": rows, "of": of, "ot": ot, "ofp": ofp, "otp": otp,
            "ecc": ecc, "ecct": ecct, "diag": diag, "x2": x2_rec}


def run_grid(jobs, workers=10):
    """Four-phase, dataset-aware, resumable scheduler (engineering only; the
    frozen shard science in _worker/run_traj is untouched):
      A. grid blocks P3 then P4 (builder order): non-aggregation Pool(2) +
         aggregation Pool(1) concurrently (3 workers total; aggregation
         n=788 shards are ~4 min each and memory-bound).
      B. aggregate P3/P4 in a subprocess and HARD-verify all six datasets are
         non-PENDING before opening the post-hoc 3D gate.
      C. start block3 (50/L0/noise) aggregation Pool(1), write
         results/.main_worker1.flag, and hold the non-aggregation pool until
         the isolated 3D supervisor writes results/.synthetic_done.flag.
      D. run block3 non-aggregation Pool(2) while aggregation is still alive
         (3 total), else Pool(3).
    Legacy callers (smoke/main, no P3/P4/thicken seed sets) keep the per-dataset
    loop with aggregation on one worker. Idempotent via done.txt."""
    import threading
    import multiprocessing as mp
    done = done_keys()
    pending = []
    for job in jobs:
        ds, method, seed_set, seed, l0, eta, alpha0, lam_mult = job
        key = f"{ds}/{method}/{seed_set}/{seed}/{l0}/{eta}/{alpha0}/{lam_mult}"
        if key in done:
            continue
        pending.append((key, job))
    total = len(pending)
    t0 = time.time()
    io_lock = threading.Lock()
    counters = {"n": 0}
    GATE = os.path.join(OUT, ".main_worker1.flag")
    DONE3D = os.path.join(OUT, ".synthetic_done.flag")

    def handle(res):
        ds, method, seed_set, seed, l0, eta, alpha0, lam_mult = res["job"]
        key = f"{ds}/{method}/{seed_set}/{seed}/{l0}/{eta}/{alpha0}/{lam_mult}"
        df = pd.DataFrame(res["rows"])
        df["dataset"] = ds; df["method"] = method; df["seed_set"] = seed_set
        df["seed"] = seed; df["l0"] = l0; df["eta"] = eta
        df["alpha0"] = alpha0; df["lam_mult"] = lam_mult
        df["outlier_rate"] = (res["of"] / res["ot"]) if res["ot"] else np.nan
        df["outlier_rate_pool"] = (res["ofp"] / res["otp"]) if res["otp"] else np.nan
        df["eccentric_rate"] = (res["ecc"] / res["ecct"]) if res["ecct"] else np.nan
        p = shard_csv_path(key)
        tmp = p + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)
        drow = {"dataset": ds, "method": method, "seed_set": seed_set,
                "seed": seed, "l0": l0, "eta": eta, **res["diag"],
                "selected_outlier_rate": (res["of"] / res["ot"]) if res["ot"] else np.nan,
                "selected_outlier_rate_pool": (res["ofp"] / res["otp"]) if res["otp"] else np.nan,
                "selected_eccentric_rate": (res["ecc"] / res["ecct"]) if res["ecct"] else np.nan}
        with io_lock:
            diag_dir = os.path.join(OUT, "diagnostics"); os.makedirs(diag_dir, exist_ok=True)
            dpath = os.path.join(diag_dir, "p2_diagnostics.csv")
            pd.DataFrame([drow]).to_csv(dpath, mode="a",
                                        header=not os.path.exists(dpath), index=False)
            if res["x2"] is not None:
                x2path = os.path.join(OUT, "exploratory", "x2_diag.csv")
                pd.DataFrame([res["x2"]]).to_csv(x2path, mode="a",
                                                 header=not os.path.exists(x2path),
                                                 index=False)
            record_done(key)
            counters["n"] += 1
            n = counters["n"]
            if n % 20 == 0:
                el = time.time() - t0
                rate = n / max(el, 1)
                eta = (total - n) / max(rate, 1e-9)
                print(f"  [{el:7.0f}s] {n}/{total}  {rate:.2f} sh/s  ETA {eta/60:.1f}min",
                      flush=True)

    def run_group(group, w, label):
        if not group:
            return 0
        print(f"[sched] {label}: n={len(group)} workers={w}", flush=True)
        n0 = counters["n"]
        if w <= 1:
            for _, j in group:
                handle(_worker(j))
        else:
            pool = mp.Pool(w)
            try:
                for res in pool.imap_unordered(_worker, [j for _, j in group]):
                    handle(res)
            finally:
                pool.close(); pool.join()
        return counters["n"] - n0

    def run_concurrent(nonagg, agg, cap_total, label):
        """aggregation always Pool(1); non-agg gets cap_total-1 while agg runs.
        Pools are created HERE in the main thread (concurrent Pool creation from
        two threads races under Windows spawn), then each consumer thread only
        drains its own imap_unordered iterator."""
        import multiprocessing as mp
        pool_ag = mp.Pool(1) if agg else None
        wn = max(1, cap_total - 1) if agg else cap_total
        pool_na = mp.Pool(wn) if nonagg else None
        threads = []

        def consume(pool, group, w, lab):
            if not group:
                if pool is not None:
                    pool.close(); pool.join()
                return
            print(f"[sched] {lab}: n={len(group)} workers={w}", flush=True)
            n0 = counters["n"]
            try:
                for res in pool.imap_unordered(_worker, [j for _, j in group]):
                    handle(res)
            finally:
                pool.close(); pool.join()
            print(f"[sched] {lab}: done ({counters['n'] - n0} shards)", flush=True)

        if pool_ag is not None:
            th = threading.Thread(target=consume,
                                  args=(pool_ag, agg, 1, label + "/aggregation[w1]"))
            th.daemon = True; th.start(); threads.append(th)
        if pool_na is not None:
            th = threading.Thread(target=consume,
                                  args=(pool_na, nonagg, wn, label + "/non-aggregation"))
            th.daemon = True; th.start(); threads.append(th)
        for th in threads:
            th.join()

    GRID_SETS = {"P3", "P4"}
    THICK_SETS = {"50", "30L0", "noise"}

    def split(sets_):
        na = [(k, j) for k, j in pending if str(j[2]) in sets_ and j[0] != "aggregation"]
        ag = [(k, j) for k, j in pending if str(j[2]) in sets_ and j[0] == "aggregation"]
        return na, ag

    is_thicken = any(str(j[2]) in (GRID_SETS | THICK_SETS) for _, j in pending)
    if not is_thicken:
        # legacy per-dataset loop (smoke / main)
        by_ds = {}
        for k, j in pending:
            by_ds.setdefault(j[0], []).append((k, j))
        order = [d for d in by_ds if d != "aggregation"] + (
            ["aggregation"] if "aggregation" in by_ds else [])
        for ds in order:
            run_group(by_ds[ds], 1 if ds == "aggregation" else min(workers, 3),
                      f"legacy:{ds}")
        print(f"[run] completed {counters['n']} new shards in {time.time()-t0:.0f}s",
              flush=True)
        return

    g_na, g_ag = split(GRID_SETS)
    b_na, b_ag = split(THICK_SETS)
    if DRY_SCHEDULE:
        print(f"[dry] pending total={total}")
        print(f"[dry] phaseA grid : non-agg={len(g_na)} Pool(2) | aggregation={len(g_ag)} Pool(1)")
        print("[dry] phaseB       : aggregate --only P3,P4 + six-dataset non-PENDING hard verify")
        print(f"[dry] phaseC block3: aggregation={len(b_ag)} Pool(1) + 3D gate (non-agg held)")
        print(f"[dry] phaseD block3: non-agg={len(b_na)} Pool(2 while agg alive else 3)")
        return

    # ---- Phase A: P3 then P4 grids (builder order preserved in lists) ------
    if g_na or g_ag:
        run_concurrent(g_na, g_ag, 3, "phaseA_grid")

    # ---- Phase B: aggregate P3/P4 and hard-verify before the gate ----------
    if g_na or g_ag or os.path.exists(GATE):
        import subprocess
        logp = os.path.join(OUT, "phase_p3p4_aggregate.log")
        with open(logp, "a", encoding="utf-8") as lf:
            lf.write(f"\n[phaseB] aggregating P3/P4 at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            rc = subprocess.run([sys.executable, os.path.join(HERE, "run_experiments.py"),
                                 "--phase", "aggregate", "--only", "P3,P4"],
                                cwd=HERE, env=os.environ.copy(),
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print("[phaseB] aggregate FAILED (see phase_p3p4_aggregate.log); gate NOT set",
                  flush=True)
            sys.exit(3)
        rows = json.load(open(os.path.join(OUT, "prereg_verdicts.json"),
                              encoding="utf-8-sig"))
        bad = []
        for pred in ("P3", "P4"):
            for ds in DATASETS:
                rr = [r for r in rows if r["prediction_id"] == pred and r["dataset"] == ds
                      and str(r.get("seed_set")) == pred]
                if not rr or any(r.get("verdict") == "PENDING" for r in rr):
                    bad.append(f"{pred}/{ds}")
        summ = [r for r in rows if r.get("prediction_id") in ("P3", "P4")
                and str(r.get("seed_set")) in ("P3", "P4")]
        json.dump(summ, open(os.path.join(OUT, "p3p4_verdict_summary.json"), "w"), indent=2)
        if bad:
            print(f"[phaseB] PENDING after aggregation: {bad}; gate NOT set", flush=True)
            sys.exit(3)
        print("[phaseB] all six datasets P3/P4 non-PENDING; summary written", flush=True)

    # ---- Phase C/D: block3 with the post-hoc 3D gate -----------------------
    if not (b_na or b_ag):
        print(f"[run] completed {counters['n']} new shards in {time.time()-t0:.0f}s",
              flush=True)
        return
    if os.path.exists(DONE3D):
        if os.path.exists(GATE):
            os.remove(GATE)
        run_concurrent(b_na, b_ag, 3, "block3")
    else:
        th_ag = None
        if b_ag:
            th_ag = threading.Thread(target=run_group,
                                     args=(b_ag, 1, "phaseC_block3/aggregation[w1]"))
            th_ag.daemon = True; th_ag.start()
        open(GATE, "w").close()
        print("[gate] .main_worker1.flag set; aggregation block3 running on Pool(1); "
              "waiting for isolated 3D supervisor (.synthetic_done.flag)", flush=True)
        while not os.path.exists(DONE3D):
            time.sleep(60)
        if os.path.exists(GATE):
            os.remove(GATE)
        wn = 2 if (th_ag is not None and th_ag.is_alive()) else 3
        if b_na:
            run_group(b_na, wn, f"phaseD_block3/non-aggregation[w{wn}]")
        if th_ag is not None:
            th_ag.join()
    print(f"[run] completed {counters['n']} new shards in {time.time()-t0:.0f}s",
          flush=True)


# ---- stats & verdicts -----------------------------------------------------#
def paired_stats(a, b, n_boot=10000, seed=0, sides="two"):
    """a,b paired arrays. sides='two' -> 95% two-sided CI; 'one' -> 95% one-sided
    lower bound (t.ppf(.95,n-1)). Returns mean, lo, hi, boot CI, dz, n."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975 if sides == "two" else 0.95, n - 1)
    lo = mean - tcrit * se
    hi = mean + tcrit * se if sides == "two" else float("nan")
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    b_lo, b_hi = np.percentile(boots, [2.5, 97.5])
    dz = mean / d.std(ddof=1)
    return {"mean": mean, "lo": lo, "hi": hi,
            "boot_lo": b_lo, "boot_hi": b_hi, "dz": dz, "n": n}


def load_long():
    frames = []
    for f in os.listdir(SHARD_DIR):
        # (ix) only result shards, NOT diagnostics
        if f.endswith(".csv") and f != "p2_diagnostics.csv" and "_" in f:
            frames.append(pd.read_csv(os.path.join(SHARD_DIR, f)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def upsert_verdicts(rows):
    # hard validation: non-PENDING rows need non-null estimate & ci_lo.
    # ci_hi is legitimately None for one-sided rows (NI/SUP/ACC).
    one_sided = {"NI", "SUP", "ACC"}
    for r in rows:
        if r["verdict"] != "PENDING":
            for k in ("estimate", "ci_lo"):
                if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k))):
                    raise ValueError(f"REFUSING null {k}: {r}")
            if r.get("test") not in one_sided:
                if r.get("ci_hi") is None or (isinstance(r.get("ci_hi"), float)
                                              and np.isnan(r.get("ci_hi"))):
                    raise ValueError(f"REFUSING null ci_hi: {r}")
    path = os.path.join(OUT, "prereg_verdicts.json")
    existing = []
    if os.path.exists(path):
        existing = json.load(open(path, encoding="utf-8-sig"))
    idx = {(r["prediction_id"], r["dataset"], r.get("seed_set", "30"),
            r.get("test", "")): i for i, r in enumerate(existing)}
    for r in rows:
        k = (r["prediction_id"], r["dataset"], r.get("seed_set", "30"),
             r.get("test", ""))
        if k in idx:
            existing[idx[k]] = r
        else:
            existing.append(r)
    json.dump(existing, open(path, "w"), indent=2)


# ---- grid-only aggregators (P3 lambda2/n, P4 alpha0 TOST); no P1/P2 rows --#
def _grid_p3_verdicts(df_all):
    rows = []
    g = df_all[df_all.seed_set.astype(str) == "P3"] if not df_all.empty else df_all
    for ds in DATASETS:
        sub = g[g.dataset == ds]
        means = sub.groupby("lam_mult").ACC.agg(["mean", "count"])
        if len(means) < len(LAM_GRID) or means["count"].min() < 30:
            rows.append({"prediction_id": "P3", "dataset": ds, "test": "LAMBDA2N",
                         "estimate": None, "ci_lo": None, "ci_hi": None,
                         "margin": str(NEIGHBOR), "verdict": "PENDING",
                         "seed_set": "P3", "theory_assumption_to_revise": "grid incomplete"})
            continue
        best_mult = float(means["mean"].idxmax())
        in_nb = NEIGHBOR[0] <= best_mult <= NEIGHBOR[1]
        in_idx = [i for i in means.index if NEIGHBOR[0] <= i <= NEIGHBOR[1]]
        out_idx = [i for i in means.index if not (NEIGHBOR[0] <= i <= NEIGHBOR[1])]
        best_in_mult = means.loc[in_idx, "mean"].idxmax()
        best_in = sub[sub.lam_mult == best_in_mult].sort_values("seed").ACC.values
        worst_out = sub[sub.lam_mult == min(out_idx, key=lambda i: means.loc[i, "mean"])
                        ].sort_values("seed").ACC.values
        s1 = paired_stats(best_in, worst_out, sides="one")
        s2 = paired_stats(best_in, worst_out, sides="two")
        v = "CONFIRMED" if (in_nb and s1["lo"] > 0.005) else "REJECTED"
        rows.append({"prediction_id": "P3", "dataset": ds, "test": "LAMBDA2N",
                     "estimate": best_mult, "ci_lo": s2["lo"], "ci_hi": s2["hi"],
                     "onesided_lo": s1["lo"], "drop_mean": s2["mean"],
                     "boot_lo": s2["boot_lo"], "boot_hi": s2["boot_hi"],
                     "margin": str(NEIGHBOR), "verdict": v, "seed_set": "P3",
                     "dz": s2["dz"],
                     "theory_assumption_to_revise": "" if v == "CONFIRMED" else
                     "Revise Thm3 (A5): lambda2/n not the right scale"})
    return rows


def _grid_p4_verdicts(df_all):
    rows = []
    g = df_all[df_all.seed_set.astype(str) == "P4"] if not df_all.empty else df_all
    for ds in DATASETS:
        sub = g[g.dataset == ds]
        counts = {a: len(sub[sub.alpha0 == a]) for a in ALPHAS}
        if not all(counts[a] >= 30 for a in ALPHAS):
            rows.append({"prediction_id": "P4", "dataset": ds, "test": "TOST",
                         "estimate": None, "ci_lo": None, "ci_hi": None,
                         "margin": 0.005, "verdict": "PENDING", "seed_set": "P4",
                         "counts": counts, "theory_assumption_to_revise": "incomplete"})
            continue
        m_by_a = {a: sub[sub.alpha0 == a].sort_values("seed").ACC.values for a in ALPHAS}
        pair_res, ok_all, maxdiff = [], True, 0.0
        worst_d = None
        keys = sorted(m_by_a)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = m_by_a[keys[i]] - m_by_a[keys[j]]
                n = len(d); se = d.std(ddof=1) / np.sqrt(n)
                p1 = stats.t.sf((d.mean() + 0.005) / se, n - 1)
                p2 = stats.t.cdf((d.mean() - 0.005) / se, n - 1)
                ok = max(p1, p2) <= 0.05
                ok_all = ok_all and ok
                if abs(d.mean()) > maxdiff:
                    maxdiff = abs(d.mean()); worst_d = d
                pair_res.append({"pair": [keys[i], keys[j]], "mean_diff": float(d.mean()),
                                 "tost_p": float(max(p1, p2)), "equiv": bool(ok)})
        v = "CONFIRMED" if ok_all else "REJECTED"
        wd = np.asarray(worst_d, float)
        nw = len(wd); sew = wd.std(ddof=1) / np.sqrt(nw)
        tcw = stats.t.ppf(0.975, nw - 1)
        rows.append({"prediction_id": "P4", "dataset": ds, "test": "TOST",
                     "estimate": maxdiff,
                     "ci_lo": float(wd.mean() - tcw * sew),
                     "ci_hi": float(wd.mean() + tcw * sew),
                     "worst_pair_mean": float(wd.mean()),
                     "margin": 0.005, "verdict": v, "seed_set": "P4",
                     "pairwise": pair_res,
                     "theory_assumption_to_revise": "" if ok_all else
                     "Revise Prop.1: C/sqrt(alpha0) bound too loose"})
    return rows


def aggregate(seed_set):
    df_all = load_long()
    # Grid tags aggregate ONLY their own prediction; they must not emit P1/P2 rows.
    if seed_set == "P3":
        rows = _grid_p3_verdicts(df_all); upsert_verdicts(rows)
        print(f"[agg seed_set=P3] wrote {len(rows)} verdict rows")
        for r in rows:
            print(f"  P3 {r['dataset']:12s} {r['verdict']:9s} best={r['estimate']}")
        return
    if seed_set == "P4":
        rows = _grid_p4_verdicts(df_all); upsert_verdicts(rows)
        print(f"[agg seed_set=P4] wrote {len(rows)} verdict rows")
        for r in rows:
            print(f"  P4 {r['dataset']:12s} {r['verdict']:9s} maxdiff={r['estimate']}")
        return
    verdicts = []
    # (B) strict isolation: main seed_set only uses l0==0 & eta==0 shards.
    df = df_all[(df_all.seed_set.astype(str) == str(seed_set)) &
                (df_all.l0 == 0) & (df_all.eta == 0.0)] if not df_all.empty else df_all
    need = {"30": 30, "30L0": 30, "50": 50}.get(seed_set, 1)

    def _pair(ds, budget=0.10):
        f = df[(df.dataset == ds) & (df.method == "P2_Full") &
               (df.budget == budget)].sort_values("seed")
        ne = df[(df.dataset == ds) & (df.method == "P2_NoExploration") &
                (df.budget == budget)].sort_values("seed")
        m = f.merge(ne, on="seed", suffixes=("_f", "_ne"))
        return m

    # ---------- P1: noninferiority (one-sided) + ecoli/glass superiority ----
    for ds in DATASETS:
        m = _pair(ds)
        if len(m) < need:
            verdicts.append({"prediction_id": "P1", "dataset": ds, "test": "NI",
                             "estimate": None, "ci_lo": None, "ci_hi": None,
                             "margin": -DELTA_NI, "verdict": "PENDING",
                             "seed_set": seed_set, "theory_assumption_to_revise": "incomplete"})
            continue
        s = paired_stats(m.ACC_ne.values, m.ACC_f.values, sides="one")
        v = "CONFIRMED" if s["lo"] > -DELTA_NI else "REJECTED"
        verdicts.append({"prediction_id": "P1", "dataset": ds, "test": "NI",
                         "estimate": s["mean"], "ci_lo": s["lo"], "ci_hi": None,
                         "margin": -DELTA_NI, "verdict": v, "seed_set": seed_set,
                         "dz": s["dz"],
                         "theory_assumption_to_revise": "" if v == "CONFIRMED" else
                         "Revise Thm1 (A4): geometry component != coverage force"})
        if ds in ("ecoli", "glass"):
            vs = "CONFIRMED" if s["lo"] > 0.0 else "REJECTED"
            verdicts.append({"prediction_id": "P1", "dataset": ds, "test": "SUP",
                             "estimate": s["mean"], "ci_lo": s["lo"], "ci_hi": None,
                             "margin": 0.0, "verdict": vs, "seed_set": seed_set,
                             "dz": s["dz"],
                             "theory_assumption_to_revise": "" if vs == "CONFIRMED" else
                             "NoExploration does not beat Full on " + ds})

    # ---------- P2 (L0) & P2b (pool): outlier frequency two-sided -----------
    for pred, col in (("P2", "outlier_rate"), ("P2b", "outlier_rate_pool")):
        for ds in DATASETS:
            sub = df[df.dataset == ds]
            f = sub[sub.method == "P2_Full"].groupby("seed")[col].mean()
            ne = sub[sub.method == "P2_NoExploration"].groupby("seed")[col].mean()
            mm = pd.concat([f, ne], axis=1, keys=["f", "ne"]).dropna()
            if len(mm) < need:
                verdicts.append({"prediction_id": pred, "dataset": ds, "test": "FREQ",
                                 "estimate": None, "ci_lo": None, "ci_hi": None,
                                 "margin": 0.0, "verdict": "PENDING",
                                 "seed_set": seed_set, "theory_assumption_to_revise": "incomplete"})
                continue
            s = paired_stats(mm["f"].values, mm["ne"].values)
            v = "CONFIRMED" if s["lo"] > 0 else "REJECTED"
            verdicts.append({"prediction_id": pred, "dataset": ds, "test": "FREQ",
                             "estimate": s["mean"], "ci_lo": s["lo"], "ci_hi": s["hi"],
                             "margin": 0.0, "verdict": v, "seed_set": seed_set, "dz": s["dz"],
                             "theory_assumption_to_revise": "" if v == "CONFIRMED" else
                             "Revise Thm2 (Eq.10/17): misalignment not observed"})

    # ecoli P2/P2b ACC directional rows
    m = _pair("ecoli")
    if len(m) >= need:
        s = paired_stats(m.ACC_ne.values, m.ACC_f.values, sides="one")
        v = "CONFIRMED" if s["lo"] > 0 else "REJECTED"
        for pred in ("P2", "P2b"):
            verdicts.append({"prediction_id": pred, "dataset": "ecoli", "test": "ACC",
                             "estimate": s["mean"], "ci_lo": s["lo"], "ci_hi": None,
                             "margin": 0.0, "verdict": v, "seed_set": seed_set, "dz": s["dz"],
                             "theory_assumption_to_revise": "" if v == "CONFIRMED" else
                             "NoExploration does not improve ecoli ACC"})

    # P2c / eccentric rate is EXPLORATORY (re-labelled 2026-10-09): it is kept
    # as a shard column + exploratory CSV, but NEVER enters prereg_verdicts.json.

    # ---------- P3: lambda2/n neighborhood, per dataset, paired drop CI ------
    g3 = df_all[df_all.seed_set == "P3"] if not df_all.empty else df_all
    for ds in DATASETS:
        sub = g3[g3.dataset == ds]
        means = sub.groupby("lam_mult").ACC.agg(["mean", "count"])
        if len(means) < len(LAM_GRID) or means["count"].min() < need:
            verdicts.append({"prediction_id": "P3", "dataset": ds, "test": "LAMBDA2N",
                             "estimate": None, "ci_lo": None, "ci_hi": None,
                             "margin": str(NEIGHBOR), "verdict": "PENDING",
                             "seed_set": seed_set, "theory_assumption_to_revise": "grid incomplete"})
            continue
        best_mult = float(means["mean"].idxmax())
        in_nb = (NEIGHBOR[0] <= best_mult <= NEIGHBOR[1])
        in_idx = [i for i in means.index if NEIGHBOR[0] <= i <= NEIGHBOR[1]]
        out_idx = [i for i in means.index if not (NEIGHBOR[0] <= i <= NEIGHBOR[1])]
        best_in_mult = means.loc[in_idx, "mean"].idxmax()
        # paired per-seed drop = ACC(best_in) - ACC(worst_out), one-sided CI>0.005
        best_in = sub[sub.lam_mult == best_in_mult].sort_values("seed").ACC.values
        worst_out = sub[sub.lam_mult == min(out_idx, key=lambda i: means.loc[i, "mean"])
                        ].sort_values("seed").ACC.values
        s = paired_stats(best_in, worst_out, sides="one")
        v = "CONFIRMED" if (in_nb and s["lo"] > 0.005) else "REJECTED"
        verdicts.append({"prediction_id": "P3", "dataset": ds, "test": "LAMBDA2N",
                         "estimate": best_mult, "ci_lo": s["lo"], "ci_hi": None,
                         "margin": str(NEIGHBOR), "verdict": v, "seed_set": seed_set,
                         "dz": s["dz"],
                         "theory_assumption_to_revise": "" if v == "CONFIRMED" else
                         "Revise Thm3 (A5): lambda2/n not the right scale"})

    # ---------- P4: alpha0 TOST, per dataset, all 4 alphas x seeds ----------
    g4 = df_all[df_all.seed_set == "P4"] if not df_all.empty else df_all
    for ds in DATASETS:
        sub = g4[g4.dataset == ds]
        counts = {a: len(sub[sub.alpha0 == a]) for a in ALPHAS}
        if not all(counts[a] >= need for a in ALPHAS):
            verdicts.append({"prediction_id": "P4", "dataset": ds, "test": "TOST",
                             "estimate": None, "ci_lo": None, "ci_hi": None,
                             "margin": 0.005, "verdict": "PENDING", "seed_set": seed_set,
                             "counts": counts, "theory_assumption_to_revise": "incomplete"})
            continue
        means = {a: sub[sub.alpha0 == a].sort_values("seed").ACC.values for a in ALPHAS}
        pair_res, ok_all, maxdiff = [], True, 0.0
        keys = sorted(means)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert len(means[keys[i]]) == len(means[keys[j]])
                d = means[keys[i]] - means[keys[j]]
                n = len(d); se = d.std(ddof=1) / np.sqrt(n)
                p1 = stats.t.sf((d.mean() + 0.005) / se, n - 1)
                p2 = stats.t.cdf((d.mean() - 0.005) / se, n - 1)
                ok = max(p1, p2) <= 0.05
                ok_all = ok_all and ok
                maxdiff = max(maxdiff, abs(d.mean()))
                pair_res.append({"pair": [keys[i], keys[j]], "mean_diff": float(d.mean()),
                                 "tost_p": float(max(p1, p2)), "equiv": bool(ok)})
        v = "CONFIRMED" if ok_all else "REJECTED"
        verdicts.append({"prediction_id": "P4", "dataset": ds, "test": "TOST",
                         "estimate": maxdiff, "ci_lo": None, "ci_hi": None,
                         "margin": 0.005, "verdict": v, "seed_set": seed_set,
                         "pairwise": pair_res,
                         "theory_assumption_to_revise": "" if ok_all else
                         "Revise Prop.1: C/sqrt(alpha0) bound too loose"})

    upsert_verdicts(verdicts)

    # ---- S.V.8 Holm step-down correction + paired_contrasts.csv ------------
    # Family definition (frozen): within each dataset, each non-Full method
    # paired against P2_Full at 10% budget; family grouped by metric
    # (ACC = confirmatory primary; NMI, ARI = secondary). Holm step-down
    # applied within (dataset, metric) family.
    contrasts = []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        for metric in ("ACC", "NMI", "ARI"):
            fam = []
            for meth in METHODS:
                if meth == "P2_Full":
                    continue
                f = sub[(sub.method == "P2_Full") & (sub.budget == 0.10)].sort_values("seed")
                o = sub[(sub.method == meth) & (sub.budget == 0.10)].sort_values("seed")
                mm = f.merge(o, on="seed", suffixes=("_f", "_o"))
                if len(mm) < need:
                    continue
                a = mm[f"{metric}_f"].values; b = mm[f"{metric}_o"].values
                s = paired_stats(a, b)
                n = len(a)
                p = 2 * stats.t.sf(abs((a - b).mean() /
                                        ((a - b).std(ddof=1) / np.sqrt(n))), n - 1)
                fam.append({"dataset": ds, "metric": metric, "budget": 0.10,
                            "contrast": f"{meth}-vs-Full", "mean_diff": s["mean"],
                            "ci95_lo_t": s["lo"], "ci95_hi_t": s["hi"],
                            "ci95_lo_boot": s.get("boot_lo", s["lo"]),
                            "ci95_hi_boot": s.get("boot_hi", s["hi"]),
                            "p_raw": p, "dz": s["dz"], "n": n})
            # Holm step-down within family
            fam_sorted = sorted(fam, key=lambda r: r["p_raw"])
            m = len(fam_sorted)
            for i, r in enumerate(fam_sorted):
                r["p_holm"] = max((m - i) * r["p_raw"] for _ in [0])
            for i in range(1, len(fam_sorted)):
                fam_sorted[i]["p_holm"] = max(fam_sorted[i]["p_holm"],
                                              fam_sorted[i - 1]["p_holm"])
            for r in fam_sorted:
                r["p_holm"] = min(r["p_holm"], 1.0)
            contrasts.extend(fam_sorted)
    if contrasts:
        cdf = pd.DataFrame(contrasts)
        cdf.to_csv(os.path.join(OUT, "paired_contrasts.csv"), index=False)
        cdf.to_csv(os.path.join(OUT, f"paired_contrasts_{seed_set}.csv"), index=False)
        # main ablation summary
        summ = (df[df.budget == 0.10].groupby(["dataset", "method"])[["NMI", "ARI", "ACC"]]
                .agg(["mean", "std"]).round(4))
        summ.to_csv(os.path.join(OUT, "main_ablation_summary.csv"))
        print(f"[agg] wrote paired_contrasts.csv ({len(cdf)} rows)")
    print(f"[agg seed_set={seed_set}] wrote {len(verdicts)} verdict rows")
    for v in verdicts:
        print(f"  {v['prediction_id']} {v['dataset']:10s} {v.get('test',''):6s} {v['verdict']}")


# ---- job builders ---------------------------------------------------------#
def jobs_main():
    return list(itertools.product(DATASETS, METHODS, ["30"], range(30),
                                  [0], [0.0], [50.0], [1.0]))


def jobs_thicken():
    jobs = []
    # P3 lambda grid FIRST (user priority): all 6 datasets, P2_NoExploration
    jobs += list(itertools.product(DATASETS, ["P2_NoExploration"], ["P3"],
                                   range(30), [0], [0.0], [50.0], LAM_GRID))
    # P4 alpha grid FIRST
    jobs += list(itertools.product(DATASETS, ["P2_NoExploration"], ["P4"],
                                   range(30), [0], [0.0], ALPHAS, [1.0]))
    # 50-seed confirmation (l0=0, eta=0, full budget)
    jobs += list(itertools.product(DATASETS, METHODS, ["50"], range(50),
                                   [0], [0.0], [50.0], [1.0]))
    # L0 robustness: l0 in [1,2] (l0=0 already covered by main/50), 30 seeds
    jobs += list(itertools.product(DATASETS, METHODS, ["30L0"], range(30),
                                   [1, 2], [0.0], [50.0], [1.0]))
    # noise: eta in {0.1,0.2,0.3} (eta=0 reused from main), all 9 methods, 10%
    jobs += list(itertools.product(DATASETS, METHODS, ["noise"], range(30),
                                   [0], (0.1, 0.2, 0.3), [50.0], [1.0]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["smoke", "main", "thicken", "aggregate"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--dry-schedule", action="store_true",
                    help="print four-phase job/pool plan and exit (no shards run)")
    args = ap.parse_args()
    if args.dry_schedule:
        global DRY_SCHEDULE
        DRY_SCHEDULE = True
        run_grid(jobs_thicken(), workers=args.workers)
        return
    if args.phase == "smoke":
        jobs = list(itertools.product(["ecoli"], ["P2_Full", "P2_NoExploration"],
                                      ["smoke"], range(3), [0], [0.0],
                                      [50.0], [1.0]))
        run_grid(jobs, workers=args.workers)
        aggregate("smoke")
    elif args.phase == "main":
        run_grid(jobs_main(), workers=args.workers)
        aggregate("30")
    elif args.phase == "thicken":
        run_grid(jobs_thicken(), workers=args.workers)
        aggregate("50")
    elif args.phase == "aggregate":
        only = getattr(args, "only", None)
        if only:
            for tag in only.split(","):
                aggregate(tag.strip())
        else:
            aggregate("30"); aggregate("50")


if __name__ == "__main__":
    main()
