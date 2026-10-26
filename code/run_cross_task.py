# -*- coding: utf-8 -*-
"""run_cross_task.py — E3 cross-task extension experiment.

6 tasks x 20 paired seeds x 7 budgets x 4 methods = 3360 checkpoint rows.
Pipeline paradigm follows add5b_binary.py / add4_large.py (LR backend, NOT FCM):
  classifier LogisticRegression(lbfgs, max_iter=1000); U = predict_proba;
  |L0| = max(c, round(0.02*n)) = 40 for n=2000; pool-transductive ACC.
Methods:
  P2_Full          = w_rob + rho*h_i - beta*red_sum   (rho=0.1)
  P2_NoExploration = w_rob            - beta*red_sum   (rho=0)
  Random           = uniform random
  Entropy          = top-b by predictive entropy
Frozen constants: M_MC=50, KAPPA=1.0, RHO=0.1, alpha0=50.0, eps_q=0.1,
lam_affine=0.5, beta=lambda2(L_R)/n recomputed on current whole-pool U.

Data caches land in data/cross_task/<task>.npz; CIFAR-10 cache as
data/cross_task/cifar10_full.npz. Results append to results/cross_task/summary.csv
with idempotent skip via results/cross_task/done.txt.

Usage:
  python run_cross_task.py            # full run, 4 workers, resumable
  python run_cross_task.py --smoke    # tiny sanity run (1 seed, 2 budgets, 1 task)
"""
import os, sys, time, pickle, json, argparse, traceback
# Pin BLAS threads BEFORE numpy import: 4 Pool workers must not each grab all
# physical cores (causes catastrophic OpenBLAS oversubscription on CIFAR d=3072).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "cross_task")
RES_DIR = os.path.join(ROOT, "results", "cross_task")
REF_DATA = os.path.join(HERE, "reference_repo", "data")
for _d in (DATA_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)

LOG_PATH = os.environ.get(
    "CROSS_TASK_LOG",
    os.path.join(RES_DIR, "cross_task_run.log"))
DONE_PATH = os.path.join(RES_DIR, "done.txt")
SUMMARY_PATH = os.path.join(RES_DIR, "summary.csv")

# ---- frozen protocol ---- #
METHODS = ["P2_Full", "P2_NoExploration", "Random", "Entropy"]
BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SEEDS = list(range(20))
N_PER_TASK = 2000
ALPHA0 = 50.0
M_MC = 50
KAPPA = 1.0
EPS_Q = 0.1
LAM_AFF = 0.5
RHO = 0.1
PREP_SEED = 20261022          # fixed seed for task construction (5-class picks)
L0_FORMULA = "RandomState((seed*1000+n_init) & 0x7FFFFFFF), choice(n, n_init, replace=False)"

TASK_ORDER = ["fashion_2", "fashion_5", "fashion_10",
              "usps_10", "cifar_5", "cifar_10"]


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _load_pkl(name):
    o = pickle.load(open(os.path.join(REF_DATA, name + ".pkl"), "rb"))
    return np.asarray(o["X"], float), np.asarray(o["y"], int)


def stratified_subsample(X, y, target_n, seed=0):
    """Balanced stratified subsample: target_n//c per class (add4_large style)."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per = target_n // len(classes)
    idx = []
    for k in classes:
        ki = np.where(y == k)[0]
        take = min(per, len(ki))
        idx.append(rng.choice(ki, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


def relabel01(y, classes):
    mp = {c: i for i, c in enumerate(classes)}
    return np.array([mp[v] for v in y], dtype=int)


def get_cifar10():
    """Return (X uint8 60000x3072, y int64 60000). Cached at DATA_DIR/cifar10_full.npz."""
    npz = os.path.join(DATA_DIR, "cifar10_full.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["X"], d["y"]
    # preferred: sklearn openml (may fail on memory/network)
    try:
        log("CIFAR-10: trying fetch_openml(CIFAR_10) ...")
        from sklearn.datasets import fetch_openml
        dd = fetch_openml("CIFAR_10", version=1, data_home=DATA_DIR, as_frame=False)
        X = np.asarray(dd.data, dtype=np.uint8)
        y = np.asarray(dd.target, dtype=int)
        np.savez(npz, X=X, y=y)
        log("CIFAR-10: openml OK, shape=%s" % (X.shape,))
        return X, y
    except Exception as e:
        log("CIFAR-10: openml failed (%r); falling back to toronto tarball" % e)
    # fallback: urllib tarball, 2 retries
    import tarfile, urllib.request
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tgz = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    last = None
    for attempt in range(2):
        try:
            if not os.path.exists(tgz) or os.path.getsize(tgz) < 10_000_000:
                log("CIFAR-10: downloading tarball attempt %d ..." % (attempt + 1))
                urllib.request.urlretrieve(url, tgz)
            Xs, ys = [], []
            with tarfile.open(tgz, "r:gz") as tf:
                names = [m for m in tf.getmembers()
                         if os.path.basename(m.name) in
                         ("data_batch_1", "data_batch_2", "data_batch_3",
                          "data_batch_4", "data_batch_5", "test_batch")]
                for m in names:
                    f = tf.extractfile(m)
                    d = pickle.load(f, encoding="bytes")
                    Xs.append(np.asarray(d[b"data"], dtype=np.uint8))
                    ys.extend(list(d[b"labels"]))
            X = np.concatenate(Xs, axis=0)
            y = np.asarray(ys, dtype=int)
            np.savez(npz, X=X, y=y)
            log("CIFAR-10: tarball OK, shape=%s" % (X.shape,))
            return X, y
        except Exception as e:
            last = e
            log("CIFAR-10: tarball attempt %d failed: %r" % (attempt + 1, e))
            time.sleep(5)
    raise RuntimeError("CIFAR-10 download failed after retries: %r" % last)


def prepare_task(task):
    """Build & cache DATA_DIR/<task>.npz with X(float32), y(int), c. Returns (X,y,c) or None."""
    npz = os.path.join(DATA_DIR, task + ".npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["X"].astype(np.float32), d["y"].astype(int), int(d["c"])

    meta = json.load(open(os.path.join(DATA_DIR, "tasks_meta.json"), "r", encoding="utf-8")) \
        if os.path.exists(os.path.join(DATA_DIR, "tasks_meta.json")) else {}

    if task in ("fashion_2", "fashion_5", "fashion_10"):
        X, y = _load_pkl("fashion")
        if task == "fashion_2":
            keep_cls = [0, 6]
            keep = np.isin(y, keep_cls)
            X, y = X[keep], y[keep]
            y = relabel01(y, keep_cls)
            c = 2
        elif task == "fashion_5":
            rng = np.random.RandomState(PREP_SEED)
            keep_cls = sorted(rng.choice(10, size=5, replace=False).tolist())
            keep = np.isin(y, keep_cls)
            X, y = X[keep], y[keep]
            y = relabel01(y, keep_cls)
            c = 5
            meta["fashion_5_classes"] = keep_cls
        else:  # fashion_10
            c = 10
        X, y = stratified_subsample(X, y, N_PER_TASK, seed=PREP_SEED)
    elif task == "usps_10":
        X, y = _load_pkl("usps")
        c = 10
        X, y = stratified_subsample(X, y, N_PER_TASK, seed=PREP_SEED)
    elif task in ("cifar_5", "cifar_10"):
        Xc, yc = get_cifar10()
        if task == "cifar_5":
            rng = np.random.RandomState(PREP_SEED + 7)
            keep_cls = sorted(rng.choice(10, size=5, replace=False).tolist())
            keep = np.isin(yc, keep_cls)
            Xc, yc = Xc[keep], yc[keep]
            yc = relabel01(yc, keep_cls)
            c = 5
            meta["cifar_5_classes"] = keep_cls
        else:
            c = 10
        X, y = stratified_subsample(Xc.astype(np.float32) / 255.0, yc,
                                    N_PER_TASK, seed=PREP_SEED)
    else:
        raise ValueError(task)

    X = X.astype(np.float32)
    np.savez(npz, X=X, y=y, c=np.int64(c))
    meta[task] = {"n": int(X.shape[0]), "d": int(X.shape[1]), "c": int(c),
                   "class_counts": {str(int(k)): int((y == k).sum())
                                    for k in np.unique(y)}}
    json.dump(meta, open(os.path.join(DATA_DIR, "tasks_meta.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    return X, y, c


# --------------------------------------------------------------------------- #
# Acquisition machinery (vectorized w_rob per add5b_binary.batched_robust_gain)
# --------------------------------------------------------------------------- #
def batched_robust_gain(U, seed=0):
    """KL-DRO dual robust gain w^rob per candidate row of U (n, c).
    Identical MC scheme to add5b_binary.batched_robust_gain: gamma/normalise =
    Dirichlet(alpha0*u), eps_q mixed uniform Dirichlet, affine soft-label update,
    logsumexp tau-grid max of -tau*kappa - tau*logmeanexp(-g/tau)."""
    n, c = U.shape
    rng = np.random.RandomState(seed)
    U_pos = np.clip(U, 1e-12, 1.0)
    conc = np.broadcast_to((ALPHA0 * U_pos)[:, None, :], (n, M_MC, c))
    gam = rng.gamma(shape=conc)
    S = gam / gam.sum(2, keepdims=True)
    gam_u = rng.gamma(shape=1.0, size=(n, M_MC, c))
    S_u = gam_u / gam_u.sum(2, keepdims=True)
    mix = rng.rand(n, M_MC) < EPS_Q
    S = np.where(mix[:, :, None], S_u, S)
    H_u = -(U_pos * np.log(U_pos)).sum(1)
    u_up = (U[:, None, :] + LAM_AFF * S) / (1.0 + LAM_AFF)
    u_up = u_up / u_up.sum(2, keepdims=True)
    u_up_c = np.clip(u_up, 1e-12, 1.0)
    H_up = -(u_up_c * np.log(u_up_c)).sum(2)
    g = H_u[:, None] - H_up
    taus = np.logspace(-3.0, 3.0, 40)
    z = -g[:, None, :] / taus[None, :, None]
    lme = logsumexp(z, axis=2) - np.log(M_MC)
    w = -taus[None, :] * KAPPA - taus[None, :] * lme
    return np.maximum(w.max(axis=1), g.min(axis=1) - 1e-12)


def lambda2_over_n(U):
    """beta = lambda_2(L_R)/n, R_ij = cosine(u_i,u_j) zero diagonal
    (add4_large.lambda2_over_n, on current whole-pool U)."""
    n = U.shape[0]
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    Un = U / norms
    R = Un @ Un.T
    np.fill_diagonal(R, 0.0)
    d = R.sum(1)
    L = -R.copy()
    np.fill_diagonal(L, d)
    ev = np.linalg.eigvalsh(L)
    lam2 = float(np.sort(ev)[1]) if len(ev) > 1 else 1.0
    return lam2 / max(n, 1)


def p2_select(U_unlab, U_all, method, b, seed):
    n = U_unlab.shape[0]
    w_rob = batched_robust_gain(U_unlab, seed=seed)
    u_bar = U_all.mean(0)
    h = np.abs(U_unlab - u_bar).sum(1)
    rho = 0.0 if method == "P2_NoExploration" else RHO
    mod = w_rob + rho * h
    beta = lambda2_over_n(U_all)
    norms = np.linalg.norm(U_unlab, axis=1, keepdims=True) + 1e-12
    Un = U_unlab / norms
    selected, avail = [], np.ones(n, bool)
    red_sum = np.zeros(n)
    for _ in range(min(b, n)):
        scores = mod - beta * red_sum
        scores[~avail] = -np.inf
        j = int(np.argmax(scores))
        if scores[j] == -np.inf:
            break
        selected.append(j)
        avail[j] = False
        red_sum += Un[j] @ Un.T
    return selected


# --------------------------------------------------------------------------- #
# Trajectory runner (top-level for Pool pickling)
# --------------------------------------------------------------------------- #
_WORKER_DATA = {}


def _init_worker(task_list):
    global _WORKER_DATA
    for t in task_list:
        X, y, c = prepare_task(t)
        _WORKER_DATA[t] = (X, y, c)


def run_trajectory(task, seed, method):
    X, y, c = _WORKER_DATA[task]
    n = X.shape[0]
    n_init = max(c, int(round(0.02 * n)))          # n=2000 -> 40
    rng0 = np.random.RandomState((seed * 1000 + n_init) & 0x7FFFFFFF)
    labeled = np.zeros(n, bool)
    labeled[rng0.choice(n, size=n_init, replace=False)] = True
    rng = np.random.RandomState((seed * 7919 + 17) & 0x7FFFFFFF)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X[labeled], y[labeled])
    U = clf.predict_proba(X)
    rows = []
    for ri, bud in enumerate(BUDGETS):
        target = int(round(bud * n))
        if int(labeled.sum()) < target:
            unlab = np.where(~labeled)[0]
            b = target - int(labeled.sum())
            if method in ("P2_Full", "P2_NoExploration"):
                sel_local = p2_select(U[unlab], U, method, b,
                                      seed=seed * 1000 + ri)
                sel = unlab[sel_local]
            elif method == "Random":
                sel = unlab[rng.choice(len(unlab), size=b, replace=False)]
            else:  # Entropy
                Uu = np.clip(U[unlab], 1e-12, 1.0)
                H = -(Uu * np.log(Uu)).sum(1)
                sel = unlab[np.argsort(-H)[:b]]
            labeled[sel] = True
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf.fit(X[labeled], y[labeled])
            U = clf.predict_proba(X)
        unlab = np.where(~labeled)[0]
        acc = float((clf.predict(X[unlab]) == y[unlab]).mean())
        rows.append({"task": task, "n_classes": int(c), "method": method,
                     "seed": int(seed), "budget": float(bud), "ACC": acc})
    return rows


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def _work_unpack(job):
    """Picklable top-level worker entry for Pool.imap_unordered."""
    task, seed, method = job
    return task, seed, method, run_trajectory(task, seed, method)


def done_keys():
    if not os.path.exists(DONE_PATH):
        return set()
    with open(DONE_PATH, "r", encoding="utf-8") as f:
        return set(l.strip() for l in f if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    t_start = time.time()
    # 1) prepare tasks (may trigger CIFAR download in parent)
    available = []
    for t in TASK_ORDER:
        try:
            X, y, c = prepare_task(t)
            available.append(t)
            cnt = np.unique(y, return_counts=True)
            log("task=%-11s n=%d d=%d c=%d  class_counts=%s"
                % (t, X.shape[0], X.shape[1], c, dict(zip(cnt[0].tolist(),
                                                           cnt[1].tolist()))))
        except Exception as e:
            log("task=%-11s UNAVAILABLE: %r" % (t, e))
            traceback.print_exc()
    if not available:
        log("no tasks available; abort.")
        return

    seeds = [0] if args.smoke else SEEDS
    budgets_save = BUDGETS
    if args.smoke:
        globals()["BUDGETS"] = [0.10, 0.20]
        available = available[:1]

    # 2) job list w/ resumable skip
    done = done_keys()
    jobs = []
    for t in available:
        for s in seeds:
            for m in METHODS:
                key = "%s|%d|%s" % (t, s, m)
                if key in done:
                    continue
                jobs.append((t, s, m))
    log("jobs to run: %d (already done: %d)" % (len(jobs), len(done)))
    log("L0 formula: %s" % L0_FORMULA)
    log("budgets=%s methods=%s seeds=%s" % (BUDGETS, METHODS, seeds))

    if not jobs:
        log("nothing to do.")
        return

    write_header = not os.path.exists(SUMMARY_PATH) or os.path.getsize(SUMMARY_PATH) == 0
    n_done = 0
    from multiprocessing import Pool
    with Pool(processes=args.workers, initializer=_init_worker,
              initargs=(available,)) as pool:
        results = pool.imap_unordered(_work_unpack, jobs, chunksize=1)
        buf = []
        done_list = []
        for (t, s, m, rows) in results:
            buf.extend(rows)
            done_list.append("%s|%d|%s" % (t, s, m))
            n_done += 1
            if n_done % 5 == 0 or n_done == len(jobs):
                df = pd.DataFrame(buf)
                df.to_csv(SUMMARY_PATH, mode="a", header=write_header,
                          index=False, encoding="utf-8")
                write_header = False
                with open(DONE_PATH, "a", encoding="utf-8") as f:
                    for k in done_list:
                        f.write(k + "\n")
                buf, done_list = [], []
                if n_done % 20 == 0 or n_done == len(jobs):
                    el = time.time() - t_start
                    eta = el / n_done * (len(jobs) - n_done)
                    log("progress %d/%d  elapsed=%.0fs  ETA=%.0fs"
                        % (n_done, len(jobs), el, eta))
    log("RUN COMPLETE: %d trajectories in %.0fs" % (n_done, time.time() - t_start))


if __name__ == "__main__":
    main()
