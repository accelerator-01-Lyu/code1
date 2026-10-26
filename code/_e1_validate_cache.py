"""_e1_validate_cache.py — bit-level validation of the init-FCM disk cache.

For 3 sampled (iid, seed) pairs:
  1. Replicate run_traj's L0 rng.choice exactly.
  2. Call the REAL plcfcm_fit twice -> U1,V1 and U2,V2 (must be bit-identical,
     proving determinism).
  3. Pickle (U1,V1) to the cache path, reload -> U3,V3 (must be bit-identical).
  4. Exercise the wrapper: set _JOB_CTX, call _cached_plcfcm_fit with a cold
     cache, then again with a warm cache; the warm call must return arrays
     bit-equal to the cold call.
"""
import os, sys, json, pickle, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_experiments import INIT_FRAC
from fcm_update import plcfcm_fit as _real

import run_synthetic_sweep as sweep

ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "synthetic")
CACHE_DIR = sweep.CACHE_DIR

SAMPLES = [(1, 0), (15, 7), (69, 13)]

def l0_labeled(X, c, seed):
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n = X.shape[0]
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    return labeled

def main():
    # Wipe any stale cache for the 3 sampled pairs to force cold path
    for iid, seed in SAMPLES:
        p = os.path.join(CACHE_DIR, f"init_{iid:03d}_seed{seed}.pkl")
        if os.path.exists(p):
            os.remove(p)

    all_ok = True
    for iid, seed in SAMPLES:
        dpath = os.path.join(DATA_DIR, f"instance_{iid:03d}")
        X = np.load(os.path.join(dpath, "X.npy"))
        y = np.load(os.path.join(dpath, "y_true.npy"))
        meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
        c = meta["c"]
        labeled = l0_labeled(X, c, seed)
        lidx = np.where(labeled)[0]
        lab = y[labeled]

        # 1. Real call x2 — determinism
        U1, V1, it1 = _real(X, c=c, labeled_idx=lidx, labels=lab, seed=seed)
        U2, V2, it2 = _real(X, c=c, labeled_idx=lidx, labels=lab, seed=seed)
        det_ok = (U1.dtype == U2.dtype and V1.dtype == V2.dtype
                  and U1.shape == U2.shape and V1.shape == V2.shape
                  and np.array_equal(U1, U2) and np.array_equal(V1, V2))
        print(f"[val] iid={iid} seed={seed}: real-plcfcm determinism "
              f"bit-identical={det_ok} (it={it1},{it2})", flush=True)

        # 2. Pickle round-trip
        pkl = os.path.join(CACHE_DIR, f"init_{iid:03d}_seed{seed}.pkl")
        tmp = pkl + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump((U1.copy(), V1.copy()), f)
        os.replace(tmp, pkl)
        with open(pkl, "rb") as f:
            U3, V3 = pickle.load(f)
        pickle_ok = (np.array_equal(U1, U3) and np.array_equal(V1, V3))
        print(f"[val] iid={iid} seed={seed}: pickle round-trip "
              f"bit-identical={pickle_ok}", flush=True)

        # 3. Wrapper cold vs warm
        sweep._JOB_CTX["iid"] = iid
        sweep._JOB_CTX["seed"] = seed
        sweep._JOB_CTX["init_done"] = False
        # warm path (cache file exists from step 2)
        Uw, Vw, _ = sweep._cached_plcfcm_fit(X, c=c, labeled_idx=lidx,
                                               labels=lab, seed=seed)
        warm_ok = (np.array_equal(U1, Uw) and np.array_equal(V1, Vw))
        print(f"[val] iid={iid} seed={seed}: wrapper warm-cache equals "
              f"real ={warm_ok}", flush=True)

        # 4. Wrapper cold path (delete file, run, check it writes + equals)
        os.remove(pkl)
        sweep._JOB_CTX["init_done"] = False
        Uc, Vc, _ = sweep._cached_plcfcm_fit(X, c=c, labeled_idx=lidx,
                                              labels=lab, seed=seed)
        cold_ok = (np.array_equal(U1, Uc) and np.array_equal(V1, Vc)
                   and os.path.exists(pkl))
        print(f"[val] iid={iid} seed={seed}: wrapper cold-path equals "
              f"real + wrote cache ={cold_ok}", flush=True)

        ok = det_ok and pickle_ok and warm_ok and cold_ok
        all_ok = all_ok and ok
        print(f"[val] iid={iid} seed={seed}: {'PASS' if ok else 'FAIL'}\n",
              flush=True)

    print(f"[val] OVERALL: {'ALL PASS' if all_ok else 'FAIL'}", flush=True)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
