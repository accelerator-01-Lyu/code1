"""fcm_update.py — Correct (P)LFCM membership update, with labeled-sample handling.

Dependencies: numpy only (standard library otherwise).

=============================================================================
ARCHIVED BUG #1 (fixed here) — wrong distance exponent
=============================================================================
Paper (p.8, source_audit_v2.md §B.1):
    "The PLCFCM routine also applies exponent -2/(m-1) to an ALREADY SQUARED
     distance, whereas the stated standard FCM objective requires -1/(m-1)."

Grounded in the archived reference repo (commit 1b20cfa, src/plcfcm.py):
    line 25: D = ||x-v||^2  ...                       # D IS ALREADY SQUARED
    line 26: exp = 2.0/(m-1.0)
    line 27: Dinv = D ** (-exp)
=> The archived code evaluates (d^2)^(-2/(m-1)) = d^(-4/(m-1)).
   On squared distances D = d^2 the textbook FCM update requires
                        D^(-1/(m-1)) = d^(-2/(m-1)).
FIX: exponent on the already-squared distance is 1/(m-1) (see `pow_` below).

Textbook update (raw distances d_ik = ||x_i - v_k||):
    u_ik = [ sum_j (d_ik / d_ij)^(2/(m-1)) ]^(-1)
Equivalently, on squared distances D_ij = d_ij^2:
    u_ik = [ sum_j (D_ik / D_ij)^(1/(m-1)) ]^(-1).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Core membership update
# --------------------------------------------------------------------------- #
def fcm_membership_update(D, m, squared=True, eps=1e-12):
    """Compute FCM membership matrix U (n, c) from distances D (n, c).

    Parameters
    ----------
    D : (n, c) array
        Pairwise distances between each sample and each centroid.
    m : float
        Fuzzifier, m > 1.
    squared : bool
        If True, D is ALREADY the squared distance d^2 (the standard internal
        form FCM uses); the exponent is -1/(m-1).
        If False, D is a raw Euclidean distance d; the exponent is -2/(m-1).
    eps : float
        Floor added to D to avoid division by zero before the explicit
        zero-distance rule below is applied.

    Returns
    -------
    U : (n, c) array, rows sum to 1.

    Zero-distance rule
    -------------------
    If a sample sits exactly on a centroid (D_ik == 0), it receives
    membership 1 on that cluster and 0 on all others.  If it sits on several
    coincident centroids, membership is split uniformly among them.
    """
    if m <= 1.0:
        raise ValueError("fuzzifier m must be > 1")
    D = np.asarray(D, dtype=np.float64)
    n, c = D.shape

    # --- FIX BUG #1: exponent depends on whether D is already squared ------ #
    if squared:
        pow_ = 1.0 / (m - 1.0)     # CORRECT on squared D = d^2
    else:
        pow_ = 2.0 / (m - 1.0)     # textbook form on raw distances d
    # ARCHIVED (buggy) line was:  pow_ = 2.0/(m-1.0)  even when D = d^2.

    D_safe = np.maximum(D, eps)
    # ratio trick: u_ik ∝ D_ik^(-pow_) = (D_ik/D_ij)^(pow_) after normalization
    U = D_safe ** (-pow_)
    U = U / U.sum(axis=1, keepdims=True)

    # --- explicit zero-distance rule -------------------------------------- #
    zero_rows = np.where(np.any(D <= 0.0, axis=1))[0]
    for i in zero_rows:
        hit = D[i] <= 0.0
        U[i] = hit / hit.sum()
    return U


# --------------------------------------------------------------------------- #
# Centroid update with hard-labeled samples
# --------------------------------------------------------------------------- #
def fcm_centroid_update(X, U, m, labeled_idx=None, labeled_clusters=None):
    """Recompute centroids V (c, d).

    Unlabeled points contribute weighted by U_ik^m.  Labeled points are hard
    constraints: their membership on their assigned cluster is forced to 1 and
    0 elsewhere (this is the standard semi-supervised FCM / PLCFCM scheme).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    c = U.shape[1]
    um = U ** m
    if labeled_idx is not None and len(labeled_idx) > 0:
        labeled_idx = np.asarray(labeled_idx, dtype=int)
        labeled_clusters = np.asarray(labeled_clusters, dtype=int)
        um[labeled_idx] = 0.0                 # wipe soft memberships ...
        um[labeled_idx, labeled_clusters] = 1.0   # ... and impose one-hot
    denom = um.sum(axis=0)
    denom[denom == 0] = 1e-12
    V = (um.T @ X) / denom[:, None]
    return V


def pairwise_sq_dist(X, V):
    """||x-v||^2 = ||x||^2 + ||v||^2 - 2 x.v  (vectorized, no (n,c,d) tensor)."""
    Xs = (X ** 2).sum(axis=1)[:, None]
    Vs = (V ** 2).sum(axis=1)[None, :]
    D = Xs + Vs - 2.0 * (X @ V.T)
    return np.maximum(D, 0.0)


def plcfcm_fit(X, c, m=2.0, labeled_idx=None, labels=None,
               max_iter=150, tol=1e-6, seed=0):
    """Self-contained PLCFCM loop (unlabeled FCM when no labels given).

    Returns U (n,c), V (c,d), n_iter.
    Labeled points: hard one-hot membership; their label ids are mapped to
    contiguous cluster ids [0, c).
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    label_map = None
    lab_clu = None
    if labeled_idx is not None and len(labeled_idx) > 0 and labels is not None:
        uniq = sorted(set(int(l) for l in labels))
        label_map = {l: i for i, l in enumerate(uniq)}
        lab_clu = np.array([label_map[int(l)] for l in labels], dtype=int)
        labeled_idx = np.asarray(labeled_idx, dtype=int)

    # k-means++ style init
    V = np.zeros((c, d))
    V[0] = X[rng.randint(n)]
    for j in range(1, c):
        D2 = pairwise_sq_dist(X, V[:j])
        d2 = D2.min(axis=1)
        p = d2 / (d2.sum() + 1e-12)
        V[j] = X[rng.choice(n, p=p)]

    U = fcm_membership_update(pairwise_sq_dist(X, V), m=m, squared=True)
    if lab_clu is not None:
        U[labeled_idx] = 0.0
        U[labeled_idx, lab_clu] = 1.0

    it = 0
    for it in range(1, max_iter + 1):
        V_old = V.copy()
        V = fcm_centroid_update(X, U, m, labeled_idx, lab_clu)
        U = fcm_membership_update(pairwise_sq_dist(X, V), m=m, squared=True)
        if lab_clu is not None:
            U[labeled_idx] = 0.0
            U[labeled_idx, lab_clu] = 1.0
        if np.linalg.norm(V - V_old) < tol:
            break
    return U, V, it


# --------------------------------------------------------------------------- #
# Smoke test (NOT a paper result)
# --------------------------------------------------------------------------- #
def _smoke_test():
    rng = np.random.RandomState(0)
    # 5 well-separated Gaussian blobs in 2D
    centers = np.array([[0, 0], [5, 0], [0, 5], [5, 5], [2.5, 2.5]])
    blobs = []
    for k, ctr in enumerate(centers):
        blobs.append(ctr + rng.randn(40, 2) * 0.35)
    X = np.vstack(blobs)
    y = np.repeat(np.arange(5), 40)

    # pick 2 labeled points per cluster as hard labels
    lab_idx, lab_lab = [], []
    for k in range(5):
        ix = np.where(y == k)[0][:2]
        lab_idx.extend(ix.tolist())
        lab_lab.extend([k, k])

    U, V, it = plcfcm_fit(X, c=5, m=2.0,
                          labeled_idx=lab_idx, labels=lab_lab, seed=0)

    from scipy.optimize import linear_sum_assignment
    row_sum_err = float(np.max(np.abs(U.sum(axis=1) - 1.0)))
    hard_ok = np.allclose(U[lab_idx].argmax(axis=1), np.array(lab_lab))
    # clustering should recover the blobs (best-acc via Hungarian matching)
    pred = U.argmax(axis=1)
    Dm = 5
    Cm = np.zeros((Dm, Dm), dtype=int)
    for t, p in zip(y, pred):
        Cm[t, p] += 1
    ri, ci = linear_sum_assignment(-Cm)
    acc = float(Cm[ri, ci].sum() / len(y))

    print(f"[smoke-test] FCM on 5 Gaussian blobs, n={X.shape[0]}, c=5")
    print(f"[smoke-test] iterations to converge = {it} (tol=1e-6)")
    print(f"[smoke-test] max |row_sum(U)-1|       = {row_sum_err:.2e}")
    print(f"[smoke-test] hard-label rows preserved = {hard_ok}")
    print(f"[smoke-test] argmax-cluster agreement with true blobs = {acc:.3f}")

    # zero-distance rule sanity: drop a point exactly on a centroid
    D_zero = np.array([[0.0, 1.0, 2.0]])
    Uz = fcm_membership_update(D_zero, m=2.0, squared=True)
    assert Uz.shape == (1, 3) and Uz[0, 0] == 1.0 and Uz[0, 1] == 0.0
    print(f"[smoke-test] zero-distance rule U=one-hot at D=0 = {Uz.tolist()}")

    assert row_sum_err < 1e-8, "membership rows must sum to 1"
    assert hard_ok, "labeled samples must stay one-hot"
    assert acc > 0.90, "well-separated blobs must be recovered"
    print("[smoke-test] ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    _smoke_test()
