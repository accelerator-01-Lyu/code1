"""acquisition.py — P2-OLSAL acquisition functions and greedy batch selection.

Dependencies: numpy, and gain.py (calculate_gain).

Objective for a selected batch S (Eq. 13, p.4):
    f(S) = sum_{i in S} ( w_i^rob + rho_expl * h_i )  -  1/2 sum_{i!=j in S} alpha_ij R_ij
    h_i = || u_i - u_bar ||_1          (Eq. 12, exploration reward)
    R_ij = cosine(u_i, u_j)            (redundancy)

Four ablations (frozen-spec §5 / §6):
    P2_Full         : w^rob + rho*h  - pairwise penalty
    P2_NoExploration: force rho_expl = 0
    P2_NoRobust      : plain entropy H(u_i) replaces w^rob
    P2_NoRedundancy  : pairwise penalty forced to 0
"""

import numpy as np
from gain import calculate_gain, entropy

MODES = ("P2_Full", "P2_NoExploration", "P2_NoRobust", "P2_NoRedundancy",
         "P2_ScoreOnly")


def cosine_redundancy(U):
    """R_ij = cosine(u_i, u_j), zero diagonal."""
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    R = (U / norms) @ (U / norms).T
    np.fill_diagonal(R, 0.0)
    return R


def exploration_reward(U, u_all=None):
    """h_i = ||u_i - u_bar||_1.  Theory: u_bar = n^-1 sum_{j=1}^n u_j over the
    WHOLE pool.  If u_all given, use its mean; else fall back to U's mean."""
    u_bar = U.mean(axis=0) if u_all is None else u_all.mean(axis=0)
    return np.abs(U - u_bar).sum(axis=1)


def modular_scores(U, mode, alpha_0=50.0, epsilon_q=0.1, M_MC=1000,
                   kappa=1.0, rho_expl=0.1, seed=0, u_all=None):
    """Per-candidate modular term w_i^rob + rho h_i, with ablation overrides."""
    n = U.shape[0]
    h = exploration_reward(U, u_all=u_all)
    w = np.empty(n)
    for i, u in enumerate(U):
        if mode == "P2_NoRobust":
            w[i] = entropy(np.clip(u, 1e-12, 1.0))   # plain entropy, no DRO
        else:
            r = calculate_gain(u, alpha_0=alpha_0, epsilon_q=epsilon_q,
                               M_MC=M_MC, kappa=kappa, order=2, seed=seed + i)
            w[i] = r["w_robust"]
    rho = 0.0 if mode in ("P2_NoExploration", "P2_ScoreOnly") else rho_expl
    return w + rho * h


def select_batch(U, X, centroids, mode, b,
                 alpha_0=50.0, epsilon_q=0.1, M_MC=1000, kappa=1.0,
                 rho_expl=0.1, lam_red=0.1, seed=0, u_all=None):
    """Greedy batch selection of size b from unlabeled pool U.

    Parameters
    ----------
    U : (n, c) membership matrix of the UNLABELED pool.
    X : (n, d) features (kept for interface completeness / future use).
    centroids : (c, d) current FCM centroids.
    mode : one of MODES.
    b : batch size.

    Returns
    -------
    selected : list of row indices (into U) chosen greedily.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    U = np.asarray(U, dtype=np.float64)
    n = U.shape[0]
    b = min(b, n)

    mod = modular_scores(U, mode, alpha_0=alpha_0, epsilon_q=epsilon_q,
                         M_MC=M_MC, kappa=kappa, rho_expl=rho_expl, seed=seed,
                         u_all=u_all)
    use_red = mode not in ("P2_NoRedundancy", "P2_ScoreOnly")
    R = cosine_redundancy(U) if use_red else np.zeros((n, n))

    selected, avail = [], np.ones(n, dtype=bool)
    red_sum = np.zeros(n)
    for _ in range(b):
        scores = mod - lam_red * red_sum
        scores[~avail] = -np.inf
        j = int(np.argmax(scores))
        if scores[j] == -np.inf:
            break
        selected.append(j)
        avail[j] = False
        red_sum += R[j]
    return selected


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _smoke_test():
    rng = np.random.RandomState(7)
    c = 4
    U = rng.dirichlet(rng.uniform(0.3, 6.0, size=c), size=25)
    X = rng.randn(25, 3)
    V = rng.randn(c, 3)

    picks = {}
    for mode in MODES:
        sel = select_batch(U, X, V, mode, b=5, M_MC=800, seed=0)
        picks[mode] = sel
        print(f"[smoke-test] {mode:18s} batch = {sel}")

    # NoRedundancy and NoExploration should still produce valid batches of b
    for mode, sel in picks.items():
        assert len(sel) == 5 and len(set(sel)) == 5, f"{mode} batch invalid"
    print("[smoke-test] all four ablations return size-b disjoint batches")
    print("[smoke-test] h_i=||u_i-ubar||_1 range:",
          np.round(exploration_reward(U), 3)[:5])
    print("[smoke-test] PASSED")


if __name__ == "__main__":
    _smoke_test()
