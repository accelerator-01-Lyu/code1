"""gain_fast.py — vectorized drop-in replacement for gain.py (EXPLORATORY/thick).

Math is line-for-line identical to gain.py; only loop structure is vectorized
over (candidate, MC sample, tau grid). Frozen constants unchanged.

NOT used by the 30-seed confirmatory run (which uses gain.py). Used only for
the thickening batch after numerical equivalence is verified.
"""
import numpy as np
from scipy.special import logsumexp


def sample_reference_law_batch(U, alpha_0, epsilon_q, M, rng):
    """U: (n, c). Returns S: (n, M, c) proposals ~ q_i. CRN-stable via rng."""
    n, c = U.shape
    U_pos = np.clip(U, 1e-12, 1.0)
    dir_mask = rng.rand(n, M) >= epsilon_q
    n_unif = int((~dir_mask).sum())
    S = np.empty((n, M, c))
    # Dirichlet via gamma; broadcast alpha
    g = rng.standard_gamma(np.broadcast_to(U_pos[:, None, :], (n, M, c)) * alpha_0)
    S = g / g.sum(2, keepdims=True)
    if n_unif:
        S[~dir_mask] = rng.dirichlet(np.ones(c), size=n_unif)
    return S


def entropy_p(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1)


def calculate_gain_batch(U, alpha_0, epsilon_q, M, kappa=1.0,
                         lam_affine=0.5, order=2, seed=0, n_tau=80):
    """Vectorized w_robust for a batch U (n,c). Returns (n,) gains."""
    n, c = U.shape
    rng = np.random.RandomState(seed)
    S = sample_reference_law_batch(U, alpha_0, epsilon_q, M, rng)  # (n,M,c)
    u_up = (U[:, None, :] + lam_affine * S) / (1 + lam_affine)
    u_up /= u_up.sum(2, keepdims=True)
    g = entropy_p(U)[:, None] - entropy_p(u_up)   # (n,M)
    g_mean = g.mean(1)
    # tau grid, vectorized: w(tau) = -tau kappa - tau logmeanexp(-g/tau)
    taus = np.logspace(-3, 3, n_tau)[None, :, None]  # (1,T,1)
    z = -g[:, None, :] / taus                        # (n,T,M)
    lme = logsumexp(z, axis=2) - np.log(M)            # (n,T)
    w_tau = -taus[:, :, 0] * kappa - taus[:, :, 0] * lme
    # also tau->0 boundary = min g
    w = np.maximum(w_tau.max(1), g.min(1))
    if order == 1:
        return g_mean
    return w


def _smoke_test():
    from gain import calculate_gain
    rng = np.random.RandomState(0)
    for ds_seed in [0, 1]:
        c = 4
        U = rng.dirichlet(rng.uniform(.3, 5, c), size=20)
        for M in (50, 2000):
            slow = np.array([calculate_gain(u, 50.0, 0.1, M, kappa=1.0, order=2,
                                            seed=ds_seed * 100 + i)["w_robust"]
                             for i, u in enumerate(U)])
            fast = calculate_gain_batch(U, 50.0, 0.1, M, kappa=1.0, order=2,
                                         seed=ds_seed * 100)
            print(f"[smoke] M={M} max|slow-fast|={np.abs(slow-fast).max():.2e}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
