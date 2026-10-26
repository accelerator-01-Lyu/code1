"""gain.py — MC reference gain + exact KL-DRO dual robust gain for P2-OLSAL.

Dependencies: numpy, scipy.

Core signature (FROZEN, do not change):
    calculate_gain(u, alpha_0, epsilon_q, M_MC)

=============================================================================
ARCHIVED BUG #4 (fixed here) — alpha_0 never reached the gain routine
=============================================================================
Paper (p.14, §8.5(v)): "that driver only changes the configuration variable
after the gain routine's default argument has been bound at import; the
selector does not pass the new concentration explicitly. The sweep therefore
[was ineffective]."

Grounded in the archived repo (commit 1b20cfa):
  src/robust_selection.py:42   def expected_entropy_gain(p_i,c,alpha0=C.DIRICHLET_ALPHA0,...)
  src/robust_selection.py:51   def robust_gain_dv(p_i,c,kappa,alpha0=C.DIRICHLET_ALPHA0,...)
  src/robust_selection.py:160  def robust_gain_dv_batch(U,c,kappa,alpha0=C.DIRICHLET_ALPHA0,...)
  src/p2_method.py:164         robust_gain_dv_batch(U[unlabeled], c, kappa=self.kappa, rng=...)
                              # <-- alpha0 NOT passed
  supplementary_experiments.py:128  cfg.DIRICHLET_ALPHA0 = alpha0   # runtime mutation

Python binds default arguments at function DEFINITION (import) time, so
`alpha0=C.DIRICHLET_ALPHA0` captured 50.0 once; mutating the module attribute
later changed nothing, and the selector never overrode it. The whole
alpha0 in {0.1,1,10,100} sweep (Fig.13b) was inert -> flat ACC.

FIX: alpha_0 is now a REQUIRED explicit positional argument and is USED inside
    the body to construct q_i. See `calculate_gain` below.

=============================================================================
ARCHIVED BUG #2 (fixed here) — clipped MC heuristic != exact KL dual
=============================================================================
Paper (p.8): archived score is a clipped MC heuristic; "headline kappa=1 is
fixed rather than adapted through Eq.(11)."

Grounded in src/robust_selection.py:61-79, the archived maximized
    F(lambda) = lambda*g_mean - lambda*kappa - log E_q exp(-lambda g)
on lambda in [0,50].  F'(0) = -kappa < 0, so the golden section collapsed to
the left boundary and the final `w_rob = max(0.0, F(.5))` clipped the DRO term
to a near-zero heuristic.  It is NOT the entropic dual.

The exact dual (Eqs. 7-9; inf_{P: KL(P||q)<=kappa} E_P g) is:
    w^rob(kappa) = sup_{tau>0} { -tau*kappa - tau * log E_q exp(-g/tau) }.
We implement it with a numerically stable logsumexp and a 1-D tau search that
explicitly handles the tau->0 (worst-case, w->min g) and tau->inf (w->-inf,
since the -tau*kappa penalty dominates) limits.
"""

import numpy as np
from scipy.special import logsumexp


# --------------------------------------------------------------------------- #
# Reference law and Monte-Carlo entropy gain
# --------------------------------------------------------------------------- #
def _sample_reference_law(u, alpha_0, epsilon_q, M_MC, rng):
    """Draw M_MC soft-label proposals s_r ~ q_i (Eq. 2).

    q_i = (1-eps_q) Dir(alpha_0 * u) + eps_q Dir(1_c).
    u : (c,) current fuzzy membership of the candidate.
    Returns (M_MC, c) proposals on the simplex.
    """
    u = np.asarray(u, dtype=np.float64)
    c = u.shape[0]
    u_pos = np.clip(u, 1e-12, 1.0)
    uniform_mask = rng.rand(M_MC) < epsilon_q
    n_unif = int(uniform_mask.sum())
    n_dir = M_MC - n_unif
    S = np.empty((M_MC, c))
    if n_dir > 0:
        S[~uniform_mask] = rng.dirichlet(alpha_0 * u_pos, size=n_dir)
    if n_unif > 0:
        S[uniform_mask] = rng.dirichlet(np.ones(c), size=n_unif)
    return S


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def _monte_carlo_gains(u, alpha_0, epsilon_q, M_MC, lam_affine, rng):
    """Return MC samples g_r = H(u) - H(u^+(s_r)), u^+ = (u+lam s)/(1+lam)."""
    u = np.asarray(u, dtype=np.float64)
    S = _sample_reference_law(u, alpha_0, epsilon_q, M_MC, rng)
    u_pos = u[None, :]
    u_up = (u_pos + lam_affine * S) / (1.0 + lam_affine)
    u_up = u_up / u_up.sum(axis=1, keepdims=True)
    g = float(entropy(u)) - entropy(u_up)
    return g  # shape (M_MC,)


# --------------------------------------------------------------------------- #
# Exact KL-DRO dual via 1-D tau optimization
# --------------------------------------------------------------------------- #
def _robust_dual_tau(g, kappa, tau):
    """w(tau) = -tau*kappa - tau*logmeanexp(-g/tau).  Vectorized in tau."""
    g = np.asarray(g, dtype=np.float64)
    M = g.shape[0]
    tau = np.asarray(tau, dtype=np.float64)[..., None]      # (T,1)
    z = -g[None, :] / tau                                  # (T,M)
    log_mean_exp = logsumexp(z, axis=1) - np.log(M)        # (T,)
    return -tau[..., 0] * kappa - tau[..., 0] * log_mean_exp


def robust_gain_from_samples(g, kappa, n_grid=80, n_refine=25):
    """Solve w^rob = sup_{tau>0} w(tau) with boundary handling.

    Boundary limits used to guard the search:
      tau -> 0  : w -> min_r g_r   (fully adversarial, tightest ambiguity set)
      tau -> inf: w -> -inf        (-tau*kappa dominates)
    We maximize on a log-spaced tau grid, then golden-section refine around the
    best grid point.  Falls back to the appropriate boundary value if the
    interior search is degenerate.
    """
    g = np.asarray(g, dtype=np.float64)
    g_min = float(g.min())
    g_mean = float(g.mean())
    if kappa is None or kappa <= 0.0:
        return g_mean          # KL radius 0 => P=q => E_q g

    # tau grid: wide log range.  As tau->0 w->min g; as tau->inf w->-inf.
    taus = np.logspace(-3.0, 3.0, n_grid)
    vals = _robust_dual_tau(g, kappa, taus)
    best = int(np.argmax(vals))

    # golden-section refinement on [lo, hi] around best grid point
    lo = taus[max(0, best - 1)]
    hi = taus[min(len(taus) - 1, best + 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def f(t):
        return float(_robust_dual_tau(g, kappa, np.array([t]))[0])

    a, b = lo, hi
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = f(c1), f(c2)
    for _ in range(n_refine):
        if f1 > f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a); f1 = f(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a); f2 = f(c2)
    w_interior = f(0.5 * (a + b))

    # w(tau) concave in tau? It is the Legendre dual -> concave in tau.
    # Guard against a grid that missed the peak by also comparing to the
    # tau->0 boundary limit min g.
    w = max(w_interior, g_min - 1e-12)
    return float(w)


# --------------------------------------------------------------------------- #
# Public API — signature is FROZEN
# --------------------------------------------------------------------------- #
def calculate_gain(u, alpha_0, epsilon_q, M_MC,
                   kappa=1.0, kappa_mode="fixed", misfit=None,
                   lam_affine=0.5, order=2, seed=0):
    """Compute the acquisition gain for one unlabeled candidate.

    Parameters (core 4 are FROZEN positional args — DO NOT REORDER/RENAMES):
        u         : (c,) fuzzy membership of the candidate (rows of U).
        alpha_0   : Dirichlet concentration of the reference law.
                    THIS IS NOW AN EXPLICIT ARG, used directly below — the
                    archived default-argument binding bug (Bug #4) is gone.
        epsilon_q : mass on the uniform Dirichlet component of q_i.
        M_MC      : number of Monte-Carlo reference-law samples.

    Optional knobs:
        kappa     : KL ambiguity radius.  scalar in 'fixed' mode; multiplied by
                    the per-point misfit in 'misfit' mode (Eq. 11 hook).
        kappa_mode: 'fixed'  -> use kappa as-is (archived headline kappa=1).
                    'misfit' -> kappa_i = kappa * misfit_i (Eq.11 budget hook;
                                archived repo had NO adaptation, Bug #2).
        misfit    : per-point misfit budget (required when kappa_mode='misfit').
        lam_affine: soft-label update step size in u^+ = (u+lam s)/(1+lam).
        order     : 1 -> first-order pure entropy gain E_q g (FMIS off);
                    2 -> second-order robust gain w^rob (FMIS on).
        seed      : RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        gain     : the score used by the selector (order=1 -> E_q g; order=2 -> w^rob)
        g_mean   : MC expected entropy gain E_q g
        w_robust : exact KL-DRO robust gain sup_tau dual
        kappa_used : the (possibly adapted) KL radius actually used.
    """
    rng = np.random.RandomState(seed)
    g = _monte_carlo_gains(u, alpha_0, epsilon_q, M_MC, lam_affine, rng)
    g_mean = float(g.mean())

    # --- Eq.(11) misfit-budget kappa adaptation hook ---------------------- #
    if kappa_mode == "misfit":
        if misfit is None:
            raise ValueError("kappa_mode='misfit' requires a misfit array")
        kappa_i = float(kappa * np.clip(np.mean(misfit), 1e-6, None))
    elif kappa_mode == "fixed":
        kappa_i = float(kappa)
    else:
        raise ValueError("kappa_mode must be 'fixed' or 'misfit'")

    w_rob = robust_gain_from_samples(g, kappa_i)

    if order == 1:
        gain = g_mean                 # first-order: pure entropy gain
    elif order == 2:
        gain = w_rob                  # second-order: KL-DRO robust gain
    else:
        raise ValueError("order must be 1 (pure entropy) or 2 (robust)")

    return {"gain": float(gain), "g_mean": g_mean,
            "w_robust": w_rob, "kappa_used": kappa_i}


# --------------------------------------------------------------------------- #
# Smoke test — regression proof for Bug #4
# --------------------------------------------------------------------------- #
def _smoke_test():
    # A boundary-ish membership (entropy high) in c=4 clusters
    u = np.array([0.32, 0.28, 0.25, 0.15])
    a0_values = [0.1, 1.0, 50.0, 1000.0]

    print("[smoke-test] alpha0 must CHANGE the gain (Bug #4 regression)")
    outs = []
    for a0 in a0_values:
        r = calculate_gain(u, alpha_0=a0, epsilon_q=0.1, M_MC=2000,
                           kappa=1.0, order=2, seed=123)
        outs.append(r["gain"])
        print(f"[smoke-test] alpha0={a0:7.2f}  E_q g={r['g_mean']:.4f}  "
              f"w^rob={r['w_robust']:.4f}  kappa={r['kappa_used']:.3f}")

    spread = float(np.max(outs) - np.min(outs))
    print(f"[smoke-test] spread of w^rob over alpha0 sweep = {spread:.4f}")
    assert spread > 1e-4, (
        "alpha0 sweep had NO effect — the archived default-argument bug is back!")

    # boundary behavior: kappa=0 must equal E_q g; larger kappa must be <= E_q g
    r0 = calculate_gain(u, 50.0, 0.1, 2000, kappa=0.0, order=2, seed=7)
    assert abs(r0["w_robust"] - r0["g_mean"]) < 1e-6, "kappa=0 => w=E_q g"
    rbig = calculate_gain(u, 50.0, 0.1, 2000, kappa=5.0, order=2, seed=7)
    assert rbig["w_robust"] <= r0["g_mean"] + 1e-6, "robust <= expected gain"
    print(f"[smoke-test] kappa=0 boundary w^rob=E_qg={r0['g_mean']:.4f} OK")
    print(f"[smoke-test] kappa=5  w^rob={rbig['w_robust']:.4f} <= E_qg OK")

    # order=1 vs order=2 give different numbers
    r1 = calculate_gain(u, 50.0, 0.1, 2000, kappa=1.0, order=1, seed=3)
    r2 = calculate_gain(u, 50.0, 0.1, 2000, kappa=1.0, order=2, seed=3)
    print(f"[smoke-test] order=1 (pure entropy)={r1['gain']:.4f}  "
          f"order=2 (robust)={r2['gain']:.4f}")
    assert abs(r1["gain"] - r2["gain"]) > 1e-6, "FMIS order flag must change gain"
    print("[smoke-test] ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    _smoke_test()
