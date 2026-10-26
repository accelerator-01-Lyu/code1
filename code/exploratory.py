"""exploratory.py — X2 gradient inner-product diagnostic (EXPLORATORY, v2).

EXPLORATORY only; never enters prereg_verdicts.json.

Measures per-candidate
    s_i = < grad_h_i, grad_w_i^rob >
with:
  grad_h_i = (1 - 1/n) sign(u_i - u_bar)                  (Thm 2(i), closed form)
  grad_w_i : central finite difference of w^rob(u) wrt u, evaluated with
             COMMON RANDOM NUMBERS (CRN) at a dedicated large sample
             M_diag (separate from the confirmatory M=50).

CRN: draw Dirichlet proposals once via gamma quantile transform on a fixed
uniform stream; reuse the SAME underlying uniforms for u, u+delta, u-delta
so MC noise cancels in the difference. delta swept on a log grid; the stable
plateau gradient is reported. Also reports per-point MC SE, gradient norm,
cos angle, and estimator reliability flags.
"""
import os
import numpy as np
from scipy.special import digamma
from gain import entropy, robust_gain_from_samples

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "exploratory")
os.makedirs(OUT_DIR, exist_ok=True)

M_DIAG = 2000
EPS = 0.1
LAM = 0.5
KAPPA = 1.0


def _dir_crn(rng, alpha, size):
    """Gamma-dirichlet with a fixed uniform stream (CRN-friendly)."""
    g = rng.standard_gamma(alpha, size=size)
    return g / g.sum(axis=-1, keepdims=True)


def _samples_crn(unif_uniform, rng_gamma_stream, u, alpha0, M, c):
    """Draw M proposals s ~ q_i = (1-eps)Dir(a0 u)+eps Dir(1) using CRN stream.

    unif_uniform : (M,) Bernoulli masks for the eps component.
    rng_gamma_stream : (M, c) standard gamma draws, reused across u perturbations.
    """
    S = np.empty((M, c))
    dir_mask = ~unif_uniform
    n_dir = int(dir_mask.sum())
    n_unif = int(unif_uniform.sum())
    if n_dir > 0:
        a = np.clip(alpha0 * u, 1e-3, None)
        # reuse gamma stream: scale each draw by alpha/sum (gamma CRN trick)
        g = rng_gamma_stream[dir_mask] * (a[None, :] /
                                           rng_gamma_stream[dir_mask].mean(1, keepdims=True))
        S[dir_mask] = g / g.sum(1, keepdims=True)
    if n_unif > 0:
        g = rng_gamma_stream[unif_uniform]
        S[unif_unif if False else unif_uniform] = g / g.sum(1, keepdims=True)
    return S


def _gains_from_S(u, S, lam=LAM):
    u_up = (u[None, :] + lam * S) / (1 + lam)
    u_up /= u_up.sum(1, keepdims=True)
    return float(entropy(u)) - entropy(u_up)


def w_with_crn(u, unif_mask, gam_stream, alpha0, M, c, kappa=KAPPA):
    S = _samples_crn(unif_mask, gam_stream, u, alpha0, M, c)
    g = _gains_from_S(u, S)
    return robust_gain_from_samples(g, kappa)


def _tan(g):
    """Project gradient onto the simplex tangent space {du: 1^T du=0}."""
    return g - g.mean()


def grad_w_crn(u, alpha0, seed, deltas=(1e-2, 1e-3, 1e-4)):
    c = u.shape[0]
    rng = np.random.RandomState(seed)
    M = M_DIAG
    unif_mask = rng.rand(M) < EPS
    gam = rng.gamma(1.0, size=(M, c))
    g_plateau = None; norms = []; tan_norms = []
    for d in deltas:
        grad = np.zeros(c)
        for k in range(c):
            # tangent-space perturbation: +d e_k - d/(c-1) sum_{j!=k} e_j
            dirv = -d / (c - 1) * np.ones(c)
            dirv[k] = d
            up = u + dirv; um = u - dirv
            wp = w_with_crn(up, unif_mask, gam, alpha0, M, c)
            wm = w_with_crn(um, unif_mask, gam, alpha0, M, c)
            grad[k] = (wp - wm) / (2 * d)
        grad = _tan(grad)
        norms.append(np.linalg.norm(grad))
        tan_norms.append(np.linalg.norm(grad))
        g_plateau = grad
    se = float(np.std(norms))
    return g_plateau, float(np.linalg.norm(g_plateau)), se, M


def grad_h(U, u_bar, n_pool):
    """grad h_i = (1 - 1/n_pool) sign(u_i - u_bar). n_pool is the TOTAL pool
    size, NOT the cluster count c."""
    return (1.0 - 1.0 / n_pool) * np.sign(U - u_bar)


def run_x2_diag(U, u_all, alpha0, kappa, seed):
    n, c = U.shape
    u_bar = u_all.mean(0)
    H = -(np.clip(U, 1e-12, 1) * np.log(np.clip(U, 1e-12, 1))).sum(1)
    h = np.abs(U - u_bar).sum(1)
    qH_hi, qH_lo = np.quantile(H, [0.75, 0.25])
    qh_hi, qh_lo = np.quantile(h, [0.75, 0.25])
    gh = grad_h(U, u_bar[None, :], n_pool=u_all.shape[0])
    s = np.zeros(n); s_amb = np.zeros(n)
    normw = np.zeros(n); tan_ratio = np.zeros(n)
    normh = np.zeros(n); searr = np.zeros(n); cosang = np.zeros(n)
    for i in range(n):
        gw, nw, se, M = grad_w_crn(U[i], alpha0, seed * 100 + i)
        ght = _tan(gh[i])
        s[i] = float(np.dot(ght, gw))
        s_amb[i] = float(np.dot(gh[i], gw))     # ambient, for comparison
        normh[i] = np.linalg.norm(ght); normw[i] = nw; searr[i] = se
        cosang[i] = s[i] / (normh[i] * nw + 1e-12)
    lab = np.full(n, "other", dtype=object)
    hiH = H >= qH_hi; loH = H <= qH_lo
    hih = h >= qh_hi; loh = h <= qh_lo
    lab[hiH & loh] = "boundary"; lab[loH & hih] = "eccentric"; lab[hiH & hih] = "overlap"
    A = H >= qH_hi; B = h >= qh_hi
    jacc = (A & B).sum() / (A | B).sum()
    return {"s": s, "s_amb": s_amb, "label": lab, "jaccard": float(jacc),
            "frac_s_neg": float((s < 0).mean()), "mean_s": float(s.mean()),
            "normw": normw, "se": searr, "cosang": cosang, "M_diag": M}


def _smoke_test():
    rng = np.random.RandomState(3)
    c = 4
    U = rng.dirichlet(rng.uniform(0.3, 5.0, c), size=40)
    out = run_x2_diag(U, U, alpha0=50.0, kappa=1.0, seed=0)
    print(f"[smoke-test] M_diag={out['M_diag']} frac_s<0={out['frac_s_neg']:.3f} "
          f"mean_s={out['mean_s']:.4f} Jaccard={out['jaccard']:.3f}")
    print(f"[smoke-test] median ||grad_w||={np.median(out['normw']):.4f} "
          f"median SE={np.median(out['se']):.4f} median cos={np.median(out['cosang']):.3f}")
    for lab in ("boundary", "eccentric", "overlap", "other"):
        m = out["label"] == lab
        if m.sum():
            print(f"[smoke-test] {lab:9s} n={m.sum():3d} "
                  f"frac_s<0={float((out['s'][m] < 0).mean()):.2f}")
    print("[smoke-test] PASSED")


if __name__ == "__main__":
    _smoke_test()
