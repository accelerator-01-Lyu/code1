"""fmis_switch.py — FMIS flag must genuinely change the gain/ranking.

Dependencies: numpy, and gain.py (calculate_gain).

=============================================================================
ARCHIVED BUG #3 (fixed here) — FMIS flag stored but never consulted
=============================================================================
Paper (p.12, §8.5): "The released selector stores the FMIS flag but does not
consult it when computing gains or penalties."

Grounded in the archived repo (commit 1b20cfa):
  src/p2_method.py:89/97   self.fmis = fmis          # stored
  src/registry.py:52        P2_NoFMIS = _make_p2(fmis=False, ...)
  src/p2_method.py:151-276  select() ... self.fmis NEVER appears in the body;
                            fmis_gain() (robust_selection.py:117) is never called.
=> P2_Full and P2_NoFMIS selected identically.

FIX: the FMIS flag now selects between
    use_fmis=False -> first-order gain g^(1) = E_q g        (pure entropy)
    use_fmis=True  -> second-order gain g^(2) = w^rob        (KL-DRO dual)
and changes the top-1 choice on data where the two disagree.
"""

import numpy as np
from gain import calculate_gain


def score_candidates(U, use_fmis, alpha_0=50.0, epsilon_q=0.1, M_MC=2000,
                     kappa=1.0, seed=0):
    """Return per-candidate score vector under FMIS on/off.

    use_fmis=False -> order=1 (first-order pure entropy gain E_q g).
    use_fmis=True  -> order=2 (second-order KL-DRO robust gain w^rob).
    """
    U = np.asarray(U, dtype=np.float64)
    order = 2 if use_fmis else 1
    scores = np.empty(U.shape[0])
    for i, u in enumerate(U):
        r = calculate_gain(u, alpha_0=alpha_0, epsilon_q=epsilon_q,
                           M_MC=M_MC, kappa=kappa, order=order, seed=seed + i)
        scores[i] = r["gain"]
    return scores


def select_top1(U, use_fmis, **kw):
    scores = score_candidates(U, use_fmis, **kw)
    return int(np.argmax(scores)), scores


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _smoke_test():
    rng = np.random.RandomState(2024)
    c = 4
    found = False
    print("[smoke-test] searching for candidate set where FMIS on/off "
          "swaps the top-1 pick ...")
    for trial in range(200):
        # candidate pool: mix of (near-uniform) and (peaked) memberships
        U = rng.dirichlet(rng.uniform(0.2, 8.0, size=c), size=12)
        top_off, s_off = select_top1(U, use_fmis=False, seed=trial)
        top_on, s_on = select_top1(U, use_fmis=True, seed=trial)
        if top_off != top_on:
            found = True
            print(f"[smoke-test] trial={trial}: top1(use_fmis=False)={top_off} "
                  f"score={s_off[top_off]:.4f}")
            print(f"[smoke-test] trial={trial}: top1(use_fmis=True )={top_on} "
                  f"score={s_on[top_on]:.4f}")
            print(f"[smoke-test] scores(off)={np.round(s_off,4)}")
            print(f"[smoke-test] scores(on )={np.round(s_on,4)}")
            # rankings must differ as ordered lists
            rank_off = list(np.argsort(-s_off))
            rank_on = list(np.argsort(-s_on))
            assert rank_off != rank_on or top_off != top_on
            break
    assert found, "FMIS switch never changed the selection — Bug #3 is back"
    print("[smoke-test] FMIS flag genuinely changes ranking; PASSED")


if __name__ == "__main__":
    _smoke_test()
