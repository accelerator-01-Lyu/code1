# POST-HOC EXPLORATORY DIAGNOSTIC. NOT PART OF PRE-REGISTERED P1-P4 VERDICTS.
"""gradient_inner_product.py — Eq.11 mechanism check on selected points.

s_i = <grad_h_i, grad_w_i^rob> on SELECTED points, ecoli/glass, 30 seeds.
Two files: gradient_inner_product.csv (P2_Full) and
gradient_inner_product_nonexplo.csv (P2_NoExploration), each with summary.
Does NOT read or write prereg_verdicts.json.

Interpretation (docstring):
  - s predominantly negative => Thm2 mechanism holds, P2 tests wrong points.
  - s predominantly positive => theory crisis (Scenario B, narrow Thm2 scope).
"""
import os, numpy as np, pandas as pd
from fcm_update import plcfcm_fit
from acquisition import select_batch, exploration_reward
from run_experiments import load_pkl, NCLASSES, INIT_FRAC, M_MC, RHO, _lambda2_over_n
import exploratory
exploratory.M_DIAG = 50   # reduced for speed; flagged post-hoc
from exploratory import run_x2_diag

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run(ds, method, seed):
    X, y = load_pkl(ds); c = NCLASSES[ds]
    rng = np.random.RandomState(seed)   # SAME as run_traj
    n = len(X); ni = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool); labeled[rng.choice(n, ni, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c, labeled_idx=np.where(labeled)[0],
                          labels=y[labeled], seed=seed)
    unlab0 = ~labeled
    H0 = -(np.clip(U, 1e-12, 1) * np.log(np.clip(U, 1e-12, 1))).sum(1)
    h0 = exploration_reward(U, u_all=U)
    qHhi, qHlo = np.quantile(H0[unlab0], [0.75, 0.25])
    qhhi, qhlo = np.quantile(h0[unlab0], [0.75, 0.25])
    rows = []
    for bud in (0.05, 0.10, 0.15, 0.20):
        target = int(round(bud * n))
        while labeled.sum() < target:
            unlab = np.where(~labeled)[0]
            b = target - int(labeled.sum())
            beta = 1.0 * _lambda2_over_n(U)
            sel = select_batch(U[unlab], X[unlab], V, method, b,
                                alpha_0=50.0, M_MC=M_MC, rho_expl=RHO,
                                lam_red=beta, seed=seed, u_all=U)
            sel = unlab[sel]
            di = run_x2_diag(U[sel], U, alpha0=50.0, kappa=1.0, seed=seed)
            for k, idx in enumerate(sel):
                # tangent-space grad_h norm (same projection as s)
                gh = (1 - 1/n) * np.sign(U[idx] - U.mean(0))
                gh_tan = gh - gh.mean()
                rows.append({"dataset": ds, "seed": seed, "budget": bud,
                             "selected_idx": int(idx), "point_class": di["label"][k],
                             "grad_h_norm": float(np.linalg.norm(gh_tan)),
                             "grad_w_norm": float(di["normw"][k]),
                             "inner_product": float(di["s"][k]),
                             "sign": int(np.sign(di["s"][k]))})
            labeled[sel] = True
            U, V, _ = plcfcm_fit(X, c, labeled_idx=np.where(labeled)[0],
                                  labels=y[labeled], seed=seed)
    return rows


if __name__ == "__main__":
    for method, suffix in (("P2_Full", ""), ("P2_NoExploration", "_nonexplo")):
        all_rows = []
        for ds in ("ecoli", "glass"):
            for seed in range(30):
                all_rows += run(ds, method, seed)
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(OUT, f"gradient_inner_product{suffix}.csv"), index=False)
        summ = df.groupby(["dataset", "budget", "point_class"]).agg(
            n=("sign", "size"),
            frac_s_neg=("sign", lambda s: float((s < 0).mean())),
            mean_s=("inner_product", "mean")).reset_index()
        summ.to_csv(os.path.join(OUT, f"gradient_inner_product{suffix}_summary.csv"),
                    index=False)
        print(f"[grad{suffix}] wrote", len(df), "rows")
