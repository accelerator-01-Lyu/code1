import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_experiments import plcfcm_fit, _lambda2_over_n, INIT_FRAC, M_MC, KAPPA
from acquisition import select_batch, exploration_reward, cosine_redundancy, modular_scores
from gain import calculate_gain

for iid in (55, 61):
    dpath = os.path.join(os.path.dirname(HERE), "data", "synthetic", f"instance_{iid:03d}")
    X = np.load(os.path.join(dpath, "X.npy")); y = np.load(os.path.join(dpath, "y_true.npy"))
    meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
    c = meta["c"]; n = X.shape[0]
    rng = np.random.RandomState(0)
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool); labeled[rng.choice(n, size=n_init, replace=False)] = True
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0], labels=y[labeled], seed=0)
    unlab = np.where(~labeled)[0]
    beta = _lambda2_over_n(U)
    h_pool = exploration_reward(U[unlab], u_all=U)
    t_bar = float(h_pool.mean())
    kappa_s = KAPPA * (1.0 + beta * t_bar)
    print(f"iid={iid} ov={meta['overlap_target']} N={n} c={c}: beta={beta:.5f} Tbar={t_bar:.5f} "
          f"kappa_slack={kappa_s:.5f} kappa_full={KAPPA}", flush=True)
    # NoExploration selection (b=40)
    sel_ne = select_batch(U[unlab], X[unlab], V, "P2_NoExploration", 40,
                          alpha_0=50.0, M_MC=M_MC, rho_expl=0.0, lam_red=beta,
                          seed=0, u_all=U)
    # SlackUniform selection: custom loop
    mod = np.empty(len(unlab))
    for i, u in enumerate(U[unlab]):
        g = calculate_gain(u, alpha_0=50.0, epsilon_q=0.1, M_MC=M_MC,
                           kappa=kappa_s, order=2, seed=0 + i)
        mod[i] = g["w_robust"]
    R = cosine_redundancy(U[unlab])
    avail = np.ones(len(unlab), bool); red_sum = np.zeros(len(unlab)); sel_sl = []
    for _ in range(40):
        sc = mod - beta * red_sum; sc[~avail] = -np.inf
        j = int(np.argmax(sc)); sel_sl.append(j); avail[j] = False; red_sum += R[j]
    print(f"  NoExploration first-10 picks: {sel_ne[:10]}", flush=True)
    print(f"  SlackUniform   first-10 picks: {sel_sl[:10]}", flush=True)
    print(f"  overlap of batches: {len(set(sel_ne) & set(sel_sl))}/40", flush=True)
    # score-level check
    w_ne = modular_scores(U[unlab], "P2_NoExploration", alpha_0=50.0, M_MC=M_MC,
                          kappa=1.0, rho_expl=0.0, seed=0, u_all=U)
    print(f"  mean |mod_slack - mod_noexpl| = {np.abs(mod - w_ne).mean():.6e}  "
          f"max = {np.abs(mod - w_ne).max():.6e}", flush=True)
