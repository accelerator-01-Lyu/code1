"""Quick profile: per-stage timing for small vs large instances."""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_experiments import run_traj, metrics, _lambda2_over_n, INIT_FRAC, M_MC, KAPPA, RHO
from fcm_update import plcfcm_fit
from acquisition import exploration_reward, cosine_redundancy
from gain import calculate_gain

ROOT = os.path.dirname(HERE)

def profile_instance(iid, method, seed=0):
    dpath = os.path.join(ROOT, "data", "synthetic", f"instance_{iid:03d}")
    X = np.load(os.path.join(dpath, "X.npy"))
    y = np.load(os.path.join(dpath, "y_true.npy"))
    meta = json.load(open(os.path.join(dpath, "meta.json"), encoding="utf-8"))
    c = meta["c"]; N = meta["N"]; d = meta["d"]
    print(f"\n=== iid={iid} N={N} d={d} c={c} method={method} seed={seed} ===", flush=True)

    t0 = time.time()
    rng = np.random.RandomState((seed * 1000 + 0) & 0x7FFFFFFF)
    n = X.shape[0]
    n_init = max(c, int(round(INIT_FRAC * n)))
    labeled = np.zeros(n, bool)
    labeled[rng.choice(n, size=n_init, replace=False)] = True
    t_rng = time.time() - t0

    t0 = time.time()
    U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                         labels=y[labeled], seed=seed)
    t_fcm0 = time.time() - t0

    budgets = (0.05, 0.10, 0.15, 0.20)
    t_modular_total = 0.0
    t_greedy_total = 0.0
    t_fcm_refit_total = 0.0
    t_metrics_total = 0.0
    n_modular_calls = 0

    for bud in budgets:
        target = int(round(bud * n))
        if labeled.sum() >= target:
            continue
        b = target - int(labeled.sum())
        unlab = np.where(~labeled)[0]
        beta = _lambda2_over_n(U)

        t0 = time.time()
        if method == "P2_Full":
            h_pool = exploration_reward(U[unlab], u_all=U)
            mod = np.empty(len(unlab))
            for i, u in enumerate(U[unlab]):
                g = calculate_gain(u, alpha_0=50.0, epsilon_q=0.1, M_MC=M_MC,
                                   kappa=KAPPA, order=2, seed=seed + i)
                mod[i] = g["w_robust"] + RHO * g.get("expl", 0.0)
                n_modular_calls += 1
        elif method == "P2_NoExploration":
            mod = np.empty(len(unlab))
            for i, u in enumerate(U[unlab]):
                g = calculate_gain(u, alpha_0=50.0, epsilon_q=0.1, M_MC=M_MC,
                                   kappa=KAPPA, order=2, seed=seed + i)
                mod[i] = g["w_robust"]
                n_modular_calls += 1
        elif method == "SlackUniform":
            h_pool = exploration_reward(U[unlab], u_all=U)
            t_bar = float(h_pool.mean())
            kappa_s = KAPPA * (1.0 + beta * t_bar)
            mod = np.empty(len(unlab))
            for i, u in enumerate(U[unlab]):
                g = calculate_gain(u, alpha_0=50.0, epsilon_q=0.1, M_MC=M_MC,
                                   kappa=kappa_s, order=2, seed=seed + i)
                mod[i] = g["w_robust"]
                n_modular_calls += 1
        t_modular = time.time() - t0
        t_modular_total += t_modular

        t0 = time.time()
        R = cosine_redundancy(U[unlab])
        avail = np.ones(len(unlab), dtype=bool)
        red_sum = np.zeros(len(unlab))
        sel_local = []
        for _ in range(min(b, len(unlab))):
            scores = mod - beta * red_sum
            scores[~avail] = -np.inf
            j = int(np.argmax(scores))
            if scores[j] == -np.inf: break
            sel_local.append(j); avail[j] = False; red_sum += R[j]
        sel = unlab[sel_local]
        t_greedy = time.time() - t0
        t_greedy_total += t_greedy

        t0 = time.time()
        labeled[sel] = True
        U, V, _ = plcfcm_fit(X, c=c, labeled_idx=np.where(labeled)[0],
                             labels=y[labeled], seed=seed)
        t_fcm_refit = time.time() - t0
        t_fcm_refit_total += t_fcm_refit

        t0 = time.time()
        _ = metrics(y, U)
        t_metrics_total += time.time() - t0

        print(f"  bud={bud}: modular={t_modular:.2f}s ({n_modular_calls} cum calls) "
              f"greedy={t_greedy:.3f}s fcm_refit={t_fcm_refit:.2f}s", flush=True)

    total = t_rng + t_fcm0 + t_modular_total + t_greedy_total + t_fcm_refit_total + t_metrics_total
    print(f"  TOTAL={total:.1f}s  rng={t_rng:.3f} fcm0={t_fcm0:.1f} "
          f"modular={t_modular_total:.1f}({100*t_modular_total/total:.0f}%) "
          f"greedy={t_greedy_total:.2f}({100*t_greedy_total/total:.0f}%) "
          f"fcm_refit={t_fcm_refit_total:.1f}({100*t_fcm_refit_total/total:.0f}%) "
          f"metrics={t_metrics_total:.2f}", flush=True)

for iid, method in [(1, "P2_NoExploration"), (1, "SlackUniform"),
                      (12, "P2_NoExploration"), (12, "SlackUniform")]:
    profile_instance(iid, method, seed=0)
