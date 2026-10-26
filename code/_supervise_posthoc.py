# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""_supervise_posthoc.py — detached watchdog for the isolated 3D post-hoc run.

LIFECYCLE: started detached via run_d.bat; survives the agent session.  Every
60 s it reads results/.main_worker1.flag (read-only).  Once the flag appears it
runs, exactly once and idempotently:
   1. run_synthetic_3d.py --run        (180 shards, NaN rows kept, resumable)
   2. make_figures_posthoc.py --only A/B/D   (C already rendered)
   3. write synthetic_3d/README.md
   4. write results/.synthetic_done.flag
All activity logged to results/synthetic_3d/supervisor.log.

ISOLATION: this script only READS .main_worker1.flag and WRITES under
results/synthetic_3d/ plus its own .synthetic_done.flag.  It never touches
prereg_verdicts.json / shards/ / done.txt.
"""
import os
import sys
import time
import subprocess
import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SYN = os.path.join(RES, "synthetic_3d")
MAIN_FLAG = os.path.join(RES, ".main_worker1.flag")
DONE_FLAG = os.path.join(RES, ".synthetic_done.flag")
LOG = os.path.join(SYN, "supervisor.log")
PY = sys.executable


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(argv, tag):
    log(f"START {tag}: {' '.join(argv)}")
    p = subprocess.run([PY] + argv, cwd=HERE, capture_output=True, text=True)
    log(f"END {tag} rc={p.returncode}")
    if p.stdout:
        log("--- " + tag + " stdout tail ---\n" + p.stdout[-3000:])
    if p.stderr:
        log("--- " + tag + " stderr tail ---\n" + p.stderr[-3000:])
    return p.returncode


def write_readme():
    import pandas as pd
    from scipy import stats
    path = os.path.join(SYN, "README.md")
    summ = pd.read_csv(os.path.join(SYN, "summary_all.csv"))
    ok = summ[summ.acc != "NaN"].copy()
    ok["acc"] = ok.acc.astype(float)
    L = []
    L.append("# Post-hoc 3-D Mechanism Validation")
    L.append("")
    L.append("**Post-hoc Mechanism Validation — not part of pre-registered "
             "P1-P4 verdicts.** Isolated 3-D synthetic follow-up to sanity-check "
             "Theorem 2 (double counting) and Proposition 2 (Fisher-Rao boundary "
             "detection). Does NOT alter or feed the six real-dataset pre-registered "
             "results.")
    L.append("")
    L.append("## Frozen protocol")
    L.append("- Data-gen seed 20261010 (independent of AL seeds).")
    L.append("- AL seeds locked 20261010..20261019 (10 seeds, no cherry-picking).")
    L.append("- P2_Full (rho=0.1 code default, untouched) vs P2_NoExploration "
             "(rho=0 forced inside select_batch).")
    L.append("- Budget 10% of 1200 normal points (=120 labeled incl. "
             "|L0|=max(c,floor(.02N))=24); one greedy batch per checkpoint, "
             "mirroring run_experiments.run_traj (M_MC=50, u_all=U, "
             "lam_red=lambda2/n).")
    L.append("- Eccentric decoys y=-1 (10/dataset, fixed coords) enter FCM fitting "
             "and acquisition; if selected pinned to current fuzzy winner cluster; "
             "EXCLUDED from ACC/NMI/ARI. Hungarian on 6 normal clusters only.")
    L.append("")
    L.append("## Reproduce")
    L.append("```")
    L.append("cd code")
    L.append("run_d.bat make_synthetic_3d.py")
    L.append("run_d.bat run_synthetic_3d.py --run")
    L.append("run_d.bat make_figures_posthoc.py")
    L.append("```")
    L.append("Smoke: `run_d.bat run_synthetic_3d.py --smoke` -> max abs diff vs "
             "main ecoli shards = 0.0.")
    L.append("")
    L.append("## Manipulation check (honest)")
    L.append("- Spiral / Swiss-roll: decoys low-entropy, high-eccentricity "
             "(h_i 2-3x boundary), umax~0.55-0.81.")
    L.append("- umax does NOT reach ~0.95: FCM m=2 spreads out-of-cloud points. "
             "Points were NOT moved to force 0.95.")
    L.append("- **Hourglass = NEGATIVE geometry control (Theorem-2 scope boundary).** "
             "One principled geometric correction still failed: after the upper-top "
             "band decoy was pushed along +z away from the z=0 junction, umax stayed "
             "~0.32-0.39 (boundary controls 0.22-0.28) and H stayed ~1.5+; no "
             "separation. Locked geometric reason: the hourglass cone-top bands are "
             "RINGS and the two cones are symmetric about the apex z=0, so an outward "
             "point is nearly equidistant to MULTIPLE ring points; under FCM m=2 its "
             "membership is shared by symmetry and no local single-cluster-dominant "
             "eccentric target forms. This DELIMITs the scope of Theorem 2: the "
             "double-counting pathology requires a LOCAL, single-cluster-dominant, "
             "ASYMMETRIC cluster structure. It did NOT manifest for hourglass. All 60 "
             "hourglass shards are kept in the supplement (negative result, not "
             "deleted). Main-text selection figure uses spiral/sigma=.6 and "
             "swissroll/sigma=.6 only; the hourglass selection map appears only as "
             "this negative supplement.")
    L.append("")
    L.append("## Results (n=10, paired ACC(NoExplo)-ACC(Full), 95% CI)")
    L.append("")
    L.append("| dataset | sigma | mean gap | CI lo | CI hi |")
    L.append("|---|---|---|---|---|")
    for (kind, s), g in ok.groupby(["dataset", "sigma"]):
        f = g[g.method == "P2_Full"].sort_values("seed")
        ne = g[g.method == "P2_NoExploration"].sort_values("seed")
        m = f.merge(ne, on="seed", suffixes=("_f", "_ne"))
        d = m.acc_ne.values - m.acc_f.values
        if len(d) >= 2:
            h = stats.t.ppf(0.975, len(d)-1) * d.std(ddof=1)/np.sqrt(len(d))
            L.append(f"| {kind} | {s} | {d.mean():.4f} | "
                     f"{d.mean()-h:.4f} | {d.mean()+h:.4f} |")
        else:
            L.append(f"| {kind} | {s} | n<2 | - | - |")
    L.append("")
    L.append("All sigma x seed combinations retained in summary_all.csv / "
             "trajectory_all.csv, including NaN / non-converged rows (none "
             "deleted, no seed picking).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("wrote " + path)


def main():
    os.makedirs(SYN, exist_ok=True)
    log("supervisor up; polling for .main_worker1.flag")
    while not os.path.exists(MAIN_FLAG):
        if os.path.exists(DONE_FLAG):
            log(".synthetic_done.flag already present; exiting.")
            return
        time.sleep(60)
    log("main_worker1.flag detected; starting post-hoc pipeline")
    rc1 = run_step(["run_synthetic_3d.py", "--run"], "batch")
    rc2 = run_step(["make_figures_posthoc.py", "--only", "A"], "figA")
    rc3 = run_step(["make_figures_posthoc.py", "--only", "B"], "figB")
    rc4 = run_step(["make_figures_posthoc.py", "--only", "D"], "figD")
    write_readme()
    with open(DONE_FLAG, "w") as f:
        f.write(datetime.datetime.now().isoformat())
    log(f"pipeline done rc batch/figA/figB/figD = "
        f"{rc1}/{rc2}/{rc3}/{rc4}; wrote .synthetic_done.flag")


if __name__ == "__main__":
    main()
