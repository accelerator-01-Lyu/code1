# POST-HOC MECHANISM VALIDATION — not part of pre-registered P1-P4 verdicts.
"""_run_posthoc_now.py — run the 180-shard 3D post-hoc batch NOW (parallel with
P4), then figures A/B/D, README, and .synthetic_done.flag.  Logs to run.log.
Read-only on main-experiment state; writes only results/synthetic_3d/ and its
own flag.
"""
import os, sys, datetime, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SYN = os.path.join(RES, "synthetic_3d")
LOG = os.path.join(SYN, "run.log")


def log(m):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("=== post-hoc run starting ===")
    import run_synthetic_3d as R
    R.run_synthetic()
    log("batch done; figures A/B/D")
    import make_figures_posthoc as F
    try:
        F.fig_a(); F.fig_b(); F.fig_d()
    except Exception:
        log("FIG ERROR\n" + traceback.format_exc())
    log("write README")
    # reuse supervisor README writer
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sup", os.path.join(HERE, "_supervise_posthoc.py"))
    sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)
    sup.write_readme()
    with open(os.path.join(RES, ".synthetic_done.flag"), "w") as f:
        f.write(datetime.datetime.now().isoformat())
    log("=== ALL DONE; .synthetic_done.flag written ===")


if __name__ == "__main__":
    main()
