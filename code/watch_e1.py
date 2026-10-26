"""watch_e1.py — wait for the E1 sweep to finish, then run analysis.

Polls results/synthetic/COMPLETED.flag every 60 s; when it appears, runs
analyze_synthetic.py and tees its output to results/synthetic/analyze.log.
Idempotent: safe to rerun.
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "synthetic")
FLAG = os.path.join(RES, "COMPLETED.flag")
DONE = os.path.join(RES, "done.txt")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

t0 = time.time()
while not os.path.exists(FLAG):
    time.sleep(60)
    try:
        n = sum(1 for _ in open(DONE, encoding="utf-8"))
    except OSError:
        n = -1
    print(f"[watch] {time.strftime('%H:%M:%S')} done={n}/5760 elapsed={(time.time()-t0)/60:.0f}min",
          flush=True)

print("[watch] COMPLETED.flag found; running analysis...", flush=True)
with open(os.path.join(RES, "analyze.log"), "w", encoding="utf-8") as lf:
    r = subprocess.run([PY, os.path.join(HERE, "analyze_synthetic.py")],
                       cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
print(f"[watch] analysis exit={r.returncode}", flush=True)
