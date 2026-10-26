"""watch_all.py - auto-restart experiment lines that go stale (no write for 30 min)
and whose process is gone. Idempotent (done-keys). Keeps E1/E2b/E2c alive."""
import os, sys, time, subprocess

ROOT = r"D:\AL_FuzzyClustering_Paper\v2_unified"
RES = os.path.join(ROOT, "results")
CODE = os.path.join(ROOT, "code")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(RES, "watch_all.log")
STALE = 30 * 60
COOLDOWN = 5 * 60

LINES = [
    ("E1",  os.path.join(RES, "synthetic", "done.txt"),
     [PY, os.path.join(CODE, "run_synthetic_sweep.py"), "--workers", "8"], 5760,
     "run_synthetic_sweep"),
    ("E2b", os.path.join(RES, "factor", "kappa_alpha.csv"),
     [PY, os.path.join(CODE, "run_factor_kappa_alpha.py")], 1920,
     "run_factor_kappa_alpha"),
    ("E2c", os.path.join(RES, "factor", "rho_beta_kappa.csv"),
     [PY, os.path.join(CODE, "run_factor_rho_beta_kappa.py")], 3240,
     "run_factor_rho_beta_kappa"),
]


def log(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def nrows(p):
    try:
        return sum(1 for _ in open(p, encoding="utf-8-sig", errors="ignore"))
    except Exception:
        return 0


def mtime(p):
    try:
        return os.path.getmtime(p)
    except Exception:
        return 0.0


def any_running(stem):
    try:
        import psutil
        for pr in psutil.process_iter(["cmdline"]):
            try:
                if stem in " ".join(pr.info["cmdline"] or []):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def main():
    log("watch_all started")
    cd = {}
    while True:
        for name, df, cmd, total, stem in LINES:
            rows = nrows(df)
            if rows >= total:
                continue
            age = time.time() - mtime(df)
            if age > STALE and not any_running(stem):
                if time.time() - cd.get(name, 0.0) < COOLDOWN:
                    continue
                log("RESTART %s rows=%d/%d age=%.0fmin" % (name, rows, total, age / 60))
                try:
                    subprocess.Popen(cmd, cwd=CODE, creationflags=0x00000008 | 0x00000200,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log("  spawned %s" % stem)
                except Exception as e:
                    log("  FAILED %s: %s" % (name, e))
                cd[name] = time.time()
            elif any_running(stem):
                cd[name] = 0.0
        time.sleep(60)


if __name__ == "__main__":
    main()
