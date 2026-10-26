# Compound-only parallel driver for the minimal noise/L0 robustness subset.
# Writes a SEPARATE csv (summary_minimal_compound.csv) so it never concurrently
# appends to the ecoli scheduler's summary_minimal.csv. Merge + de-dup later by
# primary key (dataset, method, eta, l0, seed). POST-HOC / non-preregistered.
import os, csv
from multiprocessing import Pool
import run_minimal_robust as R   # import only defines one()/constants; its Pool is under __main__, so this launches nothing

CODE, one, COLS = R.CODE, R.one, R.COLS
SEEDS, METHODS, ETAS, L0S = R.SEEDS, R.METHODS, R.ETAS, R.L0S
OUT = R.OUT
CSV = os.path.join(OUT, "summary_minimal_compound.csv")
DS = "compound"
NW = 6

if __name__ == "__main__":
    done = set()
    if os.path.exists(CSV):
        for r in csv.DictReader(open(CSV, newline="")):
            done.add((r["dataset"], r["method"], r["eta"], r["l0"], r["seed"]))
    jobs = []
    for eta in ETAS:
        for l0 in L0S:
            for m in METHODS:
                for s in SEEDS:
                    if (DS, m, str(eta), str(l0), str(s)) not in done:
                        jobs.append((DS, m, eta, l0, s))
    is_new = (not os.path.exists(CSV)) or os.path.getsize(CSV) == 0
    f = open(CSV, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS)
    if is_new:
        w.writeheader()
    with Pool(NW) as p:
        for r in p.imap_unordered(one, jobs):
            w.writerow(r); f.flush()
    f.close()
    open(os.path.join(CODE, "results", ".minimal_compound_done.flag"), "w").close()
    print("compound minimal done:", len(jobs), "new shards")
