"""run_noise_n30.py — n=30 noise robustness power-thickening (post-hoc)."""
import os, csv, numpy as np, pandas as pd
from multiprocessing import Pool
from run_experiments import run_traj, load_pkl, NCLASSES

SEEDS=list(range(20261010,20261040))
DS=["ecoli","compound"]; ETAS=[0,.1,.2]; L0S=[.02,.05]; METHODS=["P2_Full","P2_NoExploration"]
CODE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(CODE,"results","noise"); os.makedirs(OUT,exist_ok=True)
CSV=os.path.join(OUT,"noise_full_n30.csv")
COLS=["dataset","eta","L0_pct","seed","method","acc","nmi","ari"]

def one(job):
    ds,m,eta,l0,s=job
    X,y=load_pkl(ds); c=NCLASSES[ds]
    out=run_traj(X,y,c,ds,m,s,0,eta,50.,1.,budgets=(.05,.10),l0_frac=l0,run_x2=False)
    rows=out[0]; r=rows[-1]
    return dict(dataset=ds,eta=eta,L0_pct=l0,seed=s,method=m,
                acc=r["ACC"],nmi=r["NMI"],ari=r["ARI"])

if __name__=="__main__":
    done=set()
    if os.path.exists(CSV):
        for r in csv.DictReader(open(CSV)): done.add((r["dataset"],r["method"],r["eta"],r["L0_pct"],r["seed"]))
    jobs=[]
    for ds in DS:
        for eta in ETAS:
            for l0 in L0S:
                for m in METHODS:
                    for s in SEEDS:
                        if (ds,m,str(eta),str(l0),str(s)) not in done: jobs.append((ds,m,eta,l0,s))
    f=open(CSV,"a",newline=""); w=csv.DictWriter(f,fieldnames=COLS)
    if os.path.getsize(CSV)==0: w.writeheader()
    with Pool(14) as p:
        for r in p.imap_unordered(one,jobs,chunksize=1): w.writerow(r); f.flush()
    f.close(); print("done")
