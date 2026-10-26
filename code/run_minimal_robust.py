"""run_minimal_robust.py — fixed."""
import os, glob, csv, numpy as np, pandas as pd
from multiprocessing import Pool
from run_experiments import run_traj, metrics, load_pkl, NCLASSES

SEEDS=[0,1,2,3,4]
DS=["ecoli","compound"]; METHODS=["P2_Full","P2_NoExploration","Random"]
ETAS=[0,.1,.2]; L0S=[.02,.05]
CODE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(CODE,"results","minimal_robust"); os.makedirs(OUT,exist_ok=True)
CSV=os.path.join(OUT,"summary_minimal.csv")
COLS=["dataset","eta","l0","method","seed","acc_noisy","acc_clean","source"]

def reuse(ds,m,s):
    for f in glob.glob(os.path.join(CODE,"results","shards","*.csv")):
        b=os.path.basename(f)
        if "diagnostics" in b: continue
        parts=b.replace(".csv","").split("_")
        if len(parts)<7: continue
        if parts[2]=="30" and parts[0]==ds and parts[1]==m and int(parts[3])==s:
            d=pd.read_csv(f); row=d[d.budget==.10].iloc[0]; return row.ACC
    return None

def one(job):
    ds,m,eta,l0,s=job
    if eta==0 and l0==.02:
        acc=reuse(ds,m,s); return dict(dataset=ds,eta=eta,l0=l0,method=m,seed=s,
                                       acc_noisy=acc,acc_clean=acc,source="main30_reuse")
    X,y=load_pkl(ds); c=NCLASSES[ds]
    out=run_traj(X,y,c,ds,m,s,0,eta,50.,1.,budgets=(.05,.10),l0_frac=l0,return_U=True)
    rows=out[0]; U=out[-1]
    return dict(dataset=ds,eta=eta,l0=l0,method=m,seed=s,
                acc_noisy=rows[-1]["ACC"],acc_clean=metrics(y,U)["ACC"],source="minimal_run")

if __name__=="__main__":
    done=set()
    if os.path.exists(CSV):
        for r in csv.DictReader(open(CSV)): done.add((r["dataset"],r["method"],r["eta"],r["l0"],r["seed"]))
    jobs=[]
    for ds in DS:
        for eta in ETAS:
            for l0 in L0S:
                for m in METHODS:
                    for s in SEEDS:
                        if (ds,m,eta,l0,str(s)) not in done: jobs.append((ds,m,eta,l0,s))
    f=open(CSV,"a",newline=""); w=csv.DictWriter(f,fieldnames=COLS)
    if os.path.getsize(CSV)==0: w.writeheader()
    with Pool(6) as p:
        for r in p.imap_unordered(one,jobs): w.writerow(r); f.flush()
    f.close(); open(os.path.join(CODE,"results",".minimal_robust_done.flag"),"w").close()
    print("minimal done")
