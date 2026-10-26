"""_watch_p3p4.py — auto-aggregate P3/P4 when shards complete."""
import os, time, json, subprocess, sys, glob
CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results")
LOG=os.path.join(RES,"watch_p3p4.log")
FLAG=os.path.join(RES,".p3p4_done.flag")

def log(msg):
    with open(LOG,"a",encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def count(tag):
    fs=glob.glob(os.path.join(RES,"shards","*.csv"))
    n=0
    for f in fs:
        if "diagnostics" in f: continue
        try:
            import pandas as pd
            df=pd.read_csv(f)
            if len(df) and str(df.seed_set.iloc[0])==tag: n+=1
        except: pass
    return n

if os.path.exists(FLAG):
    log("already done, exit"); sys.exit(0)

while True:
    p3=count("P3"); p4=count("P4")
    log(f"P3={p3}/1440 P4={p4}/720")
    if p3>=1440 and p4>=720:
        log("both complete, aggregating")
        subprocess.run([sys.executable,os.path.join(CODE,"run_experiments.py"),
                        "--phase","aggregate","--only","P3,P4"],check=True)
        open(FLAG,"w").close()
        log("done flag written"); break
    time.sleep(300)
