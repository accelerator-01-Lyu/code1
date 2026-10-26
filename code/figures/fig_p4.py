"""fig_p4.py"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(CODE,"results"); FIG=os.path.join(CODE,"figures","final")
DS=["iris","wine","glass","ecoli","aggregation","compound"]
ALPHAS=[.1,1,10,100]
frames=[pd.read_csv(f) for f in glob.glob(os.path.join(RES,"shards","*.csv")) if "diagnostics" not in f]
long=pd.concat(frames,ignore_index=True)
p4=long[long.seed_set.astype(str)=="P4"]
fig,axes=plt.subplots(2,3,figsize=(7.1,4))
for ax,ds in zip(axes.flat,DS):
    means=[];los=[];his=[]
    for a in ALPHAS:
        v=p4[(p4.dataset==ds)&(p4.alpha0==a)&(p4.budget==.10)].ACC.values
        m=v.mean();se=v.std(ddof=1)/np.sqrt(len(v));tc=1.96
        means.append(m);los.append(m-tc*se);his.append(m+tc*se)
    ax.axhspan(means[1]-.005,means[1]+.005,color="#DDE6F3",alpha=.5)
    ax.plot(ALPHAS,means,color="#6A5A8F",lw=1.5,marker="o",ms=3)
    ax.fill_between(ALPHAS,los,his,color="#6A5A8F",alpha=.12)
    ax.set_xscale("log"); ax.set_title(ds,fontsize=7)
for ax in axes[1,:]: ax.set_xlabel("alpha0")
for ax in axes[:,0]: ax.set_ylabel("ACC@10%")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"fig_p4_alpha0.pdf")); plt.savefig(os.path.join(FIG,"fig_p4_alpha0.png"),dpi=300)
print("p4 fig done")
