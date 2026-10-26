"""make_figures.py — layered paper figures."""
import os, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palette import COLORS, LW, LS, INK

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(FIG, "final"); os.makedirs(FINAL, exist_ok=True)
DATASETS = ["iris","wine","glass","ecoli","aggregation","compound"]
ALL9 = ["Random","Entropy","CoreSet","BADGE","OriginalP2",
        "P2_Full","P2_NoExploration","P2_NoRobust","P2_NoRedundancy"]
MAIN = ["OriginalP2","Entropy","CoreSet",
        "P2_NoRobust","P2_NoRedundancy","P2_Full","P2_NoExploration"]

frames = [pd.read_csv(f) for f in glob.glob(os.path.join(CODE,"results","shards","*.csv"))
          if "diagnostics" not in f]
long = pd.concat(frames, ignore_index=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                     "axes.edgecolor":INK,"axes.labelcolor":INK,
                     "xtick.color":INK,"ytick.color":INK,"text.color":INK})

def curve(methods, fname, width):
    fig, axes = plt.subplots(2,3, figsize=(width,5.2), sharex=True)
    for ax, ds in zip(axes.flat, DATASETS):
        for mth in methods:
            means=[]; los=[]; his=[]
            for bud in [.05,.10,.15,.20]:
                v = long[(long.dataset==ds)&(long.method==mth)&(long.budget==bud)].ACC.values
                if len(v)<5: continue
                m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); tc=stats.t.ppf(.975,len(v)-1)
                means.append(m); los.append(m-tc*se); his.append(m+tc*se)
            if not means: continue
            c=COLORS[mth]; lw=LW.get(mth,1.2); ls=LS.get(mth,"-")
            mk = "o" if mth in ("P2_NoExploration","P2_Full") else None
            ax.plot([5,10,15,20],means,color=c,lw=lw,ls=ls,marker=mk,ms=3,label=mth)
            ax.fill_between([5,10,15,20],los,his,color=c,alpha=.12)
        ax.set_title(ds, fontsize=9)
    axes.flat[0].legend(fontsize=5, ncol=2)
    for ax in axes[1,:]: ax.set_xlabel("budget (%)")
    for ax in axes[:,0]: ax.set_ylabel("ACC")
    fig.text(.01,.01,"Paired seeds; 95% CI shaded.",fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL,fname+".pdf"))
    plt.savefig(os.path.join(FINAL,fname+".png"),dpi=300)

curve(MAIN, "fig_learning_curves", 7.1)
curve(ALL9, "fig_learning_curves_all9", 7.1*180/88)
print("curves done")
