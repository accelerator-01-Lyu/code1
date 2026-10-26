import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); SYN=os.path.join(HERE,"results","synthetic_3d")
FIG=os.path.join(HERE,"figures","posthoc")
INK="#3F345B"; GREY="#B8B8B8"
s=pd.read_csv(os.path.join(SYN,"summary_all.csv"))
b=pd.read_csv(os.path.join(SYN,"summary_swissroll_budget.csv"))
sw=s[s.dataset=="swissroll"].copy(); sw["budget"]=0.10
b2=pd.concat([b, sw[["sigma","method","seed","acc","nmi","ari","budget"]]],ignore_index=True)
fig,ax=plt.subplots(figsize=(88/25.4,60/25.4))
cols={0.4:"#7FA4C8",0.6:"#B59BC8",0.8:"#D08AB6"}
for sg in [0.4,0.6,0.8]:
    xs,mu,lo,hi=[],[],[],[]
    for bg in [0.02,0.03,0.10]:
        g=b2[np.isclose(b2.sigma,sg)&np.isclose(b2.budget,bg)]
        f=g[g.method=="P2_Full"].sort_values("seed"); n=g[g.method=="P2_NoExploration"].sort_values("seed")
        m=f.merge(n,on="seed",suffixes=("_f","_n")); d=m.acc_n.values-m.acc_f.values
        h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        xs.append(bg*100); mu.append(d.mean()*100); lo.append((d.mean()-h)*100); hi.append((d.mean()+h)*100)
    ax.plot(xs,mu,marker="o",ms=4,color=cols[sg],label=f"sigma={sg}",lw=1.4)
    ax.fill_between(xs,lo,hi,color=cols[sg],alpha=0.15,lw=0)
ax.axhline(0,color=INK,lw=0.8,ls="--")
ax.set_xlabel("labeled budget (%)",fontsize=8,color=INK); ax.set_ylabel("ACC gap NoExplo-Full (pp)",fontsize=8,color=INK)
ax.set_xticks([2,3,10]); ax.legend(fontsize=6,frameon=False,labelcolor=INK)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=7,colors=INK)
fig.suptitle("Swiss-roll ACC gap vs budget, 95% CI, n=10",fontsize=9,color=INK,weight="bold")
fig.tight_layout(rect=[0,0.09,1,0.93])
fig.text(0.01,0.02,"Post-hoc Mechanism Validation — not part of pre-registered P1-P4",fontsize=6,color=INK,alpha=0.8)
out=os.path.join(FIG,"synthetic_3d_swissroll_budget_curve.png")
fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white")
fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight",facecolor="white")
print(out)
