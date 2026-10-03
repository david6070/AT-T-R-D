"""make_figures.py -- every manuscript figure, drawn from the Juno result
tables in res/results/rev/ (no statistics are recomputed here).
Figures are drawn at their printed width so label sizes are true."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
R="res/results/rev/"; OUT="figures/"
plt.rcParams.update({"font.size":10,"axes.titlesize":10,"axes.labelsize":10,"xtick.labelsize":9,
    "ytick.labelsize":9,"legend.fontsize":9,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42})
C={"base":"#0072B2","p1":"#E69F00","p3":"#009E73"}          # colour-blind safe
MK={"base":"o","p1":"s","p3":"^"}
NM={"base":"LSTM baseline","p1":"Personalised, 1 branch","p3":"Personalised, 3 branches"}
SH={"base":"Baseline","p1":"1 branch","p3":"3 branches"}
W=6.3   # inches; printed at \textwidth
def save(f,n): f.savefig(OUT+n+".pdf",bbox_inches="tight"); plt.close(f)
pairs=pd.read_csv(R+"dt_pairs_expected.csv")

# 1 F1
t=pd.read_csv(R+"T1_f1.csv").set_index("config")
f,a=plt.subplots(figsize=(4.6,3.0),constrained_layout=True)
x=np.arange(3)
a.bar(x,[t.micro[c] for c in NM],yerr=[t.micro_sd[c] for c in NM],capsize=4,color=[C[c] for c in NM],width=.6)
for y,l in ((.47,"published ≈ 0.47"),(.60,"published ≈ 0.60")):
    a.axhline(y,ls="--",color="0.35",lw=1); a.text(2.45,y,l.replace("published ","published\n"),ha="left",va="center",fontsize=8.5,color="0.25")
a.set_xticks(x,["LSTM\nbaseline","Personalised\n1 branch","Personalised\n3 branches"]); a.set_ylabel("Micro-F1"); a.set_ylim(.3,.7); a.set_xlim(-.5,3.25)
save(f,"fig1_f1_comparison")

# 2 histograms (stacked rows: readable at page width)
f,ax=plt.subplots(3,1,figsize=(W*0.8,6.0),sharex=True,sharey=True,constrained_layout=True)
bins=np.arange(-72,76,6)
for a,c in zip(ax,NM):
    d=pairs.loc[pairs.config==c,"dt"].to_numpy()
    a.hist(d[d<0],bins=bins,color="#009E73",label="ΔT < 0 (before onset)")
    a.hist(d[d>=0],bins=bins,color="#D55E00",label="ΔT ≥ 0 (at or after onset)")
    a.axvline(np.median(d),color="k",ls="--",lw=1)
    a.set_title(f"{NM[c]}:  N = {len(d)},  anticipation {100*(d<0).mean():.1f}%",loc="left")
    a.set_ylabel("Matched pairs")
ax[0].legend(frameon=False,loc="upper left"); ax[-1].set_xlabel("ΔT (hours)")
save(f,"fig2_delta_t_hist")

# 3 KDE
f,a=plt.subplots(figsize=(4.8,3.0),constrained_layout=True)
xs=np.linspace(-72,72,400)
for c,ls in zip(NM,["-","--",":"]):
    a.plot(xs,stats.gaussian_kde(pairs.loc[pairs.config==c,"dt"])(xs),color=C[c],ls=ls,lw=1.8,label=NM[c])
a.axvline(0,color="0.5",lw=.8); a.set_xlabel("ΔT (hours)"); a.set_ylabel("Density"); a.legend(frameon=False,loc="upper right",fontsize=8)
save(f,"fig3_delta_t_kde")

# 4 nulls: one row per configuration, observed vs each null interval
n=pd.read_csv(R+"T5_nulls.csv").set_index("config")
f,ax=plt.subplots(1,2,figsize=(W,3.1),constrained_layout=True,sharey=True)
ys=np.arange(3)[::-1]
spec=[("ant","Anticipation rate (%)"),("align","Within ±24 h of an onset (%)")]
for a,(k,xl) in zip(ax,spec):
    for y,c in zip(ys,NM):
        for dy,nm,col,lab in ((.18,"random_cp","#56B4E9","Null: random change points"),(-.18,"shifted_labels","#CC79A7","Null: shifted labels")):
            m,lo,hi=n.loc[c,f"{nm}_{k}_mean"],n.loc[c,f"{nm}_{k}_lo"],n.loc[c,f"{nm}_{k}_hi"]
            a.plot([lo,hi],[y+dy,y+dy],color=col,lw=5,solid_capstyle="butt",alpha=.75,label=lab if y==ys[0] else None)
            a.plot(m,y+dy,"|",color="k",ms=9,mew=1.2)
        a.plot(n.loc[c,f"obs_{k}"],y,"*",color="k",ms=12,label="Observed" if y==ys[0] else None,zorder=5)
    a.set_yticks(ys,[SH[c] for c in NM]); a.set_xlabel(xl); a.set_ylim(-.6,2.6); a.grid(axis="x",color="0.9")
h,l=ax[0].get_legend_handles_labels()
f.legend(h,l,loc="outside lower center",ncol=3,frameon=False)
save(f,"fig7_null_models")

# 5 paired differences
pr=pd.read_csv(R+"T3_paired_tost.csv")
f,a=plt.subplots(figsize=(4.8,2.5),constrained_layout=True)
y=np.arange(len(pr))[::-1]; m=pr.tost_margin[0]
a.axvspan(-m,m,color="#56B4E9",alpha=.25,label=f"±{m:g} h equivalence margin")
a.errorbar(pr.mean_diff,y,xerr=[pr.mean_diff-pr.ci95_lo,pr.ci95_hi-pr.mean_diff],fmt="o",color="k",capsize=4,ms=6)
a.axvline(0,color="0.4",lw=.8)
a.set_yticks(y,["Baseline − 1 branch","Baseline − 3 branches","1 branch − 3 branches"]); a.set_ylim(-.6,2.6)
a.set_xlim(-7.5,7.5); a.set_xlabel("Paired mean ΔT difference (hours)")
f.legend(loc="outside upper center",frameon=False)
save(f,"fig4_paired_differences")

# 6 fixed k
d=pd.read_csv(R+"T4_fixed_k.csv")
f,ax=plt.subplots(1,2,figsize=(W,2.9),constrained_layout=True)
for i,c in enumerate(NM):
    g=d[d.config==c]
    ax[0].errorbar(g.k+(i-1)*.1,g.ant_rep_mean,yerr=g.ant_rep_sd,fmt=MK[c]+"-",color=C[c],capsize=3,label=NM[c])
    ax[1].plot(g.k+(i-1)*.1,g["mean"],MK[c]+"-",color=C[c])
ax[0].set_ylabel("Anticipation rate (%)"); ax[1].set_ylabel("Mean ΔT (hours)"); ax[1].set_ylim(0,12)
for a in ax: a.set_xticks([1,2,3]); a.set_xlabel("Change points per subject (k)")
h,l=ax[0].get_legend_handles_labels(); f.legend(h,l,loc="outside lower center",ncol=3,frameon=False)
save(f,"fig5_fixed_k")

# 7 asymmetric
t7=pd.read_csv(R+"T7_asymmetric.csv")
f,ax=plt.subplots(1,2,figsize=(W,2.9),constrained_layout=True)
o=["base","base_asym05","base_asym10"]; lab=["Symmetric","λ = 0.5","λ = 1.0"]; cc=["0.35","#56B4E9","#D55E00"]; lss=["-","--",":"]
for c,l,k,ls in zip(o,lab,cc,lss):
    v=pairs.loc[pairs.config==c,"dt"]; hh,_=np.histogram(v,bins=bins)
    ax[0].step(bins[:-1],100*hh/len(v),where="post",color=k,ls=ls,lw=1.6,label=l)
ax[0].axvline(0,color="0.6",lw=.8); ax[0].set_xlabel("ΔT (hours)"); ax[0].set_ylabel("Matched pairs (%)"); ax[0].legend(frameon=False)
ax[1].bar(range(3),t7.macro_f1,yerr=t7.macro_f1_sd,capsize=4,color=cc,width=.6)
ax[1].set_xticks(range(3),lab); ax[1].set_ylabel("Macro-F1"); ax[1].set_ylim(.25,.45)
save(f,"fig6_asymmetric_loss")

# 8 horizon: F1; anticipation rate, forecast-trained vs present-state on the SAME sequences; onset-paired difference
hz=pd.read_csv(R+"T8_horizon.csv"); sd=pd.read_csv(R+"T8d_same_sequence.csv")
f,ax=plt.subplots(1,3,figsize=(W*1.12,3.0),constrained_layout=True)
for k,fam in enumerate(("base","p3")):
    g=hz[hz.family==fam].sort_values("h"); b=sd[sd.family==fam].sort_values("h"); off=(k-.5)*.14
    ax[0].errorbar(g.h+off*.6,g.micro,yerr=g.micro_sd,fmt=MK[fam]+"-",color=C[fam],capsize=3,label=NM[fam])
    ax[1].plot(b.h,b.ant_forecast,MK[fam]+"-",color=C[fam])
    ax[1].plot(b.h,b.ant_present,MK[fam]+"--",color=C[fam],mfc="white",label="Present-state model, same sequences" if k==0 else None)
    ax[2].errorbar(b.h+off,b.paired_mean,yerr=[b.paired_mean-b.paired_lo,b.paired_hi-b.paired_mean],fmt=MK[fam],color=C[fam],capsize=3)
ax[2].axhspan(-6,6,color="#56B4E9",alpha=.2,label="±6 h equivalence margin"); ax[2].axhline(0,color="0.4",lw=.8); ax[2].set_ylim(-10,10)
ax[0].set_ylabel("Micro-F1"); ax[0].set_xticks([0,1,2,3])
ax[1].set_ylabel("Anticipation rate (%)"); ax[1].set_ylim(20,40)
ax[2].set_ylabel("Forecast − present, paired\nmean ΔT difference (h)")
for a in ax[1:]: a.set_xticks([1,2,3])
for a in ax: a.set_xlabel("Forecast horizon")
h1,l1=ax[0].get_legend_handles_labels(); h2,l2=ax[1].get_legend_handles_labels(); h3,l3=ax[2].get_legend_handles_labels()
f.legend(h1+h2+h3,l1+l2+l3,loc="outside lower center",ncol=2,frameon=False,fontsize=8.5)
save(f,"fig8_horizon")

# 9 grid (stacked rows so cell text is legible)
g0=pd.read_csv(R+"T6_sensitivity_grid.csv"); g0=g0[g0.pen<5]
f,ax=plt.subplots(3,1,figsize=(W*0.8,6.0),constrained_layout=True,sharex=True)
for a,c in zip(ax,NM):
    g=g0[g0.config==c].pivot(index="pen",columns="win",values="ant"); nn=g0[g0.config==c].pivot(index="pen",columns="win",values="N")
    im=a.imshow(g.values,cmap="cividis",vmin=20,vmax=34,aspect="auto")
    a.set_xticks(range(4),[f"{w:g}" for w in g.columns]); a.set_yticks(range(3),[f"{q:g}" for q in g.index])
    for i in range(3):
        for j in range(4):
            v=g.values[i,j]; a.text(j,i,f"{v:.1f}%\nN = {int(nn.values[i,j])}",ha="center",va="center",fontsize=8.5,color="w" if v<28 else "k")
    a.set_title(NM[c],loc="left"); a.set_ylabel("PELT penalty")
    for sp in a.spines.values(): sp.set_visible(False)
ax[-1].set_xlabel("Matching window (hours)")
f.colorbar(im,ax=ax,label="Anticipation rate (%)",shrink=.6)
save(f,"fig9_sensitivity")
print("figures written")
