"""Depth-averaged tidal current speed at four stations over ~4.5 days, showing the
mixed-tide modulation (unequal successive flood/ebb maxima = diurnal inequality)
and the spatial decrease from the Golden Gate into the bay."""
import os, paths as P
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10})
D=P.SFBAY
OUT=P.FIGURES
d=np.load(f"{D}/sfbay_stations.npz",allow_pickle=True)
t=d["t"]; names=list(d["names"]); spd=d["spd"]; la=d["lat"]; lo=d["lon"]
t0=6*86400; m=(t>=t0)&(t<=t0+4.5*86400); th=(t[m]-t0)/3600.0
fig,axs=plt.subplots(len(names),1,figsize=(10,7),sharex=True)
cols=["C3","C0","C2","C1"]
for ax,nm,col,i in zip(axs,names,cols,range(len(names))):
    s=spd[i][m]; ax.plot(th,s,"-",color=col,lw=1.3)
    ax.fill_between(th,0,s,color=col,alpha=0.15)
    ax.set_ylabel("speed\n(m s$^{-1}$)"); ax.set_ylim(0,max(0.4,s.max()*1.15))
    ax.text(0.005,0.86,f"{nm}  ({la[i]:.2f}$^\\circ$N, {abs(lo[i]):.2f}$^\\circ$W)  peak {s.max():.2f} m s$^{{-1}}$",
            transform=ax.transAxes,fontsize=9,va="top")
    ax.grid(alpha=0.25)
axs[-1].set_xlabel("time (hours)"); axs[0].set_xlim(0,108)
fig.suptitle("Depth-averaged tidal current speed at four stations: mixed-tide modulation "
             "(diurnal inequality) and decrease into the bay",fontsize=11,y=0.995)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sfbay_currents.png",dpi=300,bbox_inches="tight")
print("wrote fig_sfbay_currents.png; station peaks:",{nm:round(float(spd[i][m].max()),2) for i,nm in enumerate(names)})
import figure_provenance as FP
FP.record("sfbay_currents_fig.py")
