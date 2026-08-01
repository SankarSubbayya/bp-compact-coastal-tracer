"""Golden Gate sea-surface elevation over ~4.5 days from the mixed-tide model,
showing the mixed (semidiurnal + diurnal) character and the diurnal inequality."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":11})
D="sfbay_fine"
OUT="figures"
d=np.load(f"{D}/sfbay_tides.npz"); t=d["t"]; eta=d["eta_gg"]
names=list(d["amp_names"]); am=d["amp_model"]; ao=d["amp_obs"]
t0=6*86400; m=(t>=t0)&(t<=t0+4.5*86400); th=(t[m]-t0)/3600.0; e=eta[m]
fig,ax=plt.subplots(figsize=(10,3.6))
ax.plot(th,e,"-",color="navy",lw=1.4)
ax.axhline(0,color="0.6",lw=0.6)
# mark successive highs/lows to show the diurnal inequality
from scipy.signal import argrelextrema
hi=argrelextrema(e,np.greater,order=3)[0]; lo=argrelextrema(e,np.less,order=3)[0]
ax.plot(th[hi],e[hi],"r^",ms=5); ax.plot(th[lo],e[lo],"bv",ms=5)
ax.set_xlabel("time (hours)"); ax.set_ylabel("sea-surface elevation (m)")
ax.set_xlim(0,108); ax.set_title("Golden Gate tide (mixed, predominantly semidiurnal): "
            "successive highs/lows are unequal — the diurnal inequality")
F=(am[names.index("K1")]+am[names.index("O1")])/(am[names.index("M2")]+am[names.index("S2")])
txt="model vs NOAA (m):  "+",  ".join(f"{n} {am[i]:.2f}/{ao[i]:.2f}" for i,n in enumerate(names) if n in ("M2","K1","O1"))
ax.text(0.01,-0.34,txt+f"     form factor F=(K1+O1)/(M2+S2)={F:.2f}",transform=ax.transAxes,fontsize=8.5,color="0.25")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sfbay_tide.png",dpi=300,bbox_inches="tight")
print("wrote fig_sfbay_tide.png; range",f"{e.min():.2f}..{e.max():.2f} m")
