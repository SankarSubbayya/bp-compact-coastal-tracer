"""High-resolution San Francisco Bay domain figure: NOAA CRM bathymetry (200 m
grid) with landmarks/locations and depth-averaged tidal currents at peak FLOOD
and peak EBB (two panels)."""
import os, paths as P
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":12})
FINE=P.SFBAY
OUT=P.FIGURES; os.makedirs(OUT,exist_ok=True)
b=np.load(f"{FINE}/sfbay_fine.npz"); c=np.load(f"{FINE}/currents_fine.npz")
depth,wet,lat,lon=b["depth"],b["wet"],b["lat"],b["lon"]
U,V,cid,tt=c["U"],c["V"],c["cid"],c["t"]; ny,nx=depth.shape

# Golden Gate cell -> sign of along-strait (east-west) current sets flood/ebb
jg=int(np.argmin(np.abs(lat-37.817))); ig=int(np.argmin(np.abs(lon-(-122.482)))); kg=cid[jg,ig]
ugate=U[:,kg]
kf=int(np.argmax(ugate))   # peak flood  (current INTO bay, eastward at Gate)
ke=int(np.argmin(ugate))   # peak ebb    (current OUT of bay, westward at Gate)

def vel_grid(k):
    Ug=np.zeros((ny,nx)); Vg=np.zeros((ny,nx)); Ug[wet]=U[k,cid[wet]]; Vg[wet]=V[k,cid[wet]]; return Ug,Vg

LANDMARKS=[  # (lon, lat, label, dlon, dlat, ha)  -- dlon/dlat offset the text
 (-122.478,37.819,"Golden Gate",  -0.20, 0.000,"left"),
 (-122.357,37.806,"Bay Bridge\n(Cosco Busan allision)", 0.06,-0.05,"left"),
 (-122.420,37.770,"San Francisco", -0.02,-0.05,"right"),
 (-122.271,37.800,"Oakland",        0.04, 0.00,"left"),
 (-122.290,37.871,"Berkeley",       0.04, 0.01,"left"),
 (-122.372,37.930,"Richmond",       0.04, 0.02,"left"),
 (-122.422,37.827,"Alcatraz",      -0.02, 0.02,"right"),
 (-122.430,37.860,"Angel I.",      -0.02, 0.02,"right"),
 (-122.485,37.859,"Sausalito",     -0.21, 0.00,"left"),
 (-122.420,38.080,"San Pablo Bay", -0.02, 0.04,"center"),
 (-122.230,37.560,"South Bay",      0.00,-0.04,"center"),
 (-122.620,37.700,"Pacific\nOcean", 0.00, 0.00,"center"),
]

fig,axes=plt.subplots(1,2,figsize=(13,8.6),sharey=True)
dshow=np.where(wet,np.maximum(depth,0.1),np.nan)
vmax=np.nanpercentile(dshow,98)
asp=1/np.cos(np.radians(lat.mean()))
for ax,(k,ttl) in zip(axes,[(kf,"Peak FLOOD (currents into the bay)"),
                            (ke,"Peak EBB (currents out of the bay)")]):
    im=ax.pcolormesh(lon,lat,dshow,cmap="Blues",shading="auto",vmin=0,vmax=vmax)
    ax.contour(lon,lat,wet.astype(float),levels=[0.5],colors="k",linewidths=0.6)
    Ug,Vg=vel_grid(k); s=10; Lon,Lat=np.meshgrid(lon,lat); spd=np.hypot(Ug,Vg)
    q=ax.quiver(Lon[::s,::s],Lat[::s,::s],Ug[::s,::s],Vg[::s,::s],spd[::s,::s],
                cmap="autumn_r",scale=18,width=0.0030,clim=(0,1.4))
    ax.quiverkey(q,0.80,0.06,1.0,"1 m/s",labelpos="E",coordinates="axes",fontproperties={"size":9})
    for (lo,la,lab,dlo,dla,ha) in LANDMARKS:
        ax.plot(lo,la,"o",ms=3,mfc="k",mec="k")
        ax.annotate(lab,(lo,la),(lo+dlo,la+dla),fontsize=10,ha=ha,va="center",
                    color="0.15",arrowprops=dict(arrowstyle="-",lw=0.5,color="0.4"))
    ax.plot(-122.357,37.806,"*",ms=17,mfc="red",mec="k",mew=0.7,zorder=6)
    ax.set_title(ttl,fontsize=11); ax.set_xlabel("longitude"); ax.set_aspect(asp)
    jj=np.where(wet.any(axis=1))[0]; ax.set_xlim(lon.min(),lon.max())
    ax.set_ylim(lat[jj.min()]-0.02, lat[jj.max()]+0.02)
axes[0].set_ylabel("latitude")
cb=fig.colorbar(im,ax=axes,shrink=0.6,pad=0.02,location="right"); cb.set_label("water depth (m)")
fig.suptitle("San Francisco Bay: NOAA Coastal Relief Model bathymetry (200 m grid) "
             "and depth-averaged currents at peak flood and ebb",fontsize=12.5,y=0.96)
fig.savefig(f"{OUT}/fig_sfbay_domain.png",dpi=300,bbox_inches="tight"); print("wrote fig_sfbay_domain.png")
print(f"grid {ny}x{nx}, wet {int(wet.sum())}, flood k={kf} ebb k={ke}, "
      f"Gate flood U={ugate[kf]:+.2f} ebb U={ugate[ke]:+.2f} m/s, max depth {depth.max():.0f} m")
import figure_provenance as FP
FP.record("sfbay_domain.py")
