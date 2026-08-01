import os, numpy as np
from scipy.linalg import solve_banded
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":10})
FINE="sfbay_fine"; OUT="figures"
cc=np.load(f"{FINE}/currents_fine.npz")
wet,lat,lon=cc["wet"],cc["lat"],cc["lon"]; U,V,cid,tt=cc["U"],cc["V"],cc["cid"],cc["t"]
dx=float(cc["dx"]); dy=float(cc["dy"]); ny,nx=wet.shape
def vel_grid(k):
    Ug=np.zeros((ny,nx)); Vg=np.zeros((ny,nx)); Ug[wet]=U[k,cid[wet]]; Vg[wet]=V[k,cid[wet]]; return Ug,Vg
def vel_at(t):
    k=np.clip(np.searchsorted(tt,t)-1,0,len(tt)-2); a=(t-tt[k])/(tt[k+1]-tt[k])
    U0,V0=vel_grid(k); U1,V1=vel_grid(k+1); return (1-a)*U0+a*U1,(1-a)*V0+a*V1
# ---- operators (anisotropic h via axis) ----
def cmats(n):
    ab=np.zeros((3,n)); ab[1,:]=1; ab[0,1:]=0.25; ab[2,:-1]=0.25; ab[0,1]=0; ab[2,-2]=0; return ab
ABX=cmats(nx); ABY=cmats(ny)
def cd1(c,h,axis,ab):
    a=np.moveaxis(c,axis,-1); s=a.shape; a2=a.reshape(-1,s[-1]); r=np.empty_like(a2)
    r[:,1:-1]=0.75*(a2[:,2:]-a2[:,:-2])/h; r[:,0]=0; r[:,-1]=0
    return np.moveaxis(solve_banded((1,1),ab,r.T).T.reshape(s),-1,axis)
def up1(c,vel,h,axis):
    a=np.moveaxis(c,axis,-1); p=np.pad(a,((0,0),(1,1)),mode="edge")
    bk=(p[:,1:-1]-p[:,:-2])/h; fw=(p[:,2:]-p[:,1:-1])/h
    return np.where(vel>=0,np.moveaxis(bk,-1,axis),np.moveaxis(fw,-1,axis))
def up3(c,vel,h,axis):
    a=np.moveaxis(c,axis,-1); p=np.pad(a,((0,0),(2,2)),mode="edge")
    bk=(2*p[:,3:-1]+3*p[:,2:-2]-6*p[:,1:-3]+p[:,:-4])/(6*h)
    fw=-(2*p[:,1:-3]+3*p[:,2:-2]-6*p[:,3:-1]+p[:,4:])/(6*h)
    return np.where(vel>=0,np.moveaxis(bk,-1,axis),np.moveaxis(fw,-1,axis))
def dmp(c):
    p=np.pad(c,1,mode="edge"); lo=p[:-2,:-2].copy(); hi=lo.copy()
    for i in range(3):
        for j in range(3):
            w=p[i:i+ny,j:j+nx]; lo=np.minimum(lo,w); hi=np.maximum(hi,w)
    return lo,hi
def rhs(c,t,deriv):
    u,v=vel_at(t); return -(u*deriv(c,dx,1)+v*deriv(c,dy,0))
def step_rk4(c,t,dt,deriv):
    k1=rhs(c,t,deriv);k2=rhs(c+.5*dt*k1,t+.5*dt,deriv);k3=rhs(c+.5*dt*k2,t+.5*dt,deriv);k4=rhs(c+dt*k3,t+dt,deriv)
    return c+dt/6*(k1+2*k2+2*k3+k4)
def run(scheme,T0,nsteps,dt,c0):
    c=c0.copy(); hist=[]
    for n in range(nsteps):
        t=T0+n*dt
        if scheme=="up1":
            u,v=vel_at(t); c=c-dt*(u*up1(c,u,dx,1)+v*up1(c,v,dy,0))
        elif scheme=="up3": c=step_rk4(c,t,dt,lambda cc,h,ax: up3(cc,(vel_at(t)[0] if ax==1 else vel_at(t)[1]),h,ax))
        elif scheme=="compact": c=step_rk4(c,t,dt,lambda cc,h,ax: cd1(cc,h,ax,ABX if ax==1 else ABY))
        elif scheme=="bp":
            cHO=step_rk4(c,t,dt,lambda cc,h,ax: cd1(cc,h,ax,ABX if ax==1 else ABY))
            u,v=vel_at(t); cLO=c-dt*(u*up1(c,u,dx,1)+v*up1(c,v,dy,0))
            lo,hi=dmp(c); A=cHO-cLO; th=np.ones_like(A); pos=A>1e-15; neg=A<-1e-15
            th[pos]=np.minimum(1,np.maximum(0,(hi-cLO)[pos]/A[pos])); th[neg]=np.minimum(1,np.maximum(0,(cLO-lo)[neg]/(-A[neg])))
            c=cLO+th*A
        c*=wet; hist.append((c.min(),c.max(),c.sum()))
    return c,np.array(hist)
# ---- IC: sharp disk at Bay Bridge allision ----
i0=int(np.argmin(np.abs(lon-(-122.357)))); j0=int(np.argmin(np.abs(lat-37.806)))
Lon,Lat=np.meshgrid(lon,lat); R_km=3.5
dist=np.sqrt(((Lon-lon[i0])*111*np.cos(np.radians(37.8)))**2+((Lat-lat[j0])*111)**2)
c0=((dist<=R_km)&wet).astype(float)
T0=float(tt[0]); dt=60.0; nsteps=300   # 5 h
print(f"grid {ny}x{nx}, patch cells {int(c0.sum())}, run {nsteps*dt/3600:.1f} h")
res={}; 
for sch in ("up1","up3","compact","bp"):
    cf,h=run(sch,T0,nsteps,dt,c0); res[sch]=(cf,h)
    print(f"  {sch:8s} final min={cf.min():+.3f} max={cf.max():.3f} mass_drift={(h[-1,2]-h[0,2])/h[0,2]:+.3f}")
np.savez(f"{OUT}/../sfbay_transport_result.npz", **{f"c_{k}":v[0] for k,v in res.items()}, c0=c0, lon=lon, lat=lat, wet=wet)
print("saved fields")

# ---- render the comparison as a 2x3 grid (zoomed to the central bay) ----
import matplotlib.colors as mcolors
LON0,LON1,LAT0,LAT1=-122.54,-122.20,37.71,37.93   # central-bay zoom
panels=[("Initial patch (Bay Bridge release)",c0),
        ("1st-order upwind",res["up1"][0]),
        ("3rd-order upwind (Sankaranarayanan et al. 1998)",res["up3"][0]),
        ("plain compact (Padé)",res["compact"][0]),
        ("bound-preserving compact (this work)",res["bp"][0])]
fig,axs=plt.subplots(2,3,figsize=(13,9.4)); axf=axs.flat
norm=mcolors.TwoSlopeNorm(vmin=-0.3,vcenter=0.0,vmax=1.0)
im=None
for ax,(ti,fld) in zip(axf,panels):
    f=np.where(wet,fld,np.nan)
    im=ax.pcolormesh(lon,lat,f,cmap="RdBu_r",norm=norm,shading="auto")
    ax.contourf(lon,lat,(~wet).astype(float),levels=[0.5,1.5],colors="0.85")  # land
    ax.contour(lon,lat,wet.astype(float),levels=[0.5],colors="k",linewidths=0.5)
    ax.plot(-122.357,37.806,"*",ms=15,mfc="yellow",mec="k",mew=0.7,zorder=6)
    ax.set_title(f"{ti}\nrange $[{np.nanmin(fld):+.2f},\\,{np.nanmax(fld):.2f}]$",fontsize=11)
    ax.set_xlim(LON0,LON1); ax.set_ylim(LAT0,LAT1)
    ax.set_aspect(1/np.cos(np.radians(37.8))); ax.set_xticks([]); ax.set_yticks([])
axs.flat[5].axis("off")
cb=fig.colorbar(im,ax=axs.flat[5],fraction=0.5,aspect=18,extend="both")
cb.set_label("tracer concentration (blue $<0$)",fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_sfbay_transport.png",dpi=300,bbox_inches="tight"); print("wrote fig_sfbay_transport.png")
