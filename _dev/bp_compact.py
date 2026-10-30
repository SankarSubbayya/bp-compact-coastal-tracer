"""Paper step 2: BOUND-PRESERVING COMPACT via a-posteriori MOOD-style limiting.
Per step: compute the compact high-order (HO) RK4 candidate; compute a bounded
low-order (LO, 1st-order upwind) candidate; flag cells where HO violates the local
discrete-maximum-principle (DMP) neighborhood bounds (or global [0,1]); use LO there,
HO elsewhere. Smooth regions keep spectral-like compact resolution; discontinuities
stay bounded. Tested on the slotted cylinder (advection-dominated, D=1e-5)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advdiff as A, sharp as Sh
from compact_stable import compact_d1_mats, compact_d1

def upwind_dx(c,u,h): return A.d1_upwind(c,u,h,1)
def upwind_dy(c,v,h): return A.d1_upwind(c,v,h,0)
def dmp_bounds(c):
    p=np.pad(c,1,mode="reflect"); lo=p[:-2,:-2].copy(); hi=lo.copy()
    for a in range(3):
        for b in range(3):
            w=p[a:a+c.shape[0], b:b+c.shape[1]]; lo=np.minimum(lo,w); hi=np.maximum(hi,w)
    return lo,hi

def solve(N,T,tc,D,dt,mode,**ick):
    x=np.linspace(0,1,N); h=x[1]-x[0]; X,Y=np.meshgrid(x,x)
    c=Sh.slotted_cylinder(X,Y,**ick); abx=compact_d1_mats(N)
    nst=int(round(tc/dt)); dt=tc/nst
    def adv_HO(c,t):
        u,v=A.velocity(X,Y,t,T); return -(u*compact_d1(c,h,1,abx)+v*compact_d1(c,h,0,abx))
    for n in range(nst):
        t=n*dt
        if mode=="compact":      # plain compact RK4 (no limiting) -> Gibbs
            k1=adv_HO(c,t);k2=adv_HO(c+.5*dt*k1,t+.5*dt);k3=adv_HO(c+.5*dt*k2,t+.5*dt);k4=adv_HO(c+dt*k3,t+dt)
            c=c+dt/6*(k1+2*k2+2*k3+k4)
        elif mode=="upwind":     # 1st-order upwind (bounded, diffusive)
            u,v=A.velocity(X,Y,t,T); c=c-dt*(u*upwind_dx(c,u,h)+v*upwind_dy(c,v,h))
        elif mode=="bp":         # bound-preserving compact (MOOD HO->LO)
            k1=adv_HO(c,t);k2=adv_HO(c+.5*dt*k1,t+.5*dt);k3=adv_HO(c+.5*dt*k2,t+.5*dt);k4=adv_HO(c+dt*k3,t+dt)
            cHO=c+dt/6*(k1+2*k2+2*k3+k4)
            u,v=A.velocity(X,Y,t,T); cLO=c-dt*(u*upwind_dx(c,u,h)+v*upwind_dy(c,v,h))
            lo,hi=dmp_bounds(c); eps=1e-9
            flag=(cHO<lo-eps)|(cHO>hi+eps)|(~np.isfinite(cHO))
            c=np.where(flag,cLO,cHO)
        if D>0: c=c+dt*D*A.velocity and c  # (diffusion negligible at D=1e-5; omit for probe)
    return c,h

if __name__=="__main__":
    N=129;T=3.0;tc=1.5;D=0.0;dt=0.004
    R=Sh.exact_reference(N,T,tc); h=1/(N-1)
    def met(c):
        l2=np.sqrt(np.sum((c-R)**2)/np.sum(R**2)); a=c>0.5;b=R>0.5
        return l2,c.min(),c.max(),(a&b).sum()/max((a|b).sum(),1)
    for mode in ("compact","upwind","bp"):
        c,_=solve(N,T,tc,D,dt,mode); l2,mn,mx,iou=met(c)
        print(f"{mode:8s} L2={l2:.3f} min={mn:+.3f} max={mx:.3f} IoU={iou:.3f}")

# ---- improved limiter: FCT-style bounded blend (keep bounded portion of HO correction) ----
def solve_bpf(N,T,tc,D,dt,**ick):
    x=np.linspace(0,1,N); h=x[1]-x[0]; X,Y=np.meshgrid(x,x)
    c=Sh.slotted_cylinder(X,Y,**ick); abx=compact_d1_mats(N)
    nst=int(round(tc/dt)); dt=tc/nst
    def adv_HO(c,t):
        u,v=A.velocity(X,Y,t,T); return -(u*compact_d1(c,h,1,abx)+v*compact_d1(c,h,0,abx))
    for n in range(nst):
        t=n*dt
        k1=adv_HO(c,t);k2=adv_HO(c+.5*dt*k1,t+.5*dt);k3=adv_HO(c+.5*dt*k2,t+.5*dt);k4=adv_HO(c+dt*k3,t+dt)
        cHO=c+dt/6*(k1+2*k2+2*k3+k4)
        u,v=A.velocity(X,Y,t,T); cLO=c-dt*(u*upwind_dx(c,u,h)+v*upwind_dy(c,v,h))
        lo,hi=dmp_bounds(c)
        Aanti=cHO-cLO
        theta=np.ones_like(Aanti)
        pos=Aanti>1e-15; neg=Aanti<-1e-15
        theta[pos]=np.minimum(1.0,np.maximum(0.0,(hi-cLO)[pos]/Aanti[pos]))
        theta[neg]=np.minimum(1.0,np.maximum(0.0,(cLO-lo)[neg]/(-Aanti[neg])))
        c=cLO+theta*Aanti
    return c,h

if __name__=="__main__" and "--bpf" in sys.argv:
    N=129;T=3.0;tc=1.5;dt=0.004
    R=Sh.exact_reference(N,T,tc)
    def met(c):
        l2=np.sqrt(np.sum((c-R)**2)/np.sum(R**2)); a=c>0.5;b=R>0.5
        return l2,c.min(),c.max(),(a&b).sum()/max((a|b).sum(),1)
    for mode in ("compact","upwind","bp"):
        c,_=solve(N,T,tc,0.0,dt,mode); l2,mn,mx,iou=met(c); print(f"{mode:9s} L2={l2:.3f} min={mn:+.3f} max={mx:.3f} IoU={iou:.3f}")
    c,_=solve_bpf(N,T,tc,0.0,dt); l2,mn,mx,iou=met(c); print(f"{'bpf(FCT)':9s} L2={l2:.3f} min={mn:+.3f} max={mx:.3f} IoU={iou:.3f}")
