"""Paper step 1: STABLE compact (Pade) derivatives on a closed basin.
The standard Lele one-sided boundary closure is unstable for advection (we hit
this: compact4 blew up). Fix: use the PHYSICAL no-flux boundary (d c/dn = 0),
i.e. enforce f'=0 at the walls -- no extrapolation, stable, and consistent with
the closed-basin BC. Test: (a) accuracy/order on a Neumann-compatible function,
(b) STABILITY when integrating the deformational advection over the full cycle."""
import numpy as np
from scipy.linalg import solve_banded

def compact_d1_mats(n):
    # 4th-order interior Pade alpha=1/4; Neumann f'=0 at both ends (rows 0,n-1).
    ab=np.zeros((3,n)); ab[1,:]=1.0; ab[0,1:]=0.25; ab[2,:-1]=0.25
    ab[0,1]=0.0; ab[2,-2]=0.0      # decouple boundary rows (f'_0, f'_{n-1} set =0)
    return ab
def compact_d1(c, h, axis, ab):
    a=np.moveaxis(c,axis,-1); sh=a.shape; a2=a.reshape(-1,sh[-1]); n=sh[-1]
    r=np.empty_like(a2)
    r[:,1:-1]=0.75*(a2[:,2:]-a2[:,:-2])/h
    r[:,0]=0.0; r[:,-1]=0.0        # Neumann: normal derivative zero at walls
    out=solve_banded((1,1),ab,r.T).T
    return np.moveaxis(out.reshape(sh),-1,axis)

if __name__=="__main__":
    # (a) order test on cos(2 pi x): f'(0)=f'(1)=0 (Neumann-compatible)
    print("order test (interior max-err, expect ~4th):")
    prev=None
    for N in (33,65,129,257):
        x=np.linspace(0,1,N); h=x[1]-x[0]; f=np.cos(2*np.pi*x)[None,:]
        ab=compact_d1_mats(N); d=compact_d1(f,h,1,ab)[0]
        e=np.max(np.abs(d[2:-2]-(-2*np.pi*np.sin(2*np.pi*x))[2:-2]))
        rate="" if prev is None else f"rate={np.log2(prev/e):.2f}"
        print(f"  N={N:4d} err={e:.2e} {rate}"); prev=e
    # (b) stability: integrate deformational advection (smooth Gaussian), compact d1, RK4
    import sys; sys.path.insert(0,"/Users/sankar/projects/tb_science/compact-advection-diffusion/_dev")
    import advdiff as A
    N=129; T=8.0; h=1/(N-1); x=np.linspace(0,1,N); X,Y=np.meshgrid(x,x)
    abx=compact_d1_mats(N)
    c=np.exp(-((X-0.5)**2+(Y-0.75)**2)/(2*0.08**2))
    dt=1e-3; nst=int(T/dt)
    def rhs(c,t):
        u,v=A.velocity(X,Y,t,T)
        cx=compact_d1(c,h,1,abx); cy=compact_d1(c,h,0,abx)
        return -(u*cx+v*cy)
    for n in range(nst):
        t=n*dt
        k1=rhs(c,t);k2=rhs(c+.5*dt*k1,t+.5*dt);k3=rhs(c+.5*dt*k2,t+.5*dt);k4=rhs(c+dt*k3,t+dt)
        c=c+dt/6*(k1+2*k2+2*k3+k4)
        if not np.isfinite(c).all(): print(f"  BLEW UP at step {n} (t={t:.2f})"); break
    else:
        print(f"stability: STABLE over full cycle. final max={c.max():.3f} min={c.min():+.4f} (recovery err RMSE vs IC={np.sqrt(np.mean((c-np.exp(-((X-0.5)**2+(Y-0.75)**2)/(2*0.08**2)))**2)):.2e})")
