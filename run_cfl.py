# Tier-2 referee computations:
#  A. Full-cycle slotted-cylinder errors (t=T, exact reference = initial condition)
#  B. Advection-equation convergence (smooth Gaussian, full cycle, exact reference)
#  C. CFL sensitivity of the bp scheme (N=129, slotted, vary cfl)
import json, sys
import numpy as np
sys.path.insert(0, ".")
import advdiff as A
import sharp as Sh
from compact_stable import compact_d1_mats, compact_d1
from gen_deepen import muscl_adv_1d, uw3

T = 8.0

def gauss_ic(X, Y):
    return np.exp(-((X - 0.5) ** 2 + (Y - 0.35) ** 2) / (2 * 0.08 ** 2))

def run(N, scheme, ic, t_end, cfl=0.35, bounds="dmp"):
    x = np.linspace(0, 1, N); h = x[1] - x[0]; X, Y = np.meshgrid(x, x)
    c0 = ic(X, Y); c = c0.copy()
    dt = cfl * h; nst = int(round(t_end / dt)); dt = t_end / nst
    abx = compact_d1_mats(N)
    gmin, gmax = float(c0.min()), float(c0.max())
    def adv_HO(cc, t):
        u, v = A.velocity(X, Y, t, T)
        return -(u * compact_d1(cc, h, 1, abx) + v * compact_d1(cc, h, 0, abx))
    def rk4(cc, t, rhs):
        k1 = rhs(cc, t); k2 = rhs(cc + .5*dt*k1, t + .5*dt)
        k3 = rhs(cc + .5*dt*k2, t + .5*dt); k4 = rhs(cc + dt*k3, t + dt)
        return cc + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    for n in range(nst):
        t = n * dt
        if scheme == "up1":
            u, v = A.velocity(X, Y, t, T)
            c = c - dt*(u*A.d1_upwind(c, u, h, 1) + v*A.d1_upwind(c, v, h, 0))
        elif scheme == "up3":
            def r3(cc, tt):
                u, v = A.velocity(X, Y, tt, T)
                return -(u*uw3(cc, u, h, 1) + v*uw3(cc, v, h, 0))
            c = rk4(c, t, r3)
        elif scheme == "tvd":
            def L(cc, tt):
                u, v = A.velocity(X, Y, tt, T)
                return -(u*muscl_adv_1d(cc, u, h, 1) + v*muscl_adv_1d(cc, v, h, 0))
            c1 = c + dt*L(c, t); c = 0.5*c + 0.5*(c1 + dt*L(c1, t + dt))
        elif scheme == "compact":
            c = rk4(c, t, adv_HO)
        elif scheme in ("bp", "bpg"):
            cHO = rk4(c, t, adv_HO)
            u, v = A.velocity(X, Y, t, T)
            cLO = c - dt*(u*A.d1_upwind(c, u, h, 1) + v*A.d1_upwind(c, v, h, 0))
            if scheme == "bpg":
                lo, hi = gmin, gmax
                flag = (cHO < lo - 1e-9) | (cHO > hi + 1e-9)
                A_ = cHO - cLO; th = np.ones_like(A_)
                pos = A_ > 1e-15; neg = A_ < -1e-15
                th = np.where(pos, np.minimum(1, np.maximum(0, (hi - cLO)/np.where(pos, A_, 1))), th)
                th = np.where(neg, np.minimum(1, np.maximum(0, (cLO - lo)/np.where(neg, -A_, 1))), th)
                c = np.where(flag, cLO + th*A_, cHO)
            else:
                p = np.pad(c, 1, mode="reflect"); lo = p[:-2, :-2].copy(); hi = lo.copy()
                for a_ in range(3):
                    for b_ in range(3):
                        w = p[a_:a_+N, b_:b_+N]; lo = np.minimum(lo, w); hi = np.maximum(hi, w)
                flag = (cHO < lo - 1e-9) | (cHO > hi + 1e-9)
                c = np.where(flag, cLO, cHO)
    l2 = float(np.linalg.norm(c - c0)/np.linalg.norm(c0))
    return {"rel_l2": l2, "min": float(c.min()), "max": float(c.max())}


import json as _json
out = {}
for cfl in (0.175, 0.35, 0.7, 0.9, 1.1):
    m = run(129, "bp", lambda X, Y: Sh.slotted_cylinder(X, Y), T, cfl=cfl)
    out[str(cfl)] = m
    print(f"cfl={cfl} L2={m['rel_l2']:.3f} rng=[{m['min']:+.3f},{m['max']:.3f}]", flush=True)
_json.dump(out, open("cfl_results.json", "w"), indent=1)
print("wrote cfl_results.json")
