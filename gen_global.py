# Global-bounds ([0,1], the Li-Xie-Zhang property) vs local-DMP limiter variants,
# on BOTH tests: slotted cylinder at T/2 (vs ref513) and Gaussian full period (vs exact).
import json, sys
import numpy as np
sys.path.insert(0, ".")
import advdiff as A
import sharp as Sh
from compact_stable import compact_d1_mats, compact_d1
from gen_deepen import interp_to

T = 8.0

def gauss_ic(X, Y):
    return np.exp(-((X - 0.5) ** 2 + (Y - 0.35) ** 2) / (2 * 0.08 ** 2))

def run(N, ic, t_end, bounds):
    x = np.linspace(0, 1, N); h = x[1] - x[0]; X, Y = np.meshgrid(x, x)
    c0 = ic(X, Y); c = c0.copy()
    dt = 0.35 * h; nst = int(round(t_end / dt)); dt = t_end / nst
    abx = compact_d1_mats(N)
    gmin, gmax = float(c0.min()), float(c0.max())
    def adv_HO(cc, t):
        u, v = A.velocity(X, Y, t, T)
        return -(u * compact_d1(cc, h, 1, abx) + v * compact_d1(cc, h, 0, abx))
    for n in range(nst):
        t = n * dt
        k1 = adv_HO(c, t); k2 = adv_HO(c + .5*dt*k1, t + .5*dt)
        k3 = adv_HO(c + .5*dt*k2, t + .5*dt); k4 = adv_HO(c + dt*k3, t + dt)
        cHO = c + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        u, v = A.velocity(X, Y, t, T)
        cLO = c - dt*(u*A.d1_upwind(c, u, h, 1) + v*A.d1_upwind(c, v, h, 0))
        if bounds == "global":
            lo, hi = gmin, gmax
            flag = (cHO < lo - 1e-9) | (cHO > hi + 1e-9)
            A_ = cHO - cLO
            th = np.ones_like(A_)
            pos = A_ > 1e-15; neg = A_ < -1e-15
            th = np.where(pos, np.minimum(1, np.maximum(0, (hi - cLO) / np.where(pos, A_, 1))), th)
            th = np.where(neg, np.minimum(1, np.maximum(0, (cLO - lo) / np.where(neg, -A_, 1))), th)
            c = np.where(flag, cLO + th * A_, cHO)
        else:
            p = np.pad(c, 1, mode="reflect"); lo = p[:-2, :-2].copy(); hi = lo.copy()
            for a_ in range(3):
                for b_ in range(3):
                    w = p[a_:a_+N, b_:b_+N]; lo = np.minimum(lo, w); hi = np.maximum(hi, w)
            flag = (cHO < lo - 1e-9) | (cHO > hi + 1e-9)
            c = np.where(flag, cLO, cHO)
    return c, c0

cref = np.load("ref513.npy")
out = {}
for N in (129, 257):
    out[str(N)] = {}
    for b in ("global", "dmp"):
        # slotted at T/2
        c, c0 = run(N, lambda X, Y: Sh.slotted_cylinder(X, Y), T/2, b)
        ci = interp_to(cref, 513, N)
        s = {"rel_l2": float(np.linalg.norm(c-ci)/np.linalg.norm(ci)),
             "min": float(c.min()), "max": float(c.max()),
             "iou": float((((c>0.5)&(ci>0.5)).sum())/max(((c>0.5)|(ci>0.5)).sum(),1)),
             "drift": float(abs(c.sum()-c0.sum())/c0.sum())}
        # gaussian full period
        g, g0 = run(N, gauss_ic, T, b)
        s["gauss_l2"] = float(np.linalg.norm(g-g0)/np.linalg.norm(g0))
        s["gauss_peak"] = float(g.max())
        out[str(N)][b] = s
        print(f"N={N:4d} {b:6s} slotted: L2={s['rel_l2']:.3f} rng=[{s['min']:+.3f},{s['max']:.3f}] IoU={s['iou']:.2f} dM={s['drift']:.1e} | gauss: L2={s['gauss_l2']:.4f} peak={s['gauss_peak']:.3f}", flush=True)
json.dump(out, open("global_results.json", "w"), indent=1)
print("wrote global_results.json")
