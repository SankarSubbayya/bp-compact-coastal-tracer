# Smooth-field (Gaussian) deformation test: full period 0 -> T (flow reverses at
# T/2), so the exact solution at t=T is the initial condition. Exposes the
# artificial steepening of compressive limiters (superbee) on smooth data.
import json, sys
import numpy as np
sys.path.insert(0, ".")
import advdiff as A
from compact_stable import compact_d1_mats, compact_d1
from gen_deepen import muscl_adv_1d, uw3

T = 8.0

def gauss_ic(X, Y):
    return np.exp(-((X - 0.5) ** 2 + (Y - 0.35) ** 2) / (2 * 0.08 ** 2))

def run(N, scheme):
    x = np.linspace(0, 1, N); h = x[1] - x[0]; X, Y = np.meshgrid(x, x)
    c0 = gauss_ic(X, Y); c = c0.copy()
    dt = 0.35 * h; nst = int(round(T / dt)); dt = T / nst
    abx = compact_d1_mats(N)
    def adv_HO(cc, t):
        u, v = A.velocity(X, Y, t, T)
        return -(u * compact_d1(cc, h, 1, abx) + v * compact_d1(cc, h, 0, abx))
    def rk4(cc, t, rhs):
        k1 = rhs(cc, t); k2 = rhs(cc + .5 * dt * k1, t + .5 * dt)
        k3 = rhs(cc + .5 * dt * k2, t + .5 * dt); k4 = rhs(cc + dt * k3, t + dt)
        return cc + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    def dmp(cc):
        p = np.pad(cc, 1, mode="reflect"); lo = p[:-2, :-2].copy(); hi = lo.copy()
        for a_ in range(3):
            for b_ in range(3):
                w = p[a_:a_ + N, b_:b_ + N]; lo = np.minimum(lo, w); hi = np.maximum(hi, w)
        return lo, hi
    for n in range(nst):
        t = n * dt
        if scheme == "up1":
            u, v = A.velocity(X, Y, t, T)
            c = c - dt * (u * A.d1_upwind(c, u, h, 1) + v * A.d1_upwind(c, v, h, 0))
        elif scheme == "up3":
            def r3(cc, tt):
                u, v = A.velocity(X, Y, tt, T)
                return -(u * uw3(cc, u, h, 1) + v * uw3(cc, v, h, 0))
            c = rk4(c, t, r3)
        elif scheme == "tvd":
            def L(cc, tt):
                u, v = A.velocity(X, Y, tt, T)
                return -(u * muscl_adv_1d(cc, u, h, 1) + v * muscl_adv_1d(cc, v, h, 0))
            c1 = c + dt * L(c, t); c = 0.5 * c + 0.5 * (c1 + dt * L(c1, t + dt))
        elif scheme == "compact":
            c = rk4(c, t, adv_HO)
        elif scheme == "bp":
            cHO = rk4(c, t, adv_HO)
            u, v = A.velocity(X, Y, t, T)
            cLO = c - dt * (u * A.d1_upwind(c, u, h, 1) + v * A.d1_upwind(c, v, h, 0))
            lo, hi = dmp(c); flag = (cHO < lo - 1e-9) | (cHO > hi + 1e-9)
            c = np.where(flag, cLO, cHO)
    l2 = float(np.linalg.norm(c - c0) / np.linalg.norm(c0))
    return {"rel_l2": l2, "peak": float(c.max()), "min": float(c.min())}

out = {}
for N in (65, 129, 257):
    out[str(N)] = {}
    for s in ("up1", "up3", "tvd", "compact", "bp"):
        m = run(N, s)
        out[str(N)][s] = m
        print(f"N={N:4d} {s:8s} L2={m['rel_l2']:.4f} peak={m['peak']:.3f} min={m['min']:+.4f}", flush=True)
json.dump(out, open("gauss_results.json", "w"), indent=1)
print("wrote gauss_results.json")
