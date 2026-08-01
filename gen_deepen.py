# Deepening computations for the Ocean Engineering version of the manuscript:
#   (1) a TVD-MUSCL (superbee) comparator -- the flux-limited scheme class used in
#       modern operational coastal models;
#   (2) a benchmark resolution sweep (N = 65, 129, 257) for all five schemes;
#   (3) relative mass drift and wall-clock cost per step for each scheme.
# Slotted-cylinder LeVeque deformational flow at maximum deformation (t = T/2),
# matching the existing Table "deform" protocol (advection-dominated).
# Output: deepen_results.json (consumed by the manuscript tables).
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import advdiff as A
import sharp as Sh
from compact_stable import compact_d1_mats, compact_d1

ICK = {}  # slotted-cylinder defaults from sharp.py
T = 8.0
TC = T / 2.0


def velocity(X, Y, t):
    return A.velocity(X, Y, t, T)


def uw3(c, vel, h, axis):
    cm2 = np.roll(c, 2, axis); cm1 = np.roll(c, 1, axis)
    cp1 = np.roll(c, -1, axis); cp2 = np.roll(c, -2, axis)
    dpos = (2 * cp1 + 3 * c - 6 * cm1 + cm2) / (6 * h)
    dneg = (-cp2 + 6 * cp1 - 3 * c - 2 * cm1) / (6 * h)
    return np.where(vel > 0, dpos, dneg)


def muscl_adv_1d(c, u, h, axis):
    """Advective-form MUSCL derivative with superbee limiter: u * dc/dx evaluated
    from limited upwind-biased interface reconstructions. Matches the advective
    transport form used by all other schemes (valid for divergent velocity)."""
    eps = 1e-14
    cp1 = np.roll(c, -1, axis); cm1 = np.roll(c, 1, axis)
    dc_f = cp1 - c                                   # forward difference (face i+1/2)
    dc_b = c - cm1                                   # backward difference (face i-1/2)
    def phi(r):
        return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
    # left-biased reconstruction at faces (for u>0)
    rL = dc_b / np.where(np.abs(dc_f) < eps, eps, dc_f)
    cL_f = c + 0.5 * phi(rL) * dc_f                  # c^L at face i+1/2
    cL_b = np.roll(cL_f, 1, axis)                    # c^L at face i-1/2
    dpos = (cL_f - cL_b) / h
    # right-biased reconstruction at faces (for u<0)
    dc_ff = np.roll(c, -2, axis) - cp1
    rR = dc_ff / np.where(np.abs(dc_f) < eps, eps, dc_f)
    cR_f = cp1 - 0.5 * phi(rR) * dc_f                # c^R at face i+1/2
    cR_b = np.roll(cR_f, 1, axis)
    dneg = (cR_f - cR_b) / h
    return np.where(u >= 0, dpos, dneg)


def tvd_flux_1d(c, u_face, dt, h, axis):
    """Superbee flux-limited MUSCL flux divergence along one axis.
    u_face: face-normal velocity at i+1/2 faces (same shape as c, face k = between
    node k and k+1 via roll). Conservative form; periodic rolls are safe because
    the velocity vanishes at the basin walls."""
    cp1 = np.roll(c, -1, axis)
    dc = cp1 - c                                    # slope across face i+1/2
    dcm = c - np.roll(c, 1, axis)                   # upwind-side slope (u>0)
    dcp = np.roll(c, -2, axis) - cp1                # upwind-side slope (u<0)
    eps = 1e-14
    up = u_face >= 0
    r = np.where(up, dcm, dcp) / np.where(np.abs(dc) < eps, eps, dc)
    phi = np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))  # superbee
    nu = np.abs(u_face) * dt / h
    c_up = np.where(up, c, cp1)
    F = u_face * c_up + 0.5 * np.abs(u_face) * (1 - nu) * phi * dc
    return (F - np.roll(F, 1, axis)) / h            # d(F)/dx at nodes


def solve_scheme(N, scheme, cfl=0.35):
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    c0 = Sh.slotted_cylinder(X, Y, **ICK)
    c = c0.copy()
    umax = 1.0
    dt = cfl * h / umax
    nst = int(round(TC / dt)); dt = TC / nst
    abx = compact_d1_mats(N)

    def adv_HO(c, t):
        u, v = velocity(X, Y, t)
        return -(u * compact_d1(c, h, 1, abx) + v * compact_d1(c, h, 0, abx))

    def rk4(c, t, rhs):
        k1 = rhs(c, t); k2 = rhs(c + .5 * dt * k1, t + .5 * dt)
        k3 = rhs(c + .5 * dt * k2, t + .5 * dt); k4 = rhs(c + dt * k3, t + dt)
        return c + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def dmp_bounds(cc):
        p = np.pad(cc, 1, mode="reflect")
        lo = p[:-2, :-2].copy(); hi = lo.copy()
        for a in range(3):
            for b in range(3):
                w = p[a:a + N, b:b + N]
                lo = np.minimum(lo, w); hi = np.maximum(hi, w)
        return lo, hi

    t0 = time.perf_counter()
    for n in range(nst):
        t = n * dt
        if scheme == "up1":
            u, v = velocity(X, Y, t)
            c = c - dt * (u * A.d1_upwind(c, u, h, 1) + v * A.d1_upwind(c, v, h, 0))
        elif scheme == "up3":
            def rhs3(cc, tt):
                u, v = velocity(X, Y, tt)
                return -(u * uw3(cc, u, h, 1) + v * uw3(cc, v, h, 0))
            c = rk4(c, t, rhs3)
        elif scheme == "tvd":
            # SSP-RK2 (Heun), advective-form superbee MUSCL (same transport form
            # as all other schemes; valid for divergent velocity fields)
            def L(cc, tt):
                u, v = velocity(X, Y, tt)
                return -(u * muscl_adv_1d(cc, u, h, 1) + v * muscl_adv_1d(cc, v, h, 0))
            c1 = c + dt * L(c, t)
            c = 0.5 * c + 0.5 * (c1 + dt * L(c1, t + dt))
        elif scheme == "compact":
            c = rk4(c, t, adv_HO)
        elif scheme == "bp":
            cHO = rk4(c, t, adv_HO)
            u, v = velocity(X, Y, t)
            cLO = c - dt * (u * A.d1_upwind(c, u, h, 1) + v * A.d1_upwind(c, v, h, 0))
            lo, hi = dmp_bounds(c)
            eps = 1e-9
            flag = (cHO < lo - eps) | (cHO > hi + eps) | (~np.isfinite(cHO))
            c = np.where(flag, cLO, cHO)
        else:
            raise ValueError(scheme)
    wall = time.perf_counter() - t0

    # reference at max deformation = exact advection of c0 (deformation not yet reversed):
    # for error at t=T/2 the exact solution is NOT c0; use high-res compact as reference?
    # Follow the existing paper protocol: errors vs a resolved reference field computed
    # with bp at N_ref=513 are expensive; instead reuse the paper's metric = error vs the
    # analytic solution at t=T (full reversal). To stay exactly comparable to the
    # published Table (metrics AT max deformation vs the fine-grid reference), we report
    # here the same quantities the table used: range and IoU at t=T/2, plus L2 vs a
    # 513-node bp reference (computed once, interpolated).
    return c, c0, h, nst, wall


def metrics(c, cref_interp, c0):
    rel_l2 = float(np.linalg.norm(c - cref_interp) / np.linalg.norm(cref_interp))
    rel_l1 = float(np.abs(c - cref_interp).sum() / np.abs(cref_interp).sum())
    iou_num = float(((c > 0.5) & (cref_interp > 0.5)).sum())
    iou_den = float(((c > 0.5) | (cref_interp > 0.5)).sum())
    return {
        "rel_l2": rel_l2,
        "rel_l1": rel_l1,
        "min": float(c.min()),
        "max": float(c.max()),
        "iou": iou_num / max(iou_den, 1),
        "mass_drift": float(abs(c.sum() - c0.sum()) / abs(c0.sum())),
    }


def interp_to(cref, Nref, N):
    # bilinear sample of the fine reference onto the coarse grid
    idx = np.linspace(0, Nref - 1, N)
    i0 = np.clip(idx.astype(int), 0, Nref - 2)
    a = idx - i0
    row = cref[np.ix_(np.arange(Nref), i0)] * (1 - a) + cref[np.ix_(np.arange(Nref), i0 + 1)] * a
    out = row[i0, :] * (1 - a)[:, None] + row[i0 + 1, :] * a[:, None]
    return out


def main():
    import os
    Nref = 513
    if os.path.exists("ref513.npy"):
        cref = np.load("ref513.npy")
        print("reference: loaded ref513.npy", flush=True)
    else:
        print(f"reference: bp compact at N={Nref} ...", flush=True)
        cref, _, _, _, _ = solve_scheme(Nref, "bp")
        np.save("ref513.npy", cref)
    results = {}
    for N in (65, 129, 257):
        results[str(N)] = {}
        for scheme in ("up1", "up3", "tvd", "compact", "bp"):
            c, c0, h, nst, wall = solve_scheme(N, scheme)
            m = metrics(c, interp_to(cref, Nref, N), c0)
            m["ms_per_step"] = 1000.0 * wall / nst
            m["steps"] = nst
            results[str(N)][scheme] = m
            print(f"N={N:4d} {scheme:8s} L2={m['rel_l2']:.3f} "
                  f"rng=[{m['min']:+.3f},{m['max']:.3f}] IoU={m['iou']:.2f} "
                  f"dM/M={m['mass_drift']:.2e} {m['ms_per_step']:.1f} ms/step", flush=True)
    with open("deepen_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote deepen_results.json")


if __name__ == "__main__":
    main()
