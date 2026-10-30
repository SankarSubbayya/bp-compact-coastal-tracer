# Referee comparison (Ocean Modelling R1/R2): bound-preserving compact vs the positivity-preserving
# finite-volume schemes used in ROMS -- MPDATA (Smolarkiewicz 1984; nonoscillatory option,
# Smolarkiewicz & Grabowski 1990) and HSIMT-TVD (Wu & Zhu 2010; formulas as coded in ROMS
# step3d_t.F / mod_scalars.F, cc1=1/4, cc2=1/2, cc3=1/12).
# Same grid, flow, CFL and tests for every scheme. Scored at maximum deformation (t=T/2) against the
# exact solution (semi-Lagrangian backtrace of the analytic initial condition), because the full-cycle
# test rewards non-dissipative schemes whose errors undo themselves on flow reversal.
# FV schemes use node-centred control volumes (half cells at the walls) with face volume fluxes taken
# from the streamfunction, so the discrete flow is exactly non-divergent and has no flux through walls.
import os, paths as P
import json, sys, time
import numpy as np
sys.path.insert(0, P.DEV)
import advdiff as A
import sharp as Sh
from compact_stable import compact_d1_mats, compact_d1
from gen_deepen import muscl_adv_1d, uw3

T = 8.0
CFL = 0.35
EPS = 1e-15


def gauss_ic(X, Y):
    return np.exp(-((X - 0.5) ** 2 + (Y - 0.35) ** 2) / (2 * 0.08 ** 2))


def slot_ic(X, Y):
    return Sh.slotted_cylinder(X, Y)


def exact(ic, N, tc, nsub=400):
    """Backtrace every node from tc to 0 (RK4) and evaluate the analytic initial condition there."""
    x = np.linspace(0, 1, N); X, Y = np.meshgrid(x, x); px, py = X.copy(), Y.copy(); d = tc / nsub
    for n in range(nsub):
        s = tc - n * d
        u1, v1 = A.velocity(px, py, s, T)
        u2, v2 = A.velocity(px - .5*d*u1, py - .5*d*v1, s - .5*d, T)
        u3, v3 = A.velocity(px - .5*d*u2, py - .5*d*v2, s - .5*d, T)
        u4, v4 = A.velocity(px - d*u3, py - d*v3, s - d, T)
        px = px - d/6*(u1 + 2*u2 + 2*u3 + u4); py = py - d/6*(v1 + 2*v2 + 2*v3 + v4)
    return ic(np.clip(px, 0, 1), np.clip(py, 0, 1))


class Grid:
    def __init__(self, N):
        self.N = N; self.x = np.linspace(0, 1, N); self.h = self.x[1] - self.x[0]
        self.X, self.Y = np.meshgrid(self.x, self.x)
        e = np.concatenate([[0.0], 0.5 * (self.x[:-1] + self.x[1:]), [1.0]])   # CV edges, length N+1
        self.e = e; self.L = np.diff(e)                                          # CV widths (h/2 at walls)
        self.V = np.outer(self.L, self.L)                                        # V[j, i]

    def psi(self, xx, yy, t):
        return (1 / np.pi) * np.sin(np.pi * xx) ** 2 * np.sin(np.pi * yy) ** 2 * np.cos(np.pi * t / T)

    def fluxes(self, t):
        """Volume fluxes: Qx[j, f] through x-faces f=0..N (f=0, N are walls); Qy[g, i] through y-faces."""
        e = self.e
        P = self.psi(e[None, :], e[:, None], t)          # psi at CV corners, P[g, f]
        Qx = P[1:, :] - P[:-1, :]                        # u = dpsi/dy  -> shape (N, N+1)
        Qy = -(P[:, 1:] - P[:, :-1])                     # v = -dpsi/dx -> shape (N+1, N)
        return Qx, Qy


def div_update(g, c, Fx, Fy, dt):
    return c - dt / g.V * ((Fx[:, 1:] - Fx[:, :-1]) + (Fy[1:, :] - Fy[:-1, :]))


def donor(c, Qx, Qy):
    Fx = np.zeros_like(Qx); Fy = np.zeros_like(Qy)
    Fx[:, 1:-1] = np.maximum(Qx[:, 1:-1], 0) * c[:, :-1] + np.minimum(Qx[:, 1:-1], 0) * c[:, 1:]
    Fy[1:-1, :] = np.maximum(Qy[1:-1, :], 0) * c[:-1, :] + np.minimum(Qy[1:-1, :], 0) * c[1:, :]
    return Fx, Fy


def mpdata_step(g, c, t, dt, iord=2, nonosc=False):
    Qx, Qy = g.fluxes(t + 0.5 * dt)
    h = g.h
    Cx = Qx / g.L[:, None] * dt / h                  # Courant numbers at x-faces (rows j)
    Cy = Qy / g.L[None, :] * dt / h                  # at y-faces (cols i)
    cn = c
    Fx, Fy = donor(c, Qx, Qy)
    c = div_update(g, c, Fx, Fy, dt)
    for _ in range(iord - 1):
        p = np.pad(c, 1, mode="edge")
        # x-faces f=1..N-1 between nodes f-1 (L) and f (R), rows j
        cL, cR = c[:, :-1], c[:, 1:]
        Ax = (cR - cL) / (cR + cL + EPS)
        top = p[2:, 1:-2] + p[2:, 2:-1]; bot = p[:-2, 1:-2] + p[:-2, 2:-1]
        Bx = (top - bot) / (top + bot + EPS)
        cyb = 0.25 * (Cy[:-1, :-1] + Cy[1:, :-1] + Cy[:-1, 1:] + Cy[1:, 1:])
        cx = Cx[:, 1:-1]
        Ux = (np.abs(cx) - cx ** 2) * Ax - 0.5 * cx * cyb * Bx
        # y-faces g=1..N-1 between nodes g-1 (B) and g (T), cols i
        cB, cT = c[:-1, :], c[1:, :]
        Ay = (cT - cB) / (cT + cB + EPS)
        rgt = p[1:-2, 2:] + p[2:-1, 2:]; lft = p[1:-2, :-2] + p[2:-1, :-2]
        By = (rgt - lft) / (rgt + lft + EPS)
        cxb = 0.25 * (Cx[:-1, :-1] + Cx[:-1, 1:] + Cx[1:, :-1] + Cx[1:, 1:])
        cy = Cy[1:-1, :]
        Uy = (np.abs(cy) - cy ** 2) * Ay - 0.5 * cy * cxb * By
        Qax = np.zeros_like(Qx); Qay = np.zeros_like(Qy)
        Qax[:, 1:-1] = Ux * h * g.L[:, None] / dt
        Qay[1:-1, :] = Uy * h * g.L[None, :] / dt
        Fax, Fay = donor(c, Qax, Qay)
        lx = np.ones_like(Ux); ly = np.ones_like(Uy)
        if nonosc:   # Zalesak/FCT limiting of the antidiffusive fluxes (Smolarkiewicz & Grabowski 1990)
            def nb(a, f):
                q = np.pad(a, 1, mode="edge")
                return f(f(f(f(a, q[:-2, 1:-1]), q[2:, 1:-1]), q[1:-1, :-2]), q[1:-1, 2:])
            cmax = np.maximum(nb(cn, np.maximum), nb(c, np.maximum))
            cmin = np.minimum(nb(cn, np.minimum), nb(c, np.minimum))
            fin = (np.maximum(Fax[:, :-1], 0) - np.minimum(Fax[:, 1:], 0)
                   + np.maximum(Fay[:-1, :], 0) - np.minimum(Fay[1:, :], 0)) * dt / g.V
            fout = (np.maximum(Fax[:, 1:], 0) - np.minimum(Fax[:, :-1], 0)
                    + np.maximum(Fay[1:, :], 0) - np.minimum(Fay[:-1, :], 0)) * dt / g.V
            bup = (cmax - c) / (fin + EPS); bdn = (c - cmin) / (fout + EPS)
            fx = Fax[:, 1:-1]
            lx = np.where(fx > 0, np.minimum(1, np.minimum(bdn[:, :-1], bup[:, 1:])),
                          np.minimum(1, np.minimum(bup[:, :-1], bdn[:, 1:])))
            Fax[:, 1:-1] = lx * fx
            fy = Fay[1:-1, :]
            ly = np.where(fy > 0, np.minimum(1, np.minimum(bdn[:-1, :], bup[1:, :])),
                          np.minimum(1, np.minimum(bup[:-1, :], bdn[1:, :])))
            Fay[1:-1, :] = ly * fy
        c = div_update(g, c, Fax, Fay, dt)
        # Further corrective passes use the previous pass's (limited) antidiffusive pseudo-velocities.
        Cx = np.zeros_like(Cx); Cx[:, 1:-1] = lx * Ux
        Cy = np.zeros_like(Cy); Cy[1:-1, :] = ly * Uy
    return c


def hsimt_faces(c, C, axis):
    """HSIMT-TVD face values (ROMS step3d_t.F) along `axis` for interior faces; C = Courant at those faces."""
    cc1, cc2, cc3, eps1 = 0.25, 0.5, 1.0 / 12.0, 1e-12
    if axis == 0:      # work along rows: transpose so faces run along the last axis
        return hsimt_faces(c.T, C.T, 1).T
    n = c.shape[1]
    grad = np.zeros((c.shape[0], n + 1)); grad[:, 1:-1] = c[:, 1:] - c[:, :-1]   # face f: c[f]-c[f-1]
    Ka = np.zeros_like(grad); Ka[:, 1:-1] = 1.0 - np.abs(C)
    oKa = np.where(Ka <= eps1, 0.0, 1.0 / np.maximum(Ka, eps1))
    sw = np.zeros((c.shape[0], n - 1))
    f = np.arange(1, n)
    gf, kf, okf = grad[:, f], Ka[:, f], oKa[:, f]
    small = np.abs(gf) <= eps1
    a1 = cc1 * kf + cc2 - cc3 * okf; b1 = -cc1 * kf + cc2 + cc3 * okf
    with np.errstate(divide="ignore", invalid="ignore"):
        rL = np.where(small, 0.0, grad[:, f - 1] / np.where(small, 1, gf)); rkL = np.where(small, 0.0, Ka[:, f - 1] * okf)
        rR = np.where(small, 0.0, grad[:, f + 1] / np.where(small, 1, gf)); rkR = np.where(small, 0.0, Ka[:, f + 1] * okf)
    psiL = np.maximum(0, np.minimum(np.minimum(2.0, 2.0 * rL * rkL), a1 + b1 * rL))
    psiR = np.maximum(0, np.minimum(np.minimum(2.0, 2.0 * rR * rkR), a1 + b1 * rR))
    pos = C >= 0
    sw = np.where(pos, c[:, :-1] + 0.5 * psiL * gf * kf, c[:, 1:] - 0.5 * psiR * gf * kf)
    return sw


def hsimt_step(g, c, t, dt, predictor=True):
    """HSIMT as time-stepped in ROMS: pre_step3d.F forms the n+1/2 tracer with a half-step donor-cell predictor
    (Gamma = 1/2, both directions at once) and step3d_t.F computes the HSIMT face values from that state, then applies
    the flux divergence over the full step to the old tracer. predictor=False is the earlier version of this script,
    which took the face values from the old tracer (no transverse coupling; first-order in time for oblique flow)."""
    Qx, Qy = g.fluxes(t + 0.5 * dt)
    Cx = Qx[:, 1:-1] / g.L[:, None] * dt / g.h
    Cy = Qy[1:-1, :] / g.L[None, :] * dt / g.h
    cs = div_update(g, c, *donor(c, Qx, Qy), 0.5 * dt) if predictor else c
    Fx = np.zeros_like(Qx); Fy = np.zeros_like(Qy)
    Fx[:, 1:-1] = Qx[:, 1:-1] * hsimt_faces(cs, Cx, 1)
    Fy[1:-1, :] = Qy[1:-1, :] * hsimt_faces(cs, Cy, 0)
    return div_update(g, c, Fx, Fy, dt)


def hsimt_split_step(g, c, t, dt):
    """HSIMT face values in a directionally split (Strang: x dt/2, y dt, x dt/2) update. Each sweep advances the tracer
    mass and a pseudo-volume, V* = V - dt div_axis(Q), so that a uniform tracer stays uniform; after the three sweeps
    the volume returns to V because the face fluxes are divergence-free, and the final division uses that known volume
    (dividing by the round-off-perturbed pseudo-volume loses ~1e-14 of the mass over a run). Each one-dimensional sweep
    is the two-level HSIMT update (Wu and Zhu 2010) with its Courant-dependent coefficients."""
    Qx, Qy = g.fluxes(t + 0.5 * dt)
    m = g.V * c; V = g.V.copy()
    for axis, h in ((1, 0.5 * dt), (0, dt), (1, 0.5 * dt)):
        cc = m / V
        if axis == 1:
            Cx = Qx[:, 1:-1] / g.L[:, None] * h / g.h
            F = np.zeros_like(Qx); F[:, 1:-1] = Qx[:, 1:-1] * hsimt_faces(cc, Cx, 1)
            m = m - h * (F[:, 1:] - F[:, :-1]); V = V - h * (Qx[:, 1:] - Qx[:, :-1])
        else:
            Cy = Qy[1:-1, :] / g.L[None, :] * h / g.h
            F = np.zeros_like(Qy); F[1:-1, :] = Qy[1:-1, :] * hsimt_faces(cc, Cy, 0)
            m = m - h * (F[1:, :] - F[:-1, :]); V = V - h * (Qy[1:, :] - Qy[:-1, :])
    return m / g.V


def muscl_faces(c, Q, axis):
    """MUSCL face values with the superbee limiter (van Leer 1979; Sweby 1984) at interior faces along `axis`, upwinded
    by the sign of the face volume flux Q; first order (zero slope) where the upwind stencil leaves the domain."""
    if axis == 0:
        return muscl_faces(c.T, Q.T, 1).T
    n = c.shape[1]; f = np.arange(1, n)                       # face f between nodes f-1 and f
    p = np.pad(c, ((0, 0), (2, 2)), mode="edge")
    cm2, cm1, c0, c1 = p[:, f], p[:, f + 1], p[:, f + 2], p[:, f + 3]   # nodes f-2, f-1, f, f+1
    d = c0 - cm1
    def phi(r):
        return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rl = np.where(np.abs(d) > 1e-300, (cm1 - cm2) / d, 0.0)
        rr = np.where(np.abs(d) > 1e-300, (c1 - c0) / d, 0.0)
    okl = (f - 2) >= 0; okr = (f + 1) <= n - 1
    left = cm1 + 0.5 * np.where(okl, phi(rl), 0.0) * d
    right = c0 - 0.5 * np.where(okr, phi(rr), 0.0) * d
    return np.where(Q >= 0, left, right)


def muscl_fv_step(g, c, t, dt):
    """Flux-form MUSCL-superbee on the control volumes, two-stage SSP Runge-Kutta (Heun)."""
    def flux(cc, tt):
        Qx, Qy = g.fluxes(tt); Fx = np.zeros_like(Qx); Fy = np.zeros_like(Qy)
        Fx[:, 1:-1] = Qx[:, 1:-1] * muscl_faces(cc, Qx[:, 1:-1], 1)
        Fy[1:-1, :] = Qy[1:-1, :] * muscl_faces(cc, Qy[1:-1, :], 0)
        return Fx, Fy
    c1 = div_update(g, c, *flux(c, t), dt)
    return 0.5 * c + 0.5 * div_update(g, c1, *flux(c1, t + dt), dt)


def run(N, scheme, ic, t_end):
    g = Grid(N); h = g.h; X, Y = g.X, g.Y
    c0 = ic(X, Y); c = c0.copy()
    dt = CFL * h; nst = int(round(t_end / dt)); dt = t_end / nst
    abx = compact_d1_mats(N)
    gmin, gmax = float(c0.min()), float(c0.max())

    def adv_HO(cc, t):
        u, v = A.velocity(X, Y, t, T)
        return -(u * compact_d1(cc, h, 1, abx) + v * compact_d1(cc, h, 0, abx))

    def rk4(cc, t, rhs):
        k1 = rhs(cc, t); k2 = rhs(cc + .5*dt*k1, t + .5*dt)
        k3 = rhs(cc + .5*dt*k2, t + .5*dt); k4 = rhs(cc + dt*k3, t + dt)
        return cc + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    t0 = time.perf_counter()
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
        elif scheme == "mpdata":
            c = mpdata_step(g, c, t, dt, iord=2)
        elif scheme == "mpdata3":
            c = mpdata_step(g, c, t, dt, iord=3)
        elif scheme == "mpdata_no":
            c = mpdata_step(g, c, t, dt, iord=2, nonosc=True)
        elif scheme == "mpdata3_no":
            c = mpdata_step(g, c, t, dt, iord=3, nonosc=True)
        elif scheme == "hsimt":
            c = hsimt_step(g, c, t, dt)
        elif scheme == "muscl_fv":
            c = muscl_fv_step(g, c, t, dt)
        elif scheme == "donor_fv":
            Qx, Qy = g.fluxes(t + 0.5 * dt)
            c = div_update(g, c, *donor(c, Qx, Qy), dt)
        elif scheme == "hsimt_split":
            c = hsimt_split_step(g, c, t, dt)
        elif scheme == "hsimt_nopred":
            c = hsimt_step(g, c, t, dt, predictor=False)
        else:
            raise ValueError(scheme)
    wall = time.perf_counter() - t0
    return c, c0, g, wall


def score(c, ref, c0, g, wall):
    w = g.V
    d = {"rel_l2": float(np.sqrt((w*(c-ref)**2).sum() / (w*ref**2).sum())),
         "rel_l1": float((w*np.abs(c-ref)).sum() / (w*np.abs(ref)).sum()),
         "min": float(c.min()), "max": float(c.max()),
         "mass_rel": float((w*c).sum() / (w*c0).sum() - 1.0),
         "wall_s": round(wall, 2)}
    if ref.max() > 0.9 and ref.min() >= 0 and np.isin(np.unique(np.round(c0, 12)), [0, 1]).all():
        a, b = c > 0.5, ref > 0.5
        d["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
    return d


SCHEMES = ("up1", "up3", "tvd", "compact", "bp", "bpg", "mpdata", "mpdata3", "mpdata_no", "mpdata3_no", "hsimt")

if __name__ == "__main__":
    out = {"cfl": CFL, "slotted_T2": {}, "gauss_T2": {}}
    for N in (65, 129, 257):
        ref = exact(slot_ic, N, T / 2)
        out["slotted_T2"][str(N)] = {}
        for s in SCHEMES:
            c, c0, g, wall = run(N, s, slot_ic, T / 2)
            m = score(c, ref, c0, g, wall); out["slotted_T2"][str(N)][s] = m
            print(f"slot N={N} {s:10s} L2={m['rel_l2']:.3f} L1={m['rel_l1']:.3f} IoU={m.get('iou', 0):.3f} "
                  f"rng=[{m['min']:+.3f},{m['max']:.3f}] mass={m['mass_rel']:+.2e} t={m['wall_s']}s", flush=True)
    for N in (33, 65, 129, 257):
        ref = exact(gauss_ic, N, T / 2)
        out["gauss_T2"][str(N)] = {}
        for s in SCHEMES:
            c, c0, g, wall = run(N, s, gauss_ic, T / 2)
            m = score(c, ref, c0, g, wall); out["gauss_T2"][str(N)][s] = m
            print(f"gauss N={N} {s:10s} L2={m['rel_l2']:.2e} rng=[{m['min']:+.3f},{m['max']:.3f}] "
                  f"mass={m['mass_rel']:+.2e} t={m['wall_s']}s", flush=True)
    json.dump(out, open("mpdata_results.json", "w"), indent=1)
    print("wrote mpdata_results.json")
