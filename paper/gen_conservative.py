# Mass-conserving bounded compact scheme (revision for Ocean Modelling R2 point 1; JCP revision 2026-10-28).
# Flux form on the node-centred control volumes of gen_mpdata.Grid:
#   face values of c by 4th-order compact (Pade) midpoint interpolation (Lele 1992):
#       (1/4) c_{f-1} + c_f + (1/4) c_{f+1} = (3/4) (c_{node f-1} + c_{node f}),
#   with explicit one-sided closures at the first and last interior faces; for constant velocity the flux difference
#   reproduces the 4th-order Pade first derivative exactly. For comparison the face values can instead come from the
#   explicit centred fourth-order ("c4") or linear upwind fifth-order ("u5") cell-average reconstructions of
#   Chaplin and Colella (2017), with the same wall closures.
#   Face flux = face volume flux x face value, plus (TRANSVERSE) the fourth-order transverse product correction
#   (L^3/12)(dU/dtau)(dc/dtau) along each face (Chaplin and Colella 2017, eq. 29; McCorquodale and Colella 2011),
#   by centred differences of the face-averaged velocity Q/L and of the face values, omitted in the wall rows.
#   RK4 in flux form (time-averaged stage fluxes), then one Zalesak (1979) flux-corrected transport step per time step
#   (the single-stage form of Chaplin and Colella 2017) against the donor-cell flux of c^n with the RK4-averaged face
#   fluxes, with global (initial min/max), local (min/max of c^n and the low-order solution over the 5-point
#   neighbourhood) or local extremum-preserving bounds (Chaplin and Colella 2017, Sec. 4.2).
#   A grid with `moving = True` supplies time-dependent face fluxes whose divergence changes the volumes; the volumes
#   are then advanced by the same RK4 stages from the face fluxes, so a uniform tracer is preserved exactly.
#   Where the depth D = V/A varies, c = <Dc>/<D> is a depth-weighted average, and reconstructing it directly (the
#   default) is second order in the depth-weighting term. Schemes with the suffix "_da" (DEPTH_AWARE) use instead the
#   ratio r of the reconstructed area means <Dc> and <D>, with the transverse correction
#   (L^3/12)(dU/dtau - (U/D_f) dD_f/dtau)(dr/dtau), U = Q/L: fourth order for the operator, but not energy-neutral,
#   so that without the limiter grid-scale noise grows where the depth changes steeply (steep_depth.py).
#   Uniform depth: c is reconstructed directly by both (identical).
import json, sys, time
import numpy as np
from scipy.linalg import solve_banded
import gen_mpdata as M

EPS = 1e-15


REPRESENTATION = "nodal"   # "nodal": point samples (closure = cubic interpolation of nodes 0-3);
                           # "cv": control-volume averages with half-width wall volumes (closures exact for cubics)
RECON = "compact"          # face reconstruction: "compact", "c4" (explicit centred 4th order), "u5" (upwind 5th order)
TRANSVERSE = True          # fourth-order transverse product correction of the face flux
ALLOW_TRANSVERSE = True    # set False to reproduce the point-sampled protocol of the earlier version (Appendix A)
W_HALF1 = (16 / 35, 157 / 210, -53 / 210, 1 / 21)    # face 1/2 from the half-width wall volume and the next three
W_HALF2 = (-16 / 105, 73 / 105, 37 / 70, -1 / 14)    # face 3/2 from the same four volumes (needed by the c4 stencil)
W_NODAL = (5 / 16, 15 / 16, -5 / 16, 1 / 16)          # face 1/2 from point samples at nodes 0-3
CC_FLAG = False            # local_ep: no antidiffusion at extrema where the Laplacian changes sign nearby (eqs. 75-76);
                           # off: as we implemented it, it removed most antidiffusion at the deformed filament's extrema
                           # (smooth error 0.44 with it, 0.11 without, 65^2), contrary to the method's reported behaviour
CC_SMOOTH_BOUNDS = True    # local_ep: widened bounds at smooth extrema (eqs. 61-74)
DEPTH_AWARE = False        # "_da" schemes: face value = reconstructed <Dc> / reconstructed <D> (off: c directly)
H_FLOOR = 0.5              # faces whose reconstructed depth is below H_FLOOR x the smaller adjacent depth use c directly
FALLBACK = [0, 0]          # [faces that used the fallback, depth-aware faces evaluated] since the last reset


def face_interp(c, axis):
    """Compact 4th-order values at interior faces f=1..n-1 along `axis` (faces between nodes f-1 and f)."""
    if axis == 0:
        return face_interp(c.T, 1).T
    m, n = c.shape
    k = n - 1                                     # interior faces
    ab = np.zeros((3, k))
    ab[1, :] = 1.0
    ab[0, 2:] = 0.25                              # super-diagonal A[r, r+1] for rows r=1..k-2
    ab[2, :-2] = 0.25                             # sub-diagonal A[r, r-1] for rows r=1..k-2
    rhs = np.empty((k, m))
    ct = c.T                                      # (n, m)
    rhs[1:-1] = 0.75 * (ct[1:-2] + ct[2:-1])      # faces 2..n-2 (between nodes f-1, f)
    w = W_HALF1 if REPRESENTATION == "cv" else W_NODAL
    rhs[0] = w[0] * ct[0] + w[1] * ct[1] + w[2] * ct[2] + w[3] * ct[3]
    rhs[-1] = w[0] * ct[-1] + w[1] * ct[-2] + w[2] * ct[-3] + w[3] * ct[-4]
    return solve_banded((1, 1), ab, rhs).T        # (m, k)


def face_c4(c, axis):
    """Explicit centred fourth-order cell-average reconstruction (-1, 7, 7, -1)/12 at interior faces, with the
    cubic-exact half-width-volume closures at the first two and last two faces (control-volume representation)."""
    if axis == 0:
        return face_c4(c.T, 1).T
    assert REPRESENTATION == "cv"
    m, n = c.shape
    out = np.empty((m, n - 1))                    # face f in 1..n-1 at index f-1
    f = np.arange(3, n - 2)
    out[:, f - 1] = (-c[:, f - 2] + 7 * c[:, f - 1] + 7 * c[:, f] - c[:, f + 1]) / 12
    left, right = c[:, :4], c[:, ::-1][:, :4]     # the half-width wall volume first
    out[:, 0] = left @ np.array(W_HALF1); out[:, -1] = right @ np.array(W_HALF1)
    out[:, 1] = left @ np.array(W_HALF2); out[:, -2] = right @ np.array(W_HALF2)
    return out


def face_u5(c, Q, axis):
    """Linear upwind fifth-order cell-average reconstruction (2, -13, 47, 27, -3)/60 (upwind by the sign of the face
    flux Q) where the five-volume stencil contains only full-width volumes; the c4 values elsewhere."""
    if axis == 0:
        return face_u5(c.T, Q.T, 1).T
    m, n = c.shape
    out = face_c4(c, 1)
    f = np.arange(4, n - 3)
    pos = (2 * c[:, f - 3] - 13 * c[:, f - 2] + 47 * c[:, f - 1] + 27 * c[:, f] - 3 * c[:, f + 1]) / 60
    neg = (2 * c[:, f + 2] - 13 * c[:, f + 1] + 47 * c[:, f] + 27 * c[:, f - 1] - 3 * c[:, f - 2]) / 60
    out[:, f - 1] = np.where(Q[:, f - 1] >= 0, pos, neg)
    return out


def face_values(c, Q, axis):
    if RECON == "compact":
        return face_interp(c, axis)
    if RECON == "c4":
        return face_c4(c, axis)
    if RECON == "u5":
        return face_u5(c, Q, axis)
    raise ValueError(RECON)


DISSIPATION = False   # upwind-biased dissipation (as in sfbay_conservative.py); set per scheme in run()


def dissipation(c, q, axis):
    """|q| x upwind-biased correction at interior faces f=1..n-1 along `axis`: -d5/60 (6-node stencil),
    +d3/12 (4 nodes, next to walls), -d1/2 (first and last face)."""
    if axis == 0:
        return dissipation(c.T, q.T, 1).T
    n = c.shape[1]; f = np.arange(1, n)
    p = np.pad(c, ((0, 0), (3, 3)), mode="edge")
    s = [p[:, f + 3 + o] for o in (-3, -2, -1, 0, 1, 2)]           # nodes f-3 .. f+2 (face between f-1 and f)
    d5 = s[5] - 5 * s[4] + 10 * s[3] - 10 * s[2] + 5 * s[1] - s[0]
    d3 = s[4] - 3 * s[3] + 3 * s[2] - s[1]
    d1 = s[3] - s[2]
    ok5 = (f - 3 >= 0) & (f + 2 <= n - 1); ok3 = (f - 2 >= 0) & (f + 1 <= n - 1) & ~ok5
    corr = np.where(ok5, -d5 / 60.0, np.where(ok3, d3 / 12.0, -d1 / 2.0))
    return np.abs(q) * corr


def face_ratio(c, D, Q, axis):
    """Depth-aware face values along `axis`: (reconstructed <Dc>) / (reconstructed <D>), and the reconstructed face
    depth; c reconstructed directly where the face depth is below H_FLOOR x the smaller adjacent depth."""
    Df = face_values(D, Q, axis)
    lo = np.minimum(D[:, :-1], D[:, 1:]) if axis == 1 else np.minimum(D[:-1, :], D[1:, :])
    bad = ~(Df >= H_FLOOR * lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = face_values(D * c, Q, axis) / Df
    FALLBACK[0] += int(bad.sum()); FALLBACK[1] += bad.size
    if bad.any():
        r = np.where(bad, face_values(c, Q, axis), r)
    return r, Df, bad


def ho_fluxes(c, Qx, Qy, g=None, V=None):
    """High-order face fluxes of the stage concentration c; V = the stage volumes (used where the depth varies)."""
    Fx = np.zeros_like(Qx); Fy = np.zeros_like(Qy)
    D = None
    if DEPTH_AWARE and V is not None and g is not None:
        D = V / np.outer(g.L, g.L)                          # cell depth = volume / area
        if D.max() == D.min():
            D = None                                        # uniform depth: the ratio reduces to c itself
    if D is None:
        cx = face_values(c, Qx[:, 1:-1], 1); cy = face_values(c, Qy[1:-1, :], 0)
    else:
        cx, Dx, bx = face_ratio(c, D, Qx[:, 1:-1], 1); cy, Dy, by = face_ratio(c, D, Qy[1:-1, :], 0)
    Fx[:, 1:-1] = Qx[:, 1:-1] * cx
    Fy[1:-1, :] = Qy[1:-1, :] * cy
    if TRANSVERSE and ALLOW_TRANSVERSE and g is not None:
        L, h = g.L, g.h
        ux = Qx[:, 1:-1] / L[:, None]                       # face-averaged velocity on x-faces (rows j)
        vy = Qy[1:-1, :] / L[None, :]                       # on y-faces (columns i)
        dux = (ux[2:] - ux[:-2]) / (2 * h); dvy = (vy[:, 2:] - vy[:, :-2]) / (2 * h)
        if D is not None:                                   # remove the depth-tracer covariance already in r
            dux = dux - np.where(bx[1:-1], 0.0, ux[1:-1] / Dx[1:-1] * (Dx[2:] - Dx[:-2]) / (2 * h))
            dvy = dvy - np.where(by[:, 1:-1], 0.0, vy[:, 1:-1] / Dy[:, 1:-1] * (Dy[:, 2:] - Dy[:, :-2]) / (2 * h))
        Fx[1:-1, 1:-1] += L[1:-1, None] ** 3 / 12 * dux * (cx[2:] - cx[:-2]) / (2 * h)
        Fy[1:-1, 1:-1] += L[None, 1:-1] ** 3 / 12 * dvy * (cy[:, 2:] - cy[:, :-2]) / (2 * h)
    if DISSIPATION:
        Fx[:, 1:-1] += dissipation(c, Qx[:, 1:-1], 1)
        Fy[1:-1, :] += dissipation(c, Qy[1:-1, :], 0)
    return Fx, Fy


def ho_fluxes_nodal(g, cs, t):
    """Variant: compact numerical flux of the nodal flux u*c (4th order for variable velocity; a constant
    tracer is then preserved to 4th order, not exactly), times the face length."""
    u, v = M.A.velocity(g.X, g.Y, t, M.T)
    Fx = np.zeros((g.N, g.N + 1)); Fy = np.zeros((g.N + 1, g.N))
    Fx[:, 1:-1] = face_interp(u * cs, 1) * g.L[:, None]
    Fy[1:-1, :] = face_interp(v * cs, 0) * g.L[None, :]
    return Fx, Fy


def neighbourhood(a, f):
    q = np.pad(a, 1, mode="edge")
    return f(f(f(f(a, q[:-2, 1:-1]), q[2:, 1:-1]), q[1:-1, :-2]), q[1:-1, 2:])


def divergence(Fx, Fy):
    return (Fx[:, 1:] - Fx[:, :-1]) + (Fy[1:, :] - Fy[:-1, :])


def cc_bounds(cn, clo, gmin, gmax, umax_cell, dt, h):
    """Local bounds with the smooth-extremum treatment of Chaplin and Colella (2017, Sec. 4.2), as we read it:
    box maxima/minima of c^n and the low-order solution over (2s+1)^2 volumes (s = 2 where the local Courant number
    dt max|u|/h >= 0.5, else 1); at smooth extrema (their eqs. 61-67) the bound is widened to twice the distance from
    c^n to the vertex of the local parabola, deconvolved to a point value (eqs. 68-74; their eq. 74 is read as the
    mirror image of eq. 73); cells where the Laplacian of c^n changes sign within the 3-point neighbourhood of an
    extremum can be flagged to receive no antidiffusive flux (eqs. 75-76; off by default, see CC_FLAG). The bounds are then clipped to the global
    bounds [gmin, gmax] of the data, so that the scheme stays within the physical range."""
    def box(a, s, f):
        p = np.pad(a, s, mode="edge"); n0, n1 = a.shape
        out = a.copy()
        for dj in range(-s, s + 1):
            for di in range(-s, s + 1):
                out = f(out, p[s + dj:s + dj + n0, s + di:s + di + n1])
        return out
    s2 = (dt * umax_cell / h) >= 0.5
    cmax = np.where(s2, np.maximum(box(cn, 2, np.maximum), box(clo, 2, np.maximum)),
                    np.maximum(box(cn, 1, np.maximum), box(clo, 1, np.maximum)))
    cmin = np.where(s2, np.minimum(box(cn, 2, np.minimum), box(clo, 2, np.minimum)),
                    np.minimum(box(cn, 1, np.minimum), box(clo, 1, np.minimum)))
    cmax0, cmin0 = cmax.copy(), cmin.copy()
    p2 = np.pad(clo, 2, mode="edge"); pn = np.pad(cn, 1, mode="edge")
    n0, n1 = cn.shape
    ext, const, d2q, qext_d = [], [], [], []
    for d in (0, 1):                                            # 0: along rows (j), 1: along columns (i)
        sl = (lambda k: p2[2 + k:2 + k + n0, 2:2 + n1]) if d == 0 else (lambda k: p2[2:2 + n0, 2 + k:2 + k + n1])
        q = {k: sl(k) for k in (-2, -1, 0, 1, 2)}
        dq = {k: q[k] - q[k - 1] for k in (0, 1, 2)}; dq[-1] = q[-1] - q[-2]
        tv = np.abs(dq[2]) + np.abs(dq[1]) + np.abs(dq[0]) + np.abs(dq[-1])
        ext.append((np.minimum(dq[0] * dq[1], dq[-1] * dq[2]) <= 0) & (1.25 * np.abs(q[2] - q[-2]) < tv))
        const.append(np.maximum(np.abs(np.maximum(np.maximum(q[-1], q[0]), q[1]) - q[0]),
                                np.abs(np.minimum(np.minimum(q[-1], q[0]), q[1]) - q[0])) <= 1e-14)
        sn = (lambda k: pn[1 + k:1 + k + n0, 1:1 + n1]) if d == 0 else (lambda k: pn[1:1 + n0, 1 + k:1 + k + n1])
        dd = sn(1) + sn(-1) - 2 * sn(0); slope = 0.5 * (sn(1) - sn(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = np.clip(np.where(dd != 0, -slope / dd, 0.0), -0.5, 0.5)
        d2q.append(dd); qext_d.append(0.5 * dd * xc ** 2 + slope * xc + cn - dd / 24)
    smooth = (ext[0] & ext[1]) | (ext[0] & const[1]) | (ext[1] & const[0])
    lap = d2q[0] + d2q[1]
    is_max = lap <= 0
    qext = np.where(is_max, np.maximum(np.maximum(qext_d[0], qext_d[1]), cmax),
                    np.minimum(np.minimum(qext_d[0], qext_d[1]), cmin))
    cmax = np.where(smooth & is_max, cn + 2 * np.abs(qext - cn), cmax)
    cmin = np.where(smooth & ~is_max, cn - 2 * np.abs(qext - cn), cmin)
    pl = np.pad(lap, 1, mode="edge")
    nb = [pl[1:-1, 1:-1], pl[:-2, 1:-1], pl[2:, 1:-1], pl[1:-1, :-2], pl[1:-1, 2:]]
    sign_change = (np.minimum.reduce(nb) < 0) & (np.maximum.reduce(nb) > 0)
    flagged = sign_change & (ext[0] | ext[1]) if CC_FLAG else np.zeros_like(sign_change)
    if not CC_SMOOTH_BOUNDS:
        cmax, cmin = cmax0, cmin0
    return np.clip(cmin, gmin, gmax), np.clip(cmax, gmin, gmax), flagged


def _step(g, cn, Vn, t, dt, bounds, gmin, gmax, nodal=False):
    """One step c^n -> c^{n+1}; returns (c^{n+1}, V^{n+1}). Fixed grids: V = g.V throughout."""
    moving = getattr(g, "moving", False)
    b = (0.0, 0.5, 0.5, 1.0); w = (1.0, 2.0, 2.0, 1.0)
    mn = Vn * cn
    Fsx = 0.0; Fsy = 0.0; Qsx = 0.0; Qsy = 0.0
    prevF = prevQ = None
    for s in range(4):
        ts = t + b[s] * dt
        if s == 0:
            cs, Vs = cn, Vn
        else:
            ms = mn - b[s] * dt * divergence(*prevF)
            Vs = Vn - b[s] * dt * divergence(*prevQ) if moving else Vn
            cs = ms / Vs
        if nodal:
            Qx, Qy = g.fluxes(ts); Fx, Fy = ho_fluxes_nodal(g, cs, ts)
        else:
            Qx, Qy = g.fluxes(ts); Fx, Fy = ho_fluxes(cs, Qx, Qy, g, Vs)
        Fsx = Fsx + w[s] * Fx; Fsy = Fsy + w[s] * Fy; Qsx = Qsx + w[s] * Qx; Qsy = Qsy + w[s] * Qy
        prevF = (Fx, Fy); prevQ = (Qx, Qy)
    Fhx, Fhy = Fsx / 6.0, Fsy / 6.0
    Qbx, Qby = Qsx / 6.0, Qsy / 6.0                              # RK4-averaged face fluxes
    Vnp1 = Vn - dt * divergence(Qbx, Qby) if moving else Vn
    Flx, Fly = M.donor(cn, Qbx, Qby)
    mlo = mn - dt * divergence(Flx, Fly)
    cl = mlo / Vnp1
    Ax, Ay = Fhx - Flx, Fhy - Fly                 # antidiffusive fluxes (zero at walls)
    flagged = None
    if bounds == "global":
        cmax = np.full_like(cn, gmax); cmin = np.full_like(cn, gmin)
    elif bounds == "local":
        cmax = np.maximum(neighbourhood(cn, np.maximum), neighbourhood(cl, np.maximum))
        cmin = np.minimum(neighbourhood(cn, np.minimum), neighbourhood(cl, np.minimum))
    elif bounds == "local_ep":
        U = np.maximum(np.maximum(np.abs(Qbx[:, :-1]), np.abs(Qbx[:, 1:])) / g.L[:, None],
                       np.maximum(np.abs(Qby[:-1, :]), np.abs(Qby[1:, :])) / g.L[None, :])
        cmin, cmax, flagged = cc_bounds(cn, cl, gmin, gmax, U, dt, g.h)
    else:
        raise ValueError(bounds)
    pin = (np.maximum(Ax[:, :-1], 0) - np.minimum(Ax[:, 1:], 0)
           + np.maximum(Ay[:-1, :], 0) - np.minimum(Ay[1:, :], 0)) * dt
    pout = (np.maximum(Ax[:, 1:], 0) - np.minimum(Ax[:, :-1], 0)
            + np.maximum(Ay[1:, :], 0) - np.minimum(Ay[:-1, :], 0)) * dt
    rp = np.minimum(1.0, Vnp1 * (cmax - cl) / (pin + EPS))
    rm = np.minimum(1.0, Vnp1 * (cl - cmin) / (pout + EPS))
    if flagged is not None:
        rp = np.where(flagged, 0.0, rp); rm = np.where(flagged, 0.0, rm)
    ax = Ax[:, 1:-1]
    Cx = np.where(ax > 0, np.minimum(rm[:, :-1], rp[:, 1:]), np.minimum(rp[:, :-1], rm[:, 1:]))
    ay = Ay[1:-1, :]
    Cy = np.where(ay > 0, np.minimum(rm[:-1, :], rp[1:, :]), np.minimum(rp[:-1, :], rm[1:, :]))
    Lx = np.zeros_like(Ax); Ly = np.zeros_like(Ay)
    Lx[:, 1:-1] = Cx * ax; Ly[1:-1, :] = Cy * ay
    return (mlo - dt * divergence(Lx, Ly)) / Vnp1, Vnp1


def step(g, c, t, dt, bounds, gmin, gmax, nodal=False):
    return _step(g, c, g.V, t, dt, bounds, gmin, gmax, nodal)[0]


def _unlimited_step(g, cn, Vn, t, dt):
    """Conservative compact flux without limiting (diagnostic): RK4 in flux form."""
    moving = getattr(g, "moving", False)
    b = (0.0, 0.5, 0.5, 1.0); w = (1.0, 2.0, 2.0, 1.0)
    mn = Vn * cn; Fsx = Fsy = Qsx = Qsy = 0.0; prevF = prevQ = None
    for s in range(4):
        if s == 0:
            cs, Vs = cn, Vn
        else:
            Vs = Vn - b[s] * dt * divergence(*prevQ) if moving else Vn
            cs = (mn - b[s] * dt * divergence(*prevF)) / Vs
        Qx, Qy = g.fluxes(t + b[s] * dt); Fx, Fy = ho_fluxes(cs, Qx, Qy, g, Vs)
        Fsx = Fsx + w[s] * Fx; Fsy = Fsy + w[s] * Fy; Qsx = Qsx + w[s] * Qx; Qsy = Qsy + w[s] * Qy
        prevF = (Fx, Fy); prevQ = (Qx, Qy)
    Vnp1 = Vn - dt * divergence(Qsx / 6.0, Qsy / 6.0) if moving else Vn
    return (mn - dt * divergence(Fsx / 6.0, Fsy / 6.0)) / Vnp1, Vnp1


def _donor_step(g, cn, Vn, t, dt):
    b = (0.0, 0.5, 0.5, 1.0); w = (1.0, 2.0, 2.0, 1.0)
    Qbx = sum(w[s] * g.fluxes(t + b[s] * dt)[0] for s in range(4)) / 6.0
    Qby = sum(w[s] * g.fluxes(t + b[s] * dt)[1] for s in range(4)) / 6.0
    Vnp1 = Vn - dt * divergence(Qbx, Qby) if getattr(g, "moving", False) else Vn
    return (Vn * cn - dt * divergence(*M.donor(cn, Qbx, Qby))) / Vnp1, Vnp1


# scheme -> (reconstruction, bounds or None for unlimited, dissipation, transverse correction)
SCHEME_SPEC = {
    "cfctg": ("compact", "global", False, True), "cfct": ("compact", "local", False, True),
    "cfcte": ("compact", "local_ep", False, True),
    "cfctgd": ("compact", "global", True, True), "cfctd": ("compact", "local", True, True),
    "ccompact": ("compact", None, False, True), "ccompactd": ("compact", None, True, True),
    "cfctg_nt": ("compact", "global", False, False), "ccompact_nt": ("compact", None, False, False),
    "c4fctg": ("c4", "global", False, True), "c4fcte": ("c4", "local_ep", False, True),
    "u5fctg": ("u5", "global", False, True), "u5fcte": ("u5", "local_ep", False, True),
    "cdonor": ("compact", "donor", False, True),
    "cfctg_da": ("compact", "global", False, True),   # "_da": depth-aware ratio face values where the depth varies
    "ccompact_da": ("compact", None, False, True),
}


def run(N, scheme, ic, t_end, grid=None, dt=None):
    """Run `scheme` from ic(X, Y) (or an array) to t_end on M.Grid(N) or the given grid; returns (c, c0, g, seconds).
    For a moving grid the initial array must be the volume-weighted (depth-weighted) cell averages."""
    global RECON, DISSIPATION, TRANSVERSE, DEPTH_AWARE
    g = grid if grid is not None else M.Grid(N)
    DEPTH_AWARE = scheme.endswith("_da"); FALLBACK[:] = [0, 0]
    c0 = ic(g.X, g.Y); c = c0.copy()
    if dt is None:
        dt = M.CFL * g.h
    nst = int(round(t_end / dt)); dt = t_end / nst
    gmin, gmax = float(c0.min()), float(c0.max())
    nodal = scheme in ("cfct4", "cfctg4")
    if nodal:
        RECON, bounds, DISSIPATION, TRANSVERSE = "compact", ("local" if scheme == "cfct4" else "global"), False, False
    else:
        RECON, bounds, DISSIPATION, TRANSVERSE = SCHEME_SPEC[scheme]
    V = g.volume(0.0) if getattr(g, "moving", False) else g.V
    t0 = time.perf_counter()
    for n in range(nst):
        t = n * dt
        if bounds is None:
            c, V = _unlimited_step(g, c, V, t, dt)
        elif bounds == "donor":
            c, V = _donor_step(g, c, V, t, dt)
        else:
            c, V = _step(g, c, V, t, dt, bounds, gmin, gmax, nodal)
    RECON, DISSIPATION, TRANSVERSE, DEPTH_AWARE = "compact", False, True, False
    g.V_final = V                                 # the volumes as the scheme advanced them (moving grids)
    return c, c0, g, time.perf_counter() - t0


SCHEMES = ("ccompactd", "cfctd", "cfctgd")   # with the dissipation used in the bay (earlier runs kept in the json)

if __name__ == "__main__":
    res = json.load(open("mpdata_results.json"))
    for test, ic, Ns in (("slotted_T2", M.slot_ic, (65, 129, 257)), ("gauss_T2", M.gauss_ic, (33, 65, 129, 257))):
        for N in Ns:
            ref = M.exact(ic, N, M.T / 2)
            for s in SCHEMES:
                c, c0, g, wall = run(N, s, ic, M.T / 2)
                m = M.score(c, ref, c0, g, wall); res[test][str(N)][s] = m
                print(f"{test} N={N} {s:9s} L2={m['rel_l2']:.4f} IoU={m.get('iou', 0):.3f} "
                      f"rng=[{m['min']:+.2e},{m['max']:.4f}] mass={m['mass_rel']:+.1e} t={m['wall_s']}s", flush=True)
    json.dump(res, open("mpdata_results.json", "w"), indent=1)
    print("updated mpdata_results.json")
