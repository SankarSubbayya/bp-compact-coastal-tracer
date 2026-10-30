"""The single-stage flux-corrected transport algorithm of Chaplin and Colella (2017, Communications in Applied
Mathematics and Computational Science 12(1), 1-24), implemented from the published article (the 2015 arXiv
preprint differs in the bound stencil and the widening formula; its eq. 30 is the article's), for comparison with the
bounded compact scheme (JCP revision, 2026-10-28; switches for two readings of the article added 2026-10-29). Equation numbers below are those of the published article.

High-order flux: RK4 time average of the stage fluxes (eqs. 15-21), face values by the centred fourth-order ("c4",
eq. 23) or upwind fifth-order ("u5", eq. 24; "u9", eq. 27, periodic tests only) interpolation, and the product rule:
fourth order (eq. 29) with C4, sixth order (eq. 30, derivatives to fourth order in the h^2 term and second order in the
h^4 term) with the upwind values, as the article recommends for fifth- and higher-order schemes. The article asks for
centred derivatives of appropriate accuracy without saying whether they approximate point values or face averages.
Here eq. 30's coefficients 3, 3, 2, which hold for derivatives of point values, are applied to centred differences of
the face values; so applied the rule is fourth order, with a smaller error constant than eq. 29 in our test. For
derivatives of face averages the sixth-order coefficients are -2, -2, 2 (SIX_FACE_AVG = True). Low-order flux:
corner transport upwind (CTU, Sec. 3), the upwind cell value corrected by half a step of the donor-cell transverse
update of that cell, times the face flux (the first-order CTU of Colella 1990; for constant velocity the update has the
convex weights (1-a)(1-b), a(1-b), (1-a)b, ab). Limiter (Sec. 4): preconstraint of the antidiffusive flux (eqs. 76-78),
bounds from 3^D boxes of q^n and the CTU solution (s = 1, eqs. 54-59), widened at smooth extrema (eqs. 60-73, signed:
q^n + 2(q_ext - q^n)), the flag of eqs. 74-75 (no antidiffusion in or out of a smooth extremum where the Laplacian
changes sign within its 3-point neighbourhood, as the article's text states), and Zalesak's coefficients (eqs. 47-53).
The bounds are not clipped to the range of the data. Points the article leaves open, fixed here: in eq. 71 the sign of
the curvature is that of the sum over directions; in eq. 78 the antidiffusive flux per unit face length is compared
with the CTU dissipation flux (|u|/2)(1 - sigma)|d2q_i + d2q_i+1|/2 (undivided second differences), the dimensionally
consistent form of "the dissipation term, scaled by the cell size" (the printed eq. 78 has a further factor h:
EQ78_H = True); the CTU construction above for non-constant
velocity. PREPRINT = True gives the 2015 preprint's bound stencil and smooth-minimum bound (s = 2 where the local Courant
number is >= 0.5; the literal lower bound q_min = q^n at smooth minima of its eq. 74) together with the fourth-order
product rule throughout, which is an earlier implementation choice of ours, not a difference of the preprint.

Two geometries: the periodic unit square of the article's tests (cc_validation.py) and the closed basin of the
benchmark (gen_mpdata.Grid, half-width wall volumes, fixed volumes; cc_benchmark.py). There the face values are
gen_conservative's c4 / u5 reconstructions with its cubic-exact wall closures (identical to the article's away from the
walls), the product-rule terms are omitted in the wall rows and the sixth-order terms reduce to the fourth-order rule
in the rows next to them, and the low-order and limiter steps use edge padding at the walls."""
import numpy as np
import gen_conservative as C

EPS = 1e-15
PREPRINT = False       # True: the 2015 preprint's bounds, with the fourth-order product rule (see the docstring)
SIX_FACE_AVG = False   # True: eq. 30 with the coefficients for derivatives of face averages (-2, -2, 2)
EQ78_H = False         # True: eq. 78 with the printed further factor h


def pad(a, k, periodic):
    return np.pad(a, k, mode="wrap" if periodic else "edge")


def donor(c, Qx, Qy, periodic):
    """Donor-cell fluxes on faces 0..N (x: Qx[j, f], face f between cells f-1 and f; y: Qy[g, i])."""
    N = c.shape[1]; f = np.arange(N + 1)
    left, right = c[:, (f - 1) % N], c[:, f % N]
    Fx = np.where(Qx > 0, Qx * left, Qx * right)
    M_ = c.shape[0]; g = np.arange(M_ + 1)
    below, above = c[(g - 1) % M_, :], c[g % M_, :]
    Fy = np.where(Qy > 0, Qy * below, Qy * above)
    if not periodic:
        Fx[:, [0, -1]] = 0.0; Fy[[0, -1], :] = 0.0
    return Fx, Fy


def divergence(Fx, Fy):
    return (Fx[:, 1:] - Fx[:, :-1]) + (Fy[1:, :] - Fy[:-1, :])


def ctu(c, V, Qx, Qy, dt, periodic):
    """First-order corner-transport-upwind fluxes: upwind value minus half a step of its donor transverse update."""
    Gx, Gy = donor(c, Qx, Qy, periodic)
    Ty = (Gy[1:, :] - Gy[:-1, :] - c * (Qy[1:, :] - Qy[:-1, :])) / V     # transverse (y) update rate of each cell
    Tx = (Gx[:, 1:] - Gx[:, :-1] - c * (Qx[:, 1:] - Qx[:, :-1])) / V
    qy = c - 0.5 * dt * Ty                                              # value carried through x-faces
    qx = c - 0.5 * dt * Tx                                              # through y-faces
    Fx, Fy = donor(qy, Qx, np.zeros_like(Qy), periodic)[0], donor(qx, np.zeros_like(Qx), Qy, periodic)[1]
    if not periodic:
        Fx[:, [0, -1]] = 0.0; Fy[[0, -1], :] = 0.0
    return Fx, Fy


def d2(a, axis, periodic):
    p = pad(a, 1, periodic)
    if axis == 1:
        return p[1:-1, 2:] + p[1:-1, :-2] - 2 * a
    return p[2:, 1:-1] + p[:-2, 1:-1] - 2 * a


def preconstrain(A, cn, ctd, Q, L, dt, h, axis, periodic):
    """Eqs. 76-78: zero the antidiffusive flux where it steepens the low-order solution near a discontinuity."""
    if axis == 0:
        return preconstrain(A.T, cn.T, ctd.T, Q.T, L, dt, h, 1, periodic).T
    N = cn.shape[1]; f = np.arange(N + 1)
    i = (f - 1) % N                                                     # cell left of face f
    dd = d2(cn, 1, periodic)
    dpad = np.pad(dd, ((0, 0), (2, 2)), mode="wrap" if periodic else "edge")   # dpad[:, k + 2] = d2q at cell k
    di = lambda o: dpad[:, ((f - 1 + o) % N if periodic else np.clip(f - 1 + o, -2, N + 1)) + 2]
    steep = A * (ctd[:, f % N] - ctd[:, i]) <= 0.0                      # eq. 76
    sign = np.minimum(np.minimum(di(1) * di(0), di(0) * di(-1)), di(1) * di(2)) < 0.0   # eq. 77
    u = np.abs(Q) / L[:, None] if L is not None else np.abs(Q) / h
    sig = u * dt / h
    small = np.abs(A) / (L[:, None] if L is not None else h) <= \
        0.5 * u * (1 - sig) * np.abs(di(0) + di(1)) / 2 * (h if EQ78_H else 1.0)   # eq. 78
    return np.where(steep & sign & small, 0.0, A)




def bounds(cn, ctd, umax_cell, dt, h, periodic, flag="smooth", widen=True):
    """Eqs. 54-75 (gen_conservative.cc_bounds without the final clip to the data range, with periodic padding)."""
    def box(a, s, f):
        p = pad(a, s, periodic); n0, n1 = a.shape
        out = a.copy()
        for dj in range(-s, s + 1):
            for di in range(-s, s + 1):
                out = f(out, p[s + dj:s + dj + n0, s + di:s + di + n1])
        return out
    s2 = (dt * umax_cell / h) >= 0.5 if PREPRINT else np.zeros(cn.shape, bool)   # published: s = 1
    cmax = np.where(s2, np.maximum(box(cn, 2, np.maximum), box(ctd, 2, np.maximum)),
                    np.maximum(box(cn, 1, np.maximum), box(ctd, 1, np.maximum)))
    cmin = np.where(s2, np.minimum(box(cn, 2, np.minimum), box(ctd, 2, np.minimum)),
                    np.minimum(box(cn, 1, np.minimum), box(ctd, 1, np.minimum)))
    p2 = pad(ctd, 2, periodic); pn = pad(cn, 1, periodic)
    n0, n1 = cn.shape
    ext, const, d2q, qext_d = [], [], [], []
    for d in (0, 1):
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
    if widen:                                   # eqs. 72-73; the preprint's eq. 74 printed q_min = q^n
        cmax = np.where(smooth & is_max, cn + 2 * (qext - cn), cmax)
        cmin = np.where(smooth & ~is_max, cn if PREPRINT else cn + 2 * (qext - cn), cmin)
    pl = pad(lap, 1, periodic)
    nb = [pl[1:-1, 1:-1], pl[:-2, 1:-1], pl[2:, 1:-1], pl[1:-1, :-2], pl[1:-1, 2:]]
    sign_change = (np.minimum.reduce(nb) < 0) & (np.maximum.reduce(nb) > 0)
    if flag == "smooth":                      # the smooth extrema of Sec. 4.2.1 (where the bounds were widened)
        flagged = sign_change & smooth
    elif flag:                                # any one-dimensional extremum (gen_conservative.cc_bounds' reading)
        flagged = sign_change & (ext[0] | ext[1])
    else:
        flagged = np.zeros_like(sign_change)
    return cmin, cmax, flagged


def limit(cn, ctd, V, FH, FL, Q, L, dt, h, periodic, flag="smooth", pre=True, widen=True):
    """FCT step of Sec. 4: returns q^{n+1} and the number of flagged cells."""
    Ax, Ay = FH[0] - FL[0], FH[1] - FL[1]
    if pre:
        Ax = preconstrain(Ax, cn, ctd, Q[0], L, dt, h, 1, periodic)
        Ay = preconstrain(Ay, cn, ctd, Q[1], L, dt, h, 0, periodic)
    if not periodic:
        Ax[:, [0, -1]] = 0.0; Ay[[0, -1], :] = 0.0
    Lx = L[:, None] if L is not None else h; Ly = L[None, :] if L is not None else h
    U = np.maximum(np.maximum(np.abs(Q[0][:, :-1]), np.abs(Q[0][:, 1:])) / Lx,
                   np.maximum(np.abs(Q[1][:-1, :]), np.abs(Q[1][1:, :])) / Ly)
    cmin, cmax, flagged = bounds(cn, ctd, U, dt, h, periodic, flag, widen)
    pin = (np.maximum(Ax[:, :-1], 0) - np.minimum(Ax[:, 1:], 0) + np.maximum(Ay[:-1, :], 0) - np.minimum(Ay[1:, :], 0)) * dt
    pout = (np.maximum(Ax[:, 1:], 0) - np.minimum(Ax[:, :-1], 0) + np.maximum(Ay[1:, :], 0) - np.minimum(Ay[:-1, :], 0)) * dt
    rp = np.where(pin > 0, np.minimum(1.0, np.maximum(V * (cmax - ctd), 0) / (pin + EPS)), 0.0)
    rm = np.where(pout > 0, np.minimum(1.0, np.maximum(V * (ctd - cmin), 0) / (pout + EPS)), 0.0)
    rp = np.where(flagged, 0.0, rp); rm = np.where(flagged, 0.0, rm)
    N0, N1 = cn.shape; f = np.arange(N1 + 1); g = np.arange(N0 + 1)
    l, r = (f - 1) % N1, f % N1
    Cx = np.where(Ax > 0, np.minimum(rm[:, l], rp[:, r]), np.minimum(rp[:, l], rm[:, r]))
    b, a = (g - 1) % N0, g % N0
    Cy = np.where(Ay > 0, np.minimum(rm[b, :], rp[a, :]), np.minimum(rp[b, :], rm[a, :]))
    if not periodic:
        Cx[:, [0, -1]] = 0.0; Cy[[0, -1], :] = 0.0
    return ctd - dt * divergence(Cx * Ax, Cy * Ay) / V, int(flagged.sum())


# ---- periodic geometry (the paper's tests): N x N cells of width h on the unit square ----
def periodic_faces(c, Q, axis, recon):
    """Face values at faces 0..N along axis (face f between cells f-1 and f), c4 or u5 (upwind by the sign of Q)."""
    if axis == 0:
        return periodic_faces(c.T, Q.T, 1, recon).T
    N = c.shape[1]; f = np.arange(N + 1)
    q = lambda o: c[:, (f + o) % N]
    if recon == "c4":
        return (-q(-2) + 7 * q(-1) + 7 * q(0) - q(1)) / 12
    if recon == "u9":                          # ninth-order upwind (their eq. 27), for the periodic tests only
        a = np.array([4, -41, 199, -641, 1879, 1375, -305, 55, -5]) / 2520
        pos = sum(a[k] * q(k - 5) for k in range(9)); neg = sum(a[k] * q(4 - k) for k in range(9))
        return np.where(Q >= 0, pos, neg)
    pos = (2 * q(-3) - 13 * q(-2) + 47 * q(-1) + 27 * q(0) - 3 * q(1)) / 60
    neg = (2 * q(2) - 13 * q(1) + 47 * q(0) + 27 * q(-1) - 3 * q(-2)) / 60
    return np.where(Q >= 0, pos, neg)


def product(U, q, L, h, at, six):
    """Product-rule correction of the integrated face flux (eq. 29; eq. 30 if `six`). U: face-averaged velocity, q: face
    values, L: face length; at(a, k) returns `a` at tangential offset k."""
    d1 = lambda a: (at(a, 1) - at(a, -1)) / (2 * h)
    if not six:
        return L ** 3 / 12 * d1(U) * d1(q)
    d1f = lambda a: (-at(a, 2) + 8 * at(a, 1) - 8 * at(a, -1) + at(a, -2)) / (12 * h)
    d2 = lambda a: (at(a, 1) - 2 * at(a, 0) + at(a, -1)) / h ** 2
    d3 = lambda a: (at(a, 2) - 2 * at(a, 1) + 2 * at(a, -1) - at(a, -2)) / (2 * h ** 3)
    a = -2 if SIX_FACE_AVG else 3
    return L ** 3 / 12 * d1f(U) * d1f(q) + L ** 5 / 1440 * (a * d3(q) * d1(U) + a * d3(U) * d1(q) + 2 * d2(U) * d2(q))


def six_order(recon):
    return recon != "c4" and not PREPRINT


def periodic_ho(c, Qx, Qy, h, recon):
    cx, cy = periodic_faces(c, Qx, 1, recon), periodic_faces(c, Qy, 0, recon)
    roll = lambda ax: (lambda a, k: np.roll(a, -k, ax))
    six = six_order(recon)
    return (Qx * cx + product(Qx / h, cx, h, h, roll(0), six), Qy * cy + product(Qy / h, cy, h, h, roll(1), six))


def walled_ho(cs, Qx, Qy, g, recon):
    """High-order fluxes on the benchmark grid: gen_conservative.ho_fluxes (c4 / u5 faces, fourth-order product rule,
    none in the wall rows), with the sixth-order rule in rows 2..N-3 for the upwind values."""
    Fx, Fy = C.ho_fluxes(cs, Qx, Qy, g)
    if not six_order(recon):
        return Fx, Fy
    h, L = g.h, g.L
    for F, Q, axis in ((Fx, Qx, 1), (Fy, Qy, 0)):
        Qi = Q[:, 1:-1] if axis == 1 else Q[1:-1, :].T                  # interior faces, rows = tangential index
        q = C.face_values(cs, Q[:, 1:-1] if axis == 1 else Q[1:-1, :], axis)
        q = q if axis == 1 else q.T
        U = Qi / L[:, None]; n = U.shape[0]
        at = lambda a, k: a[2 + k:n - 2 + k]
        Lr = L[2:n - 2][:, None]
        old = Lr ** 3 / 12 * (at(U, 1) - at(U, -1)) / (2 * h) * (at(q, 1) - at(q, -1)) / (2 * h)
        corr = product(U, q, Lr, h, at, True) - old
        if axis == 1:
            F[2:n - 2, 1:-1] += corr
        else:
            F[1:-1, 2:n - 2] += corr.T
    return Fx, Fy


def periodic_step(c, Qx, Qy, h, dt, recon, limited=True, flag="smooth", pre=True, widen=True):
    """One step on the periodic grid with steady face fluxes Qx (N, N+1), Qy (N+1, N); returns (c, flagged cells)."""
    V = h * h; F = [0.0, 0.0]; cs = c
    for s, (bb, w) in enumerate(zip((0.0, 0.5, 0.5, 1.0), (1, 2, 2, 1))):
        if s:
            cs = c - bb * dt * divergence(*Fs) / V
        Fs = periodic_ho(cs, Qx, Qy, h, recon)
        F = [F[0] + w * Fs[0] / 6, F[1] + w * Fs[1] / 6]
    if not limited:
        return c - dt * divergence(*F) / V, 0
    FL = ctu(c, np.full_like(c, V), Qx, Qy, dt, True)
    ctd = c - dt * divergence(*FL) / V
    return limit(c, ctd, np.full_like(c, V), F, FL, (Qx, Qy), None, dt, h, True, flag, pre, widen)


# ---- closed basin of the benchmark (gen_mpdata.Grid), fixed volumes ----
def run(N, recon, c0, t_end, grid=None, dt=None, flag="smooth", pre=True):
    """Chaplin-Colella on the benchmark grid from the array c0; returns (c, flagged-cell count summed over steps)."""
    import gen_mpdata as M
    g = grid if grid is not None else M.Grid(N)
    if dt is None:
        dt = M.CFL * g.h
    nst = int(round(t_end / dt)); dt = t_end / nst
    saved = C.RECON, C.TRANSVERSE, C.DISSIPATION, C.REPRESENTATION
    C.RECON, C.TRANSVERSE, C.DISSIPATION, C.REPRESENTATION = recon, True, False, "cv"
    c = np.array(c0, dtype=float); V = g.V; nflag = 0
    try:
        for n in range(nst):
            t = n * dt; F = [0.0, 0.0]; Q = [0.0, 0.0]; cs = c
            for s, (bb, w) in enumerate(zip((0.0, 0.5, 0.5, 1.0), (1, 2, 2, 1))):
                if s:
                    cs = c - bb * dt * divergence(*Fs) / V
                Qx, Qy = g.fluxes(t + bb * dt); Fs = walled_ho(cs, Qx, Qy, g, recon)
                F = [F[0] + w * Fs[0] / 6, F[1] + w * Fs[1] / 6]; Q = [Q[0] + w * Qx / 6, Q[1] + w * Qy / 6]
            FL = ctu(c, V, Q[0], Q[1], dt, False)
            ctd = c - dt * divergence(*FL) / V
            c, nf = limit(c, ctd, V, F, FL, Q, g.L, dt, g.h, False, flag, pre); nflag += nf
    finally:
        C.RECON, C.TRANSVERSE, C.DISSIPATION, C.REPRESENTATION = saved
    return c, nflag
