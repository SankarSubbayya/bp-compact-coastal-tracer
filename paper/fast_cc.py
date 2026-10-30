"""Compiled (Numba) version of chaplin_colella.run on the closed basin of the benchmark, for timing (JCP revision,
2026-10-28). Line-for-line translation of the published configuration (PREPRINT = False, flag at smooth extrema,
preconstraint on): RK4 stage fluxes from fast_schemes.ho_fluxes (c4 or u5 face values, fourth-order product rule,
none in the wall rows) with the sixth-order rule in rows 2..N-3 for the upwind values (chaplin_colella.walled_ho),
CTU low-order fluxes, the preconstraint (eqs. 76-78), the s = 1 bounds widened at smooth extrema, the Laplacian flag
and Zalesak's coefficients; edge padding at the walls, as in chaplin_colella.py. cc_cost.py checks it against the
NumPy implementation before timing it. Arrays are indexed [j, i] (y, x)."""
import numpy as np
from numba import njit
import fast_schemes as F

EPS = 1e-15


@njit(cache=True)
def cl(k, N):
    return 0 if k < 0 else (N - 1 if k > N - 1 else k)


@njit(cache=True)
def six_face(Um2, Um1, U0, Up1, Up2, qm2, qm1, q0, qp1, qp2, Lr, h):
    """(eq. 30 - eq. 29) for one face, from the face-averaged velocity and face values at tangential offsets -2..2."""
    i2 = 1.0 / (2 * h); i12 = 1.0 / (12 * h); ih2 = 1.0 / (h * h); i2h3 = 1.0 / (2 * h * h * h)
    d1U = (Up1 - Um1) * i2; d1q = (qp1 - qm1) * i2
    d1fU = (-Up2 + 8 * Up1 - 8 * Um1 + Um2) * i12; d1fq = (-qp2 + 8 * qp1 - 8 * qm1 + qm2) * i12
    d2U = (Up1 - 2 * U0 + Um1) * ih2; d2q = (qp1 - 2 * q0 + qm1) * ih2
    d3U = (Up2 - 2 * Up1 + 2 * Um1 - Um2) * i2h3; d3q = (qp2 - 2 * qp1 + 2 * qm1 - qm2) * i2h3
    L3 = Lr * Lr * Lr
    return L3 / 12 * (d1fU * d1fq - d1U * d1q) + L3 * Lr * Lr / 1440 * (3 * d3q * d1U + 3 * d3U * d1q + 2 * d2U * d2q)


@njit(cache=True)
def six_rows(Qx, Qy, CXf, CYf, L, h, Fx, Fy, Ux, Uy):
    """Sixth-order product rule (eq. 30) in place of eq. 29 at tangential rows 2..N-3 (chaplin_colella.walled_ho);
    Ux, Uy: work arrays for the face-averaged velocities."""
    N = L.size
    for r in range(N):
        for f in range(N + 1):
            Ux[r, f] = Qx[r, f] / L[r]; Uy[f, r] = Qy[f, r] / L[r]
    for r in range(2, N - 2):
        for f in range(1, N):
            Fx[r, f] += six_face(Ux[r - 2, f], Ux[r - 1, f], Ux[r, f], Ux[r + 1, f], Ux[r + 2, f],
                                 CXf[r - 2, f], CXf[r - 1, f], CXf[r, f], CXf[r + 1, f], CXf[r + 2, f], L[r], h)
    for f in range(1, N):
        for r in range(2, N - 2):
            Fy[f, r] += six_face(Uy[f, r - 2], Uy[f, r - 1], Uy[f, r], Uy[f, r + 1], Uy[f, r + 2],
                                 CYf[f, r - 2], CYf[f, r - 1], CYf[f, r], CYf[f, r + 1], CYf[f, r + 2], L[r], h)


@njit(cache=True)
def ctu(c, V, Qx, Qy, dt, Fx, Fy, Gx, Gy, qx, qy):
    N = c.shape[0]
    F.donor(c, Qx, Qy, Gx, Gy)
    for j in range(N):
        for i in range(N):
            Ty = (Gy[j + 1, i] - Gy[j, i] - c[j, i] * (Qy[j + 1, i] - Qy[j, i])) / V[j, i]
            Tx = (Gx[j, i + 1] - Gx[j, i] - c[j, i] * (Qx[j, i + 1] - Qx[j, i])) / V[j, i]
            qy[j, i] = c[j, i] - 0.5 * dt * Ty
            qx[j, i] = c[j, i] - 0.5 * dt * Tx
    Fx[:] = 0.0; Fy[:] = 0.0
    for j in range(N):
        for f in range(1, N):
            q = Qx[j, f]
            Fx[j, f] = q * qy[j, f - 1] if q > 0 else q * qy[j, f]
    for g in range(1, N):
        for i in range(N):
            q = Qy[g, i]
            Fy[g, i] = q * qx[g - 1, i] if q > 0 else q * qx[g, i]


@njit(cache=True)
def d2(a, j, i, axis):
    N = a.shape[0]
    if axis == 1:
        return a[j, cl(i + 1, N)] + a[j, cl(i - 1, N)] - 2 * a[j, i]
    return a[cl(j + 1, N), i] + a[cl(j - 1, N), i] - 2 * a[j, i]


@njit(cache=True)
def pre_face(a, dlo, dm, dp, dhi, Q, Lr, dt, h):
    """Eqs. 77-79 for one face with steepening already established; d2 values at cells f-2, f-1, f, f+1."""
    if not min(min(dp * dm, dm * dlo), dp * dhi) < 0.0:
        return a
    u = abs(Q) / Lr; sig = u * dt / h
    if abs(a) / Lr <= 0.5 * u * (1 - sig) * abs(dm + dp) / 2:
        return 0.0
    return a


@njit(cache=True)
def preconstrain(Ax, Ay, cn, ctd, Qx, Qy, L, dt, h):
    N = cn.shape[0]
    for j in range(N):
        for f in range(1, N):
            a = Ax[j, f]
            if a * (ctd[j, f] - ctd[j, f - 1]) <= 0.0:
                Ax[j, f] = pre_face(a, d2(cn, j, cl(f - 2, N), 1), d2(cn, j, cl(f - 1, N), 1), d2(cn, j, cl(f, N), 1),
                                    d2(cn, j, cl(f + 1, N), 1), Qx[j, f], L[j], dt, h)
    for g in range(1, N):
        for i in range(N):
            a = Ay[g, i]
            if a * (ctd[g, i] - ctd[g - 1, i]) <= 0.0:
                Ay[g, i] = pre_face(a, d2(cn, cl(g - 2, N), i, 0), d2(cn, cl(g - 1, N), i, 0), d2(cn, cl(g, N), i, 0),
                                    d2(cn, cl(g + 1, N), i, 0), Qy[g, i], L[i], dt, h)


@njit(cache=True)
def at(a, j, i, d, k):
    N = a.shape[0]
    return a[cl(j + k, N), i] if d == 0 else a[j, cl(i + k, N)]


@njit(cache=True)
def bounds(cn, ctd, cmin, cmax, flagged, lap, smooth, rmax, rmin):
    N = cn.shape[0]
    for j in range(N):                                  # 3-point maxima / minima of q^n and the CTU solution along x
        for i in range(N):
            im, ip = cl(i - 1, N), cl(i + 1, N)
            rmax[j, i] = max(max(max(cn[j, im], cn[j, i]), cn[j, ip]), max(max(ctd[j, im], ctd[j, i]), ctd[j, ip]))
            rmin[j, i] = min(min(min(cn[j, im], cn[j, i]), cn[j, ip]), min(min(ctd[j, im], ctd[j, i]), ctd[j, ip]))
    for j in range(N):
        for i in range(N):
            jm, jp = cl(j - 1, N), cl(j + 1, N)
            mx = max(max(rmax[jm, i], rmax[j, i]), rmax[jp, i]); mn = min(min(rmin[jm, i], rmin[j, i]), rmin[jp, i])
            e0 = False; e1 = False; k0 = False; k1 = False; dd0 = 0.0; dd1 = 0.0; qe0 = 0.0; qe1 = 0.0
            for d in range(2):
                qm2, qm1, q0 = at(ctd, j, i, d, -2), at(ctd, j, i, d, -1), at(ctd, j, i, d, 0)
                qp1, qp2 = at(ctd, j, i, d, 1), at(ctd, j, i, d, 2)
                dqm1 = qm1 - qm2; dq0 = q0 - qm1; dq1 = qp1 - q0; dq2 = qp2 - qp1
                tv = abs(dq2) + abs(dq1) + abs(dq0) + abs(dqm1)
                ext = (min(dq0 * dq1, dqm1 * dq2) <= 0) and (1.25 * abs(qp2 - qm2) < tv)
                hi = max(max(qm1, q0), qp1); lo = min(min(qm1, q0), qp1)
                const = max(abs(hi - q0), abs(lo - q0)) <= 1e-14
                sm1, s0, sp1 = at(cn, j, i, d, -1), at(cn, j, i, d, 0), at(cn, j, i, d, 1)
                dd = sp1 + sm1 - 2 * s0; slope = 0.5 * (sp1 - sm1)
                xc = -slope / dd if dd != 0 else 0.0
                xc = min(max(xc, -0.5), 0.5)
                qe = 0.5 * dd * xc ** 2 + slope * xc + cn[j, i] - dd / 24
                if d == 0:
                    e0 = ext; k0 = const; dd0 = dd; qe0 = qe
                else:
                    e1 = ext; k1 = const; dd1 = dd; qe1 = qe
            sm = (e0 and e1) or (e0 and k1) or (e1 and k0)
            lp = dd0 + dd1
            lap[j, i] = lp; smooth[j, i] = sm
            if lp <= 0:
                qext = max(max(qe0, qe1), mx)
                if sm:
                    mx = cn[j, i] + 2 * (qext - cn[j, i])
            else:
                qext = min(min(qe0, qe1), mn)
                if sm:
                    mn = cn[j, i] + 2 * (qext - cn[j, i])
            cmax[j, i] = mx; cmin[j, i] = mn
    nf = 0
    for j in range(N):
        for i in range(N):
            a0, a1, a2 = lap[j, i], lap[cl(j - 1, N), i], lap[cl(j + 1, N), i]
            a3, a4 = lap[j, cl(i - 1, N)], lap[j, cl(i + 1, N)]
            lo = min(min(min(a0, a1), min(a2, a3)), a4); hi = max(max(max(a0, a1), max(a2, a3)), a4)
            flagged[j, i] = (lo < 0) and (hi > 0) and smooth[j, i]
            if flagged[j, i]:
                nf += 1
    return nf


@njit(cache=True)
def run_cc(c0, e, L, V, h, dt, nst, recon, six):
    N = c0.shape[0]
    CXf = np.zeros((N, N + 1)); CYf = np.zeros((N + 1, N)); qline = np.empty(N - 1)
    Qbx = np.empty((N, N + 1)); Qby = np.empty((N + 1, N))
    c = c0.copy(); cs = np.empty_like(c); ctd = np.empty_like(c)
    Qx = np.empty((N, N + 1)); Qy = np.empty((N + 1, N))
    Fx = np.empty((N, N + 1)); Fy = np.empty((N + 1, N))
    Fsx = np.empty((N, N + 1)); Fsy = np.empty((N + 1, N))
    Flx = np.empty((N, N + 1)); Fly = np.empty((N + 1, N)); Gx = np.empty((N, N + 1)); Gy = np.empty((N + 1, N))
    qx = np.empty_like(c); qy = np.empty_like(c); Ux = np.empty((N, N + 1)); Uy = np.empty((N + 1, N))
    cmin = np.empty_like(c); cmax = np.empty_like(c); lap = np.empty_like(c)
    smooth = np.zeros((N, N), np.bool_); flagged = np.zeros((N, N), np.bool_)
    rp = np.empty_like(c); rm = np.empty_like(c); rmax = np.empty_like(c); rmin = np.empty_like(c)
    line = np.empty(N); fv = np.empty(N - 1); cp = np.empty(N); dp = np.empty(N)
    a_ = (0.0, 0.5, 0.5, 1.0); w_ = (1.0, 2.0, 2.0, 1.0)
    nflag = 0
    for n in range(nst):
        t = n * dt
        Fsx[:] = 0.0; Fsy[:] = 0.0; Qbx[:] = 0.0; Qby[:] = 0.0
        for s in range(4):
            if s == 0:
                cs[:] = c
            else:
                for j in range(N):
                    for i in range(N):
                        cs[j, i] = c[j, i] - a_[s] * dt * ((Fx[j, i + 1] - Fx[j, i]) + (Fy[j + 1, i] - Fy[j, i])) / V[j, i]
            F.fluxes(e, t + a_[s] * dt, Qx, Qy)
            F.ho_fluxes(cs, Qx, Qy, Fx, Fy, line, fv, cp, dp, True, L, h, CXf, CYf, recon, qline)
            if six:
                six_rows(Qx, Qy, CXf, CYf, L, h, Fx, Fy, Ux, Uy)
            for j in range(N):
                for f in range(N + 1):
                    Fsx[j, f] += w_[s] * Fx[j, f] / 6; Qbx[j, f] += w_[s] * Qx[j, f] / 6
            for g in range(N + 1):
                for i in range(N):
                    Fsy[g, i] += w_[s] * Fy[g, i] / 6; Qby[g, i] += w_[s] * Qy[g, i] / 6
        ctu(c, V, Qbx, Qby, dt, Flx, Fly, Gx, Gy, qx, qy)
        for j in range(N):
            for i in range(N):
                ctd[j, i] = c[j, i] - dt * ((Flx[j, i + 1] - Flx[j, i]) + (Fly[j + 1, i] - Fly[j, i])) / V[j, i]
        for j in range(N):                                   # antidiffusive fluxes, overwrite Fsx/Fsy
            for f in range(N + 1):
                Fsx[j, f] = Fsx[j, f] - Flx[j, f]
        for g in range(N + 1):
            for i in range(N):
                Fsy[g, i] = Fsy[g, i] - Fly[g, i]
        preconstrain(Fsx, Fsy, c, ctd, Qbx, Qby, L, dt, h)
        for j in range(N):
            Fsx[j, 0] = 0.0; Fsx[j, N] = 0.0
        for i in range(N):
            Fsy[0, i] = 0.0; Fsy[N, i] = 0.0
        nflag += bounds(c, ctd, cmin, cmax, flagged, lap, smooth, rmax, rmin)
        for j in range(N):
            for i in range(N):
                pin = (max(Fsx[j, i], 0.0) - min(Fsx[j, i + 1], 0.0) + max(Fsy[j, i], 0.0) - min(Fsy[j + 1, i], 0.0)) * dt
                pout = (max(Fsx[j, i + 1], 0.0) - min(Fsx[j, i], 0.0) + max(Fsy[j + 1, i], 0.0) - min(Fsy[j, i], 0.0)) * dt
                rp[j, i] = min(1.0, max(V[j, i] * (cmax[j, i] - ctd[j, i]), 0.0) / (pin + EPS)) if pin > 0 else 0.0
                rm[j, i] = min(1.0, max(V[j, i] * (ctd[j, i] - cmin[j, i]), 0.0) / (pout + EPS)) if pout > 0 else 0.0
                if flagged[j, i]:
                    rp[j, i] = 0.0; rm[j, i] = 0.0
        for j in range(N):
            for f in range(1, N):
                a = Fsx[j, f]
                Cf = min(rm[j, f - 1], rp[j, f]) if a > 0 else min(rp[j, f - 1], rm[j, f])
                Fsx[j, f] = Cf * a
        for g in range(1, N):
            for i in range(N):
                a = Fsy[g, i]
                Cf = min(rm[g - 1, i], rp[g, i]) if a > 0 else min(rp[g - 1, i], rm[g, i])
                Fsy[g, i] = Cf * a
        for j in range(N):
            for i in range(N):
                c[j, i] = ctd[j, i] - dt * ((Fsx[j, i + 1] - Fsx[j, i]) + (Fsy[j + 1, i] - Fsy[j, i])) / V[j, i]
    return c, nflag


def run(N, recon, c0, t_end, dt=None, cfl=0.35):
    """Same grid, step and step count as chaplin_colella.run; recon "c4" or "u5". Returns (c, flagged-cell count)."""
    x = np.linspace(0, 1, N); h = x[1] - x[0]
    e = np.concatenate([[0.0], 0.5 * (x[:-1] + x[1:]), [1.0]]); L = np.diff(e); V = np.outer(L, L)
    if dt is None:
        dt = cfl * h
    nst = int(round(t_end / dt)); dt = t_end / nst
    c0 = np.ascontiguousarray(c0, dtype=np.float64)
    return run_cc(c0, e, L, V, h, dt, nst, 1 if recon == "c4" else 2, recon != "c4")
