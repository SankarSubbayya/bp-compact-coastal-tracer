"""Compiled (Numba) versions of four benchmark schemes, for timing (2026 revision, after the JCP-oriented review).

Line-for-line translations of gen_conservative.step (bounded compact, global bounds, no dissipation: "cfctg"),
gen_mpdata.mpdata_step (two corrective passes, nonoscillatory: "mpdata3_no"), gen_mpdata.hsimt_split_step
("hsimt_split") and gen_mpdata.muscl_fv_step ("muscl_fv") on the node-centred control volumes of gen_mpdata.Grid, with
the face volume fluxes of the deformational flow. The tridiagonal compact systems are solved by the Thomas algorithm
(the matrix is diagonally dominant). timing_benchmark.py checks every scheme against the NumPy implementation before
timing it. "cfctg" includes the fourth-order transverse flux correction (as gen_conservative does by default),
"cfctg_nt" omits it, and "c4fctg" / "u5fctg" replace the compact face values by the explicit centred fourth-order /
linear upwind fifth-order reconstructions under the same flux-corrected transport. The donor-cell flux uses the
RK4-averaged face fluxes, as in gen_conservative. Arrays are indexed [j, i] (y, x), as in gen_mpdata. Uniform depth
only (the benchmark), where gen_conservative's depth-aware face values reduce to these exactly."""
import numpy as np
from numba import njit

T = 8.0
EPS = 1e-15


@njit(cache=True)
def fluxes(e, t, Qx, Qy):
    N = e.size - 1
    ct = np.cos(np.pi * t / T) / np.pi
    s = np.sin(np.pi * e) ** 2
    for j in range(N):
        for f in range(N + 1):
            Qx[j, f] = ct * s[f] * s[j + 1] - ct * s[f] * s[j]
    for g in range(N + 1):
        for i in range(N):
            Qy[g, i] = -(ct * s[i + 1] * s[g] - ct * s[i] * s[g])


@njit(cache=True)
def donor(c, Qx, Qy, Fx, Fy):
    N = c.shape[0]
    Fx[:] = 0.0; Fy[:] = 0.0
    for j in range(N):
        for f in range(1, N):
            q = Qx[j, f]
            Fx[j, f] = q * c[j, f - 1] if q > 0 else q * c[j, f]
    for g in range(1, N):
        for i in range(N):
            q = Qy[g, i]
            Fy[g, i] = q * c[g - 1, i] if q > 0 else q * c[g, i]


@njit(cache=True)
def div_update(c, Fx, Fy, V, dt, out):
    N = c.shape[0]
    for j in range(N):
        for i in range(N):
            out[j, i] = c[j, i] - dt / V[j, i] * ((Fx[j, i + 1] - Fx[j, i]) + (Fy[j + 1, i] - Fy[j, i]))


# ---------------- bounded compact (cfctg) ----------------
W0, W1, W2, W3 = 16 / 35, 157 / 210, -53 / 210, 1 / 21          # half-width wall closure (control-volume averages)


@njit(cache=True)
def compact_line(cl, out, cp, dp):
    """Face values at interior faces f = 1..n-1 of one line (cl has n nodes); out[f-1]."""
    n = cl.size; k = n - 1
    first = W0 * cl[0] + W1 * cl[1] + W2 * cl[2] + W3 * cl[3]
    last = W0 * cl[n - 1] + W1 * cl[n - 2] + W2 * cl[n - 3] + W3 * cl[n - 4]
    # rows 1..k-2: 0.25 x_{r-1} + x_r + 0.25 x_{r+1} = 0.75 (c_r + c_{r+1}); x_0 = first, x_{k-1} = last
    m = k - 2
    for r in range(m):
        rhs = 0.75 * (cl[r + 1] + cl[r + 2])
        if r == 0:
            rhs -= 0.25 * first
        if r == m - 1:
            rhs -= 0.25 * last
        if r == 0:
            cp[r] = 0.25; dp[r] = rhs
        else:
            den = 1.0 - 0.25 * cp[r - 1]
            cp[r] = 0.25 / den; dp[r] = (rhs - 0.25 * dp[r - 1]) / den
    out[0] = first; out[k - 1] = last
    out[m] = dp[m - 1]
    for r in range(m - 2, -1, -1):
        out[r + 1] = dp[r] - cp[r] * out[r + 2]


V0, V1, V2, V3 = -16 / 105, 73 / 105, 37 / 70, -1 / 14       # second-face closure of the c4 stencil


@njit(cache=True)
def c4_line(cl, out):
    """Explicit centred fourth-order cell-average face values (gen_conservative.face_c4) of one line."""
    n = cl.size
    for f in range(3, n - 2):
        out[f - 1] = (-cl[f - 2] + 7 * cl[f - 1] + 7 * cl[f] - cl[f + 1]) / 12
    out[0] = W0 * cl[0] + W1 * cl[1] + W2 * cl[2] + W3 * cl[3]
    out[n - 2] = W0 * cl[n - 1] + W1 * cl[n - 2] + W2 * cl[n - 3] + W3 * cl[n - 4]
    out[1] = V0 * cl[0] + V1 * cl[1] + V2 * cl[2] + V3 * cl[3]
    out[n - 3] = V0 * cl[n - 1] + V1 * cl[n - 2] + V2 * cl[n - 3] + V3 * cl[n - 4]


@njit(cache=True)
def u5_line(cl, q, out):
    """Linear upwind fifth-order face values (gen_conservative.face_u5); q[f-1] is the flux at face f."""
    n = cl.size
    c4_line(cl, out)
    for f in range(4, n - 3):
        if q[f - 1] >= 0:
            out[f - 1] = (2 * cl[f - 3] - 13 * cl[f - 2] + 47 * cl[f - 1] + 27 * cl[f] - 3 * cl[f + 1]) / 60
        else:
            out[f - 1] = (2 * cl[f + 2] - 13 * cl[f + 1] + 47 * cl[f] + 27 * cl[f - 1] - 3 * cl[f - 2]) / 60


@njit(cache=True)
def line_faces(cl, q, out, cp, dp, recon):
    if recon == 0:
        compact_line(cl, out, cp, dp)
    elif recon == 1:
        c4_line(cl, out)
    else:
        u5_line(cl, q, out)


@njit(cache=True)
def ho_fluxes(c, Qx, Qy, Fx, Fy, line, fv, cp, dp, tc, L, h, CXf, CYf, recon, qline):
    """Fluxes Q c_f with face values from the compact (recon 0), c4 (1) or u5 (2) reconstruction; with tc, plus the
    transverse correction (L^3/12) (du/dt)(dc/dt) along each face (Chaplin and Colella 2017, eq. 29), central
    differences over the neighbouring faces, omitted in the wall rows."""
    N = c.shape[0]
    Fx[:] = 0.0; Fy[:] = 0.0
    for j in range(N):
        for f in range(1, N):
            qline[f - 1] = Qx[j, f]
        line_faces(c[j, :], qline, fv, cp, dp, recon)
        for f in range(1, N):
            Fx[j, f] = Qx[j, f] * fv[f - 1]; CXf[j, f] = fv[f - 1]
    for i in range(N):
        for j in range(N):
            line[j] = c[j, i]
        for g in range(1, N):
            qline[g - 1] = Qy[g, i]
        line_faces(line, qline, fv, cp, dp, recon)
        for g in range(1, N):
            Fy[g, i] = Qy[g, i] * fv[g - 1]; CYf[g, i] = fv[g - 1]
    if tc:
        for j in range(1, N - 1):
            for f in range(1, N):
                du = (Qx[j + 1, f] / L[j + 1] - Qx[j - 1, f] / L[j - 1]) / (2 * h)
                dc = (CXf[j + 1, f] - CXf[j - 1, f]) / (2 * h)
                Fx[j, f] += L[j] ** 3 / 12.0 * du * dc
        for g in range(1, N):
            for i in range(1, N - 1):
                dv = (Qy[g, i + 1] / L[i + 1] - Qy[g, i - 1] / L[i - 1]) / (2 * h)
                dc = (CYf[g, i + 1] - CYf[g, i - 1]) / (2 * h)
                Fy[g, i] += L[i] ** 3 / 12.0 * dv * dc


@njit(cache=True)
def run_cfctg(c0, e, V, t_end, dt, nst, gmin, gmax, tc, L, h, recon):
    N = c0.shape[0]
    CXf = np.zeros((N, N + 1)); CYf = np.zeros((N + 1, N)); qline = np.empty(N - 1)
    Qbx = np.empty((N, N + 1)); Qby = np.empty((N + 1, N))
    c = c0.copy(); cs = np.empty_like(c); cl = np.empty_like(c)
    Qx = np.empty((N, N + 1)); Qy = np.empty((N + 1, N))
    Fx = np.empty((N, N + 1)); Fy = np.empty((N + 1, N))
    Fsx = np.empty((N, N + 1)); Fsy = np.empty((N + 1, N))
    Flx = np.empty((N, N + 1)); Fly = np.empty((N + 1, N))
    rp = np.empty_like(c); rm = np.empty_like(c)
    line = np.empty(N); fv = np.empty(N - 1); cp = np.empty(N); dp = np.empty(N)
    ts_ = (0.0, 0.5, 0.5, 1.0); a_ = (0.0, 0.5, 0.5, 1.0); w_ = (1.0, 2.0, 2.0, 1.0)
    for n in range(nst):
        t = n * dt
        Fsx[:] = 0.0; Fsy[:] = 0.0; Qbx[:] = 0.0; Qby[:] = 0.0
        for s in range(4):
            if s == 0:
                cs[:] = c
            else:
                for j in range(N):
                    for i in range(N):
                        cs[j, i] = c[j, i] - a_[s] * dt / V[j, i] * ((Fx[j, i + 1] - Fx[j, i]) + (Fy[j + 1, i] - Fy[j, i]))
            fluxes(e, t + ts_[s] * dt, Qx, Qy)
            ho_fluxes(cs, Qx, Qy, Fx, Fy, line, fv, cp, dp, tc, L, h, CXf, CYf, recon, qline)
            for j in range(N):
                for f in range(N + 1):
                    Fsx[j, f] += w_[s] * Fx[j, f]; Qbx[j, f] += w_[s] * Qx[j, f] / 6.0
            for g in range(N + 1):
                for i in range(N):
                    Fsy[g, i] += w_[s] * Fy[g, i]; Qby[g, i] += w_[s] * Qy[g, i] / 6.0
        donor(c, Qbx, Qby, Flx, Fly)                         # donor flux with the RK4-averaged face fluxes
        div_update(c, Flx, Fly, V, dt, cl)
        for j in range(N):                                   # antidiffusive fluxes, overwrite Fsx/Fsy
            for f in range(N + 1):
                Fsx[j, f] = Fsx[j, f] / 6.0 - Flx[j, f]
        for g in range(N + 1):
            for i in range(N):
                Fsy[g, i] = Fsy[g, i] / 6.0 - Fly[g, i]
        for j in range(N):
            for i in range(N):
                pin = (max(Fsx[j, i], 0.0) - min(Fsx[j, i + 1], 0.0) + max(Fsy[j, i], 0.0) - min(Fsy[j + 1, i], 0.0)) * dt
                pout = (max(Fsx[j, i + 1], 0.0) - min(Fsx[j, i], 0.0) + max(Fsy[j + 1, i], 0.0) - min(Fsy[j, i], 0.0)) * dt
                rp[j, i] = min(1.0, V[j, i] * (gmax - cl[j, i]) / (pin + EPS))     # capacities in mass units,
                rm[j, i] = min(1.0, V[j, i] * (cl[j, i] - gmin) / (pout + EPS))    # as in gen_conservative
        for j in range(N):
            Fsx[j, 0] = 0.0; Fsx[j, N] = 0.0
            for f in range(1, N):
                a = Fsx[j, f]
                Cf = min(rm[j, f - 1], rp[j, f]) if a > 0 else min(rp[j, f - 1], rm[j, f])
                Fsx[j, f] = Cf * a
        for i in range(N):
            Fsy[0, i] = 0.0; Fsy[N, i] = 0.0
            for g in range(1, N):
                a = Fsy[g, i]
                Cf = min(rm[g - 1, i], rp[g, i]) if a > 0 else min(rp[g - 1, i], rm[g, i])
                Fsy[g, i] = Cf * a
        div_update(cl, Fsx, Fsy, V, dt, c)
    return c


# ---------------- MPDATA, two corrective passes, nonoscillatory (mpdata3_no) ----------------
@njit(cache=True)
def nbmax(a, b, j, i):
    N = a.shape[0]
    jm, jp, im, ip = max(j - 1, 0), min(j + 1, N - 1), max(i - 1, 0), min(i + 1, N - 1)
    return max(a[j, i], a[jm, i], a[jp, i], a[j, im], a[j, ip], b[j, i], b[jm, i], b[jp, i], b[j, im], b[j, ip])


@njit(cache=True)
def nbmin(a, b, j, i):
    N = a.shape[0]
    jm, jp, im, ip = max(j - 1, 0), min(j + 1, N - 1), max(i - 1, 0), min(i + 1, N - 1)
    return min(a[j, i], a[jm, i], a[jp, i], a[j, im], a[j, ip], b[j, i], b[jm, i], b[jp, i], b[j, im], b[j, ip])


@njit(cache=True)
def run_mpdata3_no(c0, e, L, V, h, t_end, dt, nst):
    N = c0.shape[0]
    c = c0.copy(); cn = np.empty_like(c); c1 = np.empty_like(c)
    Qx = np.empty((N, N + 1)); Qy = np.empty((N + 1, N))
    Cx = np.empty((N, N + 1)); Cy = np.empty((N + 1, N))
    Fx = np.empty((N, N + 1)); Fy = np.empty((N + 1, N))
    Ux = np.zeros((N, N + 1)); Uy = np.zeros((N + 1, N))
    Qax = np.zeros((N, N + 1)); Qay = np.zeros((N + 1, N))
    bup = np.empty_like(c); bdn = np.empty_like(c)
    for n in range(nst):
        t = n * dt
        fluxes(e, t + 0.5 * dt, Qx, Qy)
        for j in range(N):
            for f in range(N + 1):
                Cx[j, f] = Qx[j, f] / L[j] * dt / h
        for g in range(N + 1):
            for i in range(N):
                Cy[g, i] = Qy[g, i] / L[i] * dt / h
        cn[:] = c
        donor(c, Qx, Qy, Fx, Fy)
        div_update(cn, Fx, Fy, V, dt, c)
        for _ in range(2):
            Ux[:] = 0.0; Uy[:] = 0.0
            for j in range(N):
                for f in range(1, N):
                    cL = c[j, f - 1]; cR = c[j, f]
                    Ax = (cR - cL) / (cR + cL + EPS)
                    jp = min(j + 1, N - 1); jm = max(j - 1, 0)
                    top = c[jp, f - 1] + c[jp, f]; bot = c[jm, f - 1] + c[jm, f]
                    Bx = (top - bot) / (top + bot + EPS)
                    cyb = 0.25 * (Cy[j, f - 1] + Cy[j + 1, f - 1] + Cy[j, f] + Cy[j + 1, f])
                    cx = Cx[j, f]
                    Ux[j, f] = (abs(cx) - cx * cx) * Ax - 0.5 * cx * cyb * Bx
            for g in range(1, N):
                for i in range(N):
                    cB = c[g - 1, i]; cT = c[g, i]
                    Ay = (cT - cB) / (cT + cB + EPS)
                    ip = min(i + 1, N - 1); im = max(i - 1, 0)
                    rgt = c[g - 1, ip] + c[g, ip]; lft = c[g - 1, im] + c[g, im]
                    By = (rgt - lft) / (rgt + lft + EPS)
                    cxb = 0.25 * (Cx[g - 1, i] + Cx[g - 1, i + 1] + Cx[g, i] + Cx[g, i + 1])
                    cy = Cy[g, i]
                    Uy[g, i] = (abs(cy) - cy * cy) * Ay - 0.5 * cy * cxb * By
            for j in range(N):
                for f in range(N + 1):
                    Qax[j, f] = Ux[j, f] * h * L[j] / dt
            for g in range(N + 1):
                for i in range(N):
                    Qay[g, i] = Uy[g, i] * h * L[i] / dt
            donor(c, Qax, Qay, Fx, Fy)
            for j in range(N):
                for i in range(N):
                    fin = (max(Fx[j, i], 0.0) - min(Fx[j, i + 1], 0.0) + max(Fy[j, i], 0.0) - min(Fy[j + 1, i], 0.0)) * dt / V[j, i]
                    fout = (max(Fx[j, i + 1], 0.0) - min(Fx[j, i], 0.0) + max(Fy[j + 1, i], 0.0) - min(Fy[j, i], 0.0)) * dt / V[j, i]
                    bup[j, i] = (nbmax(cn, c, j, i) - c[j, i]) / (fin + EPS)
                    bdn[j, i] = (c[j, i] - nbmin(cn, c, j, i)) / (fout + EPS)
            for j in range(N):
                for f in range(1, N):
                    fx = Fx[j, f]
                    lx = min(1.0, min(bdn[j, f - 1], bup[j, f])) if fx > 0 else min(1.0, min(bup[j, f - 1], bdn[j, f]))
                    Fx[j, f] = lx * fx; Cx[j, f] = lx * Ux[j, f]
                Cx[j, 0] = 0.0; Cx[j, N] = 0.0
            for i in range(N):
                for g in range(1, N):
                    fy = Fy[g, i]
                    ly = min(1.0, min(bdn[g - 1, i], bup[g, i])) if fy > 0 else min(1.0, min(bup[g - 1, i], bdn[g, i]))
                    Fy[g, i] = ly * fy; Cy[g, i] = ly * Uy[g, i]
                Cy[0, i] = 0.0; Cy[N, i] = 0.0
            div_update(c, Fx, Fy, V, dt, c1)
            c[:] = c1
    return c


# ---------------- HSIMT, Strang split (hsimt_split) ----------------
@njit(cache=True)
def hsimt_line(cl, Cf, out, grad, Ka):
    """HSIMT face values at interior faces 1..n-1 of one line; Cf[f-1] = Courant number at face f."""
    cc1, cc2, cc3, eps1 = 0.25, 0.5, 1.0 / 12.0, 1e-12
    n = cl.size
    grad[0] = 0.0; grad[n] = 0.0; Ka[0] = 0.0; Ka[n] = 0.0
    for f in range(1, n):
        grad[f] = cl[f] - cl[f - 1]; Ka[f] = 1.0 - abs(Cf[f - 1])
    for f in range(1, n):
        gf = grad[f]; kf = Ka[f]
        okf = 0.0 if kf <= eps1 else 1.0 / max(kf, eps1)
        a1 = cc1 * kf + cc2 - cc3 * okf; b1 = -cc1 * kf + cc2 + cc3 * okf
        if abs(gf) <= eps1:
            rL = 0.0; rkL = 0.0; rR = 0.0; rkR = 0.0
        else:
            rL = grad[f - 1] / gf; rkL = Ka[f - 1] * okf; rR = grad[f + 1] / gf; rkR = Ka[f + 1] * okf
        if Cf[f - 1] >= 0:
            psiL = max(0.0, min(min(2.0, 2.0 * rL * rkL), a1 + b1 * rL))
            out[f - 1] = cl[f - 1] + 0.5 * psiL * gf * kf
        else:
            psiR = max(0.0, min(min(2.0, 2.0 * rR * rkR), a1 + b1 * rR))
            out[f - 1] = cl[f] - 0.5 * psiR * gf * kf


@njit(cache=True)
def run_hsimt_split(c0, e, L, V, h, t_end, dt, nst):
    N = c0.shape[0]
    c = c0.copy(); m = np.empty_like(c); Vs = np.empty_like(c)
    Qx = np.empty((N, N + 1)); Qy = np.empty((N + 1, N)); F = np.zeros(N + 1)
    line = np.empty(N); Cf = np.empty(N - 1); fv = np.empty(N - 1); grad = np.empty(N + 1); Ka = np.empty(N + 1)
    for n in range(nst):
        t = n * dt
        fluxes(e, t + 0.5 * dt, Qx, Qy)
        for j in range(N):
            for i in range(N):
                m[j, i] = V[j, i] * c[j, i]; Vs[j, i] = V[j, i]
        for sweep in range(3):
            hs = dt if sweep == 1 else 0.5 * dt
            if sweep != 1:                                  # x sweep, rows j
                for j in range(N):
                    for i in range(N):
                        line[i] = m[j, i] / Vs[j, i]
                    for f in range(1, N):
                        Cf[f - 1] = Qx[j, f] / L[j] * hs / h
                    hsimt_line(line, Cf, fv, grad, Ka)
                    F[0] = 0.0; F[N] = 0.0
                    for f in range(1, N):
                        F[f] = Qx[j, f] * fv[f - 1]
                    for i in range(N):
                        m[j, i] -= hs * (F[i + 1] - F[i]); Vs[j, i] -= hs * (Qx[j, i + 1] - Qx[j, i])
            else:                                           # y sweep, columns i
                for i in range(N):
                    for j in range(N):
                        line[j] = m[j, i] / Vs[j, i]
                    for g in range(1, N):
                        Cf[g - 1] = Qy[g, i] / L[i] * hs / h
                    hsimt_line(line, Cf, fv, grad, Ka)
                    F[0] = 0.0; F[N] = 0.0
                    for g in range(1, N):
                        F[g] = Qy[g, i] * fv[g - 1]
                    for j in range(N):
                        m[j, i] -= hs * (F[j + 1] - F[j]); Vs[j, i] -= hs * (Qy[j + 1, i] - Qy[j, i])
        for j in range(N):
            for i in range(N):
                c[j, i] = m[j, i] / V[j, i]
    return c


# ---------------- MUSCL-superbee, Heun (muscl_fv) ----------------
@njit(cache=True)
def superbee(r):
    return max(0.0, max(min(2 * r, 1.0), min(r, 2.0)))


@njit(cache=True)
def muscl_face(cl, f, n, q):
    """MUSCL value at face f (between nodes f-1 and f) of one line of n nodes."""
    cm1 = cl[f - 1]; c0 = cl[f]
    cm2 = cl[max(f - 2, 0)]; c1 = cl[min(f + 1, n - 1)]
    d = c0 - cm1
    if q >= 0:
        rl = (cm1 - cm2) / d if abs(d) > 1e-300 else 0.0
        return cm1 + 0.5 * (superbee(rl) if f - 2 >= 0 else 0.0) * d
    rr = (c1 - c0) / d if abs(d) > 1e-300 else 0.0
    return c0 - 0.5 * (superbee(rr) if f + 1 <= n - 1 else 0.0) * d


@njit(cache=True)
def muscl_fluxes(c, e, t, Qx, Qy, Fx, Fy, line):
    N = c.shape[0]
    fluxes(e, t, Qx, Qy)
    Fx[:] = 0.0; Fy[:] = 0.0
    for j in range(N):
        for f in range(1, N):
            Fx[j, f] = Qx[j, f] * muscl_face(c[j, :], f, N, Qx[j, f])
    for i in range(N):
        for j in range(N):
            line[j] = c[j, i]
        for g in range(1, N):
            Fy[g, i] = Qy[g, i] * muscl_face(line, g, N, Qy[g, i])


@njit(cache=True)
def run_muscl_fv(c0, e, V, t_end, dt, nst):
    N = c0.shape[0]
    c = c0.copy(); c1 = np.empty_like(c); c2 = np.empty_like(c)
    Qx = np.empty((N, N + 1)); Qy = np.empty((N + 1, N)); Fx = np.empty((N, N + 1)); Fy = np.empty((N + 1, N))
    line = np.empty(N)
    for n in range(nst):
        t = n * dt
        muscl_fluxes(c, e, t, Qx, Qy, Fx, Fy, line)
        div_update(c, Fx, Fy, V, dt, c1)
        muscl_fluxes(c1, e, t + dt, Qx, Qy, Fx, Fy, line)
        div_update(c1, Fx, Fy, V, dt, c2)
        for j in range(N):
            for i in range(N):
                c[j, i] = 0.5 * c[j, i] + 0.5 * c2[j, i]
    return c


def run(N, scheme, c0, t_end, cfl=0.35):
    """Same grid, time step and step count as gen_mpdata.run / gen_conservative.run."""
    x = np.linspace(0, 1, N); h = x[1] - x[0]
    e = np.concatenate([[0.0], 0.5 * (x[:-1] + x[1:]), [1.0]]); L = np.diff(e); V = np.outer(L, L)
    dt = cfl * h; nst = int(round(t_end / dt)); dt = t_end / nst
    c0 = np.ascontiguousarray(c0, dtype=np.float64)
    spec = {"cfctg": (True, 0), "cfctg_nt": (False, 0), "c4fctg": (True, 1), "u5fctg": (True, 2)}
    if scheme in spec:                       # bounded schemes with global bounds; "_nt": without the transverse correction
        tc, recon = spec[scheme]
        return run_cfctg(c0, e, V, t_end, dt, nst, float(c0.min()), float(c0.max()), tc, L, h, recon)
    if scheme == "mpdata3_no":
        return run_mpdata3_no(c0, e, L, V, h, t_end, dt, nst)
    if scheme == "hsimt_split":
        return run_hsimt_split(c0, e, L, V, h, t_end, dt, nst)
    if scheme == "muscl_fv":
        return run_muscl_fv(c0, e, V, t_end, dt, nst)
    raise ValueError(scheme)
