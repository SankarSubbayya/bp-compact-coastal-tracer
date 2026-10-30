"""Offline, mass-conserving tracer transport in San Francisco Bay with the bounded compact scheme.

Uses the exact face volume fluxes and surface elevations recorded by gen_fluxes_mixed.py. The transported quantity is
tracer mass per cell, m = A*(H+eta)*c, updated in flux form; within each recorded 120 s step the volume flux is
constant and the cell volume changes linearly, exactly as in the hydrodynamic continuity equation, so a uniform tracer
stays uniform and mass is conserved to round-off. Open-boundary (Pacific) cells are outside the tracer domain: inflow
through them carries ocean water (c = 0) and outflow is counted as export, so

    initial mass = mass in the bay + mass exported through the open boundary   (to round-off).

Face values: 4th-order compact (Pade) midpoint interpolation along each wet stretch between land, with explicit 4th-order
one-sided closures at stretch ends (2nd-order averages in stretches shorter than four cells). Time stepping: RK4 in flux
form; limiting: Zalesak (1979) flux-corrected transport against the donor-cell flux, with global (physical) or local
bounds. Schemes: donor (first-order), ccompact (conservative compact, unlimited), ccfctg / ccfct (bounded, global /
local bounds)."""
import os, paths as P
import json, os, sys, time
import numpy as np
from scipy.linalg import solve_banded

D = P.SFBAY
f = np.load(f"{D}/{os.environ.get('FLUX_FILE', 'fluxes_mixed.npz')}")   # long-run test uses fluxes_mixed_25h.npz
QX, QY, ETA = f["QX"], f["QY"], f["ETA"]
uk1, uk2, vk1, vk2, isopen = f["uk1"], f["uk2"], f["vk1"], f["vk2"], f["isopen"]
Hc, wj, wi, cid, wet, lat, lon = f["Hcell"], f["wj"], f["wi"], f["cid"], f["wet"], f["lat"], f["lon"]
dx, dy, DT = float(f["dx"]), float(f["dy"]), float(f["DT"])
N = Hc.size; A = dx * dy; ny, nx = wet.shape; nrec = QX.shape[0]
interior = ~isopen
EPS = 1e-300


# ---- face classes ----
def classify(k1, k2):
    both = interior[k1] & interior[k2]
    openf = (isopen[k1] & interior[k2]) | (interior[k1] & isopen[k2])
    return both, openf

ub, uo = classify(uk1, uk2); vb, vo = classify(vk1, vk2)
CLOSURE = "avg"       # coastline closure of the compact face system
DISSIPATION = True    # upwind-biased (5th-order-upwind type) dissipation added to the compact flux
DISS_STRENGTH = 1.0   # multiplier of the dissipation term (1 = full strength; varied only in coast_stability.py)


def compact_structure(k1, k2, both, axis):
    """Banded system for compact face values on faces where both cells are interior, ordered along `axis` lines."""
    fidx = np.where(both)[0]
    j1, i1 = wj[k1[fidx]], wi[k1[fidx]]                  # first (west/south) cell of each face
    order = np.lexsort((i1, j1)) if axis == 1 else np.lexsort((j1, i1))
    fidx = fidx[order]; j1, i1 = j1[order], i1[order]
    n = fidx.size
    # cell grid of interior wet cells (land and open cells = -1)
    g = -np.ones((ny, nx), int); g[wj[interior], wi[interior]] = np.where(interior)[0]
    def cell(jj, ii):
        ok = (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx)
        out = -np.ones(jj.shape, int); out[ok] = g[jj[ok], ii[ok]]; return out
    if axis == 1:
        step = (np.zeros_like(j1), np.ones_like(i1))
    else:
        step = (np.ones_like(j1), np.zeros_like(i1))
    sj, si = step
    cm1 = cell(j1 - sj, i1 - si); c0 = cell(j1, i1); c1 = cell(j1 + sj, i1 + si); c2 = cell(j1 + 2 * sj, i1 + 2 * si)
    has_left = cm1 >= 0; has_right = c2 >= 0          # neighbouring faces exist in the same line
    c3 = cell(j1 + 3 * sj, i1 + 3 * si); cm2 = cell(j1 - 2 * sj, i1 - 2 * si)
    ab = np.zeros((3, n)); ab[1] = 1.0
    W = np.zeros((n, 4)); K = np.zeros((n, 4), int)
    inner = has_left & has_right
    W[inner, 0] = 0.75; K[inner, 0] = c0[inner]; W[inner, 1] = 0.75; K[inner, 1] = c1[inner]
    r = np.where(inner)[0]
    ab[2, r - 1] = 0.25                                   # A[r, r-1]
    ab[0, r + 1] = 0.25                                   # A[r, r+1]
    # first face of a stretch (no left neighbour): one-sided towards +axis if four cells c0..c3 exist
    first = ~has_left & has_right
    long1 = first & (c3 >= 0)
    W[long1] = [5 / 16, 15 / 16, -5 / 16, 1 / 16]; K[long1] = np.stack([c0, c1, c2, c3], 1)[long1]
    last = has_left & ~has_right
    long2 = last & (cm2 >= 0)
    W[long2] = [5 / 16, 15 / 16, -5 / 16, 1 / 16]; K[long2] = np.stack([c1, c0, cm1, cm2], 1)[long2]
    short = (first & ~long1) | (last & ~long2) | (~has_left & ~has_right)
    W[short] = [0.5, 0.5, 0, 0]; K[short] = np.stack([c0, c1, c0, c0], 1)[short]
    if CLOSURE == "avg":     # one-sided 4th-order closures are unstable at a coastline; use the 2nd-order average
        one = np.isclose(W[:, 0], 5 / 16) & np.isclose(W[:, 3], 1 / 16)
        W[one] = [0.5, 0.5, 0, 0]; K[one] = np.stack([c0, c1, c0, c0], 1)[one]
    elif CLOSURE == "onesided_cv":   # one-sided closure exact for cubics given equal-width cell averages
        one = np.isclose(W[:, 0], 5 / 16) & np.isclose(W[:, 3], 1 / 16)
        W[one] = [1 / 4, 13 / 12, -5 / 12, 1 / 12]
    # upwind-biased dissipation stencils (cells -2..+3 around the face), with availability
    st = np.stack([cm2, cm1, c0, c1, c2, c3], 1)
    ok5 = (st >= 0).all(1); ok3 = (cm1 >= 0) & (c2 >= 0) & ~ok5
    return dict(fidx=fidx, ab=ab, W=W, K=K, st=np.where(st >= 0, st, 0), ok5=ok5, ok3=ok3)

SU = compact_structure(uk1, uk2, ub, 1)
SV = compact_structure(vk1, vk2, vb, 0)

TRANSVERSE = True   # fourth-order transverse flux correction (JCP revision) where both tangential neighbours are
                    # compact faces; omitted elsewhere (coastline, open boundary, short stretches)


def transverse_structure(k1, S, axis):
    """Positions (within S["fidx"]) of each compact face's two tangential neighbour faces, and where both exist.
    x-faces (axis 1) between cells (j, i) and (j, i+1) have neighbours in rows j-1 and j+1; y-faces (axis 0) between
    (j, i) and (j+1, i) in columns i-1 and i+1. Faces are keyed by the grid position of their first cell."""
    fidx = S["fidx"]; pos = np.full(k1.size, -1); pos[fidx] = np.arange(fidx.size)
    jj, ii = wj[k1], wi[k1]
    key = {(int(a), int(b)): f for f, (a, b) in enumerate(zip(jj, ii))}
    dj, di = (1, 0) if axis == 1 else (0, 1)
    def nb(sign):
        out = np.full(fidx.size, -1)
        for r, f in enumerate(fidx):
            g = key.get((int(jj[f]) + sign * dj, int(ii[f]) + sign * di), -1)
            out[r] = pos[g] if g >= 0 else -1
        return out
    plus, minus = nb(1), nb(-1)
    return dict(plus=plus, minus=minus, ok=(plus >= 0) & (minus >= 0))


TU = transverse_structure(uk1, SU, 1)
TV = transverse_structure(vk1, SV, 0)


def set_closure(name):
    """Rebuild the compact line systems with another coastline closure ('avg' or 'onesided')."""
    global CLOSURE, SU, SV, TU, TV
    CLOSURE = name
    SU = compact_structure(uk1, uk2, ub, 1); SV = compact_structure(vk1, vk2, vb, 0)
    TU = transverse_structure(uk1, SU, 1); TV = transverse_structure(vk1, SV, 0)


def face_values(c, S):
    rhs = (S["W"] * c[S["K"]]).sum(1)
    return solve_banded((1, 1), S["ab"], rhs)


DEPTH_AWARE = False  # on: face value = reconstructed (D c) / reconstructed D, D = H + eta (default: c directly)
H_FLOOR = 0.5        # faces whose reconstructed depth is below H_FLOOR x the smaller adjacent depth use c directly
FALLBACK = [0, 0]    # [faces that used the fallback, depth-aware faces evaluated] since the last reset


def face_ratio(c, D, S, k1):
    """Depth-aware compact face values on the faces of S (k1: the face-to-first-cell map of that direction)."""
    Df = face_values(D, S)
    fidx = S["fidx"]; k2 = uk2 if k1 is uk1 else vk2
    bad = ~(Df >= H_FLOOR * np.minimum(D[k1[fidx]], D[k2[fidx]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = face_values(D * c, S) / Df
    FALLBACK[0] += int(bad.sum()); FALLBACK[1] += bad.size
    if bad.any():
        r = np.where(bad, face_values(c, S), r)
    return r, Df, bad


def divergence(Fx, Fy):
    dv = np.zeros(N)
    np.add.at(dv, uk1, Fx); np.add.at(dv, uk2, -Fx); np.add.at(dv, vk1, Fy); np.add.at(dv, vk2, -Fy)
    dv[isopen] = 0.0
    return dv


def open_flux(c, q, k1, k2, mask):
    """Flux through open-boundary faces: outflow carries the interior value, inflow carries ocean water (c = 0)."""
    F = np.zeros_like(q)
    k_int = np.where(interior[k1], k1, k2)
    out_dir = np.where(interior[k1], 1.0, -1.0)            # +q leaves k1
    leaving = (q * out_dir) > 0
    F[mask & leaving] = q[mask & leaving] * c[k_int[mask & leaving]]
    return F


def donor(c, qx, qy):
    Fx = np.where(qx > 0, qx * c[uk1], qx * c[uk2]) * ub + open_flux(c, qx, uk1, uk2, uo)
    Fy = np.where(qy > 0, qy * c[vk1], qy * c[vk2]) * vb + open_flux(c, qy, vk1, vk2, vo)
    return Fx, Fy


def dissipation(c, q, S):
    """|q| times the upwind-biased correction: -d5/60 (6-cell stencil; compact+this ~ 5th-order-upwind dissipation),
    else +d3/12 (4 cells; 3rd-order upwind), else -d1/2 (2 cells; donor cell). Differences are taken across the face."""
    s = c[S["st"]]                                   # columns: -2, -1, 0, +1, +2, +3 (face between 0 and +1)
    d5 = s[:, 5] - 5 * s[:, 4] + 10 * s[:, 3] - 10 * s[:, 2] + 5 * s[:, 1] - s[:, 0]
    d3 = s[:, 4] - 3 * s[:, 3] + 3 * s[:, 2] - s[:, 1]
    d1 = s[:, 3] - s[:, 2]
    corr = np.where(S["ok5"], -d5 / 60.0, np.where(S["ok3"], d3 / 12.0, -d1 / 2.0))
    return np.abs(q) * corr


def high_order(c, qx, qy, V=None):
    """High-order face fluxes of the stage concentration c; V = the stage volumes (depth-aware faces when given)."""
    Fx = open_flux(c, qx, uk1, uk2, uo); Fy = open_flux(c, qy, vk1, vk2, vo)
    fu, fv = SU["fidx"], SV["fidx"]
    if DEPTH_AWARE and V is not None:
        D = V / A
        cu, Du, bu = face_ratio(c, D, SU, uk1); cv, Dv, bv = face_ratio(c, D, SV, vk1)
    else:
        cu, cv = face_values(c, SU), face_values(c, SV); Du = Dv = None
    Fx[fu] = qx[fu] * cu
    Fy[fv] = qy[fv] * cv
    if TRANSVERSE:   # (L^3/12)(d(q/L)/dtau)(dc/dtau) with centred differences on the uniform grid: dq dc / 48;
                     # depth-aware: (dq - q dD_f / D_f) dr / 48
        for F_, q, fidx, cvals, T, Df, bad in ((Fx, qx, fu, cu, TU, Du, None if Du is None else bu),
                                              (Fy, qy, fv, cv, TV, Dv, None if Dv is None else bv)):
            r = np.where(T["ok"])[0]; pp, mm = T["plus"][r], T["minus"][r]
            dq = q[fidx[pp]] - q[fidx[mm]]
            if Df is not None:
                dq = dq - np.where(bad[r], 0.0, q[fidx[r]] * (Df[pp] - Df[mm]) / Df[r])
            F_[fidx[r]] += dq * (cvals[pp] - cvals[mm]) / 48.0
    if DISSIPATION:
        Fx[fu] += DISS_STRENGTH * dissipation(c, qx[fu], SU)
        Fy[fv] += DISS_STRENGTH * dissipation(c, qy[fv], SV)
    return Fx, Fy


def neighbour_extrema(c):
    lo = c.copy(); hi = c.copy()
    for k1, k2, m in ((uk1, uk2, ub), (vk1, vk2, vb)):
        a, b = k1[m], k2[m]
        np.minimum.at(lo, a, c[b]); np.minimum.at(lo, b, c[a]); np.maximum.at(hi, a, c[b]); np.maximum.at(hi, b, c[a])
    return lo, hi


def fct(m_lo, V1, Ax, Ay, dt, cmin, cmax):
    c_lo = m_lo / V1
    pin = np.zeros(N); pout = np.zeros(N)
    for Af, k1, k2 in ((Ax, uk1, uk2), (Ay, vk1, vk2)):
        np.add.at(pin, k2, np.maximum(Af, 0)); np.add.at(pout, k1, np.maximum(Af, 0))
        np.add.at(pin, k1, np.maximum(-Af, 0)); np.add.at(pout, k2, np.maximum(-Af, 0))
    rp = np.minimum(1.0, (cmax - c_lo) * V1 / (pin * dt + EPS)); rm = np.minimum(1.0, (c_lo - cmin) * V1 / (pout * dt + EPS))
    rp = np.maximum(rp, 0.0); rm = np.maximum(rm, 0.0)
    Cx = np.where(Ax > 0, np.minimum(rm[uk1], rp[uk2]), np.minimum(rp[uk1], rm[uk2]))
    Cy = np.where(Ay > 0, np.minimum(rm[vk1], rp[vk2]), np.minimum(rp[vk1], rm[vk2]))
    return Cx * Ax, Cy * Ay


def run(scheme, c0, hours, nsub):
    m = A * (Hc + ETA[0]) * c0; m[isopen] = 0.0
    M0 = m.sum(); exported = 0.0
    hist = []
    nr = int(round(hours * 3600 / DT)); dts = DT / nsub
    t0 = time.perf_counter()
    for r in range(nr):
        qx, qy = QX[r], QY[r]
        V0 = A * (Hc + ETA[r]); V1 = A * (Hc + ETA[r + 1])
        for s in range(nsub):
            Va = V0 + (s / nsub) * (V1 - V0); Vh = V0 + ((s + 0.5) / nsub) * (V1 - V0); Vb = V0 + ((s + 1) / nsub) * (V1 - V0)
            c1 = np.where(interior, m / Va, 0.0)
            Flx, Fly = donor(c1, qx, qy)
            if scheme == "donor":
                Fx, Fy = Flx, Fly
            else:
                F1 = high_order(c1, qx, qy, Va)
                m2 = m - 0.5 * dts * divergence(*F1); F2 = high_order(np.where(interior, m2 / Vh, 0.0), qx, qy, Vh)
                m3 = m - 0.5 * dts * divergence(*F2); F3 = high_order(np.where(interior, m3 / Vh, 0.0), qx, qy, Vh)
                m4 = m - dts * divergence(*F3); F4 = high_order(np.where(interior, m4 / Vb, 0.0), qx, qy, Vb)
                Fx = (F1[0] + 2 * F2[0] + 2 * F3[0] + F4[0]) / 6.0
                Fy = (F1[1] + 2 * F2[1] + 2 * F3[1] + F4[1]) / 6.0
                if scheme in ("ccfctg", "ccfct"):
                    m_lo = m - dts * divergence(Flx, Fly)
                    if scheme == "ccfctg":
                        cmin, cmax = np.zeros(N), np.ones(N)
                    else:
                        lo1, hi1 = neighbour_extrema(c1); lo2, hi2 = neighbour_extrema(np.where(interior, m_lo / Vb, 0.0))
                        cmin, cmax = np.minimum(lo1, lo2), np.maximum(hi1, hi2)
                    Ax, Ay = Fx - Flx, Fy - Fly
                    Ax[uo] = 0.0; Ay[vo] = 0.0      # open-boundary faces carry the donor-cell flux (no correction)
                    Lx, Ly = fct(m_lo, Vb, Ax, Ay, dts, cmin, cmax)
                    Fx, Fy = Flx + Lx, Fly + Ly
            # export through the open boundary (outward-positive flux on open faces)
            ex = 0.0
            for Ff, k1, mask in ((Fx, uk1, uo), (Fy, vk1, vo)):
                sgn = np.where(interior[k1], 1.0, -1.0)
                ex += float((Ff[mask] * sgn[mask]).sum())
            exported += dts * ex
            m = m - dts * divergence(Fx, Fy); m[isopen] = 0.0
        c = np.where(interior, m / V1, 0.0)
        hist.append((float(c[interior].min()), float(c[interior].max()), float(m.sum()), exported))
    wall = time.perf_counter() - t0
    c = np.where(interior, m / (A * (Hc + ETA[nr])), 0.0)
    return c, np.array(hist), M0, wall


def max_outflow_ratio():
    worst = 0.0
    for r in range(nrec):
        qx, qy = QX[r], QY[r]; out = np.zeros(N)
        np.add.at(out, uk1, np.maximum(qx, 0)); np.add.at(out, uk2, np.maximum(-qx, 0))
        np.add.at(out, vk1, np.maximum(qy, 0)); np.add.at(out, vk2, np.maximum(-qy, 0))
        V = A * (Hc + np.minimum(ETA[r], ETA[r + 1]))
        worst = max(worst, float((out[interior] / V[interior]).max()))
    return worst


if __name__ == "__main__":
    HOURS = 5.0
    ratio = max_outflow_ratio(); nsub = max(1, int(np.ceil(ratio * DT / 0.8)))
    print(f"max outflow fraction per second {ratio:.2e}; substeps per 120 s record: {nsub} "
          f"(donor Courant <= {ratio * DT / nsub:.2f})", flush=True)
    i0 = int(np.argmin(np.abs(lon - (-122.357)))); j0 = int(np.argmin(np.abs(lat - 37.806)))
    Lon, Lat = lon[wi], lat[wj]
    dist = np.sqrt(((Lon - lon[i0]) * 111 * np.cos(np.radians(37.8))) ** 2 + ((Lat - lat[j0]) * 111) ** 2)
    c0 = ((dist <= 3.5) & interior).astype(float)
    # uniform-tracer check: a constant must stay exactly constant with these fluxes (no open-boundary contact needed)
    one = np.where(interior, 1.0, 0.0)
    out = {}
    for sch in ("donor", "ccompact", "ccfctg", "ccfct"):
        c, h, M0, wall = run(sch, c0, HOURS, nsub)
        Mf, Ex = h[-1, 2], h[-1, 3]
        res = dict(min=float(c[interior].min()), max=float(c[interior].max()), retained=Mf / M0, exported=Ex / M0,
                   budget_residual=(M0 - Mf - Ex) / M0, peak=float(c.max()), area_c_gt_0p5=int((c > 0.5).sum()),
                   area_c_gt_0p01=int((c > 0.01).sum()), area0=int((c0 > 0.5).sum()), wall_s=round(wall, 1))
        out[sch] = res
        np.save(f"sfbay_cons_{sch}.npy", c)
        print(f"{sch:9s} range [{res['min']:+.2e}, {res['max']:.4f}] retained {res['retained']:.4f} exported {res['exported']:.4f} "
              f"budget residual {res['budget_residual']:+.1e} | cells c>0.5: {res['area_c_gt_0p5']} (initial {res['area0']}), "
              f"c>0.01: {res['area_c_gt_0p01']} | {wall:.0f}s", flush=True)
    json.dump(out, open("sfbay_conservative_results.json", "w"), indent=1)
    print("wrote sfbay_conservative_results.json")
