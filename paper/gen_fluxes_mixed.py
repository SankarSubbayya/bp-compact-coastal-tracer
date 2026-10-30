"""Exact face volume fluxes for offline, mass-conserving tracer transport in San Francisco Bay.

Same grid, solver and parameters as gen_currents_mixed.py (semi-implicit Casulli C-grid, Manning drag, South Bay
sponge, six-constituent mixed tide); the forcing amplitudes are read from sfbay_calibration.npz so the currents are
reproduced exactly without repeating the calibration runs. For every step in the transport window it records the
volume flux used by the solver's own continuity equation,
    q = Hu*dy*(THETA*u_new + (1-THETA)*u_old)   [m^3/s, at u-faces; likewise at v-faces],
for which eta_new = eta_old - (DT/A) div(q) holds exactly at every non-open cell. An offline transport of
(A*(H+eta)) * c with these fluxes therefore preserves a uniform tracer exactly and conserves tracer mass to
round-off."""
import os, paths as P
import os, sys, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu

G = 9.81; DT = 120.0; THETA = 0.6; HMIN = 3.0; MANNING_N = 0.037; DEG = np.pi / 180.0
D = P.SFBAY
d = np.load(f"{D}/sfbay_fine.npz"); depth, wet, lat, lon = d["depth"], d["wet"], d["lat"], d["lon"]
dx = float(d["dx"]); dy = float(d["dy"]); ny, nx = depth.shape
H = np.where(wet, np.maximum(depth, HMIN), 0.0)
cid = -np.ones((ny, nx), int); wj, wi = np.where(wet); N = wj.size; cid[wj, wi] = np.arange(N)
um = wet[:, :-1] & wet[:, 1:]; vm = wet[:-1, :] & wet[1:, :]
uk1 = cid[:, :-1][um]; uk2 = cid[:, 1:][um]; Hu = (0.5 * (H[:, :-1] + H[:, 1:]))[um]
vk1 = cid[:-1, :][vm]; vk2 = cid[1:, :][vm]; Hv = (0.5 * (H[:-1, :] + H[1:, :]))[vm]
CDu = G * MANNING_N ** 2 / Hu ** (1.0 / 3.0); CDv = G * MANNING_N ** 2 / Hv ** (1.0 / 3.0)
LAT_SP = 37.76; latmin = float(lat.min()); RMAX = 6.0e-3
def sponge(cells):
    s = np.clip((LAT_SP - lat[wj[cells]]) / (LAT_SP - latmin), 0.0, 1.0); return RMAX * s ** 2
ru = sponge(uk1); rv = sponge(vk1)
openm = np.zeros((ny, nx), bool); openm[:, 0] = wet[:, 0]; isopen = np.zeros(N, bool); isopen[cid[openm & wet]] = True
A = dx * dy; coef = THETA ** 2 * DT ** 2 * G / A; wu = coef * Hu * dy / dx; wv = coef * Hv * dx / dy
rows = np.concatenate([uk1, uk2, vk1, vk2, np.arange(N)]); cols = np.concatenate([uk2, uk1, vk2, vk1, np.arange(N)])
diag = np.ones(N)
for a, b, w in [(uk1, uk2, wu), (vk1, vk2, wv)]: np.add.at(diag, a, w); np.add.at(diag, b, w)
vals = np.concatenate([-wu, -wu, -wv, -wv, diag])
Mx = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)).tolil()
for k in np.where(isopen)[0]: Mx.rows[k] = [k]; Mx.data[k] = [1.0]
lu = splu(Mx.tocsc())

NOAA = {"M2": 28.98410, "S2": 30.00000, "N2": 28.43973, "K1": 15.04107, "O1": 13.94304, "P1": 14.95893}
cal = np.load(f"{D}/sfbay_calibration.npz")
PH = cal["offshore_phase"] if "offshore_phase" in cal.files else cal["noaa_phase"]   # calibrated phases (2026 revision)
forcing = [(NOAA[str(nm)] * DEG / 3600.0, float(a), float(p) * DEG)
           for nm, a, p in zip(cal["names"], cal["offshore_amp"], PH)]

SPIN = 3 * 86400; WIN_H = float(os.environ.get("WIN_H", 6.0))   # 25 h for the long-run test (separate file)
nspin = int(SPIN / DT); nrec = int(WIN_H * 3600 / DT)
eta = np.zeros(N); uvel = np.zeros(uk1.size); vvel = np.zeros(vk1.size)
QX = np.zeros((nrec, uk1.size)); QY = np.zeros((nrec, vk1.size)); ETA = np.zeros((nrec + 1, N))
Ucell0 = None
t0 = time.time(); worst = 0.0
for n in range(nspin + nrec):
    t = n * DT
    gEx = (eta[uk2] - eta[uk1]) / dx; gNy = (eta[vk2] - eta[vk1]) / dy
    pu = uvel - DT * G * (1 - THETA) * gEx; pv = vvel - DT * G * (1 - THETA) * gNy
    fxu = Hu * dy * (THETA * pu + (1 - THETA) * uvel); fxv = Hv * dx * (THETA * pv + (1 - THETA) * vvel)
    div = np.zeros(N); np.add.at(div, uk1, fxu); np.add.at(div, uk2, -fxu); np.add.at(div, vk1, fxv); np.add.at(div, vk2, -fxv)
    rhs = eta - (DT / A) * div
    rhs[isopen] = sum(a * np.cos(w * t - p) for (w, a, p) in forcing)
    eta_old = eta
    eta = lu.solve(rhs)
    us = pu - DT * G * THETA * (eta[uk2] - eta[uk1]) / dx; vs = pv - DT * G * THETA * (eta[vk2] - eta[vk1]) / dy
    if n >= nspin:
        r = n - nspin
        if r == 0:
            ETA[0] = eta_old
        qx = Hu * dy * (THETA * us + (1 - THETA) * uvel); qy = Hv * dx * (THETA * vs + (1 - THETA) * vvel)
        QX[r] = qx; QY[r] = qy; ETA[r + 1] = eta
        dq = np.zeros(N); np.add.at(dq, uk1, qx); np.add.at(dq, uk2, -qx); np.add.at(dq, vk1, qy); np.add.at(dq, vk2, -qy)
        resid = (eta - eta_old) + (DT / A) * dq
        worst = max(worst, float(np.abs(resid[~isopen]).max()))
    uvel = us / (1 + DT * CDu * np.abs(us) / Hu + DT * ru); vvel = vs / (1 + DT * CDv * np.abs(vs) / Hv + DT * rv)
    if n == nspin:   # cell-centred current at the window start, as gen_currents_mixed.py stores it
        us_ = np.zeros(N); uc_ = np.zeros(N)
        np.add.at(us_, uk1, uvel); np.add.at(uc_, uk1, 1.0); np.add.at(us_, uk2, uvel); np.add.at(uc_, uk2, 1.0)
        Ucell0 = us_ / np.maximum(uc_, 1)
    if not np.all(np.isfinite(eta)): print("BLEW UP", n); sys.exit(1)
print(f"run {time.time() - t0:.0f}s; continuity residual (non-open cells, m): max {worst:.2e}")
ref = np.load(f"{D}/currents_fine.npz")
print(f"window-start U vs currents_fine.npz U[0]: max |diff| = {np.abs(Ucell0 - ref['U'][0]).max():.2e} m/s "
      f"(field max {np.abs(ref['U'][0]).max():.2f})")
np.savez(f"{D}/{os.environ.get('FLUX_OUT', 'fluxes_mixed.npz')}", QX=QX, QY=QY, ETA=ETA, uk1=uk1, uk2=uk2, vk1=vk1, vk2=vk2, isopen=isopen,
         Hcell=H[wj, wi], wj=wj, wi=wi, cid=cid, wet=wet, lat=lat, lon=lon, dx=dx, dy=dy, DT=DT, t0=SPIN)
print(f"saved fluxes_mixed.npz: {nrec} records of {DT:.0f} s, {uk1.size} u-faces, {vk1.size} v-faces")
