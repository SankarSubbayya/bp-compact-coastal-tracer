"""Recalibration of the San Francisco Bay mixed-tide model with a simultaneous harmonic analysis (2026 revision).

Replaces the band-gain calibration of gen_currents_mixed.py, which fitted one frequency at a time over 14 days and
reported a current amplitude that was not the ellipse semi-major axis. Same grid, solver and parameters as
gen_currents_mixed.py / gen_fluxes_mixed.py (linear depth-averaged shallow-water equations without momentum advection
or Coriolis, fixed-depth continuity, theta = 0.6 free-surface solve, split quadratic Manning drag n = 0.037 and a South
Bay Rayleigh sponge, minimum depth 3 m, 120 s step).

Calibration: the offshore amplitude and phase of each of the six forcing constituents are adjusted so that a
simultaneous least-squares fit (six constituents plus the M4 and MS4 overtides and a mean) of the Golden Gate elevation
over 30 days (days 3-33 of a 33-day run) matches the NOAA harmonic constants of station 9414290. A 30-day record
separates M2 from N2 and S2 (Rayleigh periods 27.6 and 14.8 days) and K1 from O1 (13.7 days); K1 and P1 (182.6 days)
are not separable by the Rayleigh criterion, but the forcing contains exactly these frequencies, so the fit is
determined by the known frequency set (its condition number is reported).

Outputs (in _dev/sfbay_fine, previous versions in archive_2026-10-27_v1/): sfbay_calibration.npz, sfbay_tides.npz,
sfbay_stations.npz, currents_fine.npz; and paper/recalibration_results.json with the harmonic and current-ellipse
results."""
import os, paths as P
import json, sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu

G = 9.81; DT = 120.0; THETA = 0.6; HMIN = 3.0; MANNING_N = 0.037; DEG = np.pi / 180.0
D = P.SFBAY
d = np.load(f"{D}/sfbay_fine.npz"); depth, wet, lat, lon = d["depth"], d["wet"], d["lat"], d["lon"]
dx = float(d["dx"]); dy = float(d["dy"]); ny, nx = depth.shape
H = np.where(wet, np.maximum(depth, HMIN), 0.0)
cid = -np.ones((ny, nx), int); wj, wi = np.where(wet); N = wj.size; cid[wj, wi] = np.arange(N)
jg, ig = 232, 96; kg = cid[jg, ig]                                   # Golden Gate cell (as in gen_currents_mixed.py)
def deepest(laq, loq, r=0.022):
    idx = np.where((np.abs(lat[wj] - laq) < r) & (np.abs(lon[wi] - loq) < r))[0]
    k = idx[np.argmax(depth[wj[idx], wi[idx]])]; return cid[wj[k], wi[k]]
STATIONS = {"Golden Gate": kg, "Oakland": deepest(37.81, -122.34), "Richmond": deepest(37.93, -122.42),
            "San Pablo Bay": deepest(38.05, -122.42)}
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

# NOAA 9414290 (San Francisco, Golden Gate): speed deg/h, amplitude m, Greenwich phase deg (as in gen_currents_mixed.py)
NOAA = {"M2": (28.98410, 0.576, 208.2), "S2": (30.00000, 0.137, 216.2), "N2": (28.43973, 0.122, 183.2),
        "K1": (15.04107, 0.370, 225.4), "O1": (13.94304, 0.230, 208.4), "P1": (14.95893, 0.114, 222.1)}
NUISANCE = {"M4": 57.96821, "MS4": 58.98410}
NAMES = list(NOAA)
def om(speed_deg_h): return speed_deg_h * DEG / 3600.0
FIT_OMEGA = [om(NOAA[n][0]) for n in NAMES] + [om(s) for s in NUISANCE.values()]


def run(forcing, ndays, rec=(), field_window=None):
    nst = int(ndays * 86400 / DT); eta = np.zeros(N); uvel = np.zeros(uk1.size); vvel = np.zeros(vk1.size)
    uf = {k: np.where((uk1 == k) | (uk2 == k))[0] for k in rec}; vf = {k: np.where((vk1 == k) | (vk2 == k))[0] for k in rec}
    ts = []; se = {k: [] for k in rec}; su = {k: [] for k in rec}; sv = {k: [] for k in rec}
    Uf = []; Vf = []; tf = []; nextf = field_window[0] if field_window else None
    for n in range(nst + 1):
        t = n * DT
        gEx = (eta[uk2] - eta[uk1]) / dx; gNy = (eta[vk2] - eta[vk1]) / dy
        pu = uvel - DT * G * (1 - THETA) * gEx; pv = vvel - DT * G * (1 - THETA) * gNy
        fxu = Hu * dy * (THETA * pu + (1 - THETA) * uvel); fxv = Hv * dx * (THETA * pv + (1 - THETA) * vvel)
        div = np.zeros(N); np.add.at(div, uk1, fxu); np.add.at(div, uk2, -fxu); np.add.at(div, vk1, fxv); np.add.at(div, vk2, -fxv)
        rhs = eta - (DT / A) * div
        rhs[isopen] = sum(a * np.cos(w * t - p) for (w, a, p) in forcing)
        eta = lu.solve(rhs)
        us = pu - DT * G * THETA * (eta[uk2] - eta[uk1]) / dx; vs = pv - DT * G * THETA * (eta[vk2] - eta[vk1]) / dy
        uvel = us / (1 + DT * CDu * np.abs(us) / Hu + DT * ru); vvel = vs / (1 + DT * CDv * np.abs(vs) / Hv + DT * rv)
        ts.append(t)
        for k in rec:
            se[k].append(eta[k]); su[k].append(uvel[uf[k]].mean()); sv[k].append(vvel[vf[k]].mean())
        if field_window and t >= nextf and t <= field_window[1] + 1e-6:
            us_ = np.zeros(N); uc_ = np.zeros(N); vs_ = np.zeros(N); vc_ = np.zeros(N)
            np.add.at(us_, uk1, uvel); np.add.at(uc_, uk1, 1.0); np.add.at(us_, uk2, uvel); np.add.at(uc_, uk2, 1.0)
            np.add.at(vs_, vk1, vvel); np.add.at(vc_, vk1, 1.0); np.add.at(vs_, vk2, vvel); np.add.at(vc_, vk2, 1.0)
            Uf.append(us_ / np.maximum(uc_, 1)); Vf.append(vs_ / np.maximum(vc_, 1)); tf.append(t); nextf += field_window[2]
        if not np.all(np.isfinite(eta)): sys.exit(f"blew up at step {n}")
    out = {"t": np.array(ts)}
    for k in rec:
        out[f"eta_{k}"] = np.array(se[k]); out[f"u_{k}"] = np.array(su[k]); out[f"v_{k}"] = np.array(sv[k])
    if field_window: out["U"] = np.array(Uf); out["V"] = np.array(Vf); out["tf"] = np.array(tf)
    return out


def hfit(t, y):
    """Simultaneous least squares: returns {name: (amp, phase_deg)} for the six constituents (y ~ A cos(w t - g)),
    the nuisance amplitudes, the raw cos/sin coefficients and the design-matrix condition number."""
    cols = [np.ones_like(t)]
    for w in FIT_OMEGA: cols += [np.cos(w * t), np.sin(w * t)]
    M = np.column_stack(cols); c, *_ = np.linalg.lstsq(M, y, rcond=None)
    out = {}; raw = {}
    for i, nm in enumerate(NAMES + list(NUISANCE)):
        a, b = c[1 + 2 * i], c[2 + 2 * i]; raw[nm] = (a, b)
        out[nm] = (float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360.0))
    return out, raw, float(np.linalg.cond(M))


def ellipse(ra_u, ra_v):
    """Tidal current ellipse from the cos/sin coefficients of u and v: semi-major, semi-minor (largest and smallest
    singular values of [[a_u, b_u], [a_v, b_v]]) and the direction of the major axis (deg clockwise from north)."""
    Mxy = np.array([[ra_u[0], ra_u[1]], [ra_v[0], ra_v[1]]])
    U, s, Vt = np.linalg.svd(Mxy)
    ax = U[:, 0]                                           # (u, v) of the major axis
    direction = float(np.degrees(np.arctan2(ax[0], ax[1])) % 180.0)
    return float(s[0]), float(s[1]), direction


def wrap(x): return (x + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    t0 = time.time(); NDAYS = 33; FIT0 = 3 * 86400
    cal_old = np.load(f"{D}/sfbay_calibration.npz")
    amp = {str(n): float(a) for n, a in zip(cal_old["names"], cal_old["offshore_amp"])}
    ph = {str(n): float(p) for n, p in zip(cal_old["names"], cal_old["noaa_phase"])}
    history = []
    for it in range(6):
        forcing = [(om(NOAA[n][0]), amp[n], ph[n] * DEG) for n in NAMES]
        r = run(forcing, NDAYS, rec=(kg,)); m = r["t"] >= FIT0
        fit, raw, cond = hfit(r["t"][m], r[f"eta_{kg}"][m])
        err = {n: (fit[n][0] / NOAA[n][1] - 1.0, wrap(fit[n][1] - NOAA[n][2])) for n in NAMES}
        history.append({"iteration": it, "offshore_amp": dict(amp), "offshore_phase": dict(ph),
                        "gate_fit": {n: fit[n] for n in NAMES}, "cond": cond})
        worst_a = max(abs(e[0]) for e in err.values()); worst_p = max(abs(e[1]) for e in err.values())
        print(f"iter {it}: worst amplitude error {100 * worst_a:.2f}%, worst phase error {worst_p:.2f} deg "
              f"(cond {cond:.1f}; {time.time() - t0:.0f}s)", flush=True)
        if worst_a < 0.005 and worst_p < 1.0:
            break
        for n in NAMES:
            amp[n] *= NOAA[n][1] / fit[n][0]; ph[n] -= err[n][1]

    # production run: stations and the figure field window, with the calibrated forcing
    forcing = [(om(NOAA[n][0]), amp[n], ph[n] * DEG) for n in NAMES]
    SPIN = 3 * 86400; WIN = (SPIN, SPIN + 26 * 3600, 1800.0)
    r = run(forcing, NDAYS, rec=tuple(STATIONS.values()), field_window=WIN)
    m = r["t"] >= FIT0
    fit, raw, cond = hfit(r["t"][m], r[f"eta_{kg}"][m])
    F_model = (fit["K1"][0] + fit["O1"][0]) / (fit["M2"][0] + fit["S2"][0])
    F_obs = (NOAA["K1"][1] + NOAA["O1"][1]) / (NOAA["M2"][1] + NOAA["S2"][1])
    currents = {}
    for nm, k in STATIONS.items():
        _, ru_, _ = hfit(r["t"][m], r[f"u_{k}"][m]); _, rv_, _ = hfit(r["t"][m], r[f"v_{k}"][m])
        smaj, smin, dirn = ellipse(ru_["M2"], rv_["M2"])
        currents[nm] = {"M2_semi_major": smaj, "M2_semi_minor": smin, "M2_major_axis_deg_from_north": dirn,
                        "lat": float(lat[wj[k]]), "lon": float(lon[wi[k]]), "depth": float(depth[wj[k], wi[k]])}
        print(f"  {nm}: M2 current ellipse semi-major {smaj:.3f} m/s, semi-minor {smin:.3f}, axis {dirn:.0f} deg", flush=True)
    res = {"record_days": NDAYS, "fit_window_days": [3, NDAYS], "fit_constituents": NAMES + list(NUISANCE),
           "condition_number": cond, "iterations": history,
           "offshore_amp": amp, "offshore_phase_deg": ph,
           "gate": {n: {"model_amp": fit[n][0], "model_phase": fit[n][1], "noaa_amp": NOAA[n][1],
                        "noaa_phase": NOAA[n][2]} for n in NAMES},
           "gate_nuisance_amp": {n: fit[n][0] for n in NUISANCE},
           "form_factor_model": F_model, "form_factor_noaa": F_obs, "currents": currents,
           "observed_M2_principal_current": {"Golden Gate (station C1)": 1.079, "Richmond": 0.708, "Oakland": 0.652,
                                             "source": "Sankaranarayanan and French McCay (2003), Tables 11, 13, 14"}}
    json.dump(res, open(os.path.join(P.PAPER, "recalibration_results.json"), "w"), indent=1)
    for n in NAMES:
        print(f"  {n}: model {fit[n][0]:.3f} m @ {fit[n][1]:.1f} deg | NOAA {NOAA[n][1]:.3f} m @ {NOAA[n][2]:.1f} deg")
    print(f"form factor model {F_model:.3f}, NOAA {F_obs:.3f}; condition number {cond:.1f}")
    st = list(STATIONS); kk = [STATIONS[s] for s in st]
    np.savez(f"{D}/currents_fine.npz", U=r["U"], V=r["V"], t=r["tf"], wj=wj, wi=wi, cid=cid, wet=wet, lat=lat, lon=lon, dx=dx, dy=dy)
    np.savez(f"{D}/sfbay_tides.npz", t=r["t"], eta_gg=r[f"eta_{kg}"], kg=kg, amp_names=NAMES,
             amp_model=[fit[n][0] for n in NAMES], amp_obs=[NOAA[n][1] for n in NAMES])
    np.savez(f"{D}/sfbay_stations.npz", t=r["t"], names=st, eta=np.array([r[f"eta_{k}"] for k in kk]),
             spd=np.array([np.hypot(r[f"u_{k}"], r[f"v_{k}"]) for k in kk]),
             u=np.array([r[f"u_{k}"] for k in kk]), v=np.array([r[f"v_{k}"] for k in kk]),
             lat=np.array([lat[wj[k]] for k in kk]), lon=np.array([lon[wi[k]] for k in kk]))
    np.savez(f"{D}/sfbay_calibration.npz", names=NAMES, noaa_amp=[NOAA[n][1] for n in NAMES],
             noaa_phase=[NOAA[n][2] for n in NAMES], offshore_amp=[amp[n] for n in NAMES],
             offshore_phase=[ph[n] for n in NAMES], model_amp=[fit[n][0] for n in NAMES],
             model_phase=[fit[n][1] for n in NAMES], manning_n=MANNING_N, hmin=HMIN, fit_days=NDAYS - 3)
    print(f"saved; total {time.time() - t0:.0f}s")
