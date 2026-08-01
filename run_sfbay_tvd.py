# TVD-MUSCL (superbee) comparator for the SF Bay tracer test -- same setup, IC,
# window and step as sfbay_transport.py; saves sfbay_tvd_result.npz for the
# bounds/retained-mass table (Table sfbaybounds) of the OE manuscript.
import numpy as np

FINE = "sfbay_fine"
cc = np.load(f"{FINE}/currents_fine.npz")
wet, lat, lon = cc["wet"], cc["lat"], cc["lon"]
U, V, cid, tt = cc["U"], cc["V"], cc["cid"], cc["t"]
dx = float(cc["dx"]); dy = float(cc["dy"]); ny, nx = wet.shape


def vel_grid(k):
    Ug = np.zeros((ny, nx)); Vg = np.zeros((ny, nx))
    Ug[wet] = U[k, cid[wet]]; Vg[wet] = V[k, cid[wet]]
    return Ug, Vg


def vel_at(t):
    k = np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2)
    a = (t - tt[k]) / (tt[k + 1] - tt[k])
    U0, V0 = vel_grid(k); U1, V1 = vel_grid(k + 1)
    return (1 - a) * U0 + a * U1, (1 - a) * V0 + a * V1


def muscl_adv_1d(c, u, h, axis):
    eps = 1e-14
    cp1 = np.roll(c, -1, axis); cm1 = np.roll(c, 1, axis)
    dc_f = cp1 - c
    dc_b = c - cm1
    def phi(r):
        return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
    rL = dc_b / np.where(np.abs(dc_f) < eps, eps, dc_f)
    cL_f = c + 0.5 * phi(rL) * dc_f
    dpos = (cL_f - np.roll(cL_f, 1, axis)) / h
    dc_ff = np.roll(c, -2, axis) - cp1
    rR = dc_ff / np.where(np.abs(dc_f) < eps, eps, dc_f)
    cR_f = cp1 - 0.5 * phi(rR) * dc_f
    dneg = (cR_f - np.roll(cR_f, 1, axis)) / h
    return np.where(u >= 0, dpos, dneg)


i0 = int(np.argmin(np.abs(lon - (-122.357)))); j0 = int(np.argmin(np.abs(lat - 37.806)))
Lon, Lat = np.meshgrid(lon, lat); R_km = 3.5
dist = np.sqrt(((Lon - lon[i0]) * 111 * np.cos(np.radians(37.8))) ** 2 + ((Lat - lat[j0]) * 111) ** 2)
c0 = ((dist <= R_km) & wet).astype(float)
T0 = float(tt[0]); dt = 60.0; nsteps = 300

c = c0.copy()
for n in range(nsteps):
    t = T0 + n * dt
    u, v = vel_at(t)
    def L(cc_, tt_):
        return -(u * muscl_adv_1d(cc_, u, dx, 1) + v * muscl_adv_1d(cc_, v, dy, 0))
    c1 = c + dt * L(c, t)
    c = 0.5 * c + 0.5 * (c1 + dt * L(c1, t + dt))
    c *= wet

print(f"tvd  final min={c.min():+.4f} max={c.max():.4f} retained={c.sum()/c0.sum():.4f}")
np.savez("sfbay_tvd_result.npz", c_tvd=c, c0=c0)
