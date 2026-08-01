# Global-bounds [0,1] bp variant on the SF Bay test (same setup as sfbay_transport.py).
import numpy as np
from scipy.linalg import solve_banded
FINE = "sfbay_fine"
cc = np.load(f"{FINE}/currents_fine.npz")
wet, lat, lon = cc["wet"], cc["lat"], cc["lon"]
U, V, cid, tt = cc["U"], cc["V"], cc["cid"], cc["t"]
dx = float(cc["dx"]); dy = float(cc["dy"]); ny, nx = wet.shape
def vel_grid(k):
    Ug = np.zeros((ny, nx)); Vg = np.zeros((ny, nx))
    Ug[wet] = U[k, cid[wet]]; Vg[wet] = V[k, cid[wet]]; return Ug, Vg
def vel_at(t):
    k = np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2)
    a = (t - tt[k]) / (tt[k + 1] - tt[k])
    U0, V0 = vel_grid(k); U1, V1 = vel_grid(k + 1)
    return (1 - a) * U0 + a * U1, (1 - a) * V0 + a * V1
def cmats(n):
    ab = np.zeros((3, n)); ab[1, :] = 1; ab[0, 1:] = 0.25; ab[2, :-1] = 0.25
    ab[0, 1] = 0; ab[2, -2] = 0; return ab
ABX = cmats(nx); ABY = cmats(ny)
def cd1(c, h, axis, ab):
    a = np.moveaxis(c, axis, -1); s = a.shape; a2 = a.reshape(-1, s[-1]); r = np.empty_like(a2)
    r[:, 1:-1] = 0.75 * (a2[:, 2:] - a2[:, :-2]) / h; r[:, 0] = 0; r[:, -1] = 0
    return np.moveaxis(solve_banded((1, 1), ab, r.T).T.reshape(s), -1, axis)
def up1(c, vel, h, axis):
    a = np.moveaxis(c, axis, -1); p = np.pad(a, ((0, 0), (1, 1)), mode="edge")
    bk = (p[:, 1:-1] - p[:, :-2]) / h; fw = (p[:, 2:] - p[:, 1:-1]) / h
    return np.where(vel >= 0, np.moveaxis(bk, -1, axis), np.moveaxis(fw, -1, axis))
def rhs(c, t):
    u, v = vel_at(t); return -(u * cd1(c, dx, 1, ABX) + v * cd1(c, dy, 0, ABY))
i0 = int(np.argmin(np.abs(lon - (-122.357)))); j0 = int(np.argmin(np.abs(lat - 37.806)))
Lon, Lat = np.meshgrid(lon, lat); R_km = 3.5
dist = np.sqrt(((Lon - lon[i0]) * 111 * np.cos(np.radians(37.8))) ** 2 + ((Lat - lat[j0]) * 111) ** 2)
c0 = ((dist <= R_km) & wet).astype(float)
T0 = float(tt[0]); dt = 60.0; nsteps = 300
c = c0.copy()
for n in range(nsteps):
    t = T0 + n * dt
    k1 = rhs(c, t); k2 = rhs(c + .5*dt*k1, t + .5*dt); k3 = rhs(c + .5*dt*k2, t + .5*dt); k4 = rhs(c + dt*k3, t + dt)
    cHO = c + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    u, v = vel_at(t); cLO = c - dt*(u*up1(c, u, dx, 1) + v*up1(c, v, dy, 0))
    flag = (cHO < -1e-9) | (cHO > 1 + 1e-9)
    A_ = cHO - cLO; th = np.ones_like(A_)
    pos = A_ > 1e-15; neg = A_ < -1e-15
    th = np.where(pos, np.minimum(1, np.maximum(0, (1 - cLO) / np.where(pos, A_, 1))), th)
    th = np.where(neg, np.minimum(1, np.maximum(0, cLO / np.where(neg, -A_, 1))), th)
    c = np.where(flag, cLO + th * A_, cHO)
    c *= wet
print(f"bp-global  final min={c.min():+.4f} max={c.max():.4f} retained={c.sum()/c0.sum():.4f}")
np.savez("sfbay_bpglobal_result.npz", c_bpg=c, c0=c0)
