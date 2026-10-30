"""DEV: 2-D advection-diffusion on a closed basin [0,1]^2 under a time-dependent
deformational (LeVeque swirl) flow. Method-of-lines + RK4. Multiple spatial
schemes so we can show a HIGH-order scheme (central4 / compact4 Pade) matches a
converged reference while LOW-order schemes (upwind1 / central2) fail.

Streamfunction  psi = (1/pi) sin^2(pi x) sin^2(pi y) cos(pi t / T)
  u =  d psi/dy =  sin^2(pi x) sin(2 pi y) cos(pi t/T)
  v = -d psi/dx = -sin(2 pi x) sin^2(pi y) cos(pi t/T)
Non-divergent; u=v=0 on the walls (closed basin). cos(pi t/T) reverses the flow,
so for pure advection (D=0) the tracer returns to its initial state at t=T, with
maximum deformation (thinnest filament) at t=T/2.

BCs: no normal flow (automatic, wall velocity = 0) + no-flux diffusion (Neumann),
enforced by reflective padding for the explicit stencils.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded


def velocity(X, Y, t, T):
    ct = np.cos(np.pi * t / T)
    u = (np.sin(np.pi * X) ** 2) * np.sin(2 * np.pi * Y) * ct
    v = -np.sin(2 * np.pi * X) * (np.sin(np.pi * Y) ** 2) * ct
    return u, v


# ----- explicit central / upwind derivative operators (reflect = Neumann) -----
def _pad(a, w):  # pad last axis, reflect (mirror, no edge repeat) -> zero normal deriv
    return np.pad(a, ((0, 0), (w, w)), mode="reflect")


def d1(c, h, axis, order):
    a = np.moveaxis(c, axis, -1)
    if order == 2:
        p = _pad(a, 1)
        r = (p[:, 2:] - p[:, :-2]) / (2 * h)
    elif order == 4:
        p = _pad(a, 2)
        r = (-p[:, 4:] + 8 * p[:, 3:-1] - 8 * p[:, 1:-3] + p[:, :-4]) / (12 * h)
    elif order == 6:
        p = _pad(a, 3)
        r = (p[:, 6:] - 9 * p[:, 5:-1] + 45 * p[:, 4:-2] - 45 * p[:, 2:-4]
             + 9 * p[:, 1:-5] - p[:, :-6]) / (60 * h)
    elif order == 8:
        p = _pad(a, 4)
        r = (-3 * p[:, 8:] + 32 * p[:, 7:-1] - 168 * p[:, 6:-2] + 672 * p[:, 5:-3]
             - 672 * p[:, 3:-5] + 168 * p[:, 2:-6] - 32 * p[:, 1:-7] + 3 * p[:, :-8]) / (840 * h)
    return np.moveaxis(r, -1, axis)


def d2(c, h, axis, order):
    a = np.moveaxis(c, axis, -1)
    if order == 2:
        p = _pad(a, 1)
        r = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / h ** 2
    elif order == 4:
        p = _pad(a, 2)
        r = (-p[:, 4:] + 16 * p[:, 3:-1] - 30 * p[:, 2:-2] + 16 * p[:, 1:-3] - p[:, :-4]) / (12 * h ** 2)
    elif order == 6:
        p = _pad(a, 3)
        r = (2 * p[:, 6:] - 27 * p[:, 5:-1] + 270 * p[:, 4:-2] - 490 * p[:, 3:-3]
             + 270 * p[:, 2:-4] - 27 * p[:, 1:-5] + 2 * p[:, :-6]) / (180 * h ** 2)
    elif order == 8:
        p = _pad(a, 4)
        r = (-9 * p[:, 8:] + 128 * p[:, 7:-1] - 1008 * p[:, 6:-2] + 8064 * p[:, 5:-3]
             - 14350 * p[:, 4:-4] + 8064 * p[:, 3:-5] - 1008 * p[:, 2:-6]
             + 128 * p[:, 1:-7] - 9 * p[:, :-8]) / (5040 * h ** 2)
    return np.moveaxis(r, -1, axis)


def d1_upwind(c, vel, h, axis):
    """First-order upwind first derivative based on sign of vel (advective)."""
    a = np.moveaxis(c, axis, -1)
    p = _pad(a, 1)
    back = (p[:, 1:-1] - p[:, :-2]) / h     # (c_i - c_{i-1})/h
    fwd = (p[:, 2:] - p[:, 1:-1]) / h       # (c_{i+1} - c_i)/h
    back = np.moveaxis(back, -1, axis)
    fwd = np.moveaxis(fwd, -1, axis)
    return np.where(vel >= 0, back, fwd)


# ----- compact (Pade) 4th-order operators, tridiagonal, Lele(1992) closures -----
class Compact:
    def __init__(self, n, h):
        self.n, self.h = n, h
        self.ab1 = self._band_d1(n)
        self.ab2 = self._band_d2(n)

    @staticmethod
    def _band_d1(n):
        ab = np.zeros((3, n))
        ab[1, :] = 1.0                 # diag
        ab[0, 1:] = 0.25               # super (interior alpha=1/4)
        ab[2, :-1] = 0.25              # sub
        ab[0, 1] = 2.0                 # row 0 super (3rd-order boundary)
        ab[2, -2] = 2.0                # row n-1 sub
        return ab

    @staticmethod
    def _band_d2(n):
        ab = np.zeros((3, n))
        ab[1, :] = 1.0
        ab[0, 1:] = 0.1                # interior alpha=1/10
        ab[2, :-1] = 0.1
        ab[0, 1] = 11.0                # boundary closure coeff
        ab[2, -2] = 11.0
        return ab

    def _rhs1(self, a):  # a: (lines, n), derivative along last axis
        h, n = self.h, self.n
        r = np.empty_like(a)
        r[:, 1:-1] = 0.75 * (a[:, 2:] - a[:, :-2]) / h
        r[:, 0] = (-2.5 * a[:, 0] + 2.0 * a[:, 1] + 0.5 * a[:, 2]) / h
        r[:, -1] = (2.5 * a[:, -1] - 2.0 * a[:, -2] - 0.5 * a[:, -3]) / h
        return r

    def _rhs2(self, a):
        h, n = self.h, self.n
        r = np.empty_like(a)
        r[:, 1:-1] = 1.2 * (a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]) / h ** 2
        r[:, 0] = (13 * a[:, 0] - 27 * a[:, 1] + 15 * a[:, 2] - a[:, 3]) / h ** 2
        r[:, -1] = (13 * a[:, -1] - 27 * a[:, -2] + 15 * a[:, -3] - a[:, -4]) / h ** 2
        return r

    def d1(self, c, axis):
        a = np.moveaxis(c, axis, -1)
        sh = a.shape
        a2 = a.reshape(-1, sh[-1])
        r = solve_banded((1, 1), self.ab1, self._rhs1(a2).T).T
        return np.moveaxis(r.reshape(sh), -1, axis)

    def d2(self, c, axis):
        a = np.moveaxis(c, axis, -1)
        sh = a.shape
        a2 = a.reshape(-1, sh[-1])
        r = solve_banded((1, 1), self.ab2, self._rhs2(a2).T).T
        return np.moveaxis(r.reshape(sh), -1, axis)


def make_rhs(scheme, X, Y, h, D, T, compact=None):
    def rhs(c, t):
        u, v = velocity(X, Y, t, T)
        if scheme == "upwind1":
            adv = u * d1_upwind(c, u, h, 1) + v * d1_upwind(c, v, h, 0)
            dif = D * (d2(c, h, 1, 2) + d2(c, h, 0, 2))
        elif scheme == "central2":
            adv = u * d1(c, h, 1, 2) + v * d1(c, h, 0, 2)
            dif = D * (d2(c, h, 1, 2) + d2(c, h, 0, 2))
        elif scheme == "central4":
            adv = u * d1(c, h, 1, 4) + v * d1(c, h, 0, 4)
            dif = D * (d2(c, h, 1, 4) + d2(c, h, 0, 4))
        elif scheme == "central6":
            adv = u * d1(c, h, 1, 6) + v * d1(c, h, 0, 6)
            dif = D * (d2(c, h, 1, 6) + d2(c, h, 0, 6))
        elif scheme == "central8":
            adv = u * d1(c, h, 1, 8) + v * d1(c, h, 0, 8)
            dif = D * (d2(c, h, 1, 8) + d2(c, h, 0, 8))
        elif scheme == "compact4":
            adv = u * compact.d1(c, 1) + v * compact.d1(c, 0)
            dif = D * (compact.d2(c, 1) + compact.d2(c, 0))
        else:
            raise ValueError(scheme)
        return -adv + dif
    return rhs


def initial(X, Y, xc=0.5, yc=0.75, sig=0.08):
    return np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * sig ** 2))


def run(N, scheme, dt, T, D, checkpoints, xc=0.5, yc=0.75, sig=0.08):
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x)            # X varies along axis=1, Y along axis=0
    c = initial(X, Y, xc, yc, sig)
    compact = Compact(N, h) if scheme == "compact4" else None
    rhs = make_rhs(scheme, X, Y, h, D, T, compact)
    nsteps = int(round(T / dt))
    dt = T / nsteps
    cps = sorted(checkpoints)
    cp_steps = {int(round(tc / dt)): tc for tc in cps}
    out = {}
    for n in range(nsteps + 1):
        t = n * dt
        if n in cp_steps:
            out[cp_steps[n]] = c.copy()
        if n == nsteps:
            break
        k1 = rhs(c, t)
        k2 = rhs(c + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(c + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(c + dt * k3, t + dt)
        c = c + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out, h


def mass(c, h):
    return float(c.sum() * h * h)
