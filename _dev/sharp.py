"""DEV PROBE (lever #1): does a SHARP (discontinuous) tracer feature under the
deformational flow require a BOUND-PRESERVING high-resolution scheme?

Pure advection (D=0). Exact reference via semi-Lagrangian back-trajectory
(integrate the characteristic ODE backward to t=0, evaluate the analytic IC) ->
no Gibbs, no grid-convergence worry, exact for a discontinuous IC.

Schemes (semi-Lagrangian, interpolation differs):
  - 'bilin' : bilinear interp        -> bounded but DIFFUSIVE (should lose the feature: low IoU)
  - 'cubic' : Catmull-Rom cubic      -> high-res but OVERSHOOTS (should violate boundedness)
  - 'mono'  : cubic clipped to local 2x2 [min,max] -> bounded + sharp (should PASS)
Plus Eulerian central (via advdiff) for a Gibbs contrast if desired.
"""
import numpy as np


def velocity(X, Y, t, T):
    ct = np.cos(np.pi * t / T)
    u = (np.sin(np.pi * X) ** 2) * np.sin(2 * np.pi * Y) * ct
    v = -np.sin(2 * np.pi * X) * (np.sin(np.pi * Y) ** 2) * ct
    return u, v


def slotted_cylinder(X, Y, cx=0.5, cy=0.65, R=0.18, sw=0.06, sh=0.22):
    """Zalesak-style: disk of value 1, with a rectangular slot cut out (-> 0)."""
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    c = (r <= R).astype(float)
    slot = (np.abs(X - cx) <= sw / 2) & (Y <= cy + R) & (Y >= cy - R + (2 * R - sh))
    c[slot] = 0.0
    return c


def exact_reference(N, T, tc, nsub=400, **ickw):
    """Backtrace each grid point from tc to 0; evaluate analytic IC there."""
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    px, py = X.copy(), Y.copy()
    dt = tc / nsub
    for n in range(nsub):
        s = tc - n * dt
        # RK4 backward (integrate dX/ds = vel, stepping s downward)
        u1, v1 = velocity(px, py, s, T)
        u2, v2 = velocity(px - 0.5 * dt * u1, py - 0.5 * dt * v1, s - 0.5 * dt, T)
        u3, v3 = velocity(px - 0.5 * dt * u2, py - 0.5 * dt * v2, s - 0.5 * dt, T)
        u4, v4 = velocity(px - dt * u3, py - dt * v3, s - dt, T)
        px = px - (dt / 6) * (u1 + 2 * u2 + 2 * u3 + u4)
        py = py - (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
    px = np.clip(px, 0, 1); py = np.clip(py, 0, 1)
    return slotted_cylinder(px, py, **ickw)


def _interp(field, xq, yq, mode, h):
    """Interpolate `field` (N,N; [j=y,i=x]) at query coords (xq,yq) in [0,1]."""
    N = field.shape[0]
    fx = np.clip(xq / h, 0, N - 1); fy = np.clip(yq / h, 0, N - 1)
    i0 = np.clip(np.floor(fx).astype(int), 0, N - 2); j0 = np.clip(np.floor(fy).astype(int), 0, N - 2)
    tx = fx - i0; ty = fy - j0

    def g(j, i):
        return field[np.clip(j, 0, N - 1), np.clip(i, 0, N - 1)]

    if mode == "bilin":
        return ((1 - tx) * (1 - ty) * g(j0, i0) + tx * (1 - ty) * g(j0, i0 + 1)
                + (1 - tx) * ty * g(j0 + 1, i0) + tx * ty * g(j0 + 1, i0 + 1))

    # cubic (Catmull-Rom) tensor product
    def cub(fm1, f0, f1, f2, t):
        return (f0 + 0.5 * t * (f1 - fm1 + t * (2 * fm1 - 5 * f0 + 4 * f1 - f2
                + t * (3 * (f0 - f1) + f2 - fm1))))
    cols = []
    for dj in (-1, 0, 1, 2):
        row = cub(g(j0 + dj, i0 - 1), g(j0 + dj, i0), g(j0 + dj, i0 + 1), g(j0 + dj, i0 + 2), tx)
        cols.append(row)
    val = cub(cols[0], cols[1], cols[2], cols[3], ty)
    if mode == "cubic":
        return val
    if mode == "mono":  # clip cubic to local bilinear-stencil bounds
        lo = np.minimum.reduce([g(j0, i0), g(j0, i0 + 1), g(j0 + 1, i0), g(j0 + 1, i0 + 1)])
        hi = np.maximum.reduce([g(j0, i0), g(j0, i0 + 1), g(j0 + 1, i0), g(j0 + 1, i0 + 1)])
        return np.clip(val, lo, hi)
    raise ValueError(mode)


def _lap(c, h):  # 2nd-order Laplacian, reflect (Neumann no-flux)
    p = np.pad(c, 1, mode="reflect")
    return (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * p[1:-1, 1:-1]) / h ** 2


def sl_solve(N, T, tc, mode, dt=0.02, D=0.0, **ickw):
    """Monotone semi-Lagrangian advection (interp `mode`) + explicit diffusion substep."""
    x = np.linspace(0, 1, N); h = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    c = slotted_cylinder(X, Y, **ickw)
    nsteps = int(round(tc / dt)); dt = tc / nsteps
    # diffusion sub-steps for explicit stability (dt_d < h^2/(4D))
    nd = 1 if D == 0 else max(1, int(np.ceil(dt / (0.2 * h * h / max(D, 1e-30)))))
    dtd = dt / nd
    for n in range(nsteps):
        t = n * dt
        u1, v1 = velocity(X, Y, t + dt, T)
        u2, v2 = velocity(X - 0.5 * dt * u1, Y - 0.5 * dt * v1, t + 0.5 * dt, T)
        u3, v3 = velocity(X - 0.5 * dt * u2, Y - 0.5 * dt * v2, t + 0.5 * dt, T)
        u4, v4 = velocity(X - dt * u3, Y - dt * v3, t, T)
        xd = X - (dt / 6) * (u1 + 2 * u2 + 2 * u3 + u4)
        yd = Y - (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
        c = _interp(c, xd, yd, mode, h)
        if D > 0:
            for _ in range(nd):
                c = c + dtd * D * _lap(c, h)
    return c, h
