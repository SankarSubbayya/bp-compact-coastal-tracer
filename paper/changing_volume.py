"""Accuracy on time-varying control volumes (JCP revision, 2026-10-28): a smooth tracer in a closed unit basin whose
water depth changes in space and time, with a reference solution.

Depth h(x, y, t) = H0 + 2 pi^2 B(t) cos(pi x) cos(pi y), B(t) = b sin(omega t), and depth-integrated transport
q = rot(psi) + grad(phi), with psi the deformational streamfunction of the benchmark and phi = A(t) cos(pi x) cos(pi y),
A = B'. Then dh/dt + div q = 0 exactly, q.n = 0 on the walls, and the face transports and cell volumes are analytic:
  x-face at x_f over [y0, y1]: psi(x_f, y1) - psi(x_f, y0) - A sin(pi x_f) (sin(pi y1) - sin(pi y0)),
  y-face at y_g over [x0, x1]: -(psi(x1, y_g) - psi(x0, y_g)) - A sin(pi y_g) (sin(pi x1) - sin(pi x0)),
  V = H0 |cell| + 2 B (sin(pi x1) - sin(pi x0)) (sin(pi y1) - sin(pi y0)).
The tracer obeys d(hc)/dt + div(q c) = 0, i.e. c is carried by u = q/h. The scheme carries tracer mass on volumes
advanced by its own RK4 stages from the face transports (gen_conservative, moving grid), starting from the analytic
volumes; data and reference are depth-weighted control-volume averages computed by sub-cell quadrature (m = 16, RK4
back-trajectories of u with 400 steps). Spatial refinement at the benchmark time step (dt = 0.35 h) and time-step
refinement on 129^2. Writes changing_volume.json."""
import json, math, time
import numpy as np
import gen_mpdata as M
import gen_conservative as C
import cv_reference as CV

C.REPRESENTATION = "cv"
H0, AMP = 1.0, 0.4                    # depth varies between H0 - AMP and H0 + AMP
OMEGA = 2 * math.pi / (M.T / 2)       # one oscillation over the run to t = T/2
BAMP = AMP / (2 * math.pi ** 2)
B = lambda t: BAMP * math.sin(OMEGA * t)
Ab = lambda t: BAMP * OMEGA * math.cos(OMEGA * t)


def depth(x, y, t):
    return H0 + 2 * math.pi ** 2 * B(t) * np.cos(math.pi * x) * np.cos(math.pi * y)


def velocity(x, y, t):
    ur, vr = M.A.velocity(x, y, t, M.T)                       # rot(psi): the benchmark's divergence-free transport
    a = Ab(t); h = depth(x, y, t)
    return (ur - math.pi * a * np.sin(math.pi * x) * np.cos(math.pi * y)) / h, \
           (vr - math.pi * a * np.cos(math.pi * x) * np.sin(math.pi * y)) / h


class MovingGrid(M.Grid):
    moving = True

    def fluxes(self, t):
        Qx, Qy = M.Grid.fluxes(self, t)
        e = self.e; s = np.sin(math.pi * e); a = Ab(t)
        Qx = Qx - a * s[None, :] * (s[1:] - s[:-1])[:, None]       # Qx[j, f]: x = e[f], y in [e[j], e[j+1]]
        Qy = Qy - a * s[:, None] * (s[1:] - s[:-1])[None, :]       # Qy[g, i]: y = e[g], x in [e[i], e[i+1]]
        return Qx, Qy

    def volume(self, t):
        s = np.sin(math.pi * self.e); ds = s[1:] - s[:-1]
        return H0 * self.V + 2 * B(t) * np.outer(ds, ds)


def weighted_average(g, t, m=16, nsub=400):
    """Depth-weighted control-volume average of the Gaussian at time t (back-trajectories from t to 0)."""
    frac = (np.arange(m) + 0.5) / m
    xs = (g.e[:-1, None] + frac[None, :] * g.L[:, None]).ravel()
    X, Y = np.meshgrid(xs, xs); px, py = X.copy(), Y.copy(); d = t / nsub
    for n in range(nsub if t > 0 else 0):
        s = t - n * d
        u1, v1 = velocity(px, py, s)
        u2, v2 = velocity(px - .5 * d * u1, py - .5 * d * v1, s - .5 * d)
        u3, v3 = velocity(px - .5 * d * u2, py - .5 * d * v2, s - .5 * d)
        u4, v4 = velocity(px - d * u3, py - d * v3, s - d)
        px = px - d / 6 * (u1 + 2 * u2 + 2 * u3 + u4); py = py - d / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
    h = depth(X, Y, t); f = h * M.gauss_ic(np.clip(px, 0, 1), np.clip(py, 0, 1)); N = g.N
    return f.reshape(N, m, N, m).sum(axis=(1, 3)) / h.reshape(N, m, N, m).sum(axis=(1, 3))


def max_outflow_fraction(g, dt, t_end):
    """Largest Delta t sum max(Qbar, 0) / V^n over the run (volumes as the scheme advances them)."""
    nst = int(round(t_end / dt)); dt = t_end / nst; V = g.volume(0.0); phi = 0.0
    b = (0.0, 0.5, 0.5, 1.0); w = (1.0, 2.0, 2.0, 1.0)
    for n in range(nst):
        Q = [g.fluxes(n * dt + bb * dt) for bb in b]
        Qx = sum(ww * q[0] for ww, q in zip(w, Q)) / 6; Qy = sum(ww * q[1] for ww, q in zip(w, Q)) / 6
        out = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0) + np.maximum(-Qy[:-1, :], 0))
        phi = max(phi, float((dt * out / V).max()))
        V = V - dt * C.divergence(Qx, Qy)
    return phi


def score(c, ref, g):
    w = g.volume(M.T / 2)
    return {"rel_l2": float(np.sqrt((w * (c - ref) ** 2).sum() / (w * ref ** 2).sum())), "min": float(c.min()),
            "max": float(c.max())}


SCHEMES = ("cfctg", "cfctg_da", "cfctg_nt", "c4fctg", "u5fctg", "cfcte", "cdonor")
if __name__ == "__main__":
    out = {"H0": H0, "amplitude": AMP, "omega": OMEGA, "cv_subcells": 16, "trajectory_steps": 400, "code": CV.CODE,
           "space": {}, "time": {}}
    for N in (33, 65, 129, 257):
        t0 = time.time(); g = MovingGrid(N)
        c0 = weighted_average(g, 0.0); ref = weighted_average(g, M.T / 2)
        rec = out["space"].setdefault(str(N), {"outflow_fraction": max_outflow_fraction(g, M.CFL * g.h, M.T / 2)})
        assert rec["outflow_fraction"] <= 1.0
        for s in SCHEMES:
            c, _, gg, _ = C.run(N, s, lambda X, Y: c0, M.T / 2, grid=g)
            r = score(c, ref, g)
            r["mass_rel"] = float((g.V_final * c).sum() / (g.volume(0.0) * c0).sum() - 1)
            r["volume_drift"] = float(np.abs(g.V_final / g.volume(M.T / 2) - 1).max())
            rec[s] = r
        print(f"N={N}: " + " ".join(f"{s} {rec[s]['rel_l2']:.3e}" for s in SCHEMES) +
              f" | phi {rec['outflow_fraction']:.2f} | {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open("changing_volume.json", "w"), indent=1)
    g = MovingGrid(129); c0 = weighted_average(g, 0.0); ref = weighted_average(g, M.T / 2)
    for f in (1.0, 0.5, 0.25):
        dt = f * M.CFL * g.h
        out["time"][str(f)] = {s: score(C.run(129, s, lambda X, Y: c0, M.T / 2, grid=g, dt=dt)[0], ref, g)
                               for s in ("cfctg", "cfctg_nt")}
        print(f"dt factor {f}: " + " ".join(f"{s} {v['rel_l2']:.4e}" for s, v in out["time"][str(f)].items()), flush=True)
    c, _, _, _ = C.run(65, "cfctg", lambda X, Y: np.ones_like(X), M.T / 2, grid=MovingGrid(65))
    out["uniform_tracer_max_dev"] = float(np.abs(c - 1).max())
    json.dump(out, open("changing_volume.json", "w"), indent=1)
    print("uniform tracer max |c - 1|:", out["uniform_tracer_max_dev"])
