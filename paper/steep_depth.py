"""Accuracy and stability across a steep shelf break (JCP revision, 2026-10-28): the stress test for the face values
where the depth varies.

As variable_depth.py (closed unit basin, steady divergence-free depth-integrated transport q = rot(psi), psi =
sin^2(pi x) sin^2(pi y) / pi, tracer carried by u = q/D, Gaussian of the benchmark), but the depth is a shelf break
D(x) = 1 - (1 - DMIN)(1 + tanh((x - 0.5)/W))/2, from 1 on the left to DMIN on the right over a width W of a few cells.
The Gaussian (centre (0.5, 0.35)) sits on the break and is carried across it. Cell volumes are the exact integrals of D.
Speeds reach max|q|/DMIN, so the step is reduced until the largest outflow fraction is PHI (0.51 in variable_depth.py).
Data and reference are depth-weighted control-volume averages by sub-cell quadrature (m = 16, RK4 back-trajectories
with NSUB steps; the reference is checked against 2 NSUB on the grids up to 65^2). Schemes as in variable_depth.py:
c reconstructed directly (default) or the depth-aware ratio ("_da"). Then the stability check: a uniform tracer 0.5 with
independent N(0, 1e-6) cell perturbations, unlimited compact flux, 65^2, on this shelf and on the smooth bathymetry of
variable_depth.py; the depth-weighted rms perturbation relative to its initial value at t = 0.25, 0.5 and 1.
Writes steep_depth.json (or the file named by --out)."""
import argparse, json, math, time
import numpy as np
import gen_mpdata as M
import gen_conservative as C
import cv_reference as CV

C.REPRESENTATION = "cv"
PHI = 0.51
P = argparse.ArgumentParser()
P.add_argument("--dmin", type=float, default=0.1)
P.add_argument("--width", type=float, default=0.05)
P.add_argument("--t_end", type=float, default=0.5)
P.add_argument("--nsub", type=int, default=1000)
P.add_argument("--grids", default="33,65,129")
P.add_argument("--schemes", default="ccompact,ccompact_da,cfctg,cfctg_da,c4fctg,u5fctg")
P.add_argument("--out", default="steep_depth.json")
A = P.parse_args()
DMIN, W = A.dmin, A.width


def depth(x, y):
    return 1.0 - (1.0 - DMIN) * 0.5 * (1.0 + np.tanh((x - 0.5) / W)) + 0 * y


def velocity(x, y):
    s = np.sin(math.pi * x); r = np.sin(math.pi * y); D = depth(x, y)
    return s ** 2 * np.sin(2 * math.pi * y) / D, -r ** 2 * np.sin(2 * math.pi * x) / D


def logcosh(z):
    return np.logaddexp(z, -z) - math.log(2.0)


class ShelfGrid(M.Grid):
    def __init__(self, N):
        M.Grid.__init__(self, N)
        a, b = 0.5 * (1 + DMIN), 0.5 * (1 - DMIN)
        Ix = a * self.L - b * W * np.diff(logcosh((self.e - 0.5) / W))      # exact integral of D over each CV in x
        self.V = np.outer(self.L, Ix)

    def psi(self, xx, yy, t):
        return (1 / np.pi) * np.sin(np.pi * xx) ** 2 * np.sin(np.pi * yy) ** 2


def weighted_average(g, t, nsub, m=16):
    frac = (np.arange(m) + 0.5) / m
    xs = (g.e[:-1, None] + frac[None, :] * g.L[:, None]).ravel()
    X, Y = np.meshgrid(xs, xs); px, py = X.copy(), Y.copy(); d = t / nsub
    for _ in range(nsub if t > 0 else 0):
        u1, v1 = velocity(px, py)
        u2, v2 = velocity(px - .5 * d * u1, py - .5 * d * v1)
        u3, v3 = velocity(px - .5 * d * u2, py - .5 * d * v2)
        u4, v4 = velocity(px - d * u3, py - d * v3)
        px = px - d / 6 * (u1 + 2 * u2 + 2 * u3 + u4); py = py - d / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
    D = depth(X, Y); f = D * M.gauss_ic(np.clip(px, 0, 1), np.clip(py, 0, 1)); N = g.N
    return f.reshape(N, m, N, m).sum(axis=(1, 3)) / D.reshape(N, m, N, m).sum(axis=(1, 3))


def score(c, ref, g):
    w = g.V
    return {"rel_l2": float(np.sqrt((w * (c - ref) ** 2).sum() / (w * ref ** 2).sum())), "min": float(c.min()),
            "max": float(c.max()), "max_err": float(np.abs(c - ref).max())}


if __name__ == "__main__":
    Ns = tuple(int(n) for n in A.grids.split(",")); schemes = A.schemes.split(",")
    out = {"dmin": DMIN, "width": W, "t_end": A.t_end, "phi": PHI, "cv_subcells": 16, "trajectory_steps": A.nsub,
           "code": CV.CODE, "space": {}}
    for N in Ns:
        t0 = time.time(); g = ShelfGrid(N)
        assert abs(g.V.sum() - (0.5 * (1 + DMIN) - 0.5 * (1 - DMIN) * W * (logcosh(0.5 / W) - logcosh(-0.5 / W)))) < 1e-12
        c0 = weighted_average(g, 0.0, A.nsub); ref = weighted_average(g, A.t_end, A.nsub)
        ref2 = weighted_average(g, A.t_end, 2 * A.nsub) if N <= 65 else None
        Qx, Qy = g.fluxes(0.0)
        outflow = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0)
                   + np.maximum(-Qy[:-1, :], 0))
        phi0 = float((M.CFL * g.h * outflow / g.V).max()); dt = M.CFL * g.h * min(1.0, PHI / phi0)
        rec = out["space"].setdefault(str(N), {"dt": dt, "steps": int(round(A.t_end / dt)), "W_over_h": W / g.h,
                                              "ref_max": float(ref.max()), "init_max": float(c0.max())})
        if ref2 is not None:
            rec["ref_nsub_change"] = float(np.sqrt((g.V * (ref - ref2) ** 2).sum() / (g.V * ref2 ** 2).sum()))
        for s in schemes:
            t1 = time.time()
            c, _, gg, _ = C.run(N, s, lambda X, Y: c0, A.t_end, grid=g, dt=dt)
            r = score(c, ref, g)
            r["mass_rel"] = float((g.V * c).sum() / (g.V * c0).sum() - 1)
            r["depth_fallback_faces"] = C.FALLBACK[0]; r["seconds"] = time.time() - t1
            rec[s] = r
        print(f"N={N}: " + " ".join(f"{s} {rec[s]['rel_l2']:.3e}" for s in schemes) +
              f" | ref {rec.get('ref_nsub_change', float('nan')):.1e} | fb " +
              " ".join(str(rec[s]["depth_fallback_faces"]) for s in schemes) + f" | {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(A.out, "w"), indent=1)
    for s in schemes:
        Es = [out["space"][str(N)][s]["rel_l2"] for N in Ns]
        print(f"{s:12s} orders " + " ".join(f"{math.log2(a / b):.2f}" for a, b in zip(Es, Es[1:])))
    import variable_depth as VD
    rng = np.random.default_rng(1); N = 65; out["noise"] = {"N": N, "amplitude": 1e-6, "seed": 1}
    for name, g in (("shelf", ShelfGrid(N)), ("cosine", VD.BathyGrid(N))):
        Qx, Qy = g.fluxes(0.0)
        outflow = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0)
                   + np.maximum(-Qy[:-1, :], 0))
        dt = M.CFL * g.h * min(1.0, PHI / float((M.CFL * g.h * outflow / g.V).max()))
        c0 = 0.5 + 1e-6 * rng.standard_normal((N, N))
        rms = lambda c: float(np.sqrt((g.V * (c - 0.5) ** 2).sum() / g.V.sum()))
        for s in ("ccompact", "ccompact_da"):
            for T in (0.25, 0.5, 1.0):
                c, *_ = C.run(N, s, lambda X, Y: c0, T, grid=g, dt=dt)
                out["noise"].setdefault(name, {}).setdefault(s, {})[str(T)] = rms(c) / rms(c0)
            print(f"noise {name:6s} {s:12s} growth " + " ".join(f"t={T}: {v:.3g}" for T, v in out["noise"][name][s].items()),
                  flush=True)
    json.dump(out, open(A.out, "w"), indent=1)
