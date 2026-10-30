"""Accuracy over steady variable bathymetry (JCP revision, 2026-10-28): the test for the depth-aware face values.

Closed unit basin with fixed depth D(x, y) = 1 + 0.4 cos(pi x) cos(pi y) (0.6 to 1.4), steady divergence-free
depth-integrated transport q = rot(psi), psi = sin^2(pi x) sin^2(pi y) / pi (the benchmark streamfunction without its
time reversal), so the cell volumes are constant (exact cell averages of D) and the tracer is carried by u = q/D. The
Gaussian of the benchmark is transported to t = T_END. Data and reference are depth-weighted control-volume averages by
sub-cell quadrature (m = 16, RK4 back-trajectories of u with 400 steps). In changing_volume.py the depth is uniform at
the start and end of the run and the deformational error dominates, so the stored-concentration (depth-weighting) error
is not visible there; here the depth varies throughout and the flow is smooth. The default schemes reconstruct the stored
concentration directly (second order where the depth varies); "_da" schemes use the depth-aware ratio. Spatial refinement
at the benchmark Courant number. Writes variable_depth.json."""
import json, math, time
import numpy as np
import gen_mpdata as M
import gen_conservative as C
import cv_reference as CV

C.REPRESENTATION = "cv"
AMP = 0.4
T_END = 1.0


def depth(x, y):
    return 1.0 + AMP * np.cos(math.pi * x) * np.cos(math.pi * y)


def velocity(x, y):
    s = np.sin(math.pi * x); r = np.sin(math.pi * y); D = depth(x, y)
    return s ** 2 * np.sin(2 * math.pi * y) / D, -r ** 2 * np.sin(2 * math.pi * x) / D


class BathyGrid(M.Grid):
    def __init__(self, N):
        M.Grid.__init__(self, N)
        s = np.sin(math.pi * self.e); ds = s[1:] - s[:-1]
        self.V = np.outer(self.L, self.L) + AMP / math.pi ** 2 * np.outer(ds, ds)   # exact integrals of D

    def psi(self, xx, yy, t):
        return (1 / np.pi) * np.sin(np.pi * xx) ** 2 * np.sin(np.pi * yy) ** 2


def weighted_average(g, t, m=16, nsub=400):
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


SCHEMES = ("ccompact", "ccompact_da", "cfctg", "cfctg_da", "c4fctg", "u5fctg")
if __name__ == "__main__":
    import sys
    Ns = tuple(int(n) for n in sys.argv[1].split(",")) if len(sys.argv) > 1 else (33, 65, 129, 257)
    out = {"amplitude": AMP, "t_end": T_END, "cv_subcells": 16, "trajectory_steps": 400, "code": CV.CODE, "space": {}}
    for N in Ns:
        t0 = time.time(); g = BathyGrid(N)
        c0 = weighted_average(g, 0.0); ref = weighted_average(g, T_END)
        Qx, Qy = g.fluxes(0.0)
        outflow = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0)
                   + np.maximum(-Qy[:-1, :], 0))
        rec = out["space"].setdefault(str(N), {"outflow_fraction": float((M.CFL * g.h * outflow / g.V).max()),
                                              "ref_max": float(ref.max()), "init_max": float(c0.max())})
        assert rec["outflow_fraction"] <= 1.0
        for s in SCHEMES:
            c, _, gg, _ = C.run(N, s, lambda X, Y: c0, T_END, grid=g)
            r = score(c, ref, g)
            r["mass_rel"] = float((g.V * c).sum() / (g.V * c0).sum() - 1)
            r["depth_fallback_faces"] = C.FALLBACK[0]
            rec[s] = r
        print(f"N={N}: " + " ".join(f"{s} {rec[s]['rel_l2']:.3e}" for s in SCHEMES) +
              f" | phi {rec['outflow_fraction']:.2f} | {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open("variable_depth.json", "w"), indent=1)
    for s in SCHEMES:
        Es = [out["space"][str(N)][s]["rel_l2"] for N in Ns]
        print(f"{s:12s} orders " + " ".join(f"{math.log2(a / b):.2f}" for a, b in zip(Es, Es[1:])))
    c, _, _, _ = C.run(65, "cfctg", lambda X, Y: np.full(X.shape, 0.37), T_END, grid=BathyGrid(65))
    out["uniform_tracer_max_dev"] = float(np.abs(c - 0.37).max())
    json.dump(out, open("variable_depth.json", "w"), indent=1)
    print("uniform tracer max |c - 0.37|:", out["uniform_tracer_max_dev"])
