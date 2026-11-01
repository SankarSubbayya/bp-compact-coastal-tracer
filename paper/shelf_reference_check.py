"""Sensitivity of the 129^2 steep-shelf reference to its sub-cell quadrature (JCP pre-submission review, round 7,
2026-11-01).

steep_depth.py integrates the depth-weighted control-volume averages of its reference with the m x m midpoint rule
(m = 16). Here the same RK4 back-trajectories (1000 steps) start from n x n Gauss-Legendre points per volume (n = 12);
the bounded compact scheme with and without the dissipation term (15) and bounded U5 are rerun on the 129^2 grid with
the initial data and step of steep_depth.py and scored against both references. The midpoint path is checked against
steep_depth.weighted_average and the midpoint scores against steep_depth_dissip.json. Writes
shelf_reference_check.json."""
import json, sys
import numpy as np
sys.argv = sys.argv[:1]                       # steep_depth.py parses its options on import: use its defaults
import steep_depth as SD
import gen_mpdata as M
import gen_conservative as C
import cv_reference as CV

N, NG, NSUB = 129, 12, 1000
SCHEMES = ("cfctg", "u5fctg", "cfctgd")


def weighted_average(g, t, nsub, frac, wq):
    """Depth-weighted averages from the points frac (fractions of each volume's width, weights wq summing to 1),
    each traced back to t = 0 along the RK4 trajectories of steep_depth.weighted_average."""
    m = len(frac)
    xs = (g.e[:-1, None] + frac[None, :] * g.L[:, None]).ravel()
    X, Y = np.meshgrid(xs, xs); px, py = X.copy(), Y.copy(); d = t / nsub
    for _ in range(nsub if t > 0 else 0):
        u1, v1 = SD.velocity(px, py)
        u2, v2 = SD.velocity(px - .5 * d * u1, py - .5 * d * v1)
        u3, v3 = SD.velocity(px - .5 * d * u2, py - .5 * d * v2)
        u4, v4 = SD.velocity(px - d * u3, py - d * v3)
        px = px - d / 6 * (u1 + 2 * u2 + 2 * u3 + u4); py = py - d / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
    w = np.tile(wq, g.N)
    W = np.outer(w, w) * SD.depth(X, Y); f = W * M.gauss_ic(np.clip(px, 0, 1), np.clip(py, 0, 1))
    return f.reshape(g.N, m, g.N, m).sum(axis=(1, 3)) / W.reshape(g.N, m, g.N, m).sum(axis=(1, 3))


if __name__ == "__main__":
    t_end = SD.A.t_end
    mid = ((np.arange(16) + 0.5) / 16, np.full(16, 1 / 16))
    xg, wg = np.polynomial.legendre.leggauss(NG); gau = ((xg + 1) / 2, wg / 2)
    small = SD.ShelfGrid(9)                   # the midpoint path reproduces steep_depth.weighted_average
    assert np.abs(weighted_average(small, t_end, 40, *mid) - SD.weighted_average(small, t_end, 40)).max() < 1e-14
    g = SD.ShelfGrid(N)
    c0 = SD.weighted_average(g, 0.0, NSUB)
    ref_mid = SD.weighted_average(g, t_end, NSUB)
    ref_gau = weighted_average(g, t_end, NSUB, *gau)
    Qx, Qy = g.fluxes(0.0)                    # the step of steep_depth.py
    outflow = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0)
               + np.maximum(-Qy[:-1, :], 0))
    dt = M.CFL * g.h * min(1.0, SD.PHI / float((M.CFL * g.h * outflow / g.V).max()))
    out = {"N": N, "t_end": t_end, "midpoint_points": 16, "gauss_points": NG, "trajectory_steps": NSUB,
           "code": CV.CODE, "dt": dt,
           "ref_change": float(np.sqrt((g.V * (ref_gau - ref_mid) ** 2).sum() / (g.V * ref_mid ** 2).sum())),
           "schemes": {}}
    for s in SCHEMES:
        c, *_ = C.run(N, s, lambda X, Y: c0, t_end, grid=g, dt=dt)
        out["schemes"][s] = {"midpoint": SD.score(c, ref_mid, g)["rel_l2"], "gauss": SD.score(c, ref_gau, g)["rel_l2"]}
        print(f"{s}: midpoint {out['schemes'][s]['midpoint']:.10f}  gauss {out['schemes'][s]['gauss']:.10f}", flush=True)
    print(f"reference change {out['ref_change']:.3e}")
    json.dump(out, open("shelf_reference_check.json", "w"), indent=1)
