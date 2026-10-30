"""Sensitivity of the control-volume-average reference to its quadrature (2026 revision, round 2).

For each test and grid, the reference at t = T/2 with m = 16 sub-cells per direction and 400 RK4 trajectory steps (the
setting of cv_results.json) is compared with m = 32 (and, on the 65^2 grid, with m = 8 and with 800 steps). The
difference delta is the relative L2 norm weighted by control volume, as in gen_mpdata.score. By the triangle
inequality, replacing one reference by the other changes a fixed solution's relative error e by at most
delta (1 + e) / (1 - delta), and two fixed solutions with errors e_a < e_b keep their order if e_b - e_a > 2 delta. delta
measures the sensitivity of the reference to its quadrature, not its distance from the exact averages.
Usage: python3 reference_sensitivity.py [65,129,257]; merges into reference_sensitivity.json."""
import json, time
import numpy as np
import gen_mpdata as M


def cv_average_tiled(g, fn, tc, m, nsub, rows=8):
    """cv_reference.cv_average computed a few control-volume rows at a time (same midpoints, same RK4 backtrace)."""
    frac = (np.arange(m) + 0.5) / m
    xs = (g.e[:-1, None] + frac[None, :] * g.L[:, None]).ravel()
    out = np.empty((g.N, g.N)); d = tc / nsub
    for j0 in range(0, g.N, rows):
        j1 = min(j0 + rows, g.N)
        X, Y = np.meshgrid(xs, xs[j0 * m:j1 * m])
        px, py = X.copy(), Y.copy()
        for n in range(nsub if tc > 0 else 0):
            s = tc - n * d
            u1, v1 = M.A.velocity(px, py, s, M.T)
            u2, v2 = M.A.velocity(px - .5*d*u1, py - .5*d*v1, s - .5*d, M.T)
            u3, v3 = M.A.velocity(px - .5*d*u2, py - .5*d*v2, s - .5*d, M.T)
            u4, v4 = M.A.velocity(px - d*u3, py - d*v3, s - d, M.T)
            px = px - d/6*(u1 + 2*u2 + 2*u3 + u4); py = py - d/6*(v1 + 2*v2 + 2*v3 + v4)
        f = fn(np.clip(px, 0, 1), np.clip(py, 0, 1))
        out[j0:j1] = f.reshape(j1 - j0, m, g.N, m).mean(axis=(1, 3))
    return out


def rel(a, b, g):
    return float(np.sqrt((g.V * (a - b) ** 2).sum() / (g.V * b ** 2).sum()))


if __name__ == "__main__":
    import os, sys
    grids = tuple(int(n) for n in sys.argv[1].split(",")) if len(sys.argv) > 1 else (65, 129, 257)
    out = json.load(open("reference_sensitivity.json")) if os.path.exists("reference_sensitivity.json") else {}
    for test, ic in (("gauss_T2", M.gauss_ic), ("slotted_T2", M.slot_ic)):
        for N in grids:
            t0 = time.perf_counter(); g = M.Grid(N)
            r16 = cv_average_tiled(g, ic, M.T / 2, 16, 400)
            r = {"subcells_16_vs_32": rel(cv_average_tiled(g, ic, M.T / 2, 32, 400), r16, g)}
            if N == 65:
                r["subcells_8_vs_16"] = rel(cv_average_tiled(g, ic, M.T / 2, 8, 400), r16, g)
                r["steps_400_vs_800_m16"] = rel(cv_average_tiled(g, ic, M.T / 2, 16, 800), r16, g)
            out.setdefault(test, {})[str(N)] = r
            print(test, N, {k: f"{v:.3e}" for k, v in r.items()}, f"{time.perf_counter() - t0:.0f}s", flush=True)
            json.dump(out, open("reference_sensitivity.json", "w"), indent=1)
