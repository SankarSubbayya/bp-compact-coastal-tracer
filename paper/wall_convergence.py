"""Boundary-inclusive convergence test (2026 revision, after the round-2 audit): steady cellular flow
psi = sin(pi x) sin(pi y) / pi, whose tangential velocity is largest along the walls, carrying a smooth Gaussian
(width 0.08, centred at (0.5, 0.15)) along the south wall to t = 0.4. Control-volume-average data and reference
(m = 8 sub-cells, RK4 back-trajectories, 400 steps). Compares the half-width-volume wall closure with the earlier
equal-width closure, and the bounded scheme with global bounds taken from the data (the extrema of the initial cell
averages, as in the benchmark) and with the physical bounds [0, 1]: the cell averages of the exact solution need not
stay within the extrema of the initial cell averages. Writes wall_convergence.json."""
import json
import numpy as np
import gen_mpdata as M
import gen_conservative as C

T_END = 0.4
_step_data_bounds = C._step


def _step_unit_bounds(g, cn, Vn, t, dt, bounds, gmin, gmax, nodal=False):
    """gen_conservative._step with the physical bounds [0, 1] in place of the extrema of the initial data."""
    return _step_data_bounds(g, cn, Vn, t, dt, bounds, 0.0, 1.0, nodal)



class CellGrid(M.Grid):
    def psi(self, xx, yy, t):
        return np.sin(np.pi * xx) * np.sin(np.pi * yy) / np.pi


def vel(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y), -np.cos(np.pi * x) * np.sin(np.pi * y)


def ic(x, y):
    return np.exp(-((x - 0.5) ** 2 + (y - 0.15) ** 2) / (2 * 0.08 ** 2))


def cv_avg(g, t, m=8, nsub=400):
    frac = (np.arange(m) + 0.5) / m
    xs = (g.e[:-1, None] + frac[None, :] * g.L[:, None]).ravel()
    X, Y = np.meshgrid(xs, xs); d = t / nsub
    for _ in range(nsub if t > 0 else 0):
        u1, v1 = vel(X, Y); u2, v2 = vel(X - .5*d*u1, Y - .5*d*v1)
        u3, v3 = vel(X - .5*d*u2, Y - .5*d*v2); u4, v4 = vel(X - d*u3, Y - d*v3)
        X = X - d/6*(u1 + 2*u2 + 2*u3 + u4); Y = Y - d/6*(v1 + 2*v2 + 2*v3 + v4)
    return ic(np.clip(X, 0, 1), np.clip(Y, 0, 1)).reshape(g.N, m, g.N, m).mean(axis=(1, 3))


if __name__ == "__main__":
    M.Grid = CellGrid
    out = {}
    for N in (33, 65, 129, 257):
        g = CellGrid(N); c0 = cv_avg(g, 0.0); ref = cv_avg(g, T_END)
        out.setdefault("extrema", {})[str(N)] = {"max_initial": float(c0.max()), "max_reference": float(ref.max())}
        for label, sch, rep, unit in (("unlimited compact, half-width wall closure", "ccompact", "cv", False),
                                      ("unlimited compact, equal-width wall closure", "ccompact", "nodal", False),
                                      ("bounded compact (global), half-width wall closure", "cfctg", "cv", False),
                                      ("bounded compact (global [0, 1]), half-width wall closure", "cfctg", "cv", True)):
            C.REPRESENTATION = rep
            C._step = _step_unit_bounds if unit else _step_data_bounds
            c, c00, gg, wall = C.run(N, sch, lambda X, Y: c0, T_END)
            C._step = _step_data_bounds
            err = float(np.sqrt((gg.V * (c - ref) ** 2).sum() / (gg.V * ref ** 2).sum()))
            near = np.zeros_like(c, bool); near[:3, :] = True          # three volumes next to the south wall
            errw = float(np.abs(c - ref)[near].max())
            out.setdefault(label, {})[str(N)] = {"rel_l2": err, "max_err_wall_rows": errw}
            print(f"N={N:3d} {label:50s} L2 {err:.3e}  max error in wall rows {errw:.3e}", flush=True)
    for label, v in out.items():
        if label == "extrema":
            continue
        Ns = sorted(v, key=int)
        print(label, "orders:", " ".join(f"{np.log2(v[a]['rel_l2'] / v[b]['rel_l2']):.2f}" for a, b in zip(Ns, Ns[1:])),
              "| wall-row orders:", " ".join(f"{np.log2(v[a]['max_err_wall_rows'] / v[b]['max_err_wall_rows']):.2f}" for a, b in zip(Ns, Ns[1:])))
    json.dump(out, open("wall_convergence.json", "w"), indent=1)
