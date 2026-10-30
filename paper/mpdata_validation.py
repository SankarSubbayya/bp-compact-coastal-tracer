"""Validation of gen_mpdata.mpdata_step against PyMPDATA (Bartman et al. 2022), 2026 revision.

A steady vortex confined away from the walls, psi = A (1 - r^2/R^2)^4 for r < R = 0.35 about (0.5, 0.5), carries a
positive Gaussian (width 0.06, centred at (0.5, 0.65)). Our routine runs on its node-centred control volumes; PyMPDATA
runs on the full interior cells with exactly the same face Courant numbers (periodic boundaries, never reached). Both
use one or two corrective passes, with and without the nonoscillatory option. Writes mpdata_validation.json."""
import json
import numpy as np
import gen_mpdata as M
from PyMPDATA import Options, ScalarField, VectorField, Stepper, Solver
from PyMPDATA.boundary_conditions import Periodic

R0, AMP = 0.35, 0.05


class VortexGrid(M.Grid):
    def psi(self, xx, yy, t):
        r2 = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / R0 ** 2
        return AMP * np.where(r2 < 1, (1 - r2) ** 4, 0.0)


def run_ours(N, c0, dt, nt, iord, nonosc):
    g = VortexGrid(N); c = c0.copy()
    for n in range(nt):
        c = M.mpdata_step(g, c, n * dt, dt, iord=iord, nonosc=nonosc)
    return c


def run_pympdata(N, c0, dt, nt, iord, nonosc):
    g = VortexGrid(N); Qx, Qy = g.fluxes(0.0); h = g.h
    Cx = (Qx[1:N - 1, 1:N] * dt / h ** 2).T          # x-faces of the interior cells, shape (nx+1, ny)
    Cy = (Qy[1:N, 1:N - 1] * dt / h ** 2).T          # y-faces, shape (nx, ny+1)
    psi = c0[1:N - 1, 1:N - 1].T.copy()              # interior cells, PyMPDATA index order [x, y]
    opts = Options(n_iters=iord, nonoscillatory=nonosc)
    bc = (Periodic(), Periodic())
    solver = Solver(stepper=Stepper(options=opts, grid=psi.shape, n_threads=1),
                    advectee=ScalarField(psi, halo=opts.n_halo, boundary_conditions=bc),
                    advector=VectorField((Cx, Cy), halo=opts.n_halo, boundary_conditions=bc))
    solver.advance(n_steps=nt)
    out = c0.copy(); out[1:N - 1, 1:N - 1] = solver.advectee.get().T
    return out, float(max(np.abs(Cx).max(), np.abs(Cy).max()))


if __name__ == "__main__":
    res = {}
    for N in (65, 129):
        g = VortexGrid(N)
        c0 = 1e-3 + np.exp(-((g.X - 0.5) ** 2 + (g.Y - 0.65) ** 2) / (2 * 0.06 ** 2))   # positive everywhere
        Qx, Qy = g.fluxes(0.0); vmax = max(np.abs(Qx).max(), np.abs(Qy).max()) / g.h
        dt = 0.4 * g.h / vmax; nt = 200
        for iord in (2, 3):
            for nonosc in (False, True):
                a = run_ours(N, c0, dt, nt, iord, nonosc)
                b, cmax = run_pympdata(N, c0, dt, nt, iord, nonosc)
                diff = float(np.abs(a - b).max()); rel = diff / float(np.abs(b).max())
                key = f"N{N}_passes{iord - 1}_{'nonosc' if nonosc else 'basic'}"
                res[key] = {"max_abs_diff": diff, "rel_diff": rel, "steps": nt, "max_courant": cmax,
                            "ours_min": float(a.min()), "pympdata_min": float(b.min())}
                print(f"{key:28s} steps {nt} max Courant {cmax:.2f}: max |ours - PyMPDATA| = {diff:.2e} (relative {rel:.1e})", flush=True)
    json.dump(res, open("mpdata_validation.json", "w"), indent=1)
    print("wrote mpdata_validation.json")
