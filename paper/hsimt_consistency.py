"""Local consistency of the three HSIMT time integrations (2026 revision, after the round-2 audit). Uniform velocity
(u, v) = (0.6, 0.4), dt = 0.25 h, exact cell averages of 2 + x^2, 2 + y^2 and 2 + xy; the coefficient
(computed - exact) / dt^2 at the central cell is zero for a scheme that is second-order consistent in time.
Writes hsimt_consistency.json."""
import json
import numpy as np
import gen_mpdata as M


class Uniform:
    def __init__(self, n):
        self.N = n; self.h = 1 / (n - 1); self.L = np.full(n, self.h); self.V = np.full((n, n), self.h ** 2)

    def fluxes(self, t):
        return np.full((self.N, self.N + 1), .6 * self.h), np.full((self.N + 1, self.N), .4 * self.h)


def cell_avg(fn, x0, x1, y0, y1, n=6):
    xg, wg = np.polynomial.legendre.leggauss(n)
    X = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * xg; Y = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * xg
    return 0.25 * np.einsum("i,j,ij->", wg, wg, fn(X[None, :], Y[:, None]))


TESTS = {"2+x^2": lambda x, y: 2 + x ** 2, "2+y^2": lambda x, y: 2 + y ** 2, "2+xy": lambda x, y: 2 + x * y}
VARIANTS = {"two-level (no predictor)": lambda g, c, dt: M.hsimt_step(g, c, 0, dt, predictor=False),
            "ROMS predictor": lambda g, c, dt: M.hsimt_step(g, c, 0, dt, predictor=True),
            "Strang split": lambda g, c, dt: M.hsimt_split_step(g, c, 0, dt)}
out = {}
for var, step in VARIANTS.items():
    for name, fn in TESTS.items():
        coef = []
        for n in (33, 65, 129):
            g = Uniform(n); h = g.h; dt = 0.25 * h; xc = np.linspace(0, 1, n)
            c = np.array([[cell_avg(fn, x - h / 2, x + h / 2, y - h / 2, y + h / 2) for x in xc] for y in xc])
            a = step(g, c, dt); i = n // 2
            ex = cell_avg(lambda x, y: fn(x - .6 * dt, y - .4 * dt), xc[i] - h / 2, xc[i] + h / 2, xc[i] - h / 2, xc[i] + h / 2)
            coef.append(float((a[i, i] - ex) / dt ** 2))
        out.setdefault(var, {})[name] = coef
        print(f"{var:26s} {name:6s} " + " ".join(f"{v:+.3f}" for v in coef))
json.dump(out, open("hsimt_consistency.json", "w"), indent=1)
