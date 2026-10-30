"""Validation of chaplin_colella.py on the tests of Chaplin and Colella (2017, CAMCoS 12(1), Sec. 5; equation and figure
numbers of the published article): periodic unit square, Courant number 0.8, one period (t = 1), so that the exact solution is the initial
condition. The article writes sigma = |u| dt / h without saying whether |u| is the speed or the largest velocity
component; both are run and recorded with the actual dt / h and step count: "published" with the speed (dt = 0.8 h /
sqrt 2 for u = (1, 1)), "published_component" with the largest component (dt = 0.8 h for u = (1, 1)). Its Table 1 puts
the ninth-order stability limit near sigma = 1.60 / D, and the component reading gives high-order errors closer to its
Fig. 7; at that step the limited ninth-order square undershoots by about 2%. Velocities (eqs. 80-81): constant u = (1, 1), centre
(0.5, 0.5); solid-body rotation u = 2 pi (y - 0.5, 0.5 - x), centre (0.5, 0.75); and u = (1, 0), which reproduces the
one-dimensional tests. Initial conditions (eqs. 82-84) as exact cell averages (Gauss quadrature): the smooth bump
cos^8(pi R / (2 R0)), R0 = 0.15; the square of half-width 0.15; the semi-ellipse of radius 0.25. The article
publishes its results as plots: max-norm errors of the smooth bump (Fig. 7; Fig. 5 in one dimension), with the limited
errors close to the unlimited ones, and limited discontinuous solutions (Figs. 6, 8, 9) described as bounded for the
ninth-order upwind values. Its rotating semi-ellipse uses the domain [0, 2]^2 and centre (1.0, 1.5) with a rotation
centre it does not state, so that case is run here on the unit square (centre (0.5, 0.75)) and is not a reproduction.
Records the published configuration (PREPRINT = False) under both readings of sigma and, for comparison, the
preprint-bounds variant (see chaplin_colella.PREPRINT) and the flag read as applying to every one-dimensional extremum. Writes cc_validation.json."""
import json, math, time
import numpy as np
import chaplin_colella as CC

R0 = 0.15
gx, gw = np.polynomial.legendre.leggauss(6)


def cell_avg(f, N):
    h = 1 / N; e = np.arange(N) * h
    X = e[None, :, None, None] + 0.5 * h * (1 + gx[None, None, None, :])
    Y = e[:, None, None, None] + 0.5 * h * (1 + gx[None, None, :, None])
    return 0.25 * (f(X, Y) * gw[None, None, :, None] * gw[None, None, None, :]).sum((-1, -2))   # [j, i]


def ic(name, xc, oned=False):
    def f(x, y):
        R = np.abs(x - xc[0]) + 0 * y if oned else np.sqrt((x - xc[0]) ** 2 + (y - xc[1]) ** 2)
        if name == "smooth":
            return np.where(R <= R0, np.cos(math.pi * R / (2 * R0)) ** 8, 0.0)
        if name == "square":
            inside = (np.abs(x - xc[0]) <= 0.15) & (0 * y == 0)
            return (inside if oned else inside & (np.abs(y - xc[1]) <= 0.15)).astype(float)
        return np.where(R <= 0.25, np.sqrt(np.clip(1 - (R / 0.25) ** 2, 0, None)), 0.0)     # semi-ellipse
    return f


def fluxes(vel, N):
    h = 1 / N; xc = (np.arange(N) + 0.5) * h
    if vel == "constant":
        return np.full((N, N + 1), h), np.full((N + 1, N), h), math.sqrt(2.0), 1.0
    if vel == "oned":
        return np.full((N, N + 1), h), np.zeros((N + 1, N)), 1.0, 1.0
    Qx = np.repeat((h * 2 * math.pi * (xc - 0.5))[:, None], N + 1, 1)          # u = 2 pi (y - 0.5) on x-faces, row j
    Qy = np.repeat((h * 2 * math.pi * (0.5 - xc))[None, :], N + 1, 0)          # v = 2 pi (0.5 - x) on y-faces, col i
    return Qx, Qy, 2 * math.pi * math.sqrt(0.5), math.pi           # speed largest at the corners, components at edges


def run(vel, name, N, recon, limited=True, flag="smooth", pre=True, sigma=0.8, conv="speed"):
    h = 1 / N; Qx, Qy, speed, comp = fluxes(vel, N); umax = speed if conv == "speed" else comp
    xc = (0.5, 0.75) if vel == "rotation" else (0.5, 0.5)
    c0 = cell_avg(ic(name, xc, vel == "oned"), N); c = c0.copy()
    nst = int(math.ceil(1.0 / (sigma * h / umax))); dt = 1.0 / nst; nflag = 0
    for _ in range(nst):
        c, nf = CC.periodic_step(c, Qx, Qy, h, dt, recon, limited, flag, pre); nflag += nf
    d = c - c0
    return {"l1": float(np.abs(d).sum() / np.abs(c0).sum()), "linf": float(np.abs(d).max()), "min": float(c.min()),
            "max": float(c.max()), "flagged": nflag, "mass": float(c.sum() / c0.sum() - 1), "dt_over_h": dt / h,
            "steps": nst}


if __name__ == "__main__":
    out = {"R0": R0, "sigma": 0.8, "conventions": {"published": "speed", "published_component": "component",
                                                   "preprint": "speed"}, "smooth": {}, "discontinuous": {}}
    t0 = time.time()
    for version, pp in (("published", False), ("published_component", False), ("preprint", True)):
        CC.PREPRINT = pp; conv = out["conventions"][version]
        variants = (("unlimited", dict(limited=False)), ("limited", {})) + \
                   ((("flag_any_extremum", dict(flag=True)),) if version == "published" else ())
        for vel in ("oned", "constant", "rotation"):
            for recon in ("c4", "u5", "u9"):
                for var, kw in variants:
                    rec = {str(N): run(vel, "smooth", N, recon, conv=conv, **kw) for N in (32, 64, 128, 256)}
                    Ns = sorted(rec, key=int)
                    rec["orders_linf"] = [math.log2(rec[a]["linf"] / rec[b]["linf"]) for a, b in zip(Ns, Ns[1:])]
                    out["smooth"][f"{version}|{vel}|{recon}|{var}"] = rec
                    print(f"smooth {version:9s} {vel:8s} {recon} {var:17s} " + " ".join(f"{rec[N]['linf']:.2e}" for N in Ns)
                          + "  orders " + " ".join(f"{o:.2f}" for o in rec["orders_linf"]) + f" | {time.time() - t0:.0f}s",
                          flush=True)
        for vel in ("oned", "constant", "rotation"):
            for name in ("square", "ellipse"):
                for recon in ("c4", "u5", "u9"):
                    r = run(vel, name, 128, recon, conv=conv)
                    out["discontinuous"][f"{version}|{vel}|{name}|{recon}"] = r
                    print(f"disc {version:9s} {vel:8s} {name:8s} {recon} L1 {r['l1']:.3e} range [{r['min']:+.2e}, "
                          f"{r['max']:.6f}] mass {r['mass']:+.1e}", flush=True)
    CC.PREPRINT = False
    json.dump(out, open("cc_validation.json", "w"), indent=1)
    print("wrote cc_validation.json")
