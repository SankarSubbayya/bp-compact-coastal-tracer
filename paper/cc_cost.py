"""Cost of the Chaplin-Colella algorithm against the bounded compact scheme, at the benchmark step and at each scheme's
largest admissible step (JCP revision, 2026-10-28).

The bounds of both algorithms rest on a low-order update that is a convex combination of old values: donor cell for
the schemes of this paper (gen_conservative), corner transport upwind for Chaplin-Colella. The largest admissible step
of each is the largest dt at which every coefficient of its low-order update is non-negative, with the face fluxes of
the deformational flow at t = 0 (their largest magnitude; the flow is reversed by cos(pi t / T)). The coefficients are
read off by applying the update to sets of unit impulses five cells apart; the CTU limit is found by bisection, the
donor-cell limit is dt = 1 / max(outflow / V) (largest outflow fraction phi = 1).

1. Validation: fast_cc.run against chaplin_colella.run (NumPy) on the 65^2 and 129^2 grids: identical counts of
   flagged cells, errors equal to 1e-12, final fields equal to 1e-9. The U5 limiter amplifies roundoff: at 129^2 the
   NumPy run itself moves by 4.2e-10 (C4: 1.6e-14) when its initial data are perturbed by 1e-16, and the ports
   differ by 1.1e-10 at the same place.
2. Timing: smooth tracer to t = T/2 on 33^2 to 257^2 from the m = 16 initial data, compiled (Numba, one thread),
   minimum of five runs after one warm-up, every sample archived; at the benchmark step 0.35 h (whose errors must equal
   the archived cv_results.json / cc_results.json values) and at each scheme's largest admissible step (the step
   count rounded up so that the step does not exceed it).
Writes cc_cost.json."""
import hashlib, json, math, platform, subprocess, time
import numpy as np
import numba
import gen_mpdata as M
import cv_reference as CV
import chaplin_colella as CC
import fast_schemes as F
import fast_cc as FC

REPEATS = 5
OURS = ("cfctg", "c4fctg", "u5fctg")
THEIRS = {"cc17_c4": "c4", "cc17_u5": "u5"}
R = json.load(open("cv_results.json")); RC = json.load(open("cc_results.json"))


def data(N):
    path = f"figure_data/gauss_m16_{N}.npz"
    try:
        d = np.load(path); return d["c0"], d["ref"]
    except FileNotFoundError:
        g = M.Grid(N)
        return CV.cv_average(g, M.gauss_ic, 0.0, m=16), CV.cv_average(g, M.gauss_ic, M.T / 2, m=16)


def ctu_min_coefficient(g, Qx, Qy, dt):
    N = g.N; w = [np.empty_like(Qx), np.empty_like(Qy), np.empty((N, N)), np.empty((N, N))]
    Fx = np.empty_like(Qx); Fy = np.empty_like(Qy); lo = np.inf
    jj, ii = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    for a in range(5):
        for b in range(5):
            c = ((jj % 5 == a) & (ii % 5 == b)).astype(float)
            FC.ctu(c, g.V, Qx, Qy, dt, Fx, Fy, *w)
            lo = min(lo, float((c - dt * ((Fx[:, 1:] - Fx[:, :-1]) + (Fy[1:, :] - Fy[:-1, :])) / g.V).min()))
    return lo


def admissible(N):
    g = M.Grid(N); Qx, Qy = g.fluxes(0.0)
    out = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0) + np.maximum(-Qy[:-1, :], 0))
    dt_donor = float(1.0 / (out / g.V).max())
    lo, hi = 0.1 * g.h, 3.0 * g.h
    assert ctu_min_coefficient(g, Qx, Qy, lo) >= -1e-14 and ctu_min_coefficient(g, Qx, Qy, hi) < 0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ctu_min_coefficient(g, Qx, Qy, mid) >= -1e-14 else (lo, mid)
    return g, dt_donor, lo


def timed(f):
    f(); ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); c = f(); ts.append(time.perf_counter() - t0)
    return c, ts


if __name__ == "__main__":
    numba.set_num_threads(1)
    out = {"machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True).stdout.strip(),
           "python": platform.python_version(), "numba": numba.__version__, "numpy": np.__version__, "threads": 1,
           "fast_cc_sha256": hashlib.sha256(open("fast_cc.py", "rb").read()).hexdigest()[:16],
           "fast_schemes_sha256": hashlib.sha256(open("fast_schemes.py", "rb").read()).hexdigest()[:16],
           "chaplin_colella_sha256": hashlib.sha256(open("chaplin_colella.py", "rb").read()).hexdigest()[:16],
           "repeats": REPEATS, "validation": {}, "admissible": {}, "timing": {}}
    for N in (65, 129):
        c0, _ = data(N)
        for name, recon in THEIRS.items():
            a, fa = CC.run(N, recon, c0, M.T / 2); b, fb = FC.run(N, recon, c0, M.T / 2)
            g = M.Grid(N); _, ref = data(N)
            d = float(np.abs(a - b).max())
            de = abs(M.score(a, ref, c0, g, 0.0)["rel_l2"] - M.score(b, ref, c0, g, 0.0)["rel_l2"])
            assert fa == fb and de < 1e-12 and d < 1e-9, (N, name, fa, fb, de, d)
            out["validation"].setdefault(str(N), {})[name] = {"max_abs_diff": d, "rel_l2_diff": de, "flagged": int(fa)}
            print(f"validate N={N} {name}: max|diff| {d:.1e}, error diff {de:.1e}, flagged {fa}", flush=True)
    for N in (33, 65, 129, 257):
        g, dt_d, dt_c = admissible(N)
        out["admissible"][str(N)] = {"donor_dt_over_h": dt_d / g.h, "ctu_dt_over_h": dt_c / g.h}
        print(f"N={N}: largest admissible dt/h: donor {dt_d / g.h:.4f}, CTU {dt_c / g.h:.4f}", flush=True)
        c0, ref = data(N); T2 = M.T / 2
        for step, dts in (("benchmark", None), ("largest", (dt_d, dt_c))):
            for s in OURS + tuple(THEIRS):
                ours = s in OURS
                if dts is None:
                    nst = int(round(T2 / (M.CFL * g.h)))
                else:
                    nst = int(math.ceil(T2 / (dts[0] if ours else dts[1]) - 1e-12))
                dt = T2 / nst
                f = (lambda: F.run(N, s, c0, T2, cfl=dt / g.h)) if ours else (lambda: FC.run(N, THEIRS[s], c0, T2, dt=dt)[0])
                c, ts = timed(f)
                m = M.score(c, ref, c0, g, 0.0)
                if step == "benchmark":
                    arch = R["gauss_T2"][str(N)][s]["rel_l2"] if ours else RC["results"]["gauss_T2"][str(N)][s]["rel_l2"]
                    assert abs(m["rel_l2"] - arch) < 1e-10, (N, s, m["rel_l2"], arch)
                out["timing"].setdefault(step, {}).setdefault(str(N), {})[s] = {
                    "dt_over_h": dt / g.h, "steps": nst, "seconds": min(ts), "samples": ts, "rel_l2": m["rel_l2"],
                    "min": float(c.min()), "max": float(c.max())}
                print(f"{step:9s} N={N} {s:8s} dt/h {dt / g.h:.4f} min {min(ts):8.3f} s  L2 {m['rel_l2']:.5f} "
                      f"range [{c.min():+.1e}, {c.max():.4f}]", flush=True)
            json.dump(out, open("cc_cost.json", "w"), indent=1)
    print("wrote cc_cost.json")
