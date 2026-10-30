"""Cost of the schemes: compiled (Numba, single thread) wall-clock time against accuracy (JCP revision 2026-10-28).

1. Validation: on the 65^2 and 129^2 grids, each compiled scheme (fast_schemes.py) is run from the same m = 16
   control-volume initial data as the NumPy implementation (cv_reference.run_from) and the final fields are compared.
2. Timing: smooth tracer to t = T/2 on 33^2 to 257^2 from the m = 16 initial data (cached in figure_data/ by
   covariance_ablation.py; computed here for 33^2), after one compilation run; five timed runs, all retained. The error
   reported for each grid is computed from the field the timed run returns, against the m = 16 reference, and must
   equal the archived cv_results.json value.
Writes timing_benchmark.json (machine, versions, the fast_schemes.py hash, validation differences, all timing samples
and errors)."""
import hashlib, json, os, platform, subprocess, time
import numpy as np
import numba
import gen_mpdata as M
import cv_reference as CV
import fast_schemes as F

SCHEMES = ("cfctg", "c4fctg", "u5fctg", "mpdata3_no", "hsimt_split", "muscl_fv")
REPEATS = 5
R = json.load(open("cv_results.json"))


def data(N):
    path = f"figure_data/gauss_m16_{N}.npz"
    if os.path.exists(path):
        d = np.load(path); return d["c0"], d["ref"]
    g = M.Grid(N)
    return CV.cv_average(g, M.gauss_ic, 0.0, m=16), CV.cv_average(g, M.gauss_ic, M.T / 2, m=16)


out = {"machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
       "python": platform.python_version(), "numba": numba.__version__, "numpy": np.__version__, "threads": 1,
       "fast_schemes_sha256": hashlib.sha256(open("fast_schemes.py", "rb").read()).hexdigest()[:16],
       "repeats": REPEATS, "validation": {}, "timing": {}}
numba.set_num_threads(1)
for N in (65, 129):
    c0, ref = data(N); g = M.Grid(N)
    for s in SCHEMES:
        c_np, c00, gg, _ = CV.run_from(N, s, c0, M.T / 2)
        c_nb = F.run(N, s, c0, M.T / 2)
        rec = {"max_abs_diff": float(np.abs(c_nb - c_np).max())}
        assert rec["max_abs_diff"] < 1e-11, (N, s, rec)
        out["validation"].setdefault(str(N), {})[s] = rec
        print(f"validate N={N} {s:12s} max|diff| {rec['max_abs_diff']:.1e}", flush=True)
for N in (33, 65, 129, 257):
    c0, ref = data(N); g = M.Grid(N)
    for s in SCHEMES:
        F.run(N, s, c0, M.T / 2)                                       # compiled already; warm caches
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter(); c = F.run(N, s, c0, M.T / 2); ts.append(time.perf_counter() - t0)
        e = M.score(c, ref, c0, g, 0.0)["rel_l2"]
        assert abs(e - R["gauss_T2"][str(N)][s]["rel_l2"]) < 1e-10, (N, s, e)
        out["timing"].setdefault(str(N), {})[s] = {"seconds": min(ts), "samples": ts, "rel_l2": e}
        print(f"time N={N} {s:12s} min {min(ts):8.3f} s (max {max(ts):.3f})  L2 {e:.5f}", flush=True)
    json.dump(out, open("timing_benchmark.json", "w"), indent=1)
print("wrote timing_benchmark.json")
