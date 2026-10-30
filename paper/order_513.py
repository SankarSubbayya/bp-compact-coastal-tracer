"""Observed order on a finer grid (2026 revision, after the JCP-oriented review): the smooth tracer of the
deformational benchmark on the 513^2 grid for the bounded compact scheme with global bounds (with and without the transverse correction) and the explicit
c4 and u5 reconstructions under the same flux-corrected transport, with initial data and
reference averaged over m = 16 sub-cells as in cv_results.json (tiled quadrature of reference_sensitivity.py, identical
to cv_reference.cv_average). Tests whether the observed order 3.3 between 129^2 and 257^2 is pre-asymptotic for a
formally second-order scheme. Writes order_513.json."""
import json, time
import gen_mpdata as M
import cv_reference as CV
from reference_sensitivity import cv_average_tiled

N = 513; t0 = time.time(); g = M.Grid(N)
import os, numpy as np
cache = "figure_data/gauss_m16_513.npz"
if os.path.exists(cache):
    d = np.load(cache); c0, ref = d["c0"], d["ref"]
else:
    c0 = cv_average_tiled(g, M.gauss_ic, 0.0, 16, CV.NSUB_TRAJ); ref = cv_average_tiled(g, M.gauss_ic, M.T / 2, 16, CV.NSUB_TRAJ)
    np.savez(cache, c0=c0, ref=ref)
print(f"references {time.time() - t0:.0f}s", flush=True)
out = {"N": N, "cv_subcells": 16, "trajectory_steps": CV.NSUB_TRAJ, "code": CV.CODE, "gauss_T2": {}}
for s in ("cfctg", "cfctg_nt", "c4fctg", "u5fctg"):
    c, c00, gg, wall = CV.run_from(N, s, c0, M.T / 2)
    out["gauss_T2"][s] = M.score(c, ref, c00, gg, wall)
    print(s, out["gauss_T2"][s]["rel_l2"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("order_513.json", "w"), indent=1)
