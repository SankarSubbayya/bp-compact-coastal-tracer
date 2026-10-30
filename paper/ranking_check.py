"""Ranking check against a finer reference quadrature (2026 revision, round 2). On the 65^2 and 129^2 grids, every
scheme of cv_reference.SCHEMES is rerun from initial data averaged over m = 32 sub-cells and scored against the m = 32
reference, so that the rankings stated in the text can be checked where the triangle-inequality bound of
reference_sensitivity.py is not decisive. Writes ranking_check.json: {test: {N: {scheme: score}}}."""
import json, time
import numpy as np
import gen_mpdata as M
import cv_reference as CV
from reference_sensitivity import cv_average_tiled

M_SUB = 32
out = {"cv_subcells": M_SUB, "trajectory_steps": CV.NSUB_TRAJ, "code": CV.CODE}
for test, ic in (("slotted_T2", M.slot_ic), ("gauss_T2", M.gauss_ic)):
    for N in (65, 129):
        t0 = time.time(); g = M.Grid(N)
        c0 = cv_average_tiled(g, ic, 0.0, M_SUB, CV.NSUB_TRAJ); ref = cv_average_tiled(g, ic, M.T / 2, M_SUB, CV.NSUB_TRAJ)
        res = out.setdefault(test, {}).setdefault(str(N), {})
        for s in CV.SCHEMES:
            c, c00, gg, wall = CV.run_from(N, s, c0, M.T / 2)
            m = M.score(c, ref, c00, gg, wall)
            a, b = c > 0.5, ref > 0.5
            m["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
            res[s] = m
        best = min(res, key=lambda k: res[k]["rel_l2"])
        print(f"{test} N={N}: lowest L2 {best} {res[best]['rel_l2']:.5f}; cfctg {res['cfctg']['rel_l2']:.5f} "
              f"IoU {res['cfctg']['iou']:.3f} | {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open("ranking_check.json", "w"), indent=1)
