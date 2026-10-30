"""What the fourth-order transverse flux correction buys on the benchmark (JCP revision 2026-10-28). The smooth tracer
of the deformational benchmark is run with the compiled bounded compact scheme (global bounds) with and without the
correction (fast_schemes "cfctg" and "cfctg_nt"), on 65^2 to 513^2, from the same m = 16 control-volume initial data and
against the same m = 16 reference (tiled quadrature, identical to cv_reference.cv_average). Both errors must equal the
NumPy results in cv_results.json (65^2-257^2) and order_513.json (513^2). References are cached in figure_data/.
Writes covariance_ablation.json with keys "cfctg" (corrected) and "cfctg_nt" (uncorrected)."""
import json, math, os, time
import numpy as np
import gen_mpdata as M
import cv_reference as CV
import fast_schemes as F
from reference_sensitivity import cv_average_tiled

R = json.load(open("cv_results.json")); O = json.load(open("order_513.json"))
os.makedirs("figure_data", exist_ok=True)
out = {"cv_subcells": 16, "trajectory_steps": CV.NSUB_TRAJ, "gauss_T2": {}}
for N in (65, 129, 257, 513):
    t0 = time.time(); g = M.Grid(N); path = f"figure_data/gauss_m16_{N}.npz"
    if os.path.exists(path):
        d = np.load(path); c0, ref = d["c0"], d["ref"]
    else:
        c0 = cv_average_tiled(g, M.gauss_ic, 0.0, 16, CV.NSUB_TRAJ); ref = cv_average_tiled(g, M.gauss_ic, M.T / 2, 16, CV.NSUB_TRAJ)
        np.savez(path, c0=c0, ref=ref)
    assert N == 513 or CV.array_hash(ref) == R["gauss_T2"][str(N)]["cfctg"]["reference_sha"]
    rec = {}
    for s in ("cfctg", "cfctg_nt"):
        c = F.run(N, s, c0, M.T / 2)
        rec[s] = M.score(c, ref, c0, g, 0.0)
        archived = R["gauss_T2"][str(N)][s]["rel_l2"] if N < 513 else O["gauss_T2"][s]["rel_l2"]
        assert abs(rec[s]["rel_l2"] - archived) < 1e-11, (N, s, rec[s]["rel_l2"], archived)
    out["gauss_T2"][str(N)] = rec
    print(f"N={N}: L2 with correction {rec['cfctg']['rel_l2']:.4e}  without {rec['cfctg_nt']['rel_l2']:.4e}  "
          f"range with [{rec['cfctg']['min']:+.1e}, {rec['cfctg']['max']:.4f}] | {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("covariance_ablation.json", "w"), indent=1)
Ns = sorted(out["gauss_T2"], key=int)
for s in ("cfctg", "cfctg_nt"):
    e = [out["gauss_T2"][N][s]["rel_l2"] for N in Ns]
    print(s, "orders", " ".join(f"{math.log2(a / b):.2f}" for a, b in zip(e, e[1:])))
