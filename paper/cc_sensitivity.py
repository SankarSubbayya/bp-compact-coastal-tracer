"""Sensitivity of the Chaplin-Colella comparison to two points where the article admits more than one reading (JCP
revision, 2026-10-29): eq. 78 with its printed further factor h (chaplin_colella.EQ78_H) and eq. 30 with the
coefficients for derivatives of face averages (chaplin_colella.SIX_FACE_AVG; it changes only the upwind variant).
On the benchmark grids 65^2 to 257^2: the smooth tracer at the benchmark step and at the largest admissible CTU step
of cc_cost.json, and the slotted cylinder at the benchmark step, each against the archived results of the reading used
in the paper (cc_results.json, cc_cost.json) and of the bounded compact scheme (cv_results.json, cc_cost.json). Times to
an error of 0.1 at the largest admissible steps are re-interpolated with the archived compiled timings and each
reading's errors, assuming unchanged timings; the alternative readings were not retimed. Writes cc_sensitivity.json.
   uv run --with numpy --with scipy python3 cc_sensitivity.py"""
import json, math, time
import numpy as np
import gen_mpdata as M
import cv_reference as CV
import chaplin_colella as CC
from cc_benchmark import score

READINGS = {"eq78_h": dict(EQ78_H=True), "six_face_avg": dict(SIX_FACE_AVG=True),
            "both": dict(EQ78_H=True, SIX_FACE_AVG=True)}
RECONS = {"eq78_h": ("c4", "u5"), "six_face_avg": ("u5",), "both": ("u5",)}
NS = (65, 129, 257)
TARGET = 0.1


def t_at(pts, target):
    """Piecewise log-log interpolation of (seconds, error) points to the time at which the error reaches target."""
    pts = sorted(pts)
    for (t0, e0), (t1, e1) in zip(pts, pts[1:]):
        if (e0 - target) * (e1 - target) <= 0:
            w = math.log(target / e0) / math.log(e1 / e0)
            return math.exp(math.log(t0) + w * math.log(t1 / t0))
    raise ValueError("target not bracketed")


if __name__ == "__main__":
    RC = json.load(open("cc_results.json"))["results"]; R = json.load(open("cv_results.json"))
    K = json.load(open("cc_cost.json")); KL = K["timing"]["largest"]
    out = {"readings": {k: v for k, v in READINGS.items()}, "grids": list(NS), "results": {}, "time_to_target": {}}
    t0 = time.time()
    for name, flags in READINGS.items():
        for r in RECONS[name]:
            s = f"cc17_{r}"; rec = out["results"].setdefault(name, {}).setdefault(s, {})
            for N in NS:
                g = M.Grid(N); d = np.load(f"figure_data/gauss_m16_{N}.npz"); c0, ref = d["c0"], d["ref"]
                for k, v in flags.items():
                    setattr(CC, k, v)
                try:
                    c, _ = CC.run(N, r, c0, M.T / 2)
                    bench = score(c, ref, c0, g, "gauss_T2")
                    dt = KL[str(N)][s]["dt_over_h"] * g.h
                    c, _ = CC.run(N, r, c0, M.T / 2, dt=dt)
                    large = score(c, ref, c0, g, "gauss_T2")
                    s0, sref = CV.cv_average(g, M.slot_ic, 0.0), CV.cv_average(g, M.slot_ic, M.T / 2)
                    c, _ = CC.run(N, r, s0, M.T / 2)
                    slot = score(c, sref, s0, g, "slotted_T2")
                finally:
                    CC.EQ78_H, CC.SIX_FACE_AVG = False, False
                rec[str(N)] = {
                    "gauss_bench": bench["rel_l2"], "gauss_bench_paper": RC["gauss_T2"][str(N)][s]["rel_l2"],
                    "gauss_bench_compact": R["gauss_T2"][str(N)]["cfctg"]["rel_l2"],
                    "gauss_largest": large["rel_l2"], "gauss_largest_paper": KL[str(N)][s]["rel_l2"],
                    "gauss_largest_compact": KL[str(N)]["cfctg"]["rel_l2"],
                    "slotted": slot["rel_l2"], "slotted_paper": RC["slotted_T2"][str(N)][s]["rel_l2"],
                    "slotted_compact": R["slotted_T2"][str(N)]["cfctg"]["rel_l2"],
                    "slotted_min": slot["min"], "slotted_max": slot["max"]}
                print(f"{name:13s} {s} N={N}: bench {bench['rel_l2']:.5f} (paper {rec[str(N)]['gauss_bench_paper']:.5f}) "
                      f"largest {large['rel_l2']:.5f} (paper {rec[str(N)]['gauss_largest_paper']:.5f}) slotted "
                      f"{slot['rel_l2']:.5f} (paper {rec[str(N)]['slotted_paper']:.5f}) | {time.time() - t0:.0f}s", flush=True)
                json.dump(out, open("cc_sensitivity.json", "w"), indent=1)
            pts = [(KL[str(N)][s]["seconds"], rec[str(N)]["gauss_largest"]) for N in NS]
            tc = t_at([(KL[str(N)]["cfctg"]["seconds"], KL[str(N)]["cfctg"]["rel_l2"]) for N in NS], TARGET)
            out["time_to_target"].setdefault(name, {})[s] = t_at(pts, TARGET) / tc
            out["time_to_target"].setdefault("paper", {})[s] = t_at(
                [(KL[str(N)][s]["seconds"], KL[str(N)][s]["rel_l2"]) for N in NS], TARGET) / tc
    json.dump(out, open("cc_sensitivity.json", "w"), indent=1)
    print("wrote cc_sensitivity.json")
