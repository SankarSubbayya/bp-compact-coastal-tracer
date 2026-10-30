"""Range of the Chaplin-Colella (2017) solutions on the deformational benchmark (U5): on the smooth tracer, the most
negative value of the CTU low-order update over the run against the final minimum (65^2, 129^2); on the slotted
cylinder, the final range with the bounds widened at smooth extrema (eqs. 72-73 of the published article) and without
that widening (65^2, 129^2). Writes cc_range.json."""
import json
import numpy as np
import gen_mpdata as M
import cv_reference as CV
import chaplin_colella as CC

rec = {}
orig = CC.bounds


def bounds(cn, ctd, *args, **kw):
    rec["ctu_min"] = min(rec["ctu_min"], float(ctd.min()))
    return orig(cn, ctd, *args, **kw)


out = {"smooth": {}, "slotted": {}}
CC.bounds = bounds
for N in (65, 129):
    d = np.load(f"figure_data/gauss_m16_{N}.npz")
    rec.clear(); rec["ctu_min"] = 0.0
    c, _ = CC.run(N, "u5", d["c0"], M.T / 2)
    out["smooth"][str(N)] = dict(rec, final_min=float(c.min()))
    print("smooth", N, out["smooth"][str(N)], flush=True)
CC.bounds = orig
for N in (65, 129):
    g = M.Grid(N); c0 = CV.cv_average(g, M.slot_ic, 0.0)
    r = {}
    for widen in (True, False):
        saved = CC.bounds
        CC.bounds = lambda *a, w=widen, **k: saved(*a[:7], w) if len(a) >= 7 else saved(*a, widen=w)
        c, _ = CC.run(N, "u5", c0, M.T / 2)
        CC.bounds = saved
        r["widened" if widen else "not_widened"] = {"min": float(c.min()), "max": float(c.max())}
    out["slotted"][str(N)] = r
    print("slotted", N, r, flush=True)
json.dump(out, open("cc_range.json", "w"), indent=1)
