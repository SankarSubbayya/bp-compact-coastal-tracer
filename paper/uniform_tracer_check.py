"""Uniform-tracer check in the bay (2026 revision, round 2): c = 1 in every interior cell, transported over one 120 s
record of fluxes_mixed.npz (five substeps) by donor cell, the bounded compact scheme (global bounds) and the unlimited
compact flux. Inflow through the open boundary carries c = 0, so cells within eight cells (in either grid direction)
of an open cell are excluded; elsewhere the largest |c - 1| measures how well the face fluxes and volumes satisfy
continuity. Merged into checks.json as uniform_tracer_bay."""
import json, os
import numpy as np
import sfbay_conservative as S

S.set_closure("avg"); S.DISSIPATION = False
far = np.zeros(S.wet.shape, bool); far[S.wj[S.isopen], S.wi[S.isopen]] = True
for _ in range(8):                                     # square (Chebyshev) dilation by one cell per pass
    p = np.pad(far, 1)
    far = far | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
keep = S.interior & ~far[S.wj, S.wi]
c0 = np.where(S.interior, 1.0, 0.0)
res = {}
for sch in ("donor", "ccfctg", "ccompact"):
    c, h, M0, wall = S.run(sch, c0, S.DT / 3600, 5)
    res[sch] = float(np.abs(c[keep] - 1).max())
    print(f"{sch:9s} max |c - 1| away from the open boundary: {res[sch]:.3e} ({int(keep.sum())} cells)", flush=True)
out = json.load(open("checks.json")) if os.path.exists("checks.json") else {}
out["uniform_tracer_bay"] = res
json.dump(out, open("checks.json", "w"), indent=1)
