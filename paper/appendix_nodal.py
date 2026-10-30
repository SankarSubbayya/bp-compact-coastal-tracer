"""Appendix A driver (2026 revision, round 3): the point-sampled protocol of the earlier version of this work. Every
scheme starts from point samples of the Gaussian initial condition and is scored against point samples of the exact
solution at t = T/2 (gen_mpdata.exact), on 33^2 to 257^2, with the earlier equal-width (point-sample) wall closure.
Runs exactly the schemes of the Appendix A table, without the transverse flux correction (as in the earlier
version). Writes appendix_nodal.json: {"gauss_T2": {N: {scheme: score}}}."""
import json
import gen_mpdata as M
import gen_conservative as C
import cv_reference as CV          # for the code hash only; it switches the closure to "cv" on import
C.REPRESENTATION = "nodal"         # point samples: (5, 15, -5, 1)/16 wall closure
C.ALLOW_TRANSVERSE = False         # the earlier version's scheme: no transverse flux correction

C_SCHEMES = ("cfctgd", "cfctg")
M_SCHEMES = ("tvd", "mpdata3_no", "hsimt", "hsimt_nopred", "up3", "up1")
out = {"protocol": "point samples", "wall_closure": C.REPRESENTATION, "cfl": M.CFL, "code": CV.CODE, "gauss_T2": {}}
for N in (33, 65, 129, 257):
    ref = M.exact(M.gauss_ic, N, M.T / 2)
    res = out["gauss_T2"].setdefault(str(N), {})
    for s in C_SCHEMES + M_SCHEMES:
        c, c0, g, wall = (C.run if s in C_SCHEMES else M.run)(N, s, M.gauss_ic, M.T / 2)
        res[s] = M.score(c, ref, c0, g, wall)
        print(f"gauss N={N} {s:12s} L2={res[s]['rel_l2']:.4e} range [{res[s]['min']:+.1e}, {res[s]['max']:.3f}]", flush=True)
    json.dump(out, open("appendix_nodal.json", "w"), indent=1)
print("wrote appendix_nodal.json")
