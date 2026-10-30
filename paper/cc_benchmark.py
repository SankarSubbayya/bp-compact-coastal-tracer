"""The complete single-stage FCT algorithm of Chaplin and Colella (2017) on the deformational benchmark (JCP revision,
2026-10-28): chaplin_colella.run with the centred fourth-order ("cc17_c4") and upwind fifth-order ("cc17_u5", their
recommended choice) reconstructions, CTU low-order flux, pre-constraint, smooth-extremum bounds and flag, as validated
on the paper's own tests by cc_validation.py. Same control-volume data and reference (m = 16, cv_reference.cv_average),
grids, final time and time step as cv_reference.py; the data hashes are recorded so that make_tables.py can check
they equal those of cv_results.json. Also a time-step sweep on 129^2 (CTU admits steps beyond the donor-cell limit).
Writes cc_results.json."""
import hashlib, json, os, time
import numpy as np
import gen_mpdata as M
import cv_reference as CV
import chaplin_colella as CC

SCHEMES = {"cc17_c4": "c4", "cc17_u5": "u5"}


def score(c, ref, c0, g, test):
    m = M.score(c, ref, c0, g, 0.0); m.pop("wall_s", None)
    if test == "slotted_T2":
        a, b = c > 0.5, ref > 0.5
        m["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
    return m


if __name__ == "__main__":
    out = {"code": CV.CODE, "cc_sha256": hashlib.sha256(open("chaplin_colella.py", "rb").read()).hexdigest()[:16],
           "cv_subcells": CV.SUB, "cfl": M.CFL, "results": {}, "sweep": {}}
    t0 = time.time()
    for test, ic, Ns in (("gauss_T2", M.gauss_ic, (33, 65, 129, 257)), ("slotted_T2", M.slot_ic, (65, 129, 257))):
        for N in Ns:
            g = M.Grid(N); cached = f"figure_data/gauss_m16_{N}.npz"
            if test == "gauss_T2" and os.path.exists(cached):
                d = np.load(cached); c0, ref = d["c0"], d["ref"]
            else:
                c0, ref = CV.cv_average(g, ic, 0.0), CV.cv_average(g, ic, M.T / 2)
            rec = out["results"].setdefault(test, {}).setdefault(str(N), {"initial_sha": CV.array_hash(c0),
                                                                          "reference_sha": CV.array_hash(ref)})
            for s, recon in SCHEMES.items():
                t1 = time.time(); c, nflag = CC.run(N, recon, c0, M.T / 2)
                m = score(c, ref, c0, g, test); m["flagged_cells"] = nflag; m["seconds"] = round(time.time() - t1, 1)
                rec[s] = m
                print(f"{test} N={N} {s}: L2 {m['rel_l2']:.4f} IoU {m.get('iou', 0):.3f} range [{m['min']:+.2e}, "
                      f"{m['max']:.4f}] mass {m['mass_rel']:+.1e} flagged {nflag} | {time.time() - t0:.0f}s", flush=True)
            json.dump(out, open("cc_results.json", "w"), indent=1)
    g = M.Grid(129); d = np.load("figure_data/gauss_m16_129.npz"); c0, ref = d["c0"], d["ref"]
    for cfl in (0.35, 0.6, 0.8):
        c, nflag = CC.run(129, "u5", c0, M.T / 2, dt=cfl * g.h)
        m = score(c, ref, c0, g, "gauss_T2"); out["sweep"][f"{cfl:.2f}"] = m
        print(f"sweep dt = {cfl} h: cc17_u5 L2 {m['rel_l2']:.4f} range [{m['min']:+.2e}, {m['max']:.4f}]", flush=True)
    json.dump(out, open("cc_results.json", "w"), indent=1)
    print("wrote cc_results.json")
