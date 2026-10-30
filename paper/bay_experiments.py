"""Archived San Francisco Bay transport experiments for the 2026 revision (run after gen_fluxes_mixed.py).

1. Coastline stability: a smooth Gaussian patch (sigma = 1.75 km) at the Bay Bridge release point, six hours, for the
   unlimited compact flux with one-sided or averaged coastline closures and dissipation strength 0, 1/4, 1/2, 1, and
   for the bounded scheme (global bounds) with and without dissipation; the unlimited flux without and with full
   dissipation, and the bounded scheme, also with the depth-aware ratio face values ("depth-aware"). Records the range of c after every 120 s
   record and, at the end, the number of cells below -0.01 and how many of those have at most three wet neighbours.
2. Ablation on the sharp release of Table C: bounded compact (global bounds) without the dissipation term.
3. Outflow test: a 2 km radius unit patch next to the Golden Gate, six hours, so that tracer leaves through the open
   boundary: exported fraction and budget residual for donor cell and the bounded scheme (global and local bounds).
Writes bay_experiments.json."""
import json, time
import numpy as np
import sfbay_conservative as S

lon, lat, wi, wj, wet, interior = S.lon, S.lat, S.wi, S.wj, S.wet, S.interior
Lon, Lat = lon[wi], lat[wj]
KM = 111.0 * np.cos(np.radians(37.8))
ratio = S.max_outflow_ratio(); NSUB = max(1, int(np.ceil(ratio * S.DT / 0.8)))
HOURS = 6.0
pad = np.pad(wet, 1)
nwet = (pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])[wj, wi]


def dist_km(lo0, la0):
    return np.sqrt(((Lon - lo0) * KM) ** 2 + ((Lat - la0) * 111.0) ** 2)


def run(scheme, c0, closure="avg", diss=True, strength=1.0, depth_aware=False):
    S.set_closure(closure); S.DISSIPATION = diss; S.DISS_STRENGTH = strength; S.DEPTH_AWARE = depth_aware
    S.FALLBACK[:] = [0, 0]
    c, h, M0, wall = S.run(scheme, c0, HOURS, NSUB)
    S.set_closure("avg"); S.DISSIPATION = True; S.DISS_STRENGTH = 1.0; S.DEPTH_AWARE = False
    neg = interior & (c < -0.01)
    return dict(depth_aware=depth_aware, depth_fallback_faces=S.FALLBACK[0], depth_faces_evaluated=S.FALLBACK[1],
                min_history=h[:, 0].tolist(), max_history=h[:, 1].tolist(), final_min=float(c[interior].min()),
                final_max=float(c[interior].max()), cells_below_minus_0p01=int(neg.sum()),
                of_which_coastal=int((neg & (nwet <= 3)).sum()), retained=float(h[-1, 2] / M0),
                exported=float(h[-1, 3] / M0), budget_residual=float((M0 - h[-1, 2] - h[-1, 3]) / M0),
                cells_c_gt_0p5=int((c > 0.5).sum()), finite=bool(np.isfinite(c).all()), wall_s=round(wall, 1)), c


def run_stability_and_ablation(out, smooth, sharp, t0):
    out["coast_stability"] = {}
    for label, sch, clo, dis, st, da in (
            ("unlimited, one-sided closures, no dissipation", "ccompact", "onesided", False, 0.0, False),
            ("unlimited, averaged closures, no dissipation", "ccompact", "avg", False, 0.0, False),
            ("unlimited, averaged closures, dissipation 1/4", "ccompact", "avg", True, 0.25, False),
            ("unlimited, averaged closures, dissipation 1/2", "ccompact", "avg", True, 0.5, False),
            ("unlimited, averaged closures, dissipation 1", "ccompact", "avg", True, 1.0, False),
            ("bounded (global), averaged closures, no dissipation", "ccfctg", "avg", False, 0.0, False),
            ("bounded (global), averaged closures, dissipation 1", "ccfctg", "avg", True, 1.0, False),
            ("depth-aware, unlimited, averaged closures, no dissipation", "ccompact", "avg", False, 0.0, True),
            ("depth-aware, unlimited, averaged closures, dissipation 1", "ccompact", "avg", True, 1.0, True),
            ("depth-aware, bounded (global), averaged closures, no dissipation", "ccfctg", "avg", False, 0.0, True)):
        r, _ = run(sch, smooth, clo, dis, st, da); out["coast_stability"][label] = r
        print(f"smooth | {label:64s} range [{r['final_min']:+.3f}, {r['final_max']:.3f}] min over run "
              f"{min(r['min_history']):+.3f}; below -0.01: {r['cells_below_minus_0p01']} ({r['of_which_coastal']} coastal) "
              f"| {time.time() - t0:.0f}s", flush=True)
    out["sharp_release"] = {}
    for label, sch, dis in (("bounded (global), no dissipation", "ccfctg", False), ("bounded (global), dissipation 1", "ccfctg", True)):
        r, _ = run(sch, sharp, "avg", dis, 1.0); out["sharp_release"][label] = r
        print(f"sharp  | {label:52s} range [{r['final_min']:+.3f}, {r['final_max']:.3f}] cells>0.5 {r['cells_c_gt_0p5']}", flush=True)


def run_outflow(out):
    # outflow test: patch 5 km inside the open (Pacific) boundary on the Golden Gate row; the window opens on the ebb
    lo0 = float(lon[wi[S.isopen]].min()) + 5.0 / KM; la0 = float(lat[232])
    dg = dist_km(lo0, la0)
    gate = ((dg <= 2.0) & interior).astype(float)
    out["outflow_test"] = {"centre_lon": lo0, "centre_lat": la0, "radius_km": 2.0, "cells": int(gate.sum())}
    for sch, dis in (("donor", True), ("ccfctg", False), ("ccfct", False), ("ccfctg", True)):
        key = sch if (sch == "donor" or not dis) else sch + " (dissipation)"
        r, _ = run(sch, gate, "avg", dis, 1.0); r.pop("min_history"); r.pop("max_history"); out["outflow_test"][key] = r
        print(f"outflow| {key:22s} exported {r['exported']:.4f} retained {r['retained']:.4f} residual {r['budget_residual']:+.1e} "
              f"range [{r['final_min']:+.2e}, {r['final_max']:.4f}]", flush=True)


if __name__ == "__main__":
    import sys
    only_outflow = len(sys.argv) > 1 and sys.argv[1] == "outflow"
    out = json.load(open("bay_experiments.json")) if only_outflow else {"substeps_per_record": NSUB, "hours": HOURS}
    i0 = int(np.argmin(np.abs(lon - (-122.357)))); j0 = int(np.argmin(np.abs(lat - 37.806)))
    d0 = dist_km(lon[i0], lat[j0])
    smooth = np.where(interior, np.exp(-d0 ** 2 / (2 * 1.75 ** 2)), 0.0)
    sharp = ((d0 <= 3.5) & interior).astype(float)
    t0 = time.time()
    if not only_outflow:
        run_stability_and_ablation(out, smooth, sharp, t0)
    run_outflow(out)
    json.dump(out, open("bay_experiments.json", "w"), indent=1)
    print("wrote bay_experiments.json")
