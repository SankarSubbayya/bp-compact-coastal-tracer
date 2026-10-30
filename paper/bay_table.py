"""Released-patch table for San Francisco Bay (2026 revision): 3.5 km radius unit patch at the Bay Bridge, five hours,
offline flux-form transport with the recorded face volume fluxes. Writes sfbay_table.json."""
import json
import numpy as np
import bay_experiments as B

B.HOURS = 5.0
i0 = int(np.argmin(np.abs(B.lon - (-122.357)))); j0 = int(np.argmin(np.abs(B.lat - 37.806)))
d0 = B.dist_km(B.lon[i0], B.lat[j0])
sharp = ((d0 <= 3.5) & B.interior).astype(float)
out = {"hours": B.HOURS, "initial_cells_c_gt_0p5": int((sharp > 0.5).sum()), "release_lon": float(B.lon[i0]),
       "release_lat": float(B.lat[j0])}
for label, sch, dis, da in (("donor cell", "donor", True, False), ("bounded compact, global bounds", "ccfctg", False, False),
                            ("bounded compact, local bounds", "ccfct", False, False),
                            ("bounded compact with dissipation, global bounds", "ccfctg", True, False),
                            ("compact with dissipation, unlimited", "ccompact", True, False),
                            ("bounded compact, global bounds, depth-aware ratio", "ccfctg", False, True)):
    r, c = B.run(sch, sharp, "avg", dis, 1.0, depth_aware=da)
    r.pop("min_history"); r.pop("max_history"); r["cells_c_gt_0p01"] = int((c > 0.01).sum())
    out[label] = r
    np.save(f"sfbay_table_{sch}{'_d' if (dis and sch != 'donor') else ''}{'_ratio' if da else ''}.npy", c)
    print(f"{label:48s} range [{r['final_min']:+.2e}, {r['final_max']:.4f}] c>0.5 {r['cells_c_gt_0p5']} "
          f"c>0.01 {r['cells_c_gt_0p01']} exported {r['exported']:.1e} residual {r['budget_residual']:+.1e}", flush=True)
json.dump(out, open("sfbay_table.json", "w"), indent=1)
print("wrote sfbay_table.json")
