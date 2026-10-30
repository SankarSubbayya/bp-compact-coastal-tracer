"""24-hour San Francisco Bay test of the bounded compact scheme with and without the dissipation term (2026 revision).
Uses fluxes_mixed_25h.npz (WIN_H=25 FLUX_OUT=fluxes_mixed_25h.npz python3 gen_fluxes_mixed.py; same spin-up, so its
first six hours equal fluxes_mixed.npz). Writes long_run.json."""
import json, os
os.environ["FLUX_FILE"] = "fluxes_mixed_25h.npz"
import numpy as np
import bay_experiments as B

B.HOURS = 24.0
i0 = int(np.argmin(np.abs(B.lon - (-122.357)))); j0 = int(np.argmin(np.abs(B.lat - 37.806)))
d0 = B.dist_km(B.lon[i0], B.lat[j0])
smooth = np.where(B.interior, np.exp(-d0 ** 2 / (2 * 1.75 ** 2)), 0.0)
sharp = ((d0 <= 3.5) & B.interior).astype(float)
out = {"hours": B.HOURS, "substeps_per_record": B.NSUB}
for name, c0 in (("smooth", smooth), ("sharp", sharp)):
    for label, sch, dis in (("bounded (global), no dissipation", "ccfctg", False),
                            ("bounded (global), dissipation 1", "ccfctg", True),
                            ("unlimited, dissipation 1", "ccompact", True)):
        r, _ = B.run(sch, c0, "avg", dis, 1.0)
        out[f"{name} | {label}"] = r
        print(f"{name:6s} | {label:34s} range [{r['final_min']:+.3e}, {r['final_max']:.4f}] cells>0.5 {r['cells_c_gt_0p5']} "
              f"exported {r['exported']:.4f} residual {r['budget_residual']:+.1e}", flush=True)
json.dump(out, open("long_run.json", "w"), indent=1)
print("wrote long_run.json")
