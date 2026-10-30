"""Coastline closure check (2026 revision, round 2). The 'one-sided' closure of bay_experiments.py uses the weights
(5, 15, -5, 1)/16, exact for cubic point samples; the bay transports equal-width cell averages, for which the
cubic-exact one-sided face value is (1/4, 13/12, -5/12, 1/12). This repeats the six-hour smooth-patch coastline test of
bay_experiments.py with the cell-average weights, unlimited and bounded (global), without dissipation.
Writes coast_closure_check.json."""
import json, time
import numpy as np
import bay_experiments as B

i0 = int(np.argmin(np.abs(B.lon - (-122.357)))); j0 = int(np.argmin(np.abs(B.lat - 37.806)))
smooth = np.where(B.interior, np.exp(-B.dist_km(B.lon[i0], B.lat[j0]) ** 2 / (2 * 1.75 ** 2)), 0.0)
out = {"hours": B.HOURS, "substeps_per_record": B.NSUB}
t0 = time.time()
for label, sch in (("unlimited, one-sided cell-average closures, no dissipation", "ccompact"),
                   ("bounded (global), one-sided cell-average closures, no dissipation", "ccfctg")):
    r, _ = B.run(sch, smooth, "onesided_cv", False, 0.0); out[label] = r
    print(f"{label:62s} min over run {min(r['min_history']):+.3e} max {max(r['max_history']):.4f} "
          f"below -0.01: {r['cells_below_minus_0p01']} ({r['of_which_coastal']} coastal) residual {r['budget_residual']:+.1e} "
          f"| {time.time() - t0:.0f}s", flush=True)
json.dump(out, open("coast_closure_check.json", "w"), indent=1)
