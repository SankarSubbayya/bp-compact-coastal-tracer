"""Cost of the transverse flux correction (2026 revision): compiled wall-clock time of the bounded compact scheme with
and without the correction (fast_schemes "cfctg" and "cfctg_nt"), smooth tracer to t = T/2 on 129^2 and 257^2, best of
five runs (all retained), same session, from the cached m = 16 initial data of covariance_ablation.py. Writes correction_cost.json."""
import json, time
import gen_mpdata as M
import fast_schemes as F
import numpy as np

out = {}
for N in (129, 257):
    c0 = np.load(f"figure_data/gauss_m16_{N}.npz")["c0"]
    for s in ("cfctg_nt", "cfctg"):
        F.run(N, s, c0, M.T / 2)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); F.run(N, s, c0, M.T / 2); ts.append(time.perf_counter() - t0)
        out.setdefault(str(N), {})[s] = {"seconds": min(ts), "samples": ts}
    print(N, {k: round(v["seconds"], 3) for k, v in out[str(N)].items()},
          f"overhead {100 * (out[str(N)]['cfctg']['seconds'] / out[str(N)]['cfctg_nt']['seconds'] - 1):.1f}%", flush=True)
json.dump(out, open("correction_cost.json", "w"), indent=1)
