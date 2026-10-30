"""Control-volume-average version of the deformational-flow benchmark (2026 revision, after the referee audit).

The flux-form schemes (bounded compact, MPDATA, HSIMT) advance control-volume averages, so here they start from the
control-volume averages of the initial condition and are scored against the control-volume averages of the exact
solution at t = T/2: each control volume (half and quarter volumes at walls) is subdivided into m x m sub-cells whose
midpoints are traced back to t = 0 with RK4 (400 steps, as in gen_mpdata.exact) and averaged. The nodal (point-sample)
results of gen_mpdata.py remain as an additional diagnostic.

Writes cv_results.json: {test: {N: {scheme: score}}}, score as in gen_mpdata.score (L2 weighted by control volume; IoU
on control volumes with c > 0.5)."""
import json, os, sys, time
import numpy as np
import gen_mpdata as M
import gen_conservative as C
C.REPRESENTATION = "cv"   # control-volume averages: half-width wall closure

SUB = int(os.environ.get("CV_SUB", 16))     # sub-cells per direction (16 for every test since round 2)
NSUB_TRAJ = 400


def cv_average(g, fn, tc, m=None, nsub=NSUB_TRAJ):
    """Average over every control volume of fn evaluated at the t = 0 foot of each sub-cell midpoint traced from tc."""
    m = SUB if m is None else m
    frac = (np.arange(m) + 0.5) / m
    xs = (g.e[:-1, None] + frac[None, :] * g.L[:, None]).ravel()          # N*m sub-cell midpoints per direction
    X, Y = np.meshgrid(xs, xs)
    px, py = X.copy(), Y.copy(); d = tc / nsub
    for n in range(nsub if tc > 0 else 0):
        s = tc - n * d
        u1, v1 = M.A.velocity(px, py, s, M.T)
        u2, v2 = M.A.velocity(px - .5*d*u1, py - .5*d*v1, s - .5*d, M.T)
        u3, v3 = M.A.velocity(px - .5*d*u2, py - .5*d*v2, s - .5*d, M.T)
        u4, v4 = M.A.velocity(px - d*u3, py - d*v3, s - d, M.T)
        px = px - d/6*(u1 + 2*u2 + 2*u3 + u4); py = py - d/6*(v1 + 2*v2 + 2*v3 + v4)
    f = fn(np.clip(px, 0, 1), np.clip(py, 0, 1))
    N = g.N
    return f.reshape(N, m, N, m).mean(axis=(1, 3))


def run_from(N, scheme, c0, t_end):
    """Run a scheme from a given initial field (the gen_mpdata / gen_conservative drivers start from nodal samples)."""
    ic = lambda X, Y: c0
    if scheme in C_SCHEMES:
        return C.run(N, scheme, ic, t_end)
    return M.run(N, scheme, ic, t_end)


C_SCHEMES = tuple(C.SCHEME_SPEC)
# compact family (transverse-corrected unless "_nt"), the controlled reconstruction comparison (c4, u5) under the
# same single-stage FCT, and the comparison schemes of ocean models
SCHEMES = ("cfctg", "cfct", "cfcte", "cfctgd", "cfctd", "ccompactd", "cfctg_nt", "c4fctg", "c4fcte", "u5fctg", "u5fcte",
           "mpdata3_no", "mpdata_no", "hsimt", "hsimt_nopred", "hsimt_split", "muscl_fv", "donor_fv")


def code_hash():
    """Hash of every source file the benchmark executes, read once when this module is imported (CODE)."""
    import hashlib
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("gen_mpdata.py", "gen_conservative.py", "gen_deepen.py", "cv_reference.py", "paths.py"):
        h.update(open(os.path.join(here, f), "rb").read())
    for f in ("advdiff.py", "sharp.py", "compact_stable.py"):
        h.update(open(os.path.join(M.P.DEV, f), "rb").read())
    return h.hexdigest()[:16]


def array_hash(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


CODE = code_hash()

if __name__ == "__main__":
    grids = {"gauss_T2": (33, 65, 129, 257), "slotted_T2": (65, 129, 257)}
    if os.environ.get("CV_TESTS"):
        grids = {k: v for k, v in grids.items() if k in os.environ["CV_TESTS"].split(",")}
    if len(sys.argv) > 1:
        grids = {k: tuple(int(n) for n in sys.argv[1].split(",") if int(n) in v) for k, v in grids.items()}
    out = {}
    if os.environ.get("CV_MERGE"):
        out = json.load(open("cv_results.json"))
    for test, Ns in grids.items():
        ic = M.gauss_ic if test == "gauss_T2" else M.slot_ic
        for N in Ns:
            t0 = time.perf_counter(); g = M.Grid(N)
            c0 = cv_average(g, ic, 0.0); ref = cv_average(g, ic, M.T / 2)
            print(f"{test} N={N}: references in {time.perf_counter() - t0:.0f}s", flush=True)
            out.setdefault(test, {}).setdefault(str(N), {})
            for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else SCHEMES):
                c, c00, gg, wall = run_from(N, s, c0, M.T / 2)
                m = M.score(c, ref, c00, gg, wall)
                if test == "slotted_T2":     # IoU on control volumes (score() only computes it for 0/1 initial data)
                    a, b = c > 0.5, ref > 0.5
                    m["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
                dt = M.CFL * gg.h; nst = int(round((M.T / 2) / dt))
                m.update(cv_subcells=SUB, trajectory_steps=NSUB_TRAJ, cfl=M.CFL, dt=(M.T / 2) / nst, steps=nst,
                         wall_closure=C.REPRESENTATION, code=CODE, initial_sha=array_hash(c0), reference_sha=array_hash(ref))
                out[test][str(N)][s] = m
                print(f"  {s:10s} L2 {m['rel_l2']:.4f} IoU {m.get('iou', 0):.3f} range [{m['min']:+.1e},{m['max']:.3f}] "
                      f"mass {m['mass_rel']:+.1e}", flush=True)
            json.dump(out, open("cv_results.json", "w"), indent=1)
    print("wrote cv_results.json")
