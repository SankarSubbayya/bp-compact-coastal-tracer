"""Time-step sweep for every flux-form scheme on the 129^2 grid, control-volume-average data (2026 revision).

For each nominal Courant number (dt = CFL * h), runs the bounded compact scheme (global and local bounds), MPDATA
(nonoscillatory, two corrective passes), HSIMT as time-stepped in ROMS, and flux-form MUSCL-superbee from the
control-volume averages of the initial condition to t = T/2, and scores them against the control-volume averages of the
exact solution (cv_reference.cv_average). Also records the largest outflow fraction of any control volume in one step,
max over t and i of dt * sum_f max(Q_out, 0) / V_i, which is the donor-cell boundedness condition (<= 1).
Writes cfl_sweep_cv.json."""
import json, time
import numpy as np
import gen_mpdata as M
import cv_reference as CV

NGRID = 129
CFLS = (0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
SCHEMES = CV.SCHEMES


def outflow_fraction(g, dt, nt=200):
    worst = 0.0
    for t in np.linspace(0, M.T / 2, nt):
        Qx, Qy = g.fluxes(t)
        out = (np.maximum(Qx[:, 1:], 0) + np.maximum(-Qx[:, :-1], 0) + np.maximum(Qy[1:, :], 0) + np.maximum(-Qy[:-1, :], 0))
        worst = max(worst, float((dt * out / g.V).max()))
    return worst


if __name__ == "__main__":
    g = M.Grid(NGRID); t0 = time.time()
    refs = {"gauss_T2": (CV.cv_average(g, M.gauss_ic, 0.0), CV.cv_average(g, M.gauss_ic, M.T / 2)),
            "slotted_T2": (CV.cv_average(g, M.slot_ic, 0.0), CV.cv_average(g, M.slot_ic, M.T / 2))}
    print(f"references {time.time() - t0:.0f}s", flush=True)
    import sys
    if len(sys.argv) > 1:                      # extra schemes, merged into the existing results
        SCHEMES = tuple(sys.argv[1].split(",")); out = json.load(open("cfl_sweep_cv.json"))
    else:
        out = {}
    for cfl in CFLS:
        M.CFL = cfl
        dt = cfl * g.h; nst = int(round((M.T / 2) / dt)); dt = (M.T / 2) / nst
        row = out.get(str(cfl), {}); row["outflow_fraction"] = outflow_fraction(g, dt)
        for test, (c0, ref) in refs.items():
            for s in SCHEMES:
                c, c00, gg, wall = CV.run_from(NGRID, s, c0, M.T / 2)
                m = M.score(c, ref, c00, gg, wall)
                m.update(cv_subcells=CV.SUB, trajectory_steps=CV.NSUB_TRAJ, cfl=cfl, dt=dt, steps=nst,
                         wall_closure=CV.C.REPRESENTATION, code=CV.CODE, initial_sha=CV.array_hash(c0),
                         reference_sha=CV.array_hash(ref))
                row[f"{test}_{s}"] = m
        out[str(cfl)] = row
        print(f"CFL {cfl}: outflow fraction {row['outflow_fraction']:.3f} | " + " ".join(
            f"{s} {row['gauss_T2_' + s]['rel_l2']:.4f}/{row['slotted_T2_' + s]['rel_l2']:.3f} min {min(row['gauss_T2_' + s]['min'], row['slotted_T2_' + s]['min']):+.1e}"
            for s in SCHEMES) + f" | {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open("cfl_sweep_cv.json", "w"), indent=1)
    print("wrote cfl_sweep_cv.json")
