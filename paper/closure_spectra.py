"""Eigenvalue spectra of the semi-discrete flux-form compact operator with different closures at the ends of a grid
line (2026 revision; boundary-stability analysis after Carpenter, Gottlieb and Abarbanel 1993).

One line of N equal-width cells (width h = 1/N) with cell averages c_i and faces f_0..f_N. Interior faces f_2..f_{N-2}
use the compact relation (1/4) c_{f-1} + c_f + (1/4) c_{f+1} = (3/4)(c_{f-1} + c_f) (face f between cells f-1 and f);
the first and last interior faces f_1, f_{N-1} use a closure:
  point     (5, 15, -5, 1)/16 on the four nearest cells (exact for cubic point samples)
  cellavg   (1/4, 13/12, -5/12, 1/12)              (exact for cubic cell averages, equal widths)
  average   (c_i + c_{i+1}) / 2                     (the bay's coastal closure)
Three end conditions:
  open  constant velocity u = 1, inflow face f_0 carries c = 0, outflow face f_N carries the upwind value c_{N-1}
        (the classical inflow-outflow setting of the boundary-stability literature);
  wall  u = sin(pi x), zero at both end faces (no flux), as along a grid line between two coastal cells;
  step  u = 1 inside and 0 at both end faces (a coast face next to moving water).
Line lengths from N = 6 (short wet stretches at a jagged coastline) to N = 256.
dc_i/dt = -(Q_{i+1} c_{i+1} - Q_i c_i)/h with the face values above; the eigenvalues lambda of this linear operator
are reported as max Re(lambda) h / max|u|. A positive value means semi-discrete modes that grow exponentially; a fully
discrete RK4 step can still damp a small positive real part at a finite time step. The spectrum alone does not bound
the evolution: for the wall and step lines the zero eigenvalue is defective (rank L^2 < rank L, a Jordan chain), so
some solutions grow algebraically. For those lines the ranks of L and L^2 and the norm of exp(t L) at t = 100 h/|u|
are recorded. Writes closure_spectra.json."""
import json
import numpy as np

W = {"point": (5 / 16, 15 / 16, -5 / 16, 1 / 16), "cellavg": (1 / 4, 13 / 12, -5 / 12, 1 / 12), "average": (0.5, 0.5, 0, 0)}


def face_matrix(N, closure):
    """Matrix R with face values (f_1..f_{N-1}) = R c."""
    nf = N - 1
    A = np.zeros((nf, nf)); B = np.zeros((nf, N))
    w = W[closure]
    for k in range(nf):                    # row k is face f_{k+1}, between cells k and k+1
        if k == 0:
            A[k, k] = 1; B[k, 0:4] = w
        elif k == nf - 1:
            A[k, k] = 1; B[k, N - 1::-1][:4] = w      # mirrored: nearest cell is N-1
        else:
            A[k, k - 1] = 0.25; A[k, k] = 1; A[k, k + 1] = 0.25; B[k, k] = 0.75; B[k, k + 1] = 0.75
    return np.linalg.solve(A, B)


def operator(N, closure, ends):
    h = 1 / N; xf = np.arange(N + 1) * h
    Q = {"open": np.ones(N + 1), "wall": np.sin(np.pi * xf), "step": np.r_[0.0, np.ones(N - 1), 0.0]}[ends]
    Fm = np.zeros((N + 1, N))              # face flux = Fm @ c
    Fm[1:N] = Q[1:N, None] * face_matrix(N, closure)
    if ends == "open":
        Fm[N, N - 1] = Q[N]                # outflow: upwind value; inflow face f_0 carries 0
    L = -(Fm[1:] - Fm[:-1]) / h
    return L, np.abs(Q).max()


if __name__ == "__main__":
    out = {}
    for ends in ("open", "wall", "step"):
        for closure in W:
            row = {}
            for N in (6, 8, 12, 16, 32, 64, 128, 256):
                L, umax = operator(N, closure, ends)
                lam = np.linalg.eigvals(L) / (N * umax)        # lambda h / max|u|
                row[str(N)] = {"max_re": float(lam.real.max()), "max_abs": float(np.abs(lam).max())}
                if ends != "open":
                    from scipy.linalg import expm
                    A = L / (N * umax)                                   # time in units of h/|u|
                    row[str(N)].update(rank_L=int(np.linalg.matrix_rank(A, tol=1e-9)),
                                       rank_L2=int(np.linalg.matrix_rank(A @ A, tol=1e-9)),
                                       norm_exp_100=float(np.linalg.norm(expm(100 * A), 2)))
            out.setdefault(ends, {})[closure] = row
            print(f"{ends:5s} {closure:8s} max Re(lambda) h/|u|: " +
                  " ".join(f"{row[n]['max_re']:+.2e}" for n in row) + " | max |lambda| h/|u|: " +
                  " ".join(f"{row[n]['max_abs']:.2f}" for n in row), flush=True)
    json.dump(out, open("closure_spectra.json", "w"), indent=1)
