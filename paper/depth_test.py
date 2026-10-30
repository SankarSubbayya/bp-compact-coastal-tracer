"""Order of the flux-form compact operator when the depth varies (JCP revision, 2026-10-28): a manufactured,
time-independent test of the spatial operator alone, as in covariance_test.py. Periodic unit square, N x N cells of
width h = 1/N, depth D = 1 + 0.2 sin(2 pi x) + 0.15 cos(2 pi y), velocity u = (sin 2 pi x cos 2 pi y,
-cos 2 pi x sin 2 pi y), transport q = D u, and the tracer c of covariance_test.py. The stored concentration is the
depth-weighted average <Dc>/<D> (<.> = cell average). Exact cell averages and the exact cell average of div(q c) (face
integrals by 8-point Gauss-Legendre) are compared with the flux divergence of
  (a) "direct": compact face values of the stored concentration, F = Q c_f + (h^3/12) U_t c_t (the transverse
      correction for uniform depth);
  (b) "ratio": r = (compact <Dc>_f) / (compact <D>_f), F = Q r, without a transverse term;
  (c) "depth-aware": F = Q r + (h^3/12) (U_t - (U / D_f) D_t) r_t, the flux of gen_conservative.py;
U = Q/h is the face-averaged transport, t the tangential derivative (central differences of face values). Also checks
that (c) gives F = c Q for a uniform tracer. Writes depth_test.json with max-norm errors and observed orders."""
import json
import numpy as np
from covariance_test import TP, cf, face_avg, cell_avg, compact_faces

Df = lambda x, y: 1 + 0.2 * np.sin(TP * x) + 0.15 * np.cos(TP * y)
uf = lambda x, y: np.sin(TP * x) * np.cos(TP * y)
vf = lambda x, y: -np.cos(TP * x) * np.sin(TP * y)


def run(N, tracer=cf):
    h = 1 / N; e = np.arange(N) * h
    Db = cell_avg(Df, N); Mb = cell_avg(lambda x, y: Df(x, y) * tracer(x, y), N); cb = Mb / Db
    XF, Y0 = np.meshgrid(e + h, e, indexing="ij"); Y1 = Y0 + h        # x-faces [i, j]
    YF, X0 = np.meshgrid(e + h, e, indexing="ij"); X1 = X0 + h        # y-faces, transposed below to [i, j]
    Qx = h * face_avg(lambda x, y: Df(x, y) * uf(x, y), XF, Y0, Y1)
    Qy = (h * face_avg(lambda x, y: Df(x, y) * vf(x, y), YF, X0, X1, along_y=False)).T
    Ex = h * face_avg(lambda x, y: Df(x, y) * uf(x, y) * tracer(x, y), XF, Y0, Y1)
    Ey = (h * face_avg(lambda x, y: Df(x, y) * vf(x, y) * tracer(x, y), YF, X0, X1, along_y=False)).T
    div = lambda Fx, Fy: (Fx - np.roll(Fx, 1, 0) + Fy - np.roll(Fy, 1, 1)) / h ** 2
    dtan = lambda a, ax: (np.roll(a, -1, ax) - np.roll(a, 1, ax)) / (2 * h)
    exact = div(Ex, Ey)
    Ux, Uy = Qx / h, Qy / h
    # (a) direct
    cx, cy = compact_faces(cb, 0), compact_faces(cb, 1)
    Fa = (Qx * cx + h ** 3 / 12 * dtan(Ux, 1) * dtan(cx, 1), Qy * cy + h ** 3 / 12 * dtan(Uy, 0) * dtan(cy, 0))
    # (b), (c) ratio of reconstructed mass and depth densities
    Dx, Dy = compact_faces(Db, 0), compact_faces(Db, 1)
    rx, ry = compact_faces(Mb, 0) / Dx, compact_faces(Mb, 1) / Dy
    Fb = (Qx * rx, Qy * ry)
    Fc = (Qx * rx + h ** 3 / 12 * (dtan(Ux, 1) - Ux / Dx * dtan(Dx, 1)) * dtan(rx, 1),
          Qy * ry + h ** 3 / 12 * (dtan(Uy, 0) - Uy / Dy * dtan(Dy, 0)) * dtan(ry, 0))
    scale = np.abs(exact).max()
    out = {k: float(np.abs(div(*F) - exact).max() / scale) for k, F in (("direct", Fa), ("ratio", Fb), ("depth_aware", Fc))}
    return out, Fc, (Qx, Qy)


if __name__ == "__main__":
    Ns = (16, 32, 64, 128, 256)
    res = {str(N): run(N)[0] for N in Ns}
    out = {"errors": res, "orders": {}}
    for k in res[str(Ns[0])]:
        out["orders"][k] = [float(np.log2(res[str(a)][k] / res[str(b)][k])) for a, b in zip(Ns, Ns[1:])]
        print(f"{k:12s} " + " ".join(f"{res[str(N)][k]:.2e}" for N in Ns) + "  orders " +
              " ".join(f"{o:.2f}" for o in out["orders"][k]))
    _, (Fx, Fy), (Qx, Qy) = run(64, tracer=lambda x, y: 0.37 + 0 * x)
    out["uniform_tracer_flux_max_dev"] = float(max(np.abs(Fx - 0.37 * Qx).max(), np.abs(Fy - 0.37 * Qy).max()))
    print("uniform tracer: max |F - 0.37 Q| =", f"{out['uniform_tracer_flux_max_dev']:.1e}")
    json.dump(out, open("depth_test.json", "w"), indent=1)
