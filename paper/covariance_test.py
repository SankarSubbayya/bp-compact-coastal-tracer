"""Order of the flux-form compact operator for variable two-dimensional flow (2026 revision): a manufactured,
time-independent test of the spatial operator alone. Periodic unit square, N x N cells of width h = 1/N, divergence-free
velocity from psi = sin(2 pi x) sin(2 pi y) / (2 pi), smooth tracer c = 1 + 0.5 sin(2 pi (x + 2y)) + 0.3 cos(2 pi (3x - y)).
Exact cell averages of c and the exact cell average of div(u c) (face integrals by 8-point Gauss-Legendre) are compared
with
  (a) the compact face values (relation (faces) applied along grid lines to exact cell averages) against the exact face
      averages of c;
  (b) the scheme's flux divergence, with F_f = Q_f c_f and Q_f the exact face-integrated volume flux;
  (c) the same with the transverse correction F_f = Q_f c_f + (h^3/12) (du/dt)(dc/dt) along the face (t tangential;
      McCorquodale and Colella 2011), derivatives by central differences of face values; used here only to show the
      size and order of the omitted covariance, not in the scheme.
Writes covariance_test.json with max-norm errors and observed orders."""
import json
import numpy as np

TP = 2 * np.pi
psi = lambda x, y: np.sin(TP * x) * np.sin(TP * y) / TP
uf = lambda x, y: np.sin(TP * x) * np.cos(TP * y)            # u = d psi / dy
vf = lambda x, y: -np.cos(TP * x) * np.sin(TP * y)           # v = -d psi / dx
cf = lambda x, y: 1 + 0.5 * np.sin(TP * (x + 2 * y)) + 0.3 * np.cos(TP * (3 * x - y))
gx, gw = np.polynomial.legendre.leggauss(8)


def face_avg(f, x0, y0, y1, along_y=True):
    """Average of f over x = x0, y in [y0, y1] (arrays broadcast), or over y = x0 with x in [y0, y1]."""
    s = 0.5 * (y0[..., None] + y1[..., None]) + 0.5 * (y1 - y0)[..., None] * gx
    vals = f(x0[..., None], s) if along_y else f(s, x0[..., None])
    return 0.5 * (vals * gw).sum(-1)


def cell_avg(f, N):
    h = 1 / N; e = np.arange(N) * h
    X = e[:, None, None, None] + 0.5 * h * (1 + gx[None, None, :, None])
    Y = e[None, :, None, None] + 0.5 * h * (1 + gx[None, None, None, :])
    return 0.25 * (f(X, Y) * gw[:, None] * gw[None, :]).sum((-1, -2))        # [i, j] = cell (x_i, y_j)


def compact_faces(cbar, axis):
    """Periodic compact face values at the faces i+1/2 along axis (face i+1/2 between cells i and i+1)."""
    n = cbar.shape[axis]
    A = np.eye(n) + 0.25 * np.roll(np.eye(n), 1, 1) + 0.25 * np.roll(np.eye(n), -1, 1)
    rhs = 0.75 * (cbar + np.roll(cbar, -1, axis))
    return np.moveaxis(np.linalg.solve(A, np.moveaxis(rhs, axis, 0).reshape(n, -1)).reshape(np.moveaxis(rhs, axis, 0).shape), 0, axis)


def run(N):
    h = 1 / N; e = np.arange(N) * h
    cbar = cell_avg(cf, N)
    XF, Y0 = np.meshgrid(e + h, e, indexing="ij"); Y1 = Y0 + h        # x-faces at x_{i+1/2} = (i+1) h
    cx = compact_faces(cbar, 0); cy = compact_faces(cbar, 1)
    exact_cx = face_avg(cf, XF, Y0, Y1)
    face_err = np.abs(cx - exact_cx).max()
    Qx = psi(XF, Y1) - psi(XF, Y0)                                       # integral of u over the x-face
    YF, X0 = np.meshgrid(e + h, e, indexing="ij"); X1 = X0 + h          # y-faces: [j, i] -> transpose below
    Qy = -(psi(X1, YF) - psi(X0, YF)).T                                  # integral of v over the y-face, [i, j]
    Fx = Qx * cx; Fy = Qy * cy
    # transverse correction (h^3/12) u_t c_t; tangential differences of face-averaged quantities
    ux = Qx / h; vy = Qy / h
    dt = lambda a, ax: (np.roll(a, -1, ax) - np.roll(a, 1, ax)) / (2 * h)
    Fx_c = Fx + h ** 3 / 12 * dt(ux, 1) * dt(cx, 1)
    Fy_c = Fy + h ** 3 / 12 * dt(vy, 0) * dt(cy, 0)
    div = lambda Fx, Fy: (Fx - np.roll(Fx, 1, 0) + Fy - np.roll(Fy, 1, 1)) / h ** 2
    # exact cell average of div(u c) from exact face integrals
    Ex = h * face_avg(lambda x, y: uf(x, y) * cf(x, y), XF, Y0, Y1)
    Ey = (h * face_avg(lambda x, y: vf(x, y) * cf(x, y), YF, X0, X1, along_y=False)).T
    exact = div(Ex, Ey)
    return {"face_value": float(face_err), "flux_divergence": float(np.abs(div(Fx, Fy) - exact).max()),
            "flux_divergence_corrected": float(np.abs(div(Fx_c, Fy_c) - exact).max())}


if __name__ == "__main__":
    Ns = (16, 32, 64, 128, 256)
    res = {str(N): run(N) for N in Ns}
    out = {"errors": res, "orders": {}}
    for k in res[str(Ns[0])]:
        out["orders"][k] = [float(np.log2(res[str(a)][k] / res[str(b)][k])) for a, b in zip(Ns, Ns[1:])]
        print(f"{k:26s} " + " ".join(f"{res[str(N)][k]:.2e}" for N in Ns) + "  orders " +
              " ".join(f"{o:.2f}" for o in out["orders"][k]))
    json.dump(out, open("covariance_test.json", "w"), indent=1)
