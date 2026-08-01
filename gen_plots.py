"""Comprehensive, JCP-style paper figures. Run with:
   uv run --with numpy --with scipy --with matplotlib python3 gen_plots.py
Writes PNGs to paper/figures/.
Figures: (1) modified-wavenumber + dispersion analysis (first deriv),
(2) second-derivative modified wavenumber (diffusion operator),
(3) operator convergence, (4) slotted-cylinder benchmark panels,
(5) filament cross-section, (6) boundedness (min/max) over time."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 11,
    "axes.labelsize": 11, "legend.fontsize": 9, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "figure.dpi": 300,
})
DEV = "."
sys.path.insert(0, DEV)
import advdiff as A, sharp as Sh, bp_compact as BP
from compact_stable import compact_d1_mats, compact_d1
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
STY = {"central2": ("C0", "-"), "central4": ("C1", "-"), "central6": ("C2", "-"),
       "compact4": ("C3", "--"), "compact6": ("C4", "--")}

# ---- first-derivative modified wavenumber ----
def mw1(kh, nm):
    s = np.sin
    return {"central2": s(kh), "central4": (8*s(kh)-s(2*kh))/6,
            "central6": (45*s(kh)-9*s(2*kh)+s(3*kh))/30,
            "compact4": 1.5*s(kh)/(1+0.5*np.cos(kh)),
            "compact6": ((14/9)*s(kh)+(1/18)*s(2*kh))/(1+(2/3)*np.cos(kh))}[nm]
# ---- second-derivative modified wavenumber (diffusion operator) ----
def mw2(kh, nm):
    c = np.cos
    return {"central2": 2*(1-c(kh)), "central4": (15-16*c(kh)+c(2*kh))/6,
            "central6": (490-540*c(kh)+54*c(2*kh)-4*c(3*kh))/180,
            "compact4": 2.4*(1-c(kh))/(1+0.2*c(kh))}[nm]

kh = np.linspace(1e-3, np.pi, 500)
# Fig 1: 2-panel first-derivative analysis
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
a1.plot(kh, kh, "k--", lw=2, label="exact (spectral)")
for nm in STY:
    a1.plot(kh, mw1(kh, nm), STY[nm][1], color=STY[nm][0], label=nm)
a1.set_xlabel(r"$k\,h$"); a1.set_ylabel(r"modified wavenumber $k'\,h$")
a1.set_title("(a) First-derivative resolution"); a1.set_xlim(0, np.pi); a1.set_ylim(0, np.pi)
a1.legend(loc="upper left"); a1.grid(alpha=.3)
for nm in STY:
    a2.semilogy(kh, np.abs(mw1(kh, nm)-kh)+1e-16, STY[nm][1], color=STY[nm][0], label=nm)
a2.axhline(1e-2, color="0.5", lw=.8, ls=":"); a2.text(0.1, 1.3e-2, "1% error", fontsize=8, color="0.4")
a2.set_xlabel(r"$k\,h$"); a2.set_ylabel(r"dispersion error $|k'h - kh|$")
a2.set_title("(b) Dispersion error (resolving efficiency)"); a2.set_xlim(0, np.pi); a2.set_ylim(1e-7, 3)
a2.legend(loc="lower right"); a2.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_wavenumber_analysis.png"); plt.close()

# Fig 2: second-derivative modified wavenumber
fig, ax = plt.subplots(figsize=(6, 4.6))
ax.plot(kh, kh**2, "k--", lw=2, label="exact $(kh)^2$")
for nm in ["central2", "central4", "central6", "compact4"]:
    ax.plot(kh, mw2(kh, nm), STY[nm][1], color=STY[nm][0], label=nm)
ax.set_xlabel(r"$k\,h$"); ax.set_ylabel(r"modified $k''\,h^2$")
ax.set_title("Second-derivative (diffusion) resolution")
ax.set_xlim(0, np.pi); ax.legend(loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_second_derivative_wavenumber.png"); plt.close()

# Fig 3: operator convergence
Ns = [17, 33, 65, 129, 257, 513]; errs = {k: [] for k in ["central2", "central4", "central6", "compact4"]}; hs = []
for N in Ns:
    x = np.linspace(0, 1, N); h = x[1]-x[0]; hs.append(h)
    f = np.cos(2*np.pi*x)[None, :]; ex = -2*np.pi*np.sin(2*np.pi*x)
    for od, nm in [(2, "central2"), (4, "central4"), (6, "central6")]:
        errs[nm].append(np.max(np.abs(A.d1(f, h, 1, od)[0][3:-3]-ex[3:-3])))
    ab = compact_d1_mats(N); errs["compact4"].append(np.max(np.abs(compact_d1(f, h, 1, ab)[0][3:-3]-ex[3:-3])))
hs = np.array(hs)
fig, ax = plt.subplots(figsize=(6, 4.8))
for nm in errs:
    ax.loglog(hs, errs[nm], "o"+STY[nm][1], color=STY[nm][0], label=nm, ms=5)
for p, c in [(2, "0.75"), (4, "0.55"), (6, "0.35")]:
    ax.loglog(hs, errs["central2"][0]*(hs/hs[0])**p, ":", color=c, lw=1.2, label=f"$h^{p}$")
ax.set_xlabel("$h$"); ax.set_ylabel("max derivative error"); ax.set_title("Operator convergence")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_convergence.png"); plt.close()

# ---- slotted-cylinder benchmark ----
N, T, tc, dt = 129, 3.0, 1.5, 0.004
x = np.linspace(0, 1, N); X, Y = np.meshgrid(x, x)
IC = Sh.slotted_cylinder(X, Y); REF = Sh.exact_reference(N, T, tc)
def uw3(c, vel, h, axis):
    a = np.moveaxis(c, axis, -1); p = np.pad(a, ((0, 0), (2, 2)), mode="reflect")
    b = (2*p[:, 3:-1]+3*p[:, 2:-2]-6*p[:, 1:-3]+p[:, :-4])/(6*h)
    f = -(2*p[:, 1:-3]+3*p[:, 2:-2]-6*p[:, 3:-1]+p[:, 4:])/(6*h)
    return np.where(vel >= 0, np.moveaxis(b, -1, axis), np.moveaxis(f, -1, axis))
def solve_uw3(N, T, tc, dt):
    xx = np.linspace(0, 1, N); h = xx[1]-xx[0]; Xl, Yl = np.meshgrid(xx, xx)
    c = Sh.slotted_cylinder(Xl, Yl); nst = int(round(tc/dt)); dt = tc/nst
    def rhs(c, t):
        u, v = A.velocity(Xl, Yl, t, T); return -(u*uw3(c, u, h, 1)+v*uw3(c, v, h, 0))
    for n in range(nst):
        t = n*dt; k1 = rhs(c, t); k2 = rhs(c+.5*dt*k1, t+.5*dt); k3 = rhs(c+.5*dt*k2, t+.5*dt); k4 = rhs(c+dt*k3, t+dt)
        c = c+dt/6*(k1+2*k2+2*k3+k4)
    return c
UP1 = BP.solve(N, T, tc, 0, dt, "upwind")[0]; UP3 = solve_uw3(N, T, tc, dt)
CMP = BP.solve(N, T, tc, 0, dt, "compact")[0]; BPC = BP.solve_bpf(N, T, tc, 0, dt)[0]
panels = [("Initial condition", IC), ("Exact reference, $t=T/2$", REF),
          ("1st-order upwind", UP1), ("3rd-order upwind (OE-98)", UP3),
          ("plain compact (Gibbs)", CMP), ("bound-preserving compact (this work)", BPC)]
fig, ax = plt.subplots(2, 3, figsize=(13, 8.2))
for a, (ti, fld) in zip(ax.flat, panels):
    im = a.imshow(fld, origin="lower", extent=[0, 1, 0, 1], vmin=-0.2, vmax=1.2, cmap="turbo")
    a.set_title(f"{ti}\nrange $[{fld.min():+.2f},\\,{fld.max():.2f}]$", fontsize=9.5)
    a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, fraction=.046, pad=.02)
fig.suptitle("Deformational transport of a slotted cylinder (advection-dominated)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_slotted_cylinder_panels.png", dpi=300); plt.close()

# Fig 5: cross-section
j = int(np.argmax(REF.sum(axis=1)))
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(x, REF[j], "k-", lw=2.5, label="exact reference")
ax.plot(x, CMP[j], "C3-", label="plain compact")
ax.plot(x, UP1[j], "C0-", label="1st-order upwind")
ax.plot(x, BPC[j], "C2-", lw=2, label="bound-preserving compact (this work)")
ax.axhline(0, color="0.6", lw=.8); ax.axhline(1, color="0.6", lw=.8)
ax.set_xlabel("$x$"); ax.set_ylabel("$c$"); ax.set_title(f"Cross-section through the filament ($y$-row {j})")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{OUT}/fig5_cross_section.png"); plt.close()

# Fig 6: boundedness over time -- broken-axis zoom on the upper bound (top panel)
# and lower bound (bottom panel). The two bounded schemes (1st-order upwind and the
# bound-preserving compact) coincide exactly on c=0 and c=1, so they are separated
# by distinct markers and a small horizontal offset; violation zones are tinted red.
from matplotlib.lines import Line2D
ts = np.linspace(0, tc, 13)[1:]
spec = [("plain compact", "C3", "None", "compact"),
        ("3rd-order upwind (OE-98)", "C1", "^", "uw3"),
        ("1st-order upwind", "C0", "o", "upwind"),
        ("bound-preserving compact", "C2", "s", "bpf")]
mn = {s[0]: [] for s in spec}; mx = {s[0]: [] for s in spec}
for t in ts:
    flds = {"compact": BP.solve(N, T, float(t), 0, dt, "compact")[0],
            "upwind":  BP.solve(N, T, float(t), 0, dt, "upwind")[0],
            "bpf":     BP.solve_bpf(N, T, float(t), 0, dt)[0],
            "uw3":     solve_uw3(N, T, float(t), dt)}
    for lbl, col, mk, key in spec:
        mn[lbl].append(float(flds[key].min())); mx[lbl].append(float(flds[key].max()))
fig, (au, al) = plt.subplots(2, 1, figsize=(8.0, 6.9), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
dxoff = (ts[1]-ts[0])*0.12
for i, (lbl, col, mk, key) in enumerate(spec):
    m = None if mk == "None" else mk
    au.plot(ts+(i-1.5)*dxoff, mx[lbl], color=col, marker=m, ms=6, mfc="white", mew=1.3, lw=1.8)
    al.plot(ts+(i-1.5)*dxoff, mn[lbl], color=col, marker=m, ms=6, mfc="white", mew=1.3, lw=1.8)
# upper-bound panel: shade the overshoot (c>1) region red
au.axhspan(1.0, 1.7, color="#fde3e3", zorder=0); au.axhline(1.0, color="0.35", lw=1.0)
au.set_ylim(0.95, 1.58); au.set_ylabel("field maximum")
au.text(ts[0], 1.49, r"overshoot region ($c>1$)", fontsize=9, color="#b22222")
au.text(ts[-1], 1.01, "$c=1$", ha="right", va="bottom", fontsize=9, color="0.35")
# lower-bound panel: shade the undershoot (c<0) region red
al.axhspan(-0.75, 0.0, color="#fde3e3", zorder=0); al.axhline(0.0, color="0.35", lw=1.0)
al.set_ylim(-0.66, 0.06); al.set_ylabel("field minimum"); al.set_xlabel("$t$")
al.text(ts[0], -0.46, r"undershoot region ($c<0$)", fontsize=9, color="#b22222")
al.text(ts[-1], -0.012, "$c=0$", ha="right", va="top", fontsize=9, color="0.35")
handles = [Line2D([0], [0], color=c, marker=(None if m == "None" else m), ms=6,
                  mfc="white", mew=1.3, lw=1.8, label=l) for l, c, m, k in spec]
au.legend(handles=handles, fontsize=9, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.03), frameon=True)
fig.savefig(f"{OUT}/fig6_boundedness_time.png", dpi=300, bbox_inches="tight"); plt.close()

print("wrote figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
