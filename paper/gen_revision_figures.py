"""Figures for the revised manuscript (control-volume-average benchmark; recalibrated bay). Run with:
   uv run --with numpy --with scipy --with matplotlib python3 gen_revision_figures.py
Reads cv_results.json, cfl_sweep_cv.json, sfbay_table.json and sfbay_table_*.npy; reruns the 129^2 slotted-cylinder
fields once from control-volume averages (cached in figure_data/slotted_cv_129.npz). Writes PDF (for LaTeX) and PNG to
figures/. Figures: (1) error vs grid size, (2) slotted cylinder at maximum deformation, (3) San Francisco Bay patch after
five hours, (4) error vs the largest outflow fraction."""
import json, os, zipfile
import paths as P
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.6,
    "lines.linewidth": 1.2, "figure.dpi": 300, "savefig.dpi": 300, "axes.edgecolor": "#52514e",
})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); CACHE = os.path.join(HERE, "figure_data")
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
INK2, GRID = "#52514e", "#e1e0d9"
# scheme key -> (label, colour, marker); categorical slots 1-5, validated (adjacent pairs) on white
# the three HSIMT time integrations share one hue and differ by line style
STY = {"cfctg": ("Bounded compact (this work)", "#2a78d6", "o", "-"),
       "u5fctg": ("U5, same limiter", "#7d7c76", "P", "-"),
       "c4fctg": ("C4, same limiter", "#7d7c76", "X", "--"),
       "cc17_u5": ("Chaplin-Colella (U5)", "#2b2a27", "*", "-"),
       "cc17_c4": ("Chaplin-Colella (C4)", "#2b2a27", "*", "--"),
       "muscl_fv": ("MUSCL-superbee", "#eb6834", "D", "-"),
       "mpdata3_no": ("MPDATA, two corrective passes", "#1baf7a", "s", "-"),
       "hsimt_split": ("HSIMT, Strang split", "#eda100", "^", "-"),
       "hsimt_nopred": ("HSIMT, two-level", "#eda100", "^", "--"),
       "hsimt": ("HSIMT, ROMS predictor", "#eda100", "^", ":"),
       "donor_fv": ("Donor cell", "#e87ba4", "v", "-")}
HS_SHORT = {"hsimt_split": "HSIMT (split)", "hsimt_nopred": "HSIMT (two-level)", "hsimt": "HSIMT (predictor)"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                 "#256abf", "#184f95", "#0d366b"])
FLAG = "#eb6834"   # values more than 0.01 outside [0, 1]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.5); ax.set_axisbelow(True)


R = json.load(open(f"{HERE}/cv_results.json"))
O5 = json.load(open(f"{HERE}/order_513.json"))          # 513^2: smooth tracer, compact, C4 and U5 only
for t_, recs in json.load(open(f"{HERE}/cc_results.json"))["results"].items():   # complete Chaplin-Colella (2017)
    for N_, rec in recs.items():
        R[t_][N_].update({k: rec[k] for k in ("cc17_c4", "cc17_u5")})


# ---- Fig 1: relative L2 error vs grid size ----
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, test, title in ((axs[0], "gauss_T2", "(a) Smooth tracer"), (axs[1], "slotted_T2", "(b) Slotted cylinder")):
    for k, (lab, col, mk, ls) in STY.items():
        Ns = sorted(int(n) for n in R[test]); e = [R[test][str(n)][k]["rel_l2"] for n in Ns]
        if test == "gauss_T2" and k in O5["gauss_T2"]:
            Ns.append(513); e.append(O5["gauss_T2"][k]["rel_l2"])
        hl = k == "cfctg"
        ax.loglog(Ns, e, ls, color=col, marker=mk, ms=5 if hl else 4, lw=2.0 if hl else 1.1,
                  mfc=col if ls == "-" else "white", mec="white" if ls == "-" else col, mew=0.6, label=lab,
                  zorder=4 if hl else 3)
    ax.set_title(title, loc="left"); style(ax); ax.set_xlabel("grid")
axs[0].set_ylabel(r"relative $L_2$ error at $t=T/2$")
for x, p, y0, txt in ((np.array([40, 110.0]), 1, 0.012, "1st order"), (np.array([40, 110.0]), 2, 0.012, "2nd order"),
                      (np.array([257, 513.0]), 4, 4e-3, "4th order")):     # slope guides (plot, not loglog: keeps the ticks)
    axs[0].plot(x, y0 * (x / x[0]) ** -p, ":", color="#898781", lw=0.8)
    axs[0].text(x[1] * 1.05, y0 * (x[1] / x[0]) ** -p, txt, fontsize=7, color=INK2, va="center")
for ax, test in ((axs[0], "gauss_T2"), (axs[1], "slotted_T2")):
    Ns = sorted(int(n) for n in R[test]) + ([513] if test == "gauss_T2" else [])
    ax.xaxis.set_major_locator(FixedLocator(Ns)); ax.xaxis.set_major_formatter(FixedFormatter([f"${n}^2$" for n in Ns]))
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
axs[0].set_xlim(28, 760); axs[0].set_ylim(1e-4, 1.2)
axs[1].set_xlim(55, 300); axs[1].set_ylim(0.15, 0.9)
axs[1].set_yticks([0.2, 0.3, 0.4, 0.6, 0.8]); axs[1].set_yticklabels(["0.2", "0.3", "0.4", "0.6", "0.8"])
axs[1].yaxis.set_minor_formatter(NullFormatter())
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.22))
fig.tight_layout()
save(fig, "fig_rev_convergence")

# ---- Fig 2: slotted cylinder at maximum deformation, 129^2, control-volume averages ----
NS = 129; cache = f"{CACHE}/slotted_cv_{NS}.npz"
HS_BEST = min(HS_SHORT, key=lambda k: R["slotted_T2"][str(NS)][k]["rel_l2"])     # best HSIMT update on this grid
KEYS = ("cfctg", "muscl_fv", "mpdata3_no", HS_BEST, "donor_fv")
CODE = R["slotted_T2"][str(NS)]["cfctg"]["code"]                                 # results' code version
os.environ["CV_SUB"] = "16"
import gen_mpdata as M, cv_reference as CV
assert CV.CODE == CODE, "the code has changed since cv_results.json"            # also when the cache is reused
g = M.Grid(NS)
c0 = CV.cv_average(g, M.slot_ic, 0.0, m=16); ref = CV.cv_average(g, M.slot_ic, M.T / 2, m=16)
l2 = lambda F, k: M.score(F[k], ref, c0, g, 0.0)["rel_l2"]


METRICS = ("rel_l2", "rel_l1", "min", "max", "mass_rel", "iou")


def cache_ok():
    """The cache holds this code version's fields on this grid, with this reference, finite and of the grid's shape,
    with the metrics of cv_results.json (relative L2 and L1 error, extrema, mass change, IoU); a missing entry or an
    unreadable file also counts as stale."""
    try:
        with np.load(cache) as F:
            if not (set(KEYS) | {"code", "edges", "exact"}) <= set(F.files) or str(F["code"]) != CODE:
                return False
            if not (np.array_equal(F["edges"], g.e) and np.array_equal(F["exact"], ref)):
                return False
            for k in KEYS:
                if F[k].shape != ref.shape or not np.isfinite(F[k]).all():
                    return False
                d, j = M.score(F[k], ref, c0, g, 0.0), R["slotted_T2"][str(NS)][k]
                a, b = F[k] > 0.5, ref > 0.5                                    # IoU on control volumes, as cv_reference
                d["iou"] = float((a & b).sum() / max((a | b).sum(), 1))
                if any(abs(d[q] - j[q]) > 1e-12 for q in METRICS):
                    return False
            return True
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile):
        return False


if os.path.exists(cache) and not cache_ok():                                    # checked on every run, also on reuse
    print("recomputing the slotted-cylinder fields (cache stale or altered)", flush=True); os.remove(cache)
if not os.path.exists(cache):
    fields = {"exact": ref, "edges": g.e, "code": CODE}
    for k in KEYS:
        fields[k] = CV.run_from(NS, k, c0, M.T / 2)[0]
    np.savez(cache, **fields)
    assert cache_ok(), "figure field differs from cv_results.json"
F = np.load(cache); e = F["edges"]; xc = np.linspace(0, 1, NS)
for k in KEYS:
    print(f"{k}: L2 {l2(F, k):.4f} (json {R['slotted_T2'][str(NS)][k]['rel_l2']:.4f})", flush=True)
fig, axs = plt.subplots(2, 3, figsize=(7.0, 5.0))
for ax, k in zip(axs.flat, ("exact",) + KEYS):
    im = ax.pcolormesh(e, e, F[k], cmap=SEQ, vmin=0, vmax=1, shading="flat", rasterized=True)
    ax.contour(xc, xc, F["exact"], levels=[0.5], colors="#0b0b0b", linewidths=0.5)
    if k == "exact":
        ax.set_title("Reference solution", fontsize=9)
    else:
        m = R["slotted_T2"][str(NS)][k]
        short = {"cfctg": "Bounded compact", "muscl_fv": "MUSCL-superbee", "mpdata3_no": "MPDATA (2 corrective)",
                 "donor_fv": "Donor cell", **HS_SHORT}[k]
        ax.set_title(f"{short}\n$L_2$ = {m['rel_l2']:.3f}, IoU = {m['iou']:.2f}", fontsize=8.5)
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1])
cb.set_label("tracer concentration"); cb.outline.set_linewidth(0.4)
save(fig, "fig_rev_slotted")

# ---- Fig 3: San Francisco Bay, released patch after five hours ----
f = np.load(os.path.join(P.SFBAY, "fluxes_mixed.npz"))
wet, lat, lon, wi, wj, isopen = f["wet"], f["lat"], f["lon"], f["wi"], f["wj"], f["isopen"]
interior = ~isopen
B = json.load(open(f"{HERE}/sfbay_table.json"))
i0 = int(np.argmin(np.abs(lon - B["release_lon"]))); j0 = int(np.argmin(np.abs(lat - B["release_lat"])))
dist = np.sqrt(((lon[wi] - lon[i0]) * 111 * np.cos(np.radians(37.8))) ** 2 + ((lat[wj] - lat[j0]) * 111) ** 2)
c0 = ((dist <= 3.5) & interior).astype(float)
panels = [("Initial patch", c0, None),
          ("Donor cell", np.load(f"{HERE}/sfbay_table_donor.npy"), "donor cell"),
          ("Bounded compact (this work)", np.load(f"{HERE}/sfbay_table_ccfctg.npy"), "bounded compact, global bounds"),
          ("Compact with dissipation, unlimited", np.load(f"{HERE}/sfbay_table_ccompact_d.npy"), "compact with dissipation, unlimited")]
cmap = SEQ.copy(); cmap.set_under(FLAG); cmap.set_over(FLAG)
LON0, LON1, LAT0, LAT1 = -122.63, -122.25, 37.71, 37.89
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9))
for ax, (ti, c, key) in zip(axs.flat, panels):
    G = np.full(wet.shape, np.nan); v = np.where(interior, c, np.nan)
    v = np.where((v < 0) & (v >= -0.01), 0.0, v); v = np.where((v > 1) & (v <= 1.01), 1.0, v)   # round-off is not a violation
    G[wj, wi] = v
    im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(G), cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contourf(lon, lat, (~wet).astype(float), levels=[0.5, 1.5], colors="#d9d8d2")
    ax.contour(lon, lat, wet.astype(float), levels=[0.5], colors=INK2, linewidths=0.4)
    ax.plot(lon[i0], lat[j0], "*", ms=8, mfc="white", mec="#0b0b0b", mew=0.6, zorder=6)
    fmt = lambda a: "0" if abs(a) < 5e-3 else f"{a:.2f}".replace("-", "−")
    if key is None:
        sub = f"{int((c > 0.5).sum())} cells with $c>0.5$"
    else:
        r = B[key]; sub = f"{r['cells_c_gt_0p5']} cells with $c>0.5$; range [{fmt(r['final_min'])}, {fmt(r['final_max'])}]"
    ax.set_title(f"{ti}\n{sub}", fontsize=8.5)
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_aspect(1 / np.cos(np.radians(37.8))); ax.set_xticks([]); ax.set_yticks([])
ax = axs[1, 1]
km = 5 / (111 * np.cos(np.radians(37.8))); xb, yb = LON1 - 0.02 - km, LAT0 + 0.012
ax.plot([xb, xb + km], [yb, yb], color="#0b0b0b", lw=1.5); ax.text(xb + km / 2, yb + 0.004, "5 km", ha="center", fontsize=7)
cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, ticks=[0, 0.5, 1], extend="both")
cb.set_label(r"concentration (orange: $>0.01$ outside $[0,1]$)"); cb.outline.set_linewidth(0.4)
save(fig, "fig_rev_sfbay")

# ---- Fig 4: error vs the largest outflow fraction (129^2) ----
Cf = json.load(open(f"{HERE}/cfl_sweep_cv.json"))
cfl = sorted(Cf, key=float); of = [Cf[k]["outflow_fraction"] for k in cfl]
SER = (("cfctg", "Bounded compact, global bounds", "#2a78d6", "o", "-"), ("cfct", "Bounded compact, local bounds", "#2a78d6", "o", "--"),
       ("muscl_fv", "MUSCL-superbee", "#eb6834", "D", "-"), ("mpdata3_no", "MPDATA, two corrective passes", "#1baf7a", "s", "-"),
       ("hsimt_split", "HSIMT, Strang split", "#eda100", "^", "-"), ("hsimt_nopred", "HSIMT, two-level", "#eda100", "^", "--"),
       ("hsimt", "HSIMT, ROMS predictor", "#eda100", "^", ":"))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
YMAX = {"gauss_T2": 0.55, "slotted_T2": 0.6}
for ax, test, title in ((axs[0], "gauss_T2", "(a) Smooth tracer"), (axs[1], "slotted_T2", "(b) Slotted cylinder")):
    for s, lab, col, mk, ls in SER:
        e = np.array([Cf[k][f"{test}_{s}"]["rel_l2"] for k in cfl])
        off = e > YMAX[test]                       # off-scale points are dropped and labelled, not drawn as a spike
        ax.plot(np.array(of)[~off], e[~off], ls, color=col, marker=mk, ms=4,
                mfc=col if ls == "-" else "white", mec="white" if ls == "-" else col, mew=0.6, label=lab)
        for x, v in zip(np.array(of)[off], e[off]):
            ax.plot(x, YMAX[test], marker="^", ms=5, color=col, mfc="white", mew=0.8, clip_on=False, zorder=5)
            ax.annotate(f"{v:.1f} (off scale)", (x, YMAX[test]), xytext=(-6, -2), textcoords="offset points",
                        ha="right", va="top", fontsize=7, color=INK2)
    ax.axvline(1.0, color="#898781", ls="--", lw=0.8)
    ax.set_xlim(0, 1.25); ax.set_ylim(0, YMAX[test]); ax.set_title(title, loc="left"); style(ax)
    ax.set_xlabel(r"largest outflow fraction $\phi$")
axs[0].set_ylabel(r"relative $L_2$ error")
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2))
save(fig, "fig_rev_cfl")
import figure_provenance as FP
FP.record("gen_revision_figures.py")
