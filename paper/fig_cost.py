"""Cost figure of the revised manuscript: smooth-tracer error at maximum deformation against compiled wall-clock time,
one point per grid 33^2 to 257^2; (a) at the benchmark Courant number (timing_benchmark.json), (b) the bounded compact
scheme and the Chaplin-Colella algorithm, each at its largest admissible time step (cc_cost.json). Run with:
   uv run --with numpy --with matplotlib python3 fig_cost.py
Writes figures/fig_rev_cost.pdf and .png."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as P

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "figure.dpi": 300})
STY = {"cfctg": ("Bounded compact (this work)", "#2a78d6", "o", "-"),
       "u5fctg": ("U5, same limiter", "#7d7c76", "P", "-"), "c4fctg": ("C4, same limiter", "#7d7c76", "X", "--"),
       "muscl_fv": ("MUSCL-superbee", "#eb6834", "D", "-"),
       "mpdata3_no": ("MPDATA, two corrective passes", "#1baf7a", "s", "-"), "hsimt_split": ("HSIMT, Strang split", "#eda100", "^", "-")}
T = json.load(open("timing_benchmark.json"))["timing"]
Ns = sorted(T, key=int)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
for s, (lab, col, mk, ls) in STY.items():
    t = [T[N][s]["seconds"] for N in Ns]; e = [T[N][s]["rel_l2"] for N in Ns]
    hl = s == "cfctg"
    ax.loglog(t, e, ls, color=col, marker=mk, ms=5 if hl else 4, lw=2.0 if hl else 1.1, mec="white", mew=0.6,
              label=lab, zorder=4 if hl else 3)
    if hl:
        for N, x, y in zip(Ns, t, e):
            ax.annotate(f"${N}^2$", (x, y), xytext=(-6, -11), textcoords="offset points", fontsize=7, color="#52514e",
                        ha="right")
L = json.load(open("cc_cost.json"))["timing"]["largest"]
LN = sorted(L, key=int)
CCS = {"cfctg": ("Bounded compact, donor-cell step", "#2a78d6", "o", "-"),
       "cc17_u5": ("Chaplin-Colella, U5 faces, CTU step", "#8e5ac8", "v", "-"),
       "cc17_c4": ("Chaplin-Colella, C4 faces, CTU step", "#8e5ac8", "<", "--")}
for s, (lab, col, mk, ls) in CCS.items():
    t = [L[N][s]["seconds"] for N in LN]; e = [L[N][s]["rel_l2"] for N in LN]
    hl = s == "cfctg"
    bx.loglog(t, e, ls, color=col, marker=mk, ms=5 if hl else 4, lw=2.0 if hl else 1.1, mec="white", mew=0.6,
              label=lab, zorder=4 if hl else 3)
    if hl:
        for N, x, y in zip(LN, t, e):
            bx.annotate(f"${N}^2$", (x, y), xytext=(-6, -11), textcoords="offset points", fontsize=7, color="#52514e",
                        ha="right")
for a, tag in ((ax, "(a) benchmark Courant number"), (bx, "(b) largest admissible step")):
    a.grid(True, which="major", color="#e1e0d9", lw=0.5); a.set_axisbelow(True)
    a.set_xlim(right=40); a.set_ylim(bottom=1.5e-3)
    a.set_xlabel("wall-clock time to $t=T/2$ (s, compiled, one thread)"); a.set_title(tag, loc="left")
    a.legend(loc="lower left", frameon=False)
ax.set_ylabel(r"relative $L_2$ error")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{P.FIGURES}/fig_rev_cost.{ext}", bbox_inches="tight")
print("wrote fig_rev_cost.pdf/.png")
import figure_provenance as FP
FP.record("fig_cost.py")
