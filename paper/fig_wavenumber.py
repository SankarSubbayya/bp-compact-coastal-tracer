"""Fig. 1 of the revised manuscript (modified wavenumber and dispersion error of first-derivative approximations), the
figure-1 block of gen_plots.py on its own so that it can be written as vector PDF without rerunning that script's
legacy simulations. Run with:  uv run --with numpy --with matplotlib python3 fig_wavenumber.py
Writes figures/fig1_wavenumber_analysis.pdf and .png (300 dpi)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as P

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 11,
    "axes.labelsize": 11, "legend.fontsize": 9, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "figure.dpi": 300,
})
STY = {"central2": ("C0", "-"), "central4": ("C1", "-"), "central6": ("C2", "-"),
       "compact4": ("C3", "--"), "compact6": ("C4", "--")}


def mw1(kh, nm):
    s = np.sin
    return {"central2": s(kh), "central4": (8*s(kh)-s(2*kh))/6,
            "central6": (45*s(kh)-9*s(2*kh)+s(3*kh))/30,
            "compact4": 1.5*s(kh)/(1+0.5*np.cos(kh)),
            "compact6": ((14/9)*s(kh)+(1/18)*s(2*kh))/(1+(2/3)*np.cos(kh))}[nm]


kh = np.linspace(1e-3, np.pi, 500)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
a1.plot(kh, kh, "k--", lw=2, label="exact (spectral)")
for nm in STY:
    a1.plot(kh, mw1(kh, nm), STY[nm][1], color=STY[nm][0], label=nm)
a1.set_xlabel(r"$k\,h$"); a1.set_ylabel(r"modified wavenumber $k'\,h$")
a1.set_title("(a) First-derivative resolution"); a1.set_xlim(0, np.pi); a1.set_ylim(0, np.pi)
a1.legend(loc="upper left"); a1.grid(alpha=.3)
for nm in STY:
    a2.semilogy(kh, np.abs(mw1(kh, nm)-kh)+1e-16, STY[nm][1], color=STY[nm][0], label=nm)
a2.axhline(1e-2 * np.pi, color="0.5", lw=.8, ls=":")                 # |k'h - kh| / pi = 0.01, as defined in the text
a2.text(0.1, 1.3e-2 * np.pi, r"1% of $\pi$", fontsize=8, color="0.4")
a2.set_xlabel(r"$k\,h$"); a2.set_ylabel(r"dispersion error $|k'h - kh|$")
a2.set_title("(b) Dispersion error (resolving efficiency)"); a2.set_xlim(0, np.pi); a2.set_ylim(1e-7, 3)
a2.legend(loc="lower right"); a2.grid(alpha=.3, which="both")
fig.tight_layout()
os.makedirs(P.FIGURES, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{P.FIGURES}/fig1_wavenumber_analysis.{ext}", dpi=300)
plt.close()
print("wrote fig1_wavenumber_analysis.pdf/.png")
import figure_provenance as FP
FP.record("fig_wavenumber.py")
