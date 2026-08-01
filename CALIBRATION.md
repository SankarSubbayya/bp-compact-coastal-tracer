# San Francisco Bay tidal hydrodynamic model — calibration documentation

This documents the depth-averaged tidal-current model used to generate the carrier
flow for the tracer-transport demonstration in the manuscript. The currents are
produced by a **single production simulation** that records the surface elevation
and depth-averaged current at all stations simultaneously; the only auxiliary runs
are two short single-constituent integrations used once to measure the friction/forcing
gains (Step 5 below).

All code is in `paper/` and is run with
`uv run --with numpy --with scipy [--with matplotlib] python3 <script>`.
The bathymetry is fetched once with `--with netcdf4`.

---

## 1. Domain and bathymetry
- **Source:** NOAA NCEI Coastal Relief Model, Volume 7 (Central Pacific), 3 arc-second
  (~90 m), retrieved via OPeNDAP (`build_sfbay_fine.py` reads `crm_raw.npz`, fetched
  from `https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol7.nc`).
- **Box:** lon −122.70…−121.95, lat 37.40…38.25.
- **Model grid:** regridded (bilinear) to a uniform **200 m** mesh (474 × 330; dlat
  = 0.00180°, dlon = 0.00227°), resolving the ~90 m-deep Golden Gate strait (~15
  cells across).
- **Wet mask:** cells below MSL connected to the Pacific (flood fill from the west
  edge); 51 477 wet cells (32.9 % of the box).

## 2. Governing equations and numerics
Linear depth-averaged (2-D) shallow-water equations solved with the **semi-implicit
θ-method of Casulli (1990)**: the free-surface gravity-wave terms (surface-gradient
in momentum, flux divergence in continuity) are treated implicitly (θ = 0.6),
removing the surface-wave CFL limit; the resulting symmetric Helmholtz system for the
free surface is factored once (sparse LU) and back-substituted each step. Bottom
friction is treated semi-implicitly. Time step Δt = 120 s.

## 3. Bottom friction (depth-dependent, Manning)
Drag coefficient `Cd = g n² / H^(1/3)` with **Manning n = 0.037**:
- deep channels (Golden Gate, H ≈ 87 m): Cd ≈ 0.0030 — consistent with the
  0.003–0.004 calibrated by Sankaranarayanan & French McCay (2003);
- shallow flats (H ≈ 3–5 m): Cd ≈ 0.008–0.009 — the enhanced drag damps the
  otherwise spurious shallow-water resonance.

## 4. Minimum depth (no wetting/drying)
A minimum working depth **HMIN = 3 m** is imposed (`H = max(depth, HMIN)`). The model
is linear (no wetting/drying), so cells must not dry: the maximum modelled surface
depression is **2.73 m**, leaving a minimum water column of HMIN − 2.73 = **0.27 m
(> 0)** everywhere over the spring cycle, so no cell dries and the integration is
stable. A **Rayleigh sponge** (Cd-like linear damping, rate RMAX = 6 × 10⁻³ s⁻¹,
ramped ∝ (distance)² south of 37.76° N) absorbs the reflection off the artificial
closed southern boundary in the under-resolved far South Bay.

## 5. Open-boundary tidal forcing — mixed tide
San Francisco Bay has a **mixed, predominantly semidiurnal** tide. The Pacific open
boundary (west edge) is forced with the six dominant constituents, amplitudes/phases
from the **NOAA Golden Gate harmonic constants (station 9414290)**:

| Const | Band | Speed (°/h) | NOAA amp (m) | NOAA phase °(G) |
|-------|------|-------------|--------------|-----------------|
| M2 | semidiurnal | 28.98410 | 0.576 | 208.2 |
| S2 | semidiurnal | 30.00000 | 0.137 | 216.2 |
| N2 | semidiurnal | 28.43973 | 0.122 | 183.2 |
| K1 | diurnal     | 15.04107 | 0.370 | 225.4 |
| O1 | diurnal     | 13.94304 | 0.230 | 208.4 |
| P1 | diurnal     | 14.95893 | 0.114 | 222.1 |

η_open(t) = Σ A′_n cos(ω_n t − g_n), with phases g_n the NOAA Greenwich epochs.

**Calibration of the offshore amplitudes A′_n:** the open boundary lies offshore of
the Gate, so the bay amplifies the forcing. Two short single-constituent runs (M2
alone, K1 alone) measure the per-band gain (Gate amplitude / offshore amplitude):
- semidiurnal gain ≈ 1.08,  diurnal gain ≈ 1.06.
Each offshore amplitude is then set to `A′_n = (NOAA amp)/gain_band × SD_CORR`, with a
small semidiurnal correction **SD_CORR = 1.10** that restores the M2 amplitude lost
to nonlinear (quadratic-drag) coupling with the strong diurnal currents.

| Const | NOAA amp (m) | offshore A′ (m) | model Gate elev (m) |
|-------|--------------|------------------|----------------------|
| M2 | 0.576 | 0.586 | 0.57 |
| S2 | 0.137 | 0.139 | 0.14 |
| N2 | 0.122 | 0.124 | (not separable)* |
| K1 | 0.370 | 0.348 | (not separable)* |
| O1 | 0.230 | 0.216 | 0.22 |
| P1 | 0.114 | 0.107 | (not separable)* |

\* A ~16-day record cannot separate N2 from M2 (Rayleigh period 27.5 d) or P1/K2
from K1 (≥ 183 d); these are forced at their observed amplitudes but their fitted
values leak from the dominant neighbour. The cleanly separable constituents (M2, S2,
O1) match observations.

## 6. Validation against observations

**Surface elevation (Golden Gate), model vs NOAA 9414290:**
- M2 0.57 / 0.58 m,  S2 0.14 / 0.14 m,  O1 0.22 / 0.23 m
- **Form factor** F = (K1+O1)/(M2+S2) ≈ **0.95** → mixed, predominantly semidiurnal;
  the modelled tide shows the **diurnal inequality** (Fig. `fig_sfbay_tide.png`).

**Principal M2 depth-averaged current amplitude, model vs observed**
(observed from Cheng et al. 1993, via Sankaranarayanan & French McCay 2003):

| Station | observed (m/s) | model (m/s) |
|---------|----------------|-------------|
| Golden Gate (C1) | 1.1 | 0.96 |
| Oakland | 0.6 | 0.53 |
| Richmond | (validated) | 0.70 |
| San Pablo Bay | — | 0.34 |

All within the ~0.2 m/s current error reported for the validated 2003 model. Station
time series in `fig_sfbay_currents.png` (one simulation) show the mixed-tide
modulation and the spatial decrease from the Gate into the bay.

## 7. Parameter summary

| Parameter | Value |
|-----------|-------|
| Grid resolution | 200 m (474 × 330) |
| Time step Δt | 120 s |
| Implicitness θ | 0.6 |
| Manning n | 0.037 |
| Minimum depth HMIN | 3 m |
| Sponge rate RMAX | 6 × 10⁻³ s⁻¹ south of 37.76° N |
| Spin-up | 3 days |
| Forcing constituents | M2, S2, N2, K1, O1, P1 (NOAA 9414290) |

## 8. Reproducibility
`build_sfbay_fine.py` (grid) → `gen_currents_mixed.py` (one production simulation;
writes `currents_fine.npz`, `sfbay_tides.npz`, `sfbay_stations.npz`,
`sfbay_calibration.npz`) → figure scripts `sfbay_domain.py`, `sfbay_currents_fig.py`,
`sfbay_tides_fig.py`, `sfbay_transport.py`.

## 9. Scope and limitations
This is a **tidal hydrodynamic** model providing a realistic carrier flow. It is **not
an oil-spill hindcast**: the observed Cosco Busan oil trajectory was wind-driven
(windage ≈ 3 % of wind speed, plus Stokes drift and weathering), and the winds of
7–13 November 2007 are not included. The tracer here is passive and tide-advected,
and the SF Bay case demonstrates the numerical transport scheme — it does not
reproduce the observed slick.

---
*This study — model development and calibration, simulations, figures, and manuscript
— was carried out using the large language model Claude Opus 4.8 (Anthropic) under the
author's direction and review.*
