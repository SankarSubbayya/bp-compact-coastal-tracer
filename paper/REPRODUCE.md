# Reproducing the revised manuscript

Python 3.12+, numpy, scipy, matplotlib (`uv run --with numpy --with scipy --with matplotlib python3 <script>`).
Run from this folder. The helper modules in `../_dev/` (advdiff.py, sharp.py, compact_stable.py) and `gen_deepen.py`
are imported by the benchmark drivers. The bay inputs live in `../_dev/sfbay_fine/`. `paths.py` resolves both folders
relative to this one (override with CAD_ROOT). The self-test block of `_dev/compact_stable.py` (never run by the
pipeline) still names an absolute path; it is left unchanged because the file is part of the code hash recorded in
every benchmark result. For the same reason the `__main__` blocks of `gen_mpdata.py` and `gen_conservative.py`, the
superseded point-sampled drivers of an earlier version (they write `mpdata_results.json`, no longer used), are left as
they were; Appendix A is produced by `appendix_nodal.py`.

| Order | Script | Produces | Used for |
|---|---|---|---|
| 1 | `fetch_crm.py`, `build_sfbay_fine.py` | `../_dev/sfbay_fine/sfbay_fine.npz` (200 m grid, CRM bathymetry) | bay model |
| 2 | `recalibrate_mixed.py` (~5 min) | `sfbay_calibration.npz`, `sfbay_tides.npz`, `sfbay_stations.npz`, `currents_fine.npz`; `recalibration_results.json` | Tables tide and currents; Figs domain, tide, currents |
| 3 | `gen_fluxes_mixed.py` (6 h window) and `WIN_H=25 FLUX_OUT=fluxes_mixed_25h.npz python3 gen_fluxes_mixed.py` | `fluxes_mixed.npz`, `fluxes_mixed_25h.npz`; `fluxes_mixed.log` | offline transport |
| 4 | `bay_table.py` | `sfbay_table.json`, `sfbay_table_*.npy` | Table bay, Fig bay |
| 5 | `bay_experiments.py`; `coast_closure_check.py` | `bay_experiments.json` (coastline stability, sharp-release ablation, outflow test); `coast_closure_check.json` (one-sided closures exact for cell averages) | Table coast; export text |
| 6 | `long_run.py` (uses `fluxes_mixed_25h.npz`) | `long_run.json` (24 h, with the range after every record) | coastline text |
| 7 | `appendix_nodal.py` (the eight schemes of the Appendix A table, point-sampled protocol) | `appendix_nodal.json` | Appendix A |
| 8 | `cv_reference.py` (CV_SUB=16 by default; the 18 schemes of `cv_reference.SCHEMES`) | `cv_results.json` (every record carries cv_subcells, trajectory_steps, cfl, dt, steps, wall_closure, code hash) | Tables smooth and slotted, Fig convergence |
| 9 | `cfl_sweep_cv.py` (same schemes and quadrature as step 8) | `cfl_sweep_cv.json` | Table and Fig time step |
| 10 | `reference_sensitivity.py 65,129` then `reference_sensitivity.py 257` | `reference_sensitivity.json` (m = 16 vs 32; m = 8 vs 16 and 400 vs 800 steps at 65^2) | Sec. 2 |
| 10e | `ranking_check.py` | `ranking_check.json` (the main comparison schemes, not the Chaplin-Colella implementation, with m = 32 data and reference on 65^2 and 129^2) | Sec. 2 ranking statement |
| 10a | `wall_convergence.py` | `wall_convergence.json` | Table wall |
| 10b | `hsimt_consistency.py` | `hsimt_consistency.json` | Table HSIMT consistency |
| 10c | `mpdata_validation.py` (needs `--with PyMPDATA==1.7.3`) | `mpdata_validation.json` | MPDATA validation sentence |
| 10d | `uniform_tracer_check.py`; `bay_checks.py` | `checks.json` (uniform tracer in the bay; station distances; fit-window sensitivity) | Secs. 5.4.2-5.4.3 |
| 10f | `closure_spectra.py`; `covariance_test.py` | `closure_spectra.json`; `covariance_test.json` | Sec. 3.2 (closure stability; flux covariance) |
| 10g | `order_513.py` (~1 h) | `order_513.json` | Sec. 5.2 (513^2 order) |
| 10h | `timing_benchmark.py` (needs `--with numba`; compiled `fast_schemes.py`, validated against the NumPy schemes) | `timing_benchmark.json` | Sec. 5.8, Fig. cost |
| 10i | `covariance_ablation.py` (needs `--with numba`; ~40 min, caches references in figure_data/) | `covariance_ablation.json` | Sec. 5.2 (transverse correction) |
| 10j | `correction_cost.py` (needs `--with numba`; after 10i) | `correction_cost.json` | Sec. 5.2 (cost of the correction) |
| 10k | `changing_volume.py` (~40 min) | `changing_volume.json` (time-varying volumes; direct and depth-aware face values) | Sec. 5.4, Table moving |
| 10l | `variable_depth.py` (~30 min); `depth_test.py` | `variable_depth.json` (steady variable depth); `depth_test.json` (manufactured operator test with variable depth) | Sec. 3.2 (depth weighting), Sec. 5.5, Table depth |
| 10n | `steep_depth.py` (~2 h) | `steep_depth.json` (steep shelf, direct and depth-aware face values; noise growth of a uniform tracer) | Sec. 3.2 (energy rate), Sec. 5.6, Table steep |
| 10n2 | `steep_depth.py --schemes cfctg,u5fctg,cfctgd --out steep_depth_dissip.json` (~15 min) | `steep_depth_dissip.json` (the dissipation term (15) added to the bounded compact scheme on the shelf; the cfctg and u5fctg runs repeat those of `steep_depth.json` exactly) | Sec. 3.4, Sec. 5.6, Table steep, Conclusions |
| 10n3 | `shelf_reference_check.py` (~10 min) | `shelf_reference_check.json` (the 129^2 shelf reference with 12x12 Gauss points per volume instead of the 16x16 midpoint rule; the three bounded schemes of step 10n2 rescored against both) | Sec. 5.6 |
| 10m | `cc_benchmark.py` (~25 min); `cc_validation.py` (~20 min); `cc_range.py`; `cc_sensitivity.py` (~40 min, after `cc_benchmark.py` and `cc_cost.py`) | `cc_results.json` (our implementation of the Chaplin-Colella 2017 algorithm, `chaplin_colella.py`, on the benchmark data); `cc_validation.json` (the article's periodic tests, with sigma on the speed and on the largest velocity component, and the one-dimensional tests); `cc_range.json`; `cc_sensitivity.json` (the benchmark under the other readings of the article's eqs. 30 and 78: `chaplin_colella.SIX_FACE_AVG`, `EQ78_H`) | Sec. 3 comparison schemes, Sec. 5.2, Sec. 5.8, Tables smooth and slotted |
| 10o | `cc_cost.py` (needs `--with numba`; compiled `fast_cc.py`, validated against `chaplin_colella.py`; single thread, load logged in `jcp6_runs.log`) | `cc_cost.json` (largest admissible steps; compiled timing at the benchmark step and at each scheme's largest admissible step) | Sec. 5.8 (cost) |
| 11 | `fig_wavenumber.py` (Fig 1), `gen_revision_figures.py`, `fig_cost.py`, `sfbay_domain.py`, `sfbay_tides_fig.py`, `sfbay_currents_fig.py` (after every result step above) | `figures/`; `figure_inputs.json` (`figure_provenance.py`: SHA-256 of every declared input of each script and of every figure it wrote; `fluxes_mixed.npz` is compared only when present; the slotted-field cache in `figure_data/` is not recorded but checked by `gen_revision_figures.py` on every run) | figures |
| 12 | `make_tables.py` (needs numpy: reads figure_data/gauss_m16_*.npz; after step 11: it refuses to run unless every figure script is recorded, its inputs are unchanged and its figures are the files it wrote) | `numbers.tex`, `tables/*.tex` | every number and results table |
| 13 | Manuscript build; not in the deposited release, which ends with step 12 (`numbers.tex`, `tables/`, `figures/` are the generated inputs of the manuscript) | PDF | manuscript |

Scheme keys: cfctg / cfct = bounded compact (global / local bounds); cfctgd / cfctd = with the optional dissipation;
ccompactd = unlimited compact with dissipation; mpdata3_no / mpdata_no = nonoscillatory MPDATA with two / one corrective
passes; hsimt / hsimt_nopred / hsimt_split = HSIMT with the ROMS predictor / two-level / Strang split; muscl_fv = flux-form MUSCL-superbee; donor_fv =
donor cell; cc17_c4 / cc17_u5 = complete Chaplin-Colella (2017) algorithm; cfcte = extremum-preserving local bounds; cfctg_nt = without the transverse term; c4fctg / u5fctg = explicit
centred fourth-order / upwind fifth-order reconstructions under the same limiter; every scheme reconstructs the stored
concentration directly, and suffix _da = depth-aware ratio face values instead (differs only where the depth varies).
Bay: ccfctg / ccfct (DISSIPATION=False) = bounded compact; ccompact = unlimited; DEPTH_AWARE=True = depth-aware ratio.

`rerun_direct.sh` reruns every benchmark and bay driver on one code hash (the compiled runs last);
`rerun_timing.sh` then reruns the three compiled timing scripts once the 5-minute load is below `LOAD` (default 1.5).
Their first pass shared the machine with up to nine container workloads, and the same compiled scheme took 14.8 s in one
script and 21.5 s in another. The final pass (2026-10-29) ran with `LOAD=12` while the load was about 8 on 16 cores
(12 performance cores), with the load before and after each script in `jcp6_runs.log`. Each time is the minimum over
repeats of a single-thread run, and make_tables.py checks that the same compiled scheme timed by different scripts
agrees to within 10%;
`rerun_cc_r3.sh` (2026-10-29) reruns the NumPy comparator drivers after the reading switches were added to
`chaplin_colella.py` (defaults unchanged: `cc_benchmark.py` and `cc_range.py` reproduce their earlier results exactly)
and runs `cc_sensitivity.py`.
`rerun_depthaware.sh` is the earlier rerun with the depth-aware values as the default, kept for the record.
