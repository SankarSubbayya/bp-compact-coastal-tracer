"""Assemble the reproducibility release of the revised manuscript (2026 revision, round 2).

Usage: python3 make_release.py <destination> [--public]
Copies, in the layout that paths.py expects (<destination>/paper, <destination>/_dev), every script of REPRODUCE.md, the
helper modules, the small bay inputs, every result file, and the manuscript sources, and writes MANIFEST.sha256 over all
of them. The offline-transport flux files (fluxes_mixed.npz, 224 MB; fluxes_mixed_25h.npz, 918 MB) are not copied:
step 3 of REPRODUCE.md regenerates them, and their checksums are listed in MANIFEST.sha256 for comparison.
With --public (the deposited release) the manuscript sources, the frozen Computers & Fluids version and the
submission-package scripts are left out (PRIVATE); every result, table and figure still regenerates."""
import hashlib, os, re, shutil, sys

import paths as P

SCRIPTS = ["paths.py", "figure_provenance.py", "fetch_crm.py", "build_sfbay_fine.py", "recalibrate_mixed.py", "gen_fluxes_mixed.py",
           "sfbay_conservative.py", "bay_table.py", "bay_experiments.py", "coast_closure_check.py", "long_run.py",
           "gen_deepen.py", "gen_mpdata.py", "gen_conservative.py", "appendix_nodal.py", "cv_reference.py", "cfl_sweep_cv.py",
           "reference_sensitivity.py", "ranking_check.py", "wall_convergence.py", "hsimt_consistency.py", "mpdata_validation.py",
           "uniform_tracer_check.py", "bay_checks.py", "make_tables.py", "gen_revision_figures.py", "sfbay_tides_fig.py",
           "sfbay_currents_fig.py", "sfbay_domain.py", "fig_wavenumber.py", "closure_spectra.py", "covariance_test.py",
           "order_513.py", "fast_schemes.py", "timing_benchmark.py", "covariance_ablation.py", "correction_cost.py",
           "fig_cost.py", "changing_volume.py", "variable_depth.py", "depth_test.py",
           "rerun_depthaware.sh", "rerun_direct.sh", "rerun_timing.sh", "rerun_cc.sh", "rerun_cc_r3.sh", "chaplin_colella.py", "cc_validation.py",
           "cc_benchmark.py", "cc_range.py", "steep_depth.py", "shelf_reference_check.py", "fast_cc.py", "cc_cost.py", "cc_sensitivity.py",
           "make_cf.sh", "make_jcp.sh", "cf_package.py", "jcp_package.py", "make_release.py"]
RESULTS = ["cv_results.json", "cfl_sweep_cv.json", "appendix_nodal.json", "recalibration_results.json", "sfbay_table.json",
           "bay_experiments.json", "coast_closure_check.json", "long_run.json", "wall_convergence.json",
           "hsimt_consistency.json", "mpdata_validation.json", "reference_sensitivity.json", "ranking_check.json", "checks.json",
           "closure_spectra.json", "covariance_test.json", "order_513.json", "timing_benchmark.json",
           "covariance_ablation.json", "correction_cost.json", "changing_volume.json", "variable_depth.json",
           "depth_test.json", "cc_results.json", "cc_validation.json", "cc_range.json",
           "fluxes_mixed.log", "sfbay_table_donor.npy", "sfbay_table_ccfctg.npy", "sfbay_table_ccompact_d.npy",
           "sfbay_table_ccfctg_ratio.npy", "steep_depth.json", "steep_depth_dissip.json", "shelf_reference_check.json", "cc_cost.json",
           "cc_sensitivity.json", "figure_inputs.json"]
DOCS = ["manuscript_rev.tex", "manuscript_jcp.tex", "manuscript_jcp.pdf", "numbers.tex", "references.bib", "REPRODUCE.md"]
DEV = ["advdiff.py", "sharp.py", "compact_stable.py", "bp_compact.py"]
BAY_INPUTS = ["crm_raw.npz", "sfbay_fine.npz", "sfbay_calibration.npz", "sfbay_tides.npz", "sfbay_stations.npz",
              "currents_fine.npz"]
BAY_LARGE = ["fluxes_mixed.npz", "fluxes_mixed_25h.npz"]
PRIVATE = {"manuscript_rev.tex", "manuscript_jcp.tex", "manuscript_jcp.pdf", "references.bib", "make_cf.sh", "make_jcp.sh",
           "cf_package.py", "jcp_package.py"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(dest, public=False):
    if os.path.exists(dest):
        sys.exit(f"{dest} exists; choose a new destination")
    rows = []
    def put(src, rel):
        out = os.path.join(dest, rel); os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy2(src, out)
        rows.append((sha(out), rel))
    for f in SCRIPTS + RESULTS + DOCS:
        if public and f in PRIVATE:
            continue
        put(os.path.join(P.PAPER, f), f"paper/{f}")
    for f in sorted(os.listdir(os.path.join(P.PAPER, "tables"))):
        put(os.path.join(P.PAPER, "tables", f), f"paper/tables/{f}")
    for N in (65, 129, 257, 513):            # benchmark data and references read by make_tables.py and the timing drivers
        put(os.path.join(P.PAPER, "figure_data", f"gauss_m16_{N}.npz"), f"paper/figure_data/gauss_m16_{N}.npz")
    tex = open(os.path.join(P.PAPER, "manuscript_rev.tex")).read()
    import figure_provenance as FP                # every figure the manuscript includes or figure_inputs.json records
    figs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}", tex))
    for f in sorted(figs | {o for _, outs in FP.EXPECTED.values() for o in outs}):
        put(os.path.join(P.PAPER, "figures", f), f"paper/figures/{f}")
    for root, _, files in ([] if public else os.walk(os.path.join(P.PAPER, "cf_version"))):   # frozen C&F version
        for f in sorted(files):
            if f.endswith((".tex", ".bib", ".pdf", ".png", ".bbl")):
                src = os.path.join(root, f); put(src, os.path.relpath(src, P.ROOT))
    for f in DEV:
        put(os.path.join(P.DEV, f), f"_dev/{f}")
    for f in BAY_INPUTS:
        put(os.path.join(P.SFBAY, f), f"_dev/sfbay_fine/{f}")
    for f in BAY_LARGE:                      # checksum only; regenerated by gen_fluxes_mixed.py
        rows.append((sha(os.path.join(P.SFBAY, f)), f"_dev/sfbay_fine/{f} (not included; regenerate)"))
    with open(os.path.join(dest, "MANIFEST.sha256"), "w") as m:
        m.writelines(f"{h}  {r}\n" for h, r in rows)
    print(f"wrote {len(rows)} entries to {dest}/MANIFEST.sha256")


if __name__ == "__main__":
    main(sys.argv[1], public="--public" in sys.argv[2:])
