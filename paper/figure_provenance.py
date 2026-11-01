"""Provenance of the manuscript figures (2026 revision, round 5). Each figure script calls record() after writing its
figures; it stores in figure_inputs.json the SHA-256 of the files the script drew from and of the figure files it
wrote. make_tables.py calls check(), which fails unless every script of EXPECTED is recorded with exactly its files,
every input is unchanged and every figure is the file its script wrote. Kept apart from paths.py, whose bytes enter
the benchmark's code hash (cv_reference.code_hash).

Not recorded: figure_data/slotted_cv_129.npz, the slotted-cylinder fields of gen_revision_figures.py, which that
script checks on every run, also when it reuses them (code hash, grid, reference field, shapes, finite values, and
the relative L2 and L1 errors, extrema, mass change and IoU against cv_results.json to 1e-12; an unreadable or
incomplete file counts as stale), and recomputes when absent or when any check fails. OMITTED inputs are not in
the release (step 3 of REPRODUCE.md regenerates them; MANIFEST.sha256 lists their checksums); check() compares them
only when present."""
import hashlib, json, os
import paths as P

RECORD = os.path.join(P.PAPER, "figure_inputs.json")
_R = lambda *f: [f"paper/{x}" for x in f]
_D = lambda *f: [f"_dev/sfbay_fine/{x}" for x in f]
EXPECTED = {          # script: (inputs relative to P.ROOT, figure files in P.FIGURES)
    "fig_wavenumber.py": ([], ["fig1_wavenumber_analysis.pdf", "fig1_wavenumber_analysis.png"]),
    "gen_revision_figures.py": (
        _R("cv_results.json", "order_513.json", "cc_results.json", "cfl_sweep_cv.json", "sfbay_table.json",
           "sfbay_table_donor.npy", "sfbay_table_ccfctg.npy", "sfbay_table_ccompact_d.npy") + _D("fluxes_mixed.npz"),
        [f"fig_rev_{n}.{x}" for n in ("convergence", "slotted", "sfbay", "cfl") for x in ("pdf", "png")]),
    "fig_cost.py": (_R("timing_benchmark.json", "cc_cost.json"), ["fig_rev_cost.pdf", "fig_rev_cost.png"]),
    "sfbay_domain.py": (_D("sfbay_fine.npz", "currents_fine.npz"), ["fig_sfbay_domain.png"]),
    "sfbay_tides_fig.py": (_D("sfbay_tides.npz"), ["fig_sfbay_tide.png"]),
    "sfbay_currents_fig.py": (_D("sfbay_stations.npz"), ["fig_sfbay_currents.png"]),
}
OMITTED = set(_D("fluxes_mixed.npz"))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def record(script):
    inputs, outputs = EXPECTED[script]
    rec = json.load(open(RECORD)) if os.path.exists(RECORD) else {}
    rec[script] = {"inputs": {f: _sha(os.path.join(P.ROOT, f)) for f in inputs},
                   "outputs": {f: _sha(os.path.join(P.FIGURES, f)) for f in outputs}}
    json.dump(rec, open(RECORD, "w"), indent=1, sort_keys=True)


def check():
    rec = json.load(open(RECORD))
    assert set(rec) == set(EXPECTED), f"figure_inputs.json records {sorted(rec)}, not the scripts {sorted(EXPECTED)}"
    for s, (inputs, outputs) in EXPECTED.items():
        assert set(rec[s]["inputs"]) == set(inputs) and set(rec[s]["outputs"]) == set(outputs), f"{s}: record lacks files"
        for f, h in rec[s]["inputs"].items():
            if f in OMITTED and not os.path.exists(os.path.join(P.ROOT, f)):
                continue
            assert _sha(os.path.join(P.ROOT, f)) == h, f"{s} drew its figures from an older {f}; rerun it"
        for f, h in rec[s]["outputs"].items():
            path = os.path.join(P.FIGURES, f)
            assert os.path.exists(path) and _sha(path) == h, f"figures/{f} is not the file {s} wrote; rerun it"
