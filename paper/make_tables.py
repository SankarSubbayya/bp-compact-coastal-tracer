"""Generate every table and in-text number of the revised manuscript from the result files, with one rounding step.
Upper bounds ("within", "at most") are rounded outward, never down.

Inputs: cv_results.json, cfl_sweep_cv.json, appendix_nodal.json (point-sampled appendix), recalibration_results.json,
sfbay_table.json, bay_experiments.json, long_run.json, wall_convergence.json, hsimt_consistency.json,
mpdata_validation.json, checks.json, reference_sensitivity.json, ranking_check.json, coast_closure_check.json,
covariance_test.json, closure_spectra.json, timing_benchmark.json, order_513.json, covariance_ablation.json,
correction_cost.json, changing_volume.json, depth_test.json, variable_depth.json, fluxes_mixed.log.
Outputs: numbers.tex (\\newcommand macros used in the text) and tables/*.tex (\\input by the manuscript)."""
import json, math, os, re

os.makedirs("tables", exist_ok=True)
J = lambda f: json.load(open(f))
CV, CF, NOD = J("cv_results.json"), J("cfl_sweep_cv.json"), J("appendix_nodal.json")
RC, BT, BE, LR = J("recalibration_results.json"), J("sfbay_table.json"), J("bay_experiments.json"), J("long_run.json")
WC, HC, MV, CK = J("wall_convergence.json"), J("hsimt_consistency.json"), J("mpdata_validation.json"), J("checks.json")
CC = J("coast_closure_check.json")
# the complete Chaplin-Colella (2017) algorithm (cc_benchmark.py) on the same data, merged into CV after a data check
CCB = J("cc_results.json")
for t in CCB["results"]:
    for N, rec in CCB["results"][t].items():
        base = CV[t][N]["cfctg"]
        assert (rec["initial_sha"], rec["reference_sha"]) == (base["initial_sha"], base["reference_sha"]), (t, N)
        assert CCB["code"] == base["code"] and CCB["cv_subcells"] == base["cv_subcells"]
        for s in ("cc17_c4", "cc17_u5"):
            CV[t][N][s] = dict(rec[s], code=CCB["code"], cv_subcells=CCB["cv_subcells"])
GRID = {"a": "33", "b": "65", "c": "129", "d": "257"}
CODE = {"BCG": "cfctg", "BCE": "cfcte", "BCL": "cfct", "BCU": "cfctg_nt", "BCD": "cfctgd", "BCDL": "cfctd",
        "UCD": "ccompactd", "CFG": "c4fctg", "CFE": "c4fcte", "UFG": "u5fctg", "UFE": "u5fcte", "MPD": "mpdata3_no",
        "MPO": "mpdata_no", "HST": "hsimt", "HSN": "hsimt_nopred", "HSS": "hsimt_split", "MUS": "muscl_fv", "DON": "donor_fv",
        "CCC": "cc17_c4", "CCU": "cc17_u5"}
LABEL = {"BCG": "\\textbf{bounded compact, global bounds}", "BCE": "bounded compact, extremum-preserving bounds",
         "BCL": "bounded compact, local bounds", "BCU": "bounded compact, global, no transverse correction",
         "BCD": "bounded compact + dissipation, global", "BCDL": "bounded compact + dissipation, local",
         "UCD": "compact + dissipation, unlimited", "CFG": "bounded C4, global bounds", "UFG": "bounded U5, global bounds",
         "CCC": "Chaplin--Colella, C4", "CCU": "Chaplin--Colella, U5",
         "MUS": "MUSCL-superbee", "MPD": "MPDATA, two corrective passes",
         "MPO": "MPDATA, one corrective pass", "HST": "HSIMT, ROMS predictor", "HSN": "HSIMT, two-level",
         "HSS": "HSIMT, Strang split", "DON": "donor cell"}
HS = ("HST", "HSN", "HSS")
COMPACT = {CODE[c] for c in ("BCG", "BCE", "BCL", "BCU", "BCD", "BCDL", "UCD")}   # compact face values, any bounds
macros = {}


def f3(x):        # three significant figures, plain decimal
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    return f"{x:.{max(0, 2 - e)}f}"


def sci(x, digits=1):
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def up(x, sig=2):  # outward (upward) rounding of a positive bound to `sig` significant digits, scientific
    e = math.floor(math.log10(x)); m = math.ceil(x / 10 ** (e - sig + 1)) / 10 ** (sig - 1)
    if m >= 10:
        m /= 10; e += 1
    return f"{m:.{sig - 1}f}\\times10^{{{e}}}"


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    if value[:1] in "+-" and "$" not in value:     # signed numbers get a true minus sign in running text
        value = f"\\ensuremath{{{value}}}"
    macros[name] = value


def l2(test, N, code):
    return CV[test][N][CODE[code]]["rel_l2"]


# ---- benchmark numbers (control-volume averages) ----
for code in CODE:
    for g, N in GRID.items():
        put(f"sm{code}{g}", f3(l2("gauss_T2", N, code)))
        if N in CV["slotted_T2"]:
            put(f"sl{code}{g}", f3(l2("slotted_T2", N, code)))
            put(f"io{code}{g}", f"{CV['slotted_T2'][N][CODE[code]]['iou']:.2f}")
def rng(vals):
    lo, hi = min(vals), max(vals)
    return f"{lo:.1f}", (f"{hi:.0f}" if hi >= 10 else f"{hi:.1f}")
for other in ("MPD", "MUS", "DON", "BCD", "CFG", "UFG", "CCC", "CCU") + HS:
    lo, hi = rng([l2("gauss_T2", N, other) / l2("gauss_T2", N, "BCG") for N in ("65", "129", "257")])
    put(f"r{other}min", lo); put(f"r{other}max", hi)
for other in ("CFG", "UFG"):                                    # same limiter, other reconstruction: every grid, both tracers
    assert all(l2(t, N, other) > l2(t, N, "BCG") and l2(t, N, other[:2] + "E") > l2(t, N, "BCE")
               for t in CV for N in CV[t]), f"text: compact more accurate than {other} under the same limiter"
    put(f"sr{other}d", f"{l2('slotted_T2', '257', other) / l2('slotted_T2', '257', 'BCG'):.2f}")
best_hs = {N: min(l2("gauss_T2", N, c) for c in HS) for N in GRID.values()}
lo, hi = rng([best_hs[N] / l2("gauss_T2", N, "BCG") for N in ("65", "129", "257")]); put("rHSBmin", lo); put("rHSBmax", hi)
best_hs_sl = {N: min(l2("slotted_T2", N, c) for c in HS) for N in CV["slotted_T2"]}
put("slHSBd", f3(best_hs_sl["257"]))
assert min(HS, key=lambda c: l2("slotted_T2", "129", c)) == "HSS", "slotted-figure caption names the split update"
rld = {g: l2("gauss_T2", GRID[g], "BCDL") / l2("gauss_T2", GRID[g], "BCL") for g in "abcd"}   # local bounds, with/without
for g in "bcd":
    put(f"rLD{g}", f"{rld[g]:.2f}")
assert rld["a"] > 1 and rld["b"] > 1 and rld["c"] < 1 and rld["d"] < 1, rld     # text: helps on 129^2 and 257^2 only
assert all(l2("gauss_T2", N, "BCD") > l2("gauss_T2", N, "BCG") for N in GRID.values())  # text: costs accuracy, global
def order(code, test="gauss_T2", a="129", b="257"):
    return f"{math.log2(l2(test, a, code) / l2(test, b, code)):.1f}"
for code in ("BCG", "BCE", "BCL", "BCU", "BCD", "CFG", "UFG", "CCC", "CCU", "MPD", "MUS", "UCD") + HS:
    put(f"ord{code}", order(code))
rows_main = ["BCG", "BCE", "BCL", "BCU", "CFG", "UFG", "CCC", "CCU", "BCD", "BCDL", "MUS", "MPD", "MPO", "HSS", "HSN", "HST", "DON"]
# local bounds clip smooth extrema; the extremum-preserving bounds recover the global-bounds accuracy (smooth tracer)
for g in "abcd":
    put(f"rEG{g}", f"{l2('gauss_T2', GRID[g], 'BCE') / l2('gauss_T2', GRID[g], 'BCG'):.2f}")
assert all(l2("gauss_T2", N, "BCL") > 1.3 * l2("gauss_T2", N, "BCE") for N in ("65", "129", "257")), "text: local clips, EP not"
assert all(abs(l2("gauss_T2", N, "BCE") / l2("gauss_T2", N, "BCG") - 1) < 0.05 for N in ("65", "129", "257"))
put("rLGd", f"{l2('gauss_T2', '257', 'BCL') / l2('gauss_T2', '257', 'BCG'):.1f}")
mass_fv = max(abs(CV[t][N][CODE[c]]["mass_rel"]) for t in CV for N in CV[t] for c in rows_main)
put("massFV", up(mass_fv))
hs_min = min(CV[t][N][CODE["HST"]]["min"] for t in CV for N in CV[t])
put("hsMinNeg", sci(-hs_min))                                   # an attained value, not a bound
assert all(CV[t][N][CODE[c]]["min"] >= -1e-12 for t in CV for N in CV[t] for c in ("HSS", "HSN")), "split/two-level negative"
others_ok = all(CV[t][N][CODE[c]]["min"] >= -1e-12 and CV[t][N][CODE[c]]["max"] <= 1 + 1e-12
                for t in CV for N in CV[t] for c in rows_main if c not in ("HST", "CCC", "CCU"))
assert others_ok, "a scheme other than predictor HSIMT and the two Chaplin-Colella implementations left [0, 1]"
codes = {CV[t][N][s].get("code") for t in CV for N in CV[t] for s in CV[t][N]} | {CF[c][k].get("code") for c in CF for k in CF[c] if k != "outflow_fraction"}
assert len(codes) == 1 and None not in codes, f"results come from more than one code version: {codes}"
assert NOD["code"] in codes and NOD["wall_closure"] == "nodal", "appendix results from another code version or closure"
sub = {CV[t][N][s].get("cv_subcells") for t in CV for N in CV[t] for s in CV[t][N]}
assert len(sub) == 1 and None not in sub, f"mixed or missing reference quadrature: {sub}"
put("subcells", str(sub.pop()))
RS = J("reference_sensitivity.json")
put("refSmoothSub", up(max(RS["gauss_T2"][N]["subcells_16_vs_32"] for N in RS["gauss_T2"]), 2))
put("refSlotSub", up(RS["slotted_T2"]["65"]["subcells_16_vs_32"], 2))
put("refSlotSubB", up(RS["slotted_T2"]["129"]["subcells_16_vs_32"], 2))
put("refSteps", up(max(RS[t]["65"]["steps_400_vs_800_m16"] for t in RS), 1))
# ranking claims of the text, each asserted. With delta = |r32 - r16| / |r16| and e = |u - r16| / |r16|, replacing r16
# by r32 changes e by at most delta (1 + e) / (1 - delta); two fixed solutions with e_a < e_b keep their order if
# e_b - e_a > 2 delta (their new errors share the denominator |r32|). Where that is not decisive, ranking_check.json
# (the cv_reference schemes, not Chaplin-Colella, rerun with initial data and reference at m = 32) must give the same ranking.
RK = J("ranking_check.json")
assert RK["code"] in codes, "ranking_check.json comes from another code version"
def bounded(rec):
    return rec["min"] >= -1e-12 and rec["max"] <= 1 + 1e-12
def claim(t, N, key, best="cfctg", among=None, higher=False):
    """`best` beats every scheme outside the compact family (and, if given, inside `among`)."""
    recs = CV[t][N]; pool = [s for s in recs if s not in COMPACT and (among is None or among(recs[s]))]
    val = lambda r, s: r[s][key]
    ok = all((val(recs, best) > val(recs, s)) if higher else (val(recs, best) < val(recs, s)) for s in pool)
    assert ok, f"{t} {N}: {best} is not the {'highest' if higher else 'lowest'} {key} outside the compact family"
    if key != "rel_l2" or N not in RS[t]:
        return "n/a"
    d = RS[t][N]["subcells_16_vs_32"]; a = val(recs, best)
    if all(val(recs, s) - a > 2 * d for s in pool):
        return "bound"
    r = RK[t][N]; assert all(s in r for s in pool if not s.startswith("cc17")), "m = 32 check lacks a scheme"
    assert all(r[best][key] < r[s][key] for s in pool if s in r), f"{t} {N}: ranking changes at m = 32"
    return "m32"
how = {}
for t in ("gauss_T2", "slotted_T2"):
    for N in CV[t]:
        how[(t, N)] = claim(t, N, "rel_l2")             # lower error than every scheme with another reconstruction
for N in CV["slotted_T2"]:
    claim("slotted_T2", N, "iou", among=bounded, higher=True)       # higher IoU than every bounded non-compact scheme
for N in RK["slotted_T2"]:                                           # text: at m = 32 on 65^2 and 129^2 both hold
    r = RK["slotted_T2"][N]; pool = [s for s in r if s not in COMPACT]
    assert all(r["cfctg"]["iou"] > r[s]["iou"] for s in pool if bounded(r[s])), f"IoU ranking at m = 32, {N}"
    assert all(r["cfctg"]["rel_l2"] < r[s]["rel_l2"] for s in pool), f"L2 ranking at m = 32, {N}"
assert sorted(RK["slotted_T2"], key=int) == ["65", "129"], "text names the m = 32 check grids"
assert all(h == "bound" for (t, N), h in how.items() if N != "33"), how     # text: every stated L2 ranking survives delta
# within the compact family the differences are not resolved: on 257^2 the smooth global / extremum-preserving errors
# and the slotted corrected / uncorrected errors differ by less than 2 delta
d257 = {t: RS[t]["257"]["subcells_16_vs_32"] for t in RS}
assert abs(l2("gauss_T2", "257", "BCE") - l2("gauss_T2", "257", "BCG")) < 2 * d257["gauss_T2"]
assert abs(l2("slotted_T2", "257", "BCU") - l2("slotted_T2", "257", "BCG")) < 2 * d257["slotted_T2"]
put("refSlotSubC", up(RS["slotted_T2"]["257"]["subcells_16_vs_32"], 2))
dsl = {N: RS["slotted_T2"][N]["subcells_16_vs_32"] for N in RS["slotted_T2"]}
assert all(abs(l2("slotted_T2", N, "BCU") - l2("slotted_T2", N, "BCG")) < 2 * dsl[N] for N in dsl), "text: BCU within delta"
assert l2("slotted_T2", "257", "BCE") - l2("slotted_T2", "257", "BCG") > 2 * dsl["257"], "text: EP bounds larger on 257^2"

# ---- manufactured covariance test, closure spectra ----
CT = J("covariance_test.json")
for key, name in (("covFace", "face_value"), ("covFlux", "flux_divergence"), ("covCorr", "flux_divergence_corrected")):
    put(key, f"{CT['orders'][name][-1]:.1f}")
put("covGain", f"{CT['errors']['128']['flux_divergence'] / CT['errors']['128']['flux_divergence_corrected']:.0f}")
SP = J("closure_spectra.json")
re_ = lambda ends, clo: {int(N): v["max_re"] for N, v in SP[ends][clo].items()}
assert all(v > 0 for v in re_("open", "cellavg").values()), "text: cell-average closure unstable on every length"
assert all((v > 0) == (N >= 16) for N, v in re_("open", "point").items()), "text: point closure unstable from 16 cells"
assert all(v < 0 for v in re_("open", "average").values()), "text: averaged closure stable on every length"
assert all(abs(v) < 1e-7 for e in ("wall", "step") for c in SP[e] for v in re_(e, c).values()), "text: wall spectra neutral"
assert all(v["rank_L2"] < v["rank_L"] for e in ("wall", "step") for c in SP[e] for v in SP[e][c].values()), \
    "text: the zero eigenvalue is defective on every closed line"
put("specCA", sci(max(re_("open", "cellavg").values())))
nx = SP["wall"]["average"]; nmax = max(nx, key=int)
put("specExpN", nmax); put("specExpNorm", f"{nx[nmax]['norm_exp_100']:.0f}")
put("peakBCGd", f"{CV['gauss_T2']['257']['cfctg']['max']:.3f}"); put("peakBCLd", f"{CV['gauss_T2']['257']['cfct']['max']:.3f}")

# ---- every figure drawn by its script from the current inputs (figure_provenance.py, figure_inputs.json) ----
import figure_provenance as FP
FP.check()

# ---- Chaplin-Colella: validation on the paper's tests (cc_validation.py) and the benchmark (cc_benchmark.py) ----
CCV = J("cc_validation.json")["smooth"]
diff = max(abs(CCV[f"{v}|{vel}|{rc}|limited"][N]["linf"] / CCV[f"{v}|{vel}|{rc}|unlimited"][N]["linf"] - 1)
           for v in ("published", "published_component") for vel in ("oned", "constant", "rotation")
           for rc in ("c4", "u5", "u9") for N in ("128", "256"))
assert diff < 0.05, "text: with either reading of sigma, limited errors close to the unlimited ones (128^2, 256^2)"
put("ccValDiff", f"{math.ceil(1000 * diff) / 10:.1f}")
put("ccValUnl", sci(CCV["published|constant|c4|unlimited"]["256"]["linf"]))
put("ccValUnlC", sci(CCV["published_component|constant|c4|unlimited"]["256"]["linf"]))
put("ccNineS", sci(CCV["published|constant|u9|unlimited"]["256"]["linf"]))
put("ccNineC", sci(CCV["published_component|constant|u9|unlimited"]["256"]["linf"]))
assert all(CCV[f"published_component|constant|{rc}|unlimited"]["256"]["linf"] >
           CCV[f"published|constant|{rc}|unlimited"]["256"]["linf"] for rc in ("c4", "u9")), \
    "text: the component reading (longer step) gives larger errors"
anyx = CCV["published|constant|c4|flag_any_extremum"]["256"]["linf"]
assert anyx > 10 * CCV["published|constant|c4|limited"]["256"]["linf"], "text: flagging every 1-D extremum destroys the accuracy"
put("ccValAny", f"{anyx:.2f}")
CD = J("cc_validation.json")["discontinuous"]
exc = max(max(-r["min"], r["max"] - 1, 0.0) for vel in ("oned", "constant") for name in ("square", "ellipse")
          for rc in ("c4", "u5", "u9") for r in [CD[f"published|{vel}|{name}|{rc}"]])
assert exc < 1e-2, "text: the paper's discontinuous profiles stay close to [0, 1] in 1-D and under constant velocity"
put("ccValExc", up(max(exc, 1e-16), 1))
sq9 = J("cc_validation.json")["discontinuous"]["published_component|constant|square|u9"]
assert sq9["min"] < -0.01, "text: at the component-reading step the limited ninth-order square undershoots"
put("ccNineSq", f"{-sq9['min']:.3f}")
# sensitivity to the readings of eqs. 30 and 78 (cc_sensitivity.py)
SNS = J("cc_sensitivity.json")
srec = [(rd, sc, N, v) for rd in SNS["results"] for sc in SNS["results"][rd] for N, v in SNS["results"][rd][sc].items()]
assert len(srec) == 12 and all(abs(v["gauss_bench_paper"] - CCB["results"]["gauss_T2"][N][sc]["rel_l2"]) < 1e-15
                               for rd, sc, N, v in srec), "cc_sensitivity read stale comparator results"
assert all(v["gauss_bench"] > v["gauss_bench_compact"] and v["gauss_largest"] > v["gauss_largest_compact"] and
           v["slotted"] > v["slotted_compact"] for rd, sc, N, v in srec), \
    "text: under every reading the compact scheme remains more accurate on every grid and for both tracers"
put("ccSensBench", f"{math.ceil(1000 * max(abs(v['gauss_bench'] / v['gauss_bench_paper'] - 1) for *_, v in srec)) / 10:.1f}")
put("ccSensSlot", f"{math.ceil(1000 * max(abs(v['slotted'] / v['slotted_paper'] - 1) for *_, v in srec)) / 10:.1f}")
TTT = SNS["time_to_target"]
ttt = [abs(f / TTT["paper"][sc] - 1) for rd in TTT if rd != "paper" for sc, f in TTT[rd].items()]
assert min(f for rd in TTT for f in TTT[rd].values()) > 1, "text: under every reading the compact scheme reaches 0.1 faster"
put("ckSensPct", f"{math.ceil(1000 * max(ttt)) / 10:.1f}")                    # largest change of a time factor
ccmin = min(CV[t][N][s]["min"] for t in CV for N in CV[t] for s in ("cc17_c4", "cc17_u5"))
ccmax = max(CV[t][N][s]["max"] for t in CV for N in CV[t] for s in ("cc17_c4", "cc17_u5"))
assert ccmin < -1e-12 and ccmax > 1 + 1e-3, "text: the Chaplin-Colella solutions leave the data range"
put("ccUnder", sci(-ccmin)); put("ccOverAbs", f"{ccmax - 1:.3f}")              # attained values
CRG = J("cc_range.json")["slotted"]
assert sorted(CRG, key=int) == ["65", "129"] and all(
    v["not_widened"]["min"] >= -1e-12 and v["not_widened"]["max"] <= 1 + 1e-12 and
    (v["widened"]["min"] < -1e-12 or v["widened"]["max"] > 1 + 1e-12) for v in CRG.values()), \
    "text: without the eq. 72-73 widening the slotted solutions stay within [0, 1] (65^2, 129^2)"
assert all(CV["slotted_T2"][N]["cfctg"]["iou"] > CV["slotted_T2"][N][s]["iou"] for N in CV["slotted_T2"]
           for s in ("cc17_c4", "cc17_u5")), "text: compact IoU higher than Chaplin-Colella"
for code in ("CCC", "CCU"):
    assert all(l2(t, N, code) > l2(t, N, "BCG") for t in CV for N in CV[t]), f"text: compact more accurate than {code}"
    put(f"srCC{code[-1]}d", f"{l2('slotted_T2', '257', code) / l2('slotted_T2', '257', 'BCG'):.2f}")
SW = CCB["sweep"]
for k, v in SW.items():
    put("ccSweep" + {"0.35": "A", "0.60": "B", "0.80": "C"}[k], f3(v["rel_l2"]))

# ---- 513^2 (order_513.json) ----
O5 = J("order_513.json")
assert O5["code"] in codes and O5["cv_subcells"] == 16
E513 = {c: O5["gauss_T2"][CODE[c]]["rel_l2"] for c in ("BCG", "BCU", "CFG", "UFG")}
for c, e in E513.items():
    put(f"sm{c}e", sci(e, 2)); put(f"ord{c}e", f"{math.log2(l2('gauss_T2', '257', c) / e):.1f}")
    r = O5["gauss_T2"][CODE[c]]; assert r["min"] >= -1e-12 and r["max"] <= 1 + 1e-12, f"{c} left the bounds at 513"
for c in ("CFG", "UFG"):
    put(f"r{c}e", f"{E513[c] / E513['BCG']:.1f}")
    assert E513[c] > E513["BCG"]
assert abs(math.log2(l2("gauss_T2", "257", "BCG") / E513["BCG"]) - 4) < 0.1, "text: fourth order between 257^2 and 513^2"

# ---- covariance ablation (covariance_ablation.json): compiled scheme with and without the transverse correction ----
CA = J("covariance_ablation.json")["gauss_T2"]
for N in CA:
    for c in ("BCG", "BCU"):
        ref_e = l2("gauss_T2", N, c) if N != "513" else E513[c]
        assert abs(CA[N][CODE[c]]["rel_l2"] - ref_e) < 1e-11, f"ablation {N} {c} does not reproduce the archived error"
gain = {N: 1 - E513["BCG"] / E513["BCU"] if N == "513" else 1 - l2("gauss_T2", N, "BCG") / l2("gauss_T2", N, "BCU")
        for N in ("65", "129", "257", "513")}
assert all(b > a for a, b in zip(list(gain.values()), list(gain.values())[1:])), "text: the gain grows with refinement"
put("tcGainB", f"{100 * gain['65']:.0f}"); put("tcGainD", f"{100 * gain['257']:.0f}")
put("tcGain", f"{E513['BCU'] / E513['BCG']:.1f}")

CC2 = J("correction_cost.json")
ov = [100 * (CC2[N][CODE["BCG"]]["seconds"] / CC2[N][CODE["BCU"]]["seconds"] - 1) for N in CC2]
TBG = J("timing_benchmark.json")["timing"]                         # same compiled scheme timed by two scripts:
assert all(abs(CC2[N][CODE["BCG"]]["seconds"] / TBG[N][CODE["BCG"]]["seconds"] - 1) < 0.1 for N in CC2), \
    "correction_cost and timing_benchmark disagree: rerun both on an idle machine (rerun_timing.sh)"
put("corrCostMin", f"{min(ov):.0f}"); put("corrCostMax", f"{max(ov):.0f}")

# ---- compiled cost (timing_benchmark.json) ----
TB = J("timing_benchmark.json")
assert all(v["max_abs_diff"] < 1e-11 for N in TB["validation"] for v in TB["validation"][N].values())
TT = TB["timing"]; TN = sorted(TT, key=int)
OTHERS = ("CFG", "UFG", "MPD", "HSS", "MUS")
assert set(CODE[c] for c in ("BCG",) + OTHERS) <= set(TT["257"]), "timing lacks a scheme named in the text"
for c in OTHERS:                                                 # same-grid cost factors, 257^2
    put(f"cR{c}", f"{TT['257'][CODE['BCG']]['seconds'] / TT['257'][CODE[c]]['seconds']:.1f}")
def t_pts(pts, target, what):
    """Time to reach `target` error from (seconds, error) per grid, log-log interpolation between bracketing grids."""
    for (t0, e0), (t1, e1) in zip(pts, pts[1:]):
        if e0 >= target >= e1:
            w = math.log(target / e0) / math.log(e1 / e0)
            return math.exp(math.log(t0) + w * math.log(t1 / t0))
    raise ValueError(f"{what} does not reach {target}")
def t_at(code, target):
    return t_pts([(TT[N][CODE[code]]["seconds"], TT[N][CODE[code]]["rel_l2"]) for N in TN], target, code)
fmt = lambda r: f"{r:.0f}" if r >= 10 else f"{r:.1f}"
TARGET, COARSE = 0.1, 0.2
put("tTarget", f"{TARGET}"); put("tCoarse", f"{COARSE}")
tb = t_at("BCG", TARGET); put("tTgtBCG", f"{tb:.2f}")
for c in OTHERS:
    tc_ = t_at(c, TARGET); put(f"tTgt{c}", f"{tc_:.2f}" if tc_ < 1 else f"{tc_:.1f}"); put(f"sp{c}", fmt(tc_ / tb))
    assert tc_ > tb, f"text: {c} takes longer than the compact scheme to reach {TARGET}"
    put(f"spc{c}", fmt(t_at(c, COARSE) / t_at("BCG", COARSE)))
assert t_at("MUS", COARSE) < t_at("BCG", COARSE), "text: MUSCL-superbee is faster at the coarse target"
assert all(t_at(c, COARSE) > t_at("BCG", COARSE) for c in ("CFG", "UFG", "MPD", "HSS")), "text: the others are slower there"
put("cpuName", TB["machine"].replace("Apple ", "")); put("numbaVer", TB["numba"])
put("valMax", up(max(v["max_abs_diff"] for N in TB["validation"] for v in TB["validation"][N].values()), 1))
for c in ("MUS", "CFG", "UFG"):                                  # at each scheme's own finest error
    e_ = TT["257"][CODE[c]]["rel_l2"]
    put(f"spf{c}", fmt(TT["257"][CODE[c]]["seconds"] / t_at("BCG", e_)))
put("tBCGc", f"{TT['129'][CODE['BCG']]['seconds']:.1f}")
for c in ("MPD", "HSS", "MUS"):
    put(f"t{c}d", f"{TT['257'][CODE[c]]['seconds']:.1f}")
assert all(TT["129"][CODE["BCG"]]["rel_l2"] < TT["257"][CODE[c]]["rel_l2"] for c in ("MPD", "HSS", "MUS"))
assert all(TT["129"][CODE["BCG"]]["seconds"] < TT["257"][CODE[c]]["seconds"] for c in ("MPD", "HSS"))
put("repeats", str(TB["repeats"]))

# ---- Chaplin-Colella cost at the benchmark step and at each scheme's largest admissible step (cc_cost.json) ----
CCK = J("cc_cost.json"); KT = CCK["timing"]; KN = sorted(KT["largest"], key=int)
assert all(v["max_abs_diff"] < 1e-9 and v["rel_l2_diff"] < 1e-12 for N in CCK["validation"]
           for v in CCK["validation"][N].values())
assert CCK["fast_schemes_sha256"] == TB["fast_schemes_sha256"] and CCK["repeats"] == TB["repeats"]
assert all(abs(KT["benchmark"][N]["cfctg"]["seconds"] / TT[N]["cfctg"]["seconds"] - 1) < 0.1 for N in KN), \
    "cc_cost and timing_benchmark disagree: rerun both on an idle machine (rerun_timing.sh)"
put("ckVal", up(max(v["max_abs_diff"] for N in CCK["validation"] for v in CCK["validation"][N].values()), 1))
AD = {N: CCK["admissible"][N] for N in KN if N != "33"}
for key, name in (("donor_dt_over_h", "ckDonor"), ("ctu_dt_over_h", "ckCTU")):
    assert len({f"{a[key]:.2f}" for a in AD.values()}) == 1, "text: one admissible step to two digits on 65^2-257^2"
    put(name, f"{AD['257'][key]:.2f}")
put("ckStep", f"{AD['257']['ctu_dt_over_h'] / AD['257']['donor_dt_over_h']:.2f}")
KL = KT["largest"]
kpts = lambda s: [(KL[N][s]["seconds"], KL[N][s]["rel_l2"]) for N in KN]
assert all(KL[N][s]["rel_l2"] < KL[N][t]["rel_l2"] for N in KN[1:] for s, t in (("cfctg", "cc17_u5"), ("cfctg", "cc17_c4"))), \
    "text: at the largest steps the compact scheme stays more accurate than Chaplin-Colella"
tk = t_pts(kpts("cfctg"), TARGET, "cfctg")
for s, key in (("cc17_u5", "U"), ("cc17_c4", "C")):
    put(f"ckR{key}", f"{KL['257']['cfctg']['seconds'] / KL['257'][s]['seconds']:.1f}")
    put(f"ckSp{key}", fmt(t_pts(kpts(s), TARGET, s) / tk))
    assert t_pts(kpts(s), TARGET, s) > tk, f"text: {s} takes longer to reach {TARGET} at its largest step"
    e_ = KL["257"][s]["rel_l2"]; put(f"ckE{key}", f"{e_:.4f}")
    put(f"ckSpf{key}", fmt(KL["257"][s]["seconds"] / t_pts(kpts("cfctg"), e_, "cfctg")))
assert all(abs(SNS["time_to_target"]["paper"][s] / (t_pts(kpts(s), TARGET, s) / tk) - 1) < 1e-9 for s in ("cc17_u5", "cc17_c4")), \
    "cc_sensitivity's interpolation of the paper's reading disagrees with the cost section"

# ---- wall convergence ----
def wc(label, key, N):
    return WC[label][N][key]
LH, LE = "unlimited compact, half-width wall closure", "unlimited compact, equal-width wall closure"
LB = "bounded compact (global), half-width wall closure"
o2 = lambda lab, key, a="129", b="257": f"{math.log2(wc(lab, key, a) / wc(lab, key, b)):.1f}"
put("wcOrdNew", o2(LH, "rel_l2")); put("wcWallNew", o2(LH, "max_err_wall_rows"))
put("wcOrdOld", o2(LE, "rel_l2")); put("wcWallOld", o2(LE, "max_err_wall_rows"))
put("wcOrdBounded", o2(LB, "rel_l2"))
put("wcGain", f"{wc(LE, 'max_err_wall_rows', '257') / wc(LH, 'max_err_wall_rows', '257'):.0f}")
LU = "bounded compact (global [0, 1]), half-width wall closure"
put("wcOrdUnit", o2(LU, "rel_l2"))
ext = WC["extrema"]["257"]
assert ext["max_reference"] > ext["max_initial"], "text: on 257^2 the reference peak exceeds the initial maximum"
put("wcMaxInit", f"{ext['max_initial']:.6f}"); put("wcMaxRef", f"{ext['max_reference']:.6f}")
assert abs(wc(LU, "rel_l2", "257") / wc(LH, "rel_l2", "257") - 1) < 1e-3, "text: with [0, 1] the limiter is inactive on 257^2"
for g in "bd":
    put(f"wcB{g}", sci(wc(LB, "rel_l2", GRID[g]))); put(f"wcU{g}", sci(wc(LU, "rel_l2", GRID[g])))
import numpy as np                                   # text: benchmark reference peak below the initial peak, 65^2-513^2
for N in ("65", "129", "257", "513"):
    fd = np.load(f"figure_data/gauss_m16_{N}.npz")
    assert fd["ref"].max() < fd["c0"].max(), f"benchmark reference peak exceeds the initial peak on {N}^2"

# ---- changing control volumes (changing_volume.json) ----
CVV = J("changing_volume.json")
assert CVV["code"] in codes and CVV["cv_subcells"] == 16
CVC = {"BCG": "cfctg", "BCA": "cfctg_da", "BCU": "cfctg_nt", "BCE": "cfcte", "CFG": "c4fctg", "UFG": "u5fctg",
       "DON": "cdonor"}                                  # BCA: depth-aware ratio face values
SPC = CVV["space"]; CVN = sorted(SPC, key=int)
cve = lambda c, N: SPC[N][CVC[c]]["rel_l2"]
for c in CVC:
    for g, N in GRID.items():
        put(f"cv{c}{g}", f3(cve(c, N)))
    put(f"cvOrd{c}", f"{math.log2(cve(c, '129') / cve(c, '257')):.1f}")
cvdw = max(abs(cve("BCA", N) / cve("BCG", N) - 1) for N in CVN)
put("cvDA", f"{math.ceil(1000 * cvdw) / 10:.1f}")       # largest relative difference, percent (rounded up)
assert cvdw < 0.01, "text: depth-aware and direct face values differ by under 1% in the changing-volume test"
for c in ("BCU", "CFG", "UFG", "DON"):
    assert all(cve(c, N) > cve("BCG", N) for N in CVN), f"text: compact more accurate than {c} with moving volumes"
    put(f"cvR{c}", f"{cve(c, '257') / cve('BCG', '257'):.1f}")
put("cvGainD", f"{100 * (1 - cve('BCG', '257') / cve('BCU', '257')):.0f}")
recs = [r for N in CVN for c, r in SPC[N].items() if isinstance(r, dict)]
assert all(r["min"] >= -1e-12 and r["max"] <= 1 + 1e-12 for r in recs), "text: bounds kept with moving volumes"
put("cvMass", up(max(max(abs(r["mass_rel"]) for r in recs), 1e-16)))
put("cvUniform", up(CVV["uniform_tracer_max_dev"], 1))
phis = {SPC[N]["outflow_fraction"] for N in CVN}
put("cvPhi", f"{math.ceil(100 * max(phis)) / 100:.2f}")
put("cvHmin", f"{CVV['H0'] - CVV['amplitude']:.1f}"); put("cvHmax", f"{CVV['H0'] + CVV['amplitude']:.1f}")
TM = CVV["time"]
for c, key in (("BCG", "cfctg"), ("BCU", "cfctg_nt")):
    v = [TM[f][key]["rel_l2"] for f in TM]
    put(f"cvDt{c}", up((max(v) - min(v)) / min(v), 1))
assert max(abs(TM[f]["cfctg"]["rel_l2"] / cve("BCG", "129") - 1) for f in TM) < 1e-3, "text: error is spatial"

# ---- depth weighting: manufactured operator test (depth_test.json), steady bathymetry (variable_depth.json), ----
# ---- steep shelf and noise growth (steep_depth.json); "_da" = depth-aware ratio, default = c reconstructed directly ----
DT_ = J("depth_test.json")
for key, name in (("dtDirect", "direct"), ("dtRatio", "ratio"), ("dtAware", "depth_aware")):
    put(key, f"{DT_['orders'][name][-1]:.2f}")
assert abs(DT_["orders"]["direct"][-1] - 2) < 0.05 and abs(DT_["orders"]["depth_aware"][-1] - 4) < 0.05
put("dtGain", f"{DT_['errors']['256']['direct'] / DT_['errors']['256']['depth_aware']:.0f}")
put("dtUniform", up(max(DT_["uniform_tracer_flux_max_dev"], 1e-18), 1))
VD = J("variable_depth.json")
assert VD["code"] in codes and VD["cv_subcells"] == 16
VDC = {"UCW": "ccompact", "UCA": "ccompact_da", "BCG": "cfctg", "BCA": "cfctg_da", "CFG": "c4fctg", "UFG": "u5fctg"}
VS = VD["space"]; VN = sorted(VS, key=int)
vde = lambda c, N: VS[N][VDC[c]]["rel_l2"]
for c in VDC:
    put(f"vdOrd{c}", f"{math.log2(vde(c, '129') / vde(c, '257')):.1f}")
vrec = [r for N in VN for k, r in VS[N].items() if isinstance(r, dict)]
assert all(r["depth_fallback_faces"] == 0 for r in vrec), "text: no fallback faces over the smooth bathymetry"
assert all(r["min"] >= -1e-12 and r["max"] <= 1 + 1e-12 for N in VN for c in ("BCG", "BCA", "CFG", "UFG")
           for r in [VS[N][VDC[c]]]), "text: bounded schemes stay in [0, 1] over the bathymetry"
put("vdMass", up(max(max(abs(r["mass_rel"]) for r in vrec), 1e-16)))
put("vdUniform", up(max(VD["uniform_tracer_max_dev"], 1e-17), 1))
PAIRS = (("UCA", "UCW"), ("BCA", "BCG"))                # (depth-aware, direct)
low = max(vde(a, N) / vde(w, N) - 1 for N in ("33", "65", "129") for a, w in PAIRS)
assert 0 < low < 0.05, "text: on 33^2-129^2 the direct values are up to a few per cent more accurate"
put("vdLow", f"{math.ceil(100 * low)}")
assert all(vde("BCA", N) >= vde("BCG", N) for N in ("33", "65", "129")), \
    "text: with the limiter the ratio reduces the smooth-depth error on no grid from 33^2 to 129^2"
for a, w in PAIRS:
    assert vde(a, "257") < vde(w, "257"), "text: on 257^2 the depth-aware values are more accurate"
    assert float(macros[f"vdOrd{a}"]) > float(macros[f"vdOrd{w}"]), "text: the direct values' order falls on 257^2"
    put(f"vdGain{a}", f"{100 * (1 - vde(a, '257') / vde(w, '257')):.0f}")
for c in ("CFG", "UFG"):
    assert all(vde(c, N) > vde("BCG", N) for N in VN), f"text: compact more accurate than {c} over the bathymetry"
    put(f"vdR{c}", f"{vde(c, '257') / vde('BCG', '257'):.1f}")
put("vdAmp", f"{VD['amplitude']:.1f}"); put("vdTend", f"{VD['t_end']:g}")
put("vdPhi", f"{math.ceil(100 * max(VS[N]['outflow_fraction'] for N in VN)) / 100:.2f}")
SD = J("steep_depth.json")
assert SD["code"] in codes and SD["cv_subcells"] == 16
SS = SD["space"]; SN = sorted(SS, key=int)
sde = lambda c, N: SS[N][VDC[c]]["rel_l2"]
put("sdDmin", f"{SD['dmin']:g}"); put("sdW", f"{SD['width']:g}"); put("sdTend", f"{SD['t_end']:g}")
put("sdPhi", f"{SD['phi']:.2f}"); put("sdRef", up(max(SS[N]["ref_nsub_change"] for N in SN if "ref_nsub_change" in SS[N]), 1))
put("sdSpeed", f"{1 / SD['dmin']:.0f}")
for g, N in zip("abc", SN):
    put(f"sdWh{g}", f"{SS[N]['W_over_h']:.1f}")
for c in VDC:
    for g, N in zip("abc", SN):
        put(f"sd{c}{g}", f"{sde(c, N):#.3g}")
srec = [SS[N][VDC[c]] for N in SN for c in ("BCG", "BCA", "CFG", "UFG")]
assert all(r["min"] >= -1e-12 and r["max"] <= 1 + 1e-12 for r in srec), "text: bounded schemes stay in [0, 1] on the shelf"
assert all(sde("UCA", N) > 5 * sde("UCW", N) for N in SN), "text: the unlimited depth-aware flux fails on the shelf"
assert all(sde("UCA", N) > 1 for N in SN[:2]), "text: the depth-aware unlimited flux loses the solution on 33^2, 65^2"
put("sdUCAr", f"{sde('UCA', SN[-1]) / sde('UCW', SN[-1]):.0f}")
assert all(sde("BCA", N) > sde("BCG", N) for N in SN), "text: with the limiter the ratio reduces the error on no grid"
put("sdBA", f"{math.ceil(100 * max(sde('BCA', N) / sde('BCG', N) - 1 for N in SN))}")
assert int(macros["sdBA"]) <= 5
assert all(sde("UFG", N) < sde("BCG", N) < sde("CFG", N) for N in SN), "text: on the shelf U5 < compact < C4 on every grid"
put("sdGainUFG", f"{100 * (1 - sde('UFG', SN[-1]) / sde('BCG', SN[-1])):.0f}")
put("sdRCFG", f"{sde('CFG', SN[-1]) / sde('BCG', SN[-1]):.1f}")
sdo = [math.log2(sde(c, SN[-2]) / sde(c, SN[-1])) for c in VDC if c != "UCA"]
assert max(sdo) < 1.5, "text: orders near one on the shelf, except the depth-aware unlimited flux"
put("sdOrdMax", f"{max(sdo):.1f}")
put("sdMass", up(max(max(max(abs(SS[N][s]["mass_rel"]) for s in VDC.values()) for N in SN), 1e-16)))
NZ = SD["noise"]
for bathy, key in (("shelf", "Sh"), ("cosine", "Co")):
    for c, s in (("W", "ccompact"), ("A", "ccompact_da")):
        v = NZ[bathy][s]["1.0"]
        put(f"nz{key}{c}", f"{v:.2f}" if v < 10 else sci(v, 1))
assert NZ["shelf"]["ccompact_da"]["1.0"] > 100 * NZ["shelf"]["ccompact"]["1.0"], "text: noise grows with the ratio"
assert all(NZ[b]["ccompact"]["1.0"] < 1.25 for b in NZ if b in ("shelf", "cosine")), "text: direct values stay near 1"

# ---- HSIMT consistency, MPDATA validation ----
for var, key in (("two-level (no predictor)", "N"), ("ROMS predictor", "R"), ("Strang split", "S")):
    for test, tk in (("2+x^2", "xx"), ("2+y^2", "yy"), ("2+xy", "xy")):
        v = HC[var][test][-1]
        put(f"hc{key}{tk}", "0" if abs(v) < 5e-4 else f"{v:+.2f}")
put("mpVal", up(max(v["rel_diff"] for v in MV.values()), 1))

# ---- time-step sweep ----
cfls = sorted(CF, key=float)
ok = [c for c in cfls if CF[c]["outflow_fraction"] <= 1.0]
def spread(code, test="gauss_T2"):
    v = [CF[c][f"{test}_{CODE[code]}"]["rel_l2"] for c in ok]
    return 100 * (max(v) - min(v)) / min(v)
for code in ("BCG", "BCE", "BCL", "BCU", "CFG", "UFG", "MPD", "MPO", "MUS", "DON") + HS:
    sp = spread(code); put(f"cf{code}spread", f"{sp:.2f}" if sp < 0.1 else f"{sp:.1f}" if sp < 10 else f"{sp:.0f}")
    sps = spread(code, "slotted_T2"); put(f"cfs{code}spread", f"{sps:.1f}" if sps < 10 else f"{sps:.0f}")
put("cfOutMin", f"{CF[cfls[0]]['outflow_fraction']:.2f}"); put("cfOutMax", f"{CF[cfls[-1]]['outflow_fraction']:.2f}")
neg_beyond = min(min(CF[c][k]["min"] for k in CF[c] if k != "outflow_fraction" and ("cfct" in k and "cfctgd" not in k and "cfctd" not in k))
                 for c in cfls if CF[c]["outflow_fraction"] > 1.0)
put("cfNegBeyond", up(-neg_beyond, 1))
for code in HS:
    v = [CF[c][f"gauss_T2_{CODE[code]}"]["rel_l2"] for c in ok]
    put(f"cf{code}lo", f3(v[0])); put(f"cf{code}hi", f3(v[-1]))
    mono = all(b > a for a, b in zip(v, v[1:])) if code != "HSS" else all(b < a for a, b in zip(v, v[1:]))
    assert mono, f"{code}: the text states a monotone trend over the admissible steps"
mus_min = min(CF[c][f"slotted_T2_muscl_fv"]["min"] for c in ok)
assert mus_min < -1e-12, "text: MUSCL's slotted solution goes negative at the larger admissible steps"
put("cfMUSneg", sci(-mus_min))                                  # an attained value, not a bound
neg_phi = [CF[c]["outflow_fraction"] for c in ok if CF[c]["slotted_T2_muscl_fv"]["min"] < -1e-12]
assert all(CF[c]["slotted_T2_muscl_fv"]["min"] >= -1e-12 for c in ok if CF[c]["outflow_fraction"] < min(neg_phi))
put("cfMUSphi", f"{math.floor(100 * min(neg_phi)) / 100:.2f}")
assert spread("BCL") > spread("BCG") and spread("BCG", "slotted_T2") > spread("BCG"), "text: local / slotted more sensitive"
mp = [CF[c]["gauss_T2_mpdata3_no"]["rel_l2"] for c in ok]
assert abs(CF[ok[mp.index(min(mp))]]["outflow_fraction"] - 0.8) < 0.05, "MPDATA minimum not near 0.8"

# ---- tide and currents ----
put("tideCond", f"{RC['condition_number']:.0f}"); put("tideFm", f"{RC['form_factor_model']:.2f}"); put("tideFo", f"{RC['form_factor_noaa']:.2f}")
it0 = RC["iterations"][0]
put("tideOldAmpErr", f"{100 * max(abs(it0['gate_fit'][n][0] / RC['gate'][n]['noaa_amp'] - 1) for n in RC['gate']):.1f}")
put("tideOldPhErr", f"{max(abs((it0['gate_fit'][n][1] - RC['gate'][n]['noaa_phase'] + 180) % 360 - 180) for n in RC['gate']):.0f}")
amp_err = max(abs(RC["gate"][n]["model_amp"] / RC["gate"][n]["noaa_amp"] - 1) for n in RC["gate"])
ph_err = max(abs((RC["gate"][n]["model_phase"] - RC["gate"][n]["noaa_phase"] + 180) % 360 - 180) for n in RC["gate"])
put("tideAmpErr", f"{math.ceil(1e4 * amp_err) / 1e2:.2f}"); put("tidePhErr", f"{math.ceil(1e2 * ph_err) / 1e2:.2f}")
OBS = {"Golden Gate": 1.079, "Richmond": 0.708, "Oakland": 0.652}
for st, key in (("Golden Gate", "GG"), ("Richmond", "RI"), ("Oakland", "OA")):
    m = RC["currents"][st]["M2_semi_major"]
    put(f"cur{key}", f"{m:.2f}"); put(f"cur{key}obs", f"{OBS[st]:.2f}"); put(f"cur{key}err", f"{100 * (m / OBS[st] - 1):+.1f}")
put("curGGalt", f"{100 * (RC['currents']['Golden Gate']['M2_semi_major'] / 0.880 - 1):+.1f}")
d = {k: CK["station_distance_km"][k] for k in ("Richmond", "Oakland")}; put("stDistMin", f"{math.floor(10 * min(d.values())) / 10:.1f}"); put("stDistMax", f"{math.ceil(10 * max(d.values())) / 10:.1f}")
a, b = CK["fit_window_amp_days3_33"], CK["fit_window_amp_days8_33"]
put("fitWinP", f"{100 * abs(b['P1'] / a['P1'] - 1):.0f}"); put("fitWinK", f"{100 * abs(b['K1'] / a['K1'] - 1):.0f}")
log = open("fluxes_mixed.log").read()
put("contRes", up(float(re.search(r"max ([0-9.e+-]+)", log).group(1))))
put("bayUniform", up(max(CK["uniform_tracer_bay"].values()), 1))

# ---- bay ----
for key, lab in (("DON", "donor cell"), ("BCG", "bounded compact, global bounds"), ("BCL", "bounded compact, local bounds"),
                 ("BCD", "bounded compact with dissipation, global bounds"), ("UCD", "compact with dissipation, unlimited")):
    r = BT[lab]
    put(f"bay{key}", str(r["cells_c_gt_0p5"])); put(f"bayw{key}", str(r["cells_c_gt_0p01"]))
assert max(BT[k]["exported"] for k in BT if isinstance(BT[k], dict)) < 1e-35, "text: Bay Bridge export < 1e-35"
put("bayInit", str(BT["initial_cells_c_gt_0p5"])); put("bayRatio", f"{BT['bounded compact, global bounds']['cells_c_gt_0p5'] / BT['donor cell']['cells_c_gt_0p5']:.1f}")
BDA = "bounded compact, global bounds, depth-aware ratio"
put("bayBCA", str(BT[BDA]["cells_c_gt_0p5"])); put("baywBCA", str(BT[BDA]["cells_c_gt_0p01"]))
put("bayCostDA", f"{BT[BDA]['wall_s'] / BT['bounded compact, global bounds']['wall_s']:.1f}")
fb = [v for v in list(BT.values()) + [x for fam in ("coast_stability", "sharp_release", "outflow_test") for x in BE[fam].values()]
      + list(LR.values()) + list(CC.values()) if isinstance(v, dict) and v.get("depth_faces_evaluated")]
allbay = [v for v in list(BT.values()) + [x for fam in ("coast_stability", "sharp_release", "outflow_test") for x in BE[fam].values()]
          + list(LR.values()) + list(CC.values()) if isinstance(v, dict) and "depth_aware" in v]
assert len(fb) == 4 and all(v["depth_aware"] for v in fb), "the four depth-aware bay runs report their fallback"
assert all(not v["depth_aware"] for v in allbay if v not in fb), "every other bay run reconstructs c directly"
fbmax = max(v["depth_fallback_faces"] / v["depth_faces_evaluated"] for v in fb)
put("bayFallback", up(fbmax, 1) if fbmax > 0 else "0")
u = BT["compact with dissipation, unlimited"]; put("bayUCDmin", f"{u['final_min']:.2f}"); put("bayUCDmax", f"{u['final_max']:.2f}")
residuals = [abs(BT[k]["budget_residual"]) for k in BT if isinstance(BT[k], dict)]
for fam in ("coast_stability", "sharp_release", "outflow_test"):
    residuals += [abs(v["budget_residual"]) for v in BE[fam].values() if isinstance(v, dict) and "budget_residual" in v]
residuals += [abs(v["budget_residual"]) for v in LR.values() if isinstance(v, dict) and "budget_residual" in v]
residuals += [abs(v["budget_residual"]) for v in CC.values() if isinstance(v, dict)]
put("bayBudget", up(max(residuals)))
ot = BE["outflow_test"]
put("outDON", f"{100 * ot['donor']['exported']:.1f}"); put("outBCG", f"{100 * ot['ccfctg']['exported']:.1f}")
put("outBCL", f"{100 * ot['ccfct']['exported']:.1f}"); put("outCells", str(ot["cells"]))
cs = BE["coast_stability"]
put("csOne", f"{min(cs['unlimited, one-sided closures, no dissipation']['min_history']):.2f}")
put("csAvg", f"{min(cs['unlimited, averaged closures, no dissipation']['min_history']):.2f}")
put("csQuarter", f"{min(cs['unlimited, averaged closures, dissipation 1/4']['min_history']):.2f}")
put("csHalf", f"{min(cs['unlimited, averaged closures, dissipation 1/2']['min_history']):.2f}")
put("csFull", f"{min(cs['unlimited, averaged closures, dissipation 1']['min_history']):.2f}")
put("csDAavg", f"{min(cs['depth-aware, unlimited, averaged closures, no dissipation']['min_history']):.1f}")
put("csDAfull", f"{min(cs['depth-aware, unlimited, averaged closures, dissipation 1']['min_history']):.2f}")
assert min(cs["depth-aware, unlimited, averaged closures, no dissipation"]["min_history"]) < 5 * min(
    cs["unlimited, averaged closures, no dissipation"]["min_history"]), "text: the ratio worsens the unlimited undershoots"
assert min(cs["depth-aware, bounded (global), averaged closures, no dissipation"]["min_history"]) > -1e-12
cc_u = CC["unlimited, one-sided cell-average closures, no dissipation"]
put("csOneCV", f"{min(cc_u['min_history']):.1f}"); put("csOneCVmax", f"{max(cc_u['max_history']):.1f}")
cc_b = CC["bounded (global), one-sided cell-average closures, no dissipation"]
assert min(cc_b["min_history"]) > -1e-12 and max(cc_b["max_history"]) < 1 + 1e-12
lrb = LR["smooth | bounded (global), no dissipation"]
put("lrPeakNo", f"{lrb['final_max']:.3f}")
put("lrPeakD", f"{LR['smooth | bounded (global), dissipation 1']['final_max']:.2f}")
put("lrCellsNo", str(LR["sharp | bounded (global), no dissipation"]["cells_c_gt_0p5"]))
put("lrCellsD", str(LR["sharp | bounded (global), dissipation 1"]["cells_c_gt_0p5"]))
put("lrUmin", f"{LR['smooth | unlimited, dissipation 1']['final_min']:.2f}")
nod = [v for k, v in LR.items() if isinstance(v, dict) and "bounded (global), no dissipation" in k]
assert len(nod) == 2 and all(len(v["min_history"]) == 720 for v in nod)
put("lrLowest", up(-min(min(v["min_history"]) for v in nod), 1)); put("lrOver", up(max(max(v["max_history"]) for v in nod) - 1, 1))
put("lrUminRun", f"{min(LR['smooth | unlimited, dissipation 1']['min_history']):.2f}")

with open("numbers.tex", "w") as f:
    f.write("% generated by make_tables.py from the result files -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- tables ----
def table(path, body):
    open(f"tables/{path}", "w").write("% generated by make_tables.py -- do not edit\n" + body)

body = "\\begin{tabular}{lccccc}\n\\toprule\nscheme & $33^2$ & $65^2$ & $129^2$ & $257^2$ & $513^2$\\\\\n\\midrule\n"
for c in rows_main:
    body += (LABEL[c] + " & " + " & ".join(f3(l2("gauss_T2", GRID[g], c)) for g in "abcd")
             + " & " + (f"${sci(E513[c], 2)}$" if c in E513 else "--") + "\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_smooth.tex", body)
body = "\\begin{tabular}{lccccc}\n\\toprule\nscheme & $33^2$ & $65^2$ & $129^2$ & $257^2$ & order\\\\\n\\midrule\n"
LABEL_DW = {"BCA": "bounded compact, global, depth-aware", "UCA": "compact, unlimited, depth-aware",
            "UCW": "compact, unlimited"}
for c in ("BCG", "BCA", "BCE", "BCU", "CFG", "UFG", "DON"):
    body += ({**LABEL, **LABEL_DW}[c] + " & " + " & ".join(f3(cve(c, GRID[g])) for g in "abcd")
             + f" & {macros['cvOrd' + c]}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_moving.tex", body)
body = ("\\begin{tabular}{llcccc}\n\\toprule\ndepth & operator & $64^2$ & $128^2$ & $256^2$ & order\\\\\n\\midrule\n")
OPROWS = (("uniform", "face values \\eqref{eq:faces}", CT, "face_value"),
          ("uniform", "flux $Q_f\\hat c_f$", CT, "flux_divergence"),
          ("uniform", "flux \\eqref{eq:transverse}", CT, "flux_divergence_corrected"),
          ("variable", "$c$ directly, \\eqref{eq:transverse}", DT_, "direct"),
          ("variable", "ratio $r_f$, no transverse term", DT_, "ratio"),
          ("variable", "flux \\eqref{eq:depthaware}", DT_, "depth_aware"))
for dep, lab, src, key in OPROWS:
    body += (f"{dep} & {lab} & " + " & ".join(f"${sci(src['errors'][N][key])}$" for N in ("64", "128", "256"))
             + f" & {src['orders'][key][-1]:.2f}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_operator.tex", body)
body = "\\begin{tabular}{lccccc}\n\\toprule\nscheme & $33^2$ & $65^2$ & $129^2$ & $257^2$ & order\\\\\n\\midrule\n"
for c in ("UCW", "UCA", "BCG", "BCA", "CFG", "UFG"):
    body += ({**LABEL, **LABEL_DW}[c] + " & " + " & ".join(f"${sci(vde(c, GRID[g]), 1)}$" for g in "abcd")
             + f" & {macros['vdOrd' + c]}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_depth.tex", body)
body = ("\\begin{tabular}{lccc}\n\\toprule\nscheme & $33^2$ & $65^2$ & $129^2$\\\\\n"
        + "$W/h$ & " + " & ".join(macros[f"sdWh{g}"] for g in "abc") + "\\\\\n\\midrule\n")
for c in ("UCW", "UCA", "BCG", "BCA", "CFG", "UFG"):
    body += {**LABEL, **LABEL_DW}[c] + " & " + " & ".join(f"${macros[f'sd{c}{g}']}$" for g in "abc") + "\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_steep.tex", body)
body = "\\begin{tabular}{lccc}\n\\toprule\nscheme & $65^2$ & $129^2$ & $257^2$\\\\\n\\midrule\n"
for c in rows_main:
    body += LABEL[c] + " & " + " & ".join(f"{f3(l2('slotted_T2', GRID[g], c))} / {CV['slotted_T2'][GRID[g]][CODE[c]]['iou']:.2f}" for g in "bcd") + "\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_slotted.tex", body)
pick = [c for c in cfls if f"{float(c):.2f}" in ("0.10", "0.35", "0.75", "0.90")]
body = ("\\begin{tabular}{l" + "c" * len(pick) + "cc}\n\\toprule\n & \\multicolumn{" + str(len(pick)) + "}{c}{largest outflow fraction $\\phi$} & \\multicolumn{2}{c}{spread (\\%)}\\\\\n"
        "scheme & " + " & ".join(f"{CF[c]['outflow_fraction']:.2f}" for c in pick) + " & smooth & slotted\\\\\n\\midrule\n")
SHORTLAB = {"BCG": "\\textbf{bounded compact, global}", "BCE": "bounded compact, extr.-pres.", "BCL": "bounded compact, local",
            "BCU": "bounded compact, uncorrected", "CFG": "bounded C4, global", "UFG": "bounded U5, global", "MUS": "MUSCL-superbee",
            "MPD": "MPDATA, 2 passes", "MPO": "MPDATA, 1 pass", "HSS": "HSIMT, split", "HSN": "HSIMT, two-level",
            "HST": "HSIMT, predictor", "DON": "donor cell"}
for c in ["BCG", "BCE", "BCL", "BCU", "CFG", "UFG", "MUS", "MPD", "MPO", "HSS", "HSN", "HST", "DON"]:
    body += SHORTLAB[c] + " & " + " & ".join(f3(CF[k][f'gauss_T2_{CODE[c]}']['rel_l2']) for k in pick)
    body += f" & {macros['cf' + c + 'spread']} & {macros['cfs' + c + 'spread']}\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_cfl.tex", body)
body = ("\\begin{tabular}{rcccccccc}\n\\toprule\n & \\multicolumn{3}{c}{half-width wall closure} & \\multicolumn{3}{c}{equal-width wall closure} & \\multicolumn{2}{c}{bounded}\\\\\n"
        "$N$ & rel.\\ $L_2$ & wall rows & order & rel.\\ $L_2$ & wall rows & order & data bounds & $[0,1]$\\\\\n\\midrule\n")
Ns = sorted(WC[LH], key=int)
for i, N in enumerate(Ns):
    oh = "" if i == 0 else o2(LH, "max_err_wall_rows", Ns[i - 1], N)
    oe = "" if i == 0 else o2(LE, "max_err_wall_rows", Ns[i - 1], N)
    body += (f"${N}^2$ & ${sci(wc(LH, 'rel_l2', N))}$ & ${sci(wc(LH, 'max_err_wall_rows', N))}$ & {oh} & "
             f"${sci(wc(LE, 'rel_l2', N))}$ & ${sci(wc(LE, 'max_err_wall_rows', N))}$ & {oe} & ${sci(wc(LB, 'rel_l2', N))}$"
             f" & ${sci(wc(LU, 'rel_l2', N))}$\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_wall.tex", body)
body = "\\begin{tabular}{lccc}\n\\toprule\nHSIMT time integration & $2+x^2$ & $2+y^2$ & $2+xy$\\\\\n\\midrule\n"
for var, key in (("two-level (no predictor)", "N"), ("ROMS predictor", "R"), ("Strang split", "S")):
    body += f"{var} & " + " & ".join(f"${macros['hc' + key + tk]}$" for tk in ("xx", "yy", "xy")) + "\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_hsimt.tex", body)
body = "\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{NOAA 9414290} & \\multicolumn{2}{c}{model, Golden Gate} & \\multicolumn{2}{c}{offshore forcing}\\\\\nconstituent & amp.\\ (m) & phase & amp.\\ (m) & phase & amp.\\ (m) & phase\\\\\n\\midrule\n"
for n in RC["gate"]:
    gte = RC["gate"][n]
    body += (f"{n[0]}$_{n[1]}$ & {gte['noaa_amp']:.3f} & {gte['noaa_phase']:.1f} & {gte['model_amp']:.3f} & {gte['model_phase']:.1f} & "
             f"{RC['offshore_amp'][n]:.3f} & {RC['offshore_phase_deg'][n] % 360:.1f}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_tide.tex", body)
body = "\\begin{tabular}{lccc}\n\\toprule\nstation & observed (m\\,s$^{-1}$) & model (m\\,s$^{-1}$) & difference\\\\\n\\midrule\n"
for st, src in (("Golden Gate", "C1$^{a}$"), ("Richmond", "ADP 2000"), ("Oakland", "ADP 2000")):
    m = RC["currents"][st]["M2_semi_major"]
    body += f"{st} ({src}) & {OBS[st]:.2f} & {m:.2f} & {100 * (m / OBS[st] - 1):+.1f}\\%\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_currents.tex", body)
body = "\\begin{tabular}{lccc}\n\\toprule\nscheme & range & $c>0.5$ & $c>0.01$\\\\\n\\midrule\n"
for key, lab in (("DON", "donor cell"), ("BCG", "bounded compact, global bounds"),
                 ("BCA", "bounded compact, global bounds, depth-aware ratio"), ("BCL", "bounded compact, local bounds"),
                 ("BCD", "bounded compact with dissipation, global bounds"), ("UCD", "compact with dissipation, unlimited")):
    r = BT[lab]; lo = r["final_min"]; hi = r["final_max"]
    rng_ = f"$[{'0' if abs(lo) < 5e-3 else f'{lo:.2f}'},\\,{hi:.2f}]$"
    short = {"bounded compact, global bounds": "\\textbf{bounded compact, global}", "bounded compact, local bounds": "bounded compact, local",
             "bounded compact, global bounds, depth-aware ratio": "bounded compact, global, depth-aware",
             "bounded compact with dissipation, global bounds": "bounded compact + dissipation, global",
             "compact with dissipation, unlimited": "compact + dissipation, unlimited"}.get(lab, lab)
    body += f"{short} & {rng_} & {r['cells_c_gt_0p5']} & {r['cells_c_gt_0p01']}\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_bay.tex", body)
SHORT = {"unlimited, one-sided closures, no dissipation": "unlimited, one-sided (point-sample weights)",
         "unlimited, one-sided cell-average closures, no dissipation": "unlimited, one-sided (cell-average weights)",
         "bounded (global), one-sided cell-average closures, no dissipation": "bounded, one-sided (cell-average weights)",
         "unlimited, averaged closures, no dissipation": "unlimited, averaged",
         "unlimited, averaged closures, dissipation 1/4": "unlimited, averaged, dissipation $\\times\\tfrac14$",
         "unlimited, averaged closures, dissipation 1/2": "unlimited, averaged, dissipation $\\times\\tfrac12$",
         "unlimited, averaged closures, dissipation 1": "unlimited, averaged, dissipation $\\times1$",
         "bounded (global), averaged closures, no dissipation": "\\textbf{bounded, averaged}",
         "bounded (global), averaged closures, dissipation 1": "bounded, averaged, dissipation $\\times1$",
         "depth-aware, unlimited, averaged closures, no dissipation": "depth-aware: unlimited, averaged",
         "depth-aware, unlimited, averaged closures, dissipation 1": "depth-aware: unlimited, averaged, dissipation $\\times1$",
         "depth-aware, bounded (global), averaged closures, no dissipation": "depth-aware: bounded, averaged"}
body = "\\begin{tabular}{lcc}\n\\toprule\nflux, coastal closure & min.\\ of $c$ & cells $<-0.01$ (coastal)\\\\\n\\midrule\n"
rows_cs = list(cs.items())
rows_cs = rows_cs[:1] + [(k, v) for k, v in CC.items() if isinstance(v, dict)] + rows_cs[1:]
for lab, v in rows_cs:
    mn = min(v['min_history'])
    body += f"{SHORT[lab]} & ${'0' if abs(mn) < 5e-3 else f'{mn:.2f}'}$ & {v['cells_below_minus_0p01']} ({v['of_which_coastal']})\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_coast.tex", body)
nrows = (("bounded compact + dissipation, global", "cfctgd"), ("bounded compact, global", "cfctg"), ("MUSCL-superbee, advective form", "tvd"),
         ("MPDATA, two corrective passes", "mpdata3_no"), ("HSIMT, ROMS predictor", "hsimt"),
         ("HSIMT, two-level", "hsimt_nopred"), ("third-order upwind, advective form", "up3"), ("first-order upwind, advective form", "up1"))
body = "\\begin{tabular}{lcccc}\n\\toprule\nscheme & $33^2$ & $65^2$ & $129^2$ & $257^2$\\\\\n\\midrule\n"
for lab, s in nrows:
    body += lab + " & " + " & ".join(f3(NOD["gauss_T2"][N][s]["rel_l2"]) for N in ("33", "65", "129", "257")) + "\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"; table("tab_nodal.tex", body)
print(f"wrote numbers.tex ({len(macros)} macros) and", sorted(os.listdir("tables")))
