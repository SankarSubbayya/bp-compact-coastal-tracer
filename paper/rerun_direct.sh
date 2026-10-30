#!/bin/bash
# Full rerun on one code hash after reverting the default to direct reconstruction of the stored concentration, with
# the depth-aware ratio as the "_da" variant (2026-10-28, steep_depth.py). Logs jcp5_*.log; status jcp5_runs.log.
# cc_validation.json is not rerun: the periodic tests do not use gen_conservative and chaplin_colella.py is unchanged.
export PYTHONDONTWRITEBYTECODE=1
U="uv run -q --with numpy --with scipy --with numba"
: > jcp5_runs.log
r() { name=$1; shift; ( "$@" > jcp5_$name.log 2>&1; echo "$name exit $? $(date +%H:%M)" >> jcp5_runs.log ); }
r cv_reference $U python3 cv_reference.py &
r cfl_sweep $U python3 cfl_sweep_cv.py &
r ranking $U python3 ranking_check.py &
r appendix $U python3 appendix_nodal.py &
r wall $U python3 wall_convergence.py &
r order513 $U python3 order_513.py &
r changing_volume $U python3 changing_volume.py &
r variable_depth $U python3 variable_depth.py &
r steep_depth $U python3 steep_depth.py &
( r cc_benchmark $U python3 cc_benchmark.py; r cc_range $U python3 cc_range.py ) &
( r bay_table $U python3 bay_table.py; r bay_experiments $U python3 bay_experiments.py;
  r coast_closure $U python3 coast_closure_check.py; r long_run $U python3 long_run.py;
  r uniform $U python3 uniform_tracer_check.py ) &
( r depth_test $U python3 depth_test.py; r covariance_test $U python3 covariance_test.py;
  r closure_spectra $U python3 closure_spectra.py ) &
wait
echo "MAIN-DONE $(date +%H:%M)" >> jcp5_runs.log
# compiled runs last, on an otherwise idle machine
r ablation $U python3 covariance_ablation.py
r timing $U python3 timing_benchmark.py
r correction_cost $U python3 correction_cost.py
r cc_cost $U python3 cc_cost.py
echo "ALL-DONE $(date +%H:%M)" >> jcp5_runs.log
