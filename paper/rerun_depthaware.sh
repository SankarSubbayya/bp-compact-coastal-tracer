#!/bin/bash
# Full rerun on one code hash after the depth-aware face values (2026-10-28). Logs jcp2_*.log; status jcp2_runs.log.
export PYTHONDONTWRITEBYTECODE=1
U="uv run -q --with numpy --with scipy --with numba"
: > jcp2_runs.log
r() { name=$1; shift; ( "$@" > jcp2_$name.log 2>&1; echo "$name exit $?" >> jcp2_runs.log ); }
r cv_reference $U python3 cv_reference.py &
r cfl_sweep $U python3 cfl_sweep_cv.py &
r ranking $U python3 ranking_check.py &
r appendix $U python3 appendix_nodal.py &
r wall $U python3 wall_convergence.py &
r order513 $U python3 order_513.py &
r changing_volume $U python3 changing_volume.py &
r variable_depth $U python3 variable_depth.py &
( r bay_table $U python3 bay_table.py; r bay_experiments $U python3 bay_experiments.py;
  r coast_closure $U python3 coast_closure_check.py; r long_run $U python3 long_run.py;
  r uniform $U python3 uniform_tracer_check.py ) &
( r depth_test $U python3 depth_test.py; r covariance_test $U python3 covariance_test.py;
  r closure_spectra $U python3 closure_spectra.py ) &
wait
echo MAIN-DONE >> jcp2_runs.log
# compiled runs last, on an otherwise idle machine
r ablation $U python3 covariance_ablation.py
r timing $U python3 timing_benchmark.py
r correction_cost $U python3 correction_cost.py
echo ALL-DONE >> jcp2_runs.log
