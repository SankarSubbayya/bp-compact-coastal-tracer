#!/bin/bash
# Pre-submission review, round 3 (2026-10-29): chaplin_colella.py gained two opt-in reading switches (defaults unchanged) and
# corrected comments; cc_validation.py records both readings of sigma and the one-dimensional smooth test. Reruns the
# NumPy comparator drivers (cc_benchmark and cc_range must reproduce their archived numbers) and the new sensitivity
# run. Logs jcp7_*.log; status jcp7_runs.log.
export PYTHONDONTWRITEBYTECODE=1
U="uv run -q --with numpy --with scipy"
: > jcp7_runs.log
r() { name=$1; shift; echo "$name start $(date +%H:%M) load $(sysctl -n vm.loadavg)" >> jcp7_runs.log
      "$@" > jcp7_$name.log 2>&1; echo "$name exit $? $(date +%H:%M)" >> jcp7_runs.log; }
r cc_benchmark $U python3 cc_benchmark.py
r cc_range $U python3 cc_range.py
r cc_validation $U python3 cc_validation.py
r cc_sensitivity $U python3 cc_sensitivity.py
echo "ALL-DONE $(date +%H:%M)" >> jcp7_runs.log
