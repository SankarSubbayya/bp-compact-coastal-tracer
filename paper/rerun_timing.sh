#!/bin/bash
# Compiled timing runs on an idle machine (2026-10-28: the first pass of rerun_direct.sh shared the machine with
# container workloads, and the same compiled scheme took 14.8 s in timing_benchmark.py and 21.5 s in
# correction_cost.py). Waits for any running cc_cost.py to exit and for the 5-minute load average to fall below
# LOAD, then runs the three timing scripts in sequence, logging the load averages before and after each.
# Logs jcp6_*.log; status jcp6_runs.log. The accuracy results are unaffected and are not rerun.
export PYTHONDONTWRITEBYTECODE=1
U="uv run -q --with numpy --with scipy --with numba"
LOAD=${LOAD:-1.5}
: > jcp6_runs.log
while pgrep -f "python3 cc_cost.py" > /dev/null; do sleep 60; done
load5() { sysctl -n vm.loadavg | awk '{print $3}'; }
while awk -v l="$(load5)" -v m="$LOAD" 'BEGIN { exit !(l >= m) }'; do sleep 60; done
echo "idle $(date +%H:%M) load $(sysctl -n vm.loadavg)" >> jcp6_runs.log
r() { name=$1; shift; echo "$name start $(date +%H:%M) load $(sysctl -n vm.loadavg)" >> jcp6_runs.log
      "$@" > jcp6_$name.log 2>&1; echo "$name exit $? $(date +%H:%M) load $(sysctl -n vm.loadavg)" >> jcp6_runs.log; }
r timing $U python3 timing_benchmark.py
r correction_cost $U python3 correction_cost.py
r cc_cost $U python3 cc_cost.py
echo "ALL-DONE $(date +%H:%M)" >> jcp6_runs.log
