#!/bin/bash
# Chaplin-Colella (published article) runs: validation, benchmark, range diagnostics. Logs jcp4_cc_*.log.
export PYTHONDONTWRITEBYTECODE=1
U="uv run -q --with numpy --with scipy"
: > jcp4_cc_runs.log
( $U python3 cc_validation.py > jcp4_cc_validation.log 2>&1; echo "cc_validation exit $?" >> jcp4_cc_runs.log ) &
( $U python3 cc_benchmark.py > jcp4_cc_benchmark.log 2>&1; echo "cc_benchmark exit $?" >> jcp4_cc_runs.log ) &
( $U python3 cc_range.py > jcp4_cc_range.log 2>&1; echo "cc_range exit $?" >> jcp4_cc_runs.log ) &
wait; echo ALL-DONE >> jcp4_cc_runs.log
