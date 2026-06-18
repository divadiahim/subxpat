#!/bin/bash
# Submit the full-run SubXPAT benchmark (per-stage Z3 calls, time, area) to Hábrók.
# One array task per benchmark in cluster/benchmarks_fullrun.txt (18 benchmarks).
# Each task runs all 6 labelling modes at 2 error thresholds.
#
# Usage:
#   ./cluster/submit_fullrun.sh            # all benchmarks
#   ./cluster/submit_fullrun.sh 1-9        # subset (array range)
#
# After completion:
#   python cluster/merge_results.py

set -euo pipefail
mkdir -p cluster/logs

ARRAY_RANGE="${1:-1-18}"
CPUS=32
MEM="32G"
TIME="24:00:00"

echo "Submitting full-run benchmark: array=${ARRAY_RANGE}, cpus=${CPUS}, mem=${MEM}, time=${TIME}"
sbatch \
    --array="${ARRAY_RANGE}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --parsable \
    cluster/fullrun.job

echo ""
echo "Monitor:  squeue -u \$USER"
echo "After completion:  python cluster/merge_results.py"
