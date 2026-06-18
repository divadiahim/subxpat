#!/bin/bash
# Submit the full-run SubXPAT benchmark (per-stage Z3 calls, time, area) to Hábrók
# with tiered resources, matching submit_all.sh.
#
# Tiers (same core/mem/time as the other experiments):
#   small  : 16 cores, 16G,  4h
#   medium : 48 cores, 48G, 24h
#
# One array task per benchmark in cluster/benchmarks_fullrun.txt; each task runs
# all 6 labelling modes at 2 error thresholds. Tiers are cumulative (like
# submit_all.sh): --medium submits small + medium.
#
# Array ranges (line numbers in cluster/benchmarks_fullrun.txt):
#   1-3   adder i4,i6,i8       7-8   mul i4,i6      10-11 abs_diff i4,i8   13 madd i6     -> small
#   4-6   adder i10,i12,i16    9     mul i8         12    abs_diff i12     14 madd i9
#   15-18 dct4c0..c3 (i16)                                                                -> medium
#
# Usage:
#   ./cluster/submit_fullrun.sh             # all tiers (small + medium)
#   ./cluster/submit_fullrun.sh --small     # small tier only
#   ./cluster/submit_fullrun.sh --medium    # small + medium (default)
#
# After completion:
#   python cluster/merge_results.py

set -euo pipefail
mkdir -p cluster/logs

SMALL_RANGE="1-3,7-8,10-11,13"
SMALL_CPUS=16
SMALL_MEM="16G"
SMALL_TIME="04:00:00"

MEDIUM_RANGE="4-6,9,12,14-18"
MEDIUM_CPUS=48
MEDIUM_MEM="48G"
MEDIUM_TIME="24:00:00"

MAX_TIER="medium"
for arg in "$@"; do
    case "${arg}" in
        --small)  MAX_TIER="small" ;;
        --medium) MAX_TIER="medium" ;;
        --large)  MAX_TIER="medium" ;;   # no large circuits in the fullrun set
        *) echo "Unknown flag: ${arg}"; echo "Usage: $0 [--small|--medium]"; exit 1 ;;
    esac
done

submit_tier() {
    local name=$1 range=$2 cpus=$3 mem=$4 tl=$5
    echo "  ${name}: array=${range}, cpus=${cpus}, mem=${mem}, time=${tl}"
    sbatch \
        --array="${range}" \
        --cpus-per-task="${cpus}" \
        --mem="${mem}" \
        --time="${tl}" \
        --parsable \
        cluster/fullrun.job
}

echo ""
echo "=== Submitting full-run benchmark, max tier: ${MAX_TIER} ==="
echo ""

S_JOB=$(submit_tier "small" "${SMALL_RANGE}" ${SMALL_CPUS} "${SMALL_MEM}" "${SMALL_TIME}")
echo "    Job ID: ${S_JOB}"

if [[ "${MAX_TIER}" == "medium" ]]; then
    M_JOB=$(submit_tier "medium" "${MEDIUM_RANGE}" ${MEDIUM_CPUS} "${MEDIUM_MEM}" "${MEDIUM_TIME}")
    echo "    Job ID: ${M_JOB}"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "After completion:  python cluster/merge_results.py"
