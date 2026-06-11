#!/bin/bash
# Submit the pre-filter benchmark array job to Hábrók with tiered resources.
# Same tiers as submit_all.sh.
#
# Tiers (based on circuit input size):
#   small  (i4-i16):   16 cores,  16G,  4h
#   medium (i20-i32):  48 cores,  48G, 24h
#   large  (i36+):     64 cores,  64G, 72h
#
# Usage:
#   ./cluster/submit_prefilter.sh             # all tiers
#   ./cluster/submit_prefilter.sh --small     # small tier only
#   ./cluster/submit_prefilter.sh --medium    # small + medium
#
# After completion:
#   python cluster/merge_results.py

set -euo pipefail

mkdir -p cluster/logs

# ── Tier definitions (keep in sync with submit_all.sh) ────────────────
SMALL_RANGE="1-3,15-20,33-34,38-40,46"
SMALL_CPUS=16
SMALL_MEM="16G"
SMALL_TIME="04:00:00"

MEDIUM_RANGE="4-7,21-24,35-37,41-43,47"
MEDIUM_CPUS=48
MEDIUM_MEM="48G"
MEDIUM_TIME="24:00:00"

LARGE_RANGE="8-14,25-32,44-45,48-50"
LARGE_CPUS=64
LARGE_MEM="64G"
LARGE_TIME="72:00:00"

JOB_SCRIPT="cluster/benchmark_prefilter.job"

# ── Parse flags ───────────────────────────────────────────────────────
MAX_TIER="large"
for arg in "$@"; do
    case "${arg}" in
        --small)  MAX_TIER="small" ;;
        --medium) MAX_TIER="medium" ;;
        --large)  MAX_TIER="large" ;;
        *)
            echo "Unknown flag: ${arg}"
            echo "Usage: $0 [--small|--medium|--large]"
            exit 1
            ;;
    esac
done

submit_tier() {
    local tier_name=$1 array_range=$2 cpus=$3 mem=$4 time_limit=$5

    echo "  ${tier_name}: array=${array_range}, cpus=${cpus}, mem=${mem}, time=${time_limit}"
    sbatch \
        --array="${array_range}" \
        --cpus-per-task="${cpus}" \
        --mem="${mem}" \
        --time="${time_limit}" \
        --parsable \
        "${JOB_SCRIPT}"
}

echo ""
echo "=== Submitting prefilter benchmark, max tier: ${MAX_TIER} ==="
echo ""

SMALL_JOB=$(submit_tier "small" "${SMALL_RANGE}" ${SMALL_CPUS} "${SMALL_MEM}" "${SMALL_TIME}")
echo "    Job ID: ${SMALL_JOB}"

if [[ "${MAX_TIER}" == "medium" || "${MAX_TIER}" == "large" ]]; then
    MED_JOB=$(submit_tier "medium" "${MEDIUM_RANGE}" ${MEDIUM_CPUS} "${MEDIUM_MEM}" "${MEDIUM_TIME}")
    echo "    Job ID: ${MED_JOB}"
fi

if [[ "${MAX_TIER}" == "large" ]]; then
    LRG_JOB=$(submit_tier "large" "${LARGE_RANGE}" ${LARGE_CPUS} "${LARGE_MEM}" "${LARGE_TIME}")
    echo "    Job ID: ${LRG_JOB}"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "After completion:  python cluster/merge_results.py"
