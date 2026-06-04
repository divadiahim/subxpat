#!/bin/bash
# Submit benchmark and profiling array jobs to Hábrók with tiered resources.
#
# Tiers (based on circuit input size):
#   small  (i4-i16):   16 cores,  16G,  4h
#   medium (i20-i32):  48 cores,  48G, 24h
#   large  (i36+):     64 cores,  64G, 72h
#
# Usage:
#   ./cluster/submit_all.sh                  # all tiers, both scripts
#   ./cluster/submit_all.sh --small          # small tier only
#   ./cluster/submit_all.sh --medium         # small + medium
#   ./cluster/submit_all.sh --large          # all tiers (same as no flag)
#   ./cluster/submit_all.sh --bench-only     # benchmark script only
#   ./cluster/submit_all.sh --prof-only      # profiling script only
#
# After completion:
#   python cluster/merge_results.py

set -euo pipefail

mkdir -p cluster/logs

# ── Tier definitions ──────────────────────────────────────────────────
# Array ranges match line numbers in cluster/benchmarks.txt:
#   1-14:  abs_diff  (i4..i48)     → small: 1-3,  medium: 4-7,  large: 8-14
#   15-32: adder     (i4..i64)     → small: 15-20, medium: 21-24, large: 25-32
#   33-37: madd      (i6..i18)     → small: 33-34, medium: 35-37
#   38-45: mul       (i4..i20)     → small: 38-40, medium: 41-43, large: 44-45
#   46-50: sad       (i10..i50)    → small: 46,    medium: 47,    large: 48-50

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

# ── Parse flags ───────────────────────────────────────────────────────
MAX_TIER="large"
RUN_BENCH=true
RUN_PROF=true

for arg in "$@"; do
    case "${arg}" in
        --small)       MAX_TIER="small" ;;
        --medium)      MAX_TIER="medium" ;;
        --large)       MAX_TIER="large" ;;
        --bench-only)  RUN_PROF=false ;;
        --prof-only)   RUN_BENCH=false ;;
        *)
            echo "Unknown flag: ${arg}"
            echo "Usage: $0 [--small|--medium|--large] [--bench-only|--prof-only]"
            exit 1
            ;;
    esac
done

# ── Submit function ───────────────────────────────────────────────────
submit_tier() {
    local tier_name=$1 array_range=$2 cpus=$3 mem=$4 time_limit=$5 job_script=$6

    echo "  ${tier_name}: array=${array_range}, cpus=${cpus}, mem=${mem}, time=${time_limit}"
    sbatch \
        --array="${array_range}" \
        --cpus-per-task="${cpus}" \
        --mem="${mem}" \
        --time="${time_limit}" \
        --parsable \
        "${job_script}"
}

echo ""
echo "=== Submitting with max tier: ${MAX_TIER} ==="
echo ""

for script_label in bench prof; do
    if [[ "${script_label}" == "bench" && "${RUN_BENCH}" == false ]]; then continue; fi
    if [[ "${script_label}" == "prof"  && "${RUN_PROF}"  == false ]]; then continue; fi

    if [[ "${script_label}" == "bench" ]]; then
        JOB_SCRIPT="cluster/benchmark_labeling.job"
        echo "── benchmark_labeling ──"
    else
        JOB_SCRIPT="cluster/profile_labeling.job"
        echo "── profile_labeling ──"
    fi

    SMALL_JOB=$(submit_tier "small" "${SMALL_RANGE}" ${SMALL_CPUS} "${SMALL_MEM}" "${SMALL_TIME}" "${JOB_SCRIPT}")
    echo "    Job ID: ${SMALL_JOB}"

    if [[ "${MAX_TIER}" == "medium" || "${MAX_TIER}" == "large" ]]; then
        MED_JOB=$(submit_tier "medium" "${MEDIUM_RANGE}" ${MEDIUM_CPUS} "${MEDIUM_MEM}" "${MEDIUM_TIME}" "${JOB_SCRIPT}")
        echo "    Job ID: ${MED_JOB}"
    fi

    if [[ "${MAX_TIER}" == "large" ]]; then
        LRG_JOB=$(submit_tier "large" "${LARGE_RANGE}" ${LARGE_CPUS} "${LARGE_MEM}" "${LARGE_TIME}" "${JOB_SCRIPT}")
        echo "    Job ID: ${LRG_JOB}"
    fi

    echo ""
done

echo "Monitor:  squeue -u \$USER"
echo "Details:  sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS"
echo ""
echo "After completion:  python cluster/merge_results.py"
