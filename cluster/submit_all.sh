#!/bin/bash
# Submit benchmark and profiling array jobs to Hábrók.
#
# Usage:
#   bash cluster/submit_all.sh              # submit both jobs (all 50 benchmarks)
#   bash cluster/submit_all.sh --small      # only small circuits (i4-i16)
#   bash cluster/submit_all.sh --adders     # only adder family
#
# After jobs finish:
#   python cluster/merge_results.py         # merge per-benchmark CSVs

set -euo pipefail

mkdir -p cluster/logs

# Parse subset flags
ARRAY_RANGE="1-50"
case "${1:-}" in
    --small)
        # abs_diff i4-i16 (1-3), adder i4-i16 (15-20), madd i6 (33), mul i4-i8 (38-40), sad i10 (46)
        ARRAY_RANGE="1-3,15-20,33,38-40,46"
        echo "Submitting SMALL benchmarks only: ${ARRAY_RANGE}"
        ;;
    --adders)
        ARRAY_RANGE="15-32"
        echo "Submitting ADDER benchmarks only: ${ARRAY_RANGE}"
        ;;
    --medium)
        # Everything up to ~i24
        ARRAY_RANGE="1-7,15-22,33-35,38-42,46-47"
        echo "Submitting MEDIUM benchmarks: ${ARRAY_RANGE}"
        ;;
    "")
        echo "Submitting ALL 50 benchmarks"
        ;;
    *)
        echo "Unknown flag: $1"
        echo "Usage: $0 [--small|--medium|--adders]"
        exit 1
        ;;
esac

echo ""
echo "=== Submitting benchmark_labeling ==="
BENCH_JOB=$(sbatch --array="${ARRAY_RANGE}" --parsable cluster/benchmark_labeling.job)
echo "Job ID: ${BENCH_JOB}"

echo ""
echo "=== Submitting profile_labeling ==="
PROF_JOB=$(sbatch --array="${ARRAY_RANGE}" --parsable cluster/profile_labeling.job)
echo "Job ID: ${PROF_JOB}"

echo ""
echo "Jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${BENCH_JOB} --format=JobID,JobName,State,Elapsed,MaxRSS"
echo "  sacct -j ${PROF_JOB} --format=JobID,JobName,State,Elapsed,MaxRSS"
echo ""
echo "After completion, merge results:"
echo "  python cluster/merge_results.py"
