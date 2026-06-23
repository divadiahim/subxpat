#!/bin/bash
# Submit the DCT case study with 1A pre-filter labelling to Hábrók.
#
# One job: all four DCT coefficient circuits at ET in {16, 64} with the 1A
# structural pre-filter labelling (SXPAT_LABELING_METHOD=prefilter), followed by
# the image-quality pipeline. The DCT runs dominate wall time (the i16 circuits
# take up to ~15 min each), so a single multi-core node is used.
#
# Resources: 48 cores, 48G, 24h (matches the fullrun medium tier).
#
# Usage:
#   ./cluster/submit_dct.sh
#
# After completion, results land in output/report/habrok_dct/:
#   dct_area_1a.csv          (WP5 area-vs-ET, 1A labelling)
#   dct_image_quality_1a.csv (WP6 PSNR/SSIM)
#   dct_approx_1a/           (approximate circuits)

set -euo pipefail
mkdir -p cluster/logs

CPUS=48
MEM="48G"
TIME="24:00:00"

echo "=== Submitting DCT 1A case study ==="
echo "  cpus=${CPUS}, mem=${MEM}, time=${TIME}"
JOB=$(sbatch \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --parsable \
    cluster/dct.job)
echo "  Job ID: ${JOB}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     cluster/logs/dct1a_${JOB}.out"
echo "Results:  output/report/habrok_dct/"
