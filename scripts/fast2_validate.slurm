#!/bin/bash
# ---------------------------------------------------------------------------
#  Validation of viral_tree_fast2.c (threads inside a sample + fused kernel)
#  on a compute node.
#
#      suite:  sbatch --job-name=viral_fast2_suite --cpus-per-task=8 --mem=8G \
#                     --time=03:00:00 scripts/fast2_validate.slurm suite
#      large:  sbatch --job-name=viral_fast2_large --cpus-per-task=8 --mem=8G \
#                     --time=01:30:00 scripts/fast2_validate.slurm large
#
#  suite compares against the frozen campaign binary for p = 1..24 at 1, 4
#  and 8 threads; large compares against the validated viral_tree_fast at
#  p = 26, 28, 30 with 1, 2, 4, 8 threads and records time and peak RSS.
#  QoS cpu168: 2 x 8G fits beside the running p=32 tasks (62 x 32G of the
#  2000G cap), so they lose no slot, and p=30's cpu48 quota is not touched.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast2
#SBATCH --account=torch_pr_577_general
#SBATCH --qos=cpu168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
MODE="${1:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${SCRATCH:?SCRATCH must be set}/viral-fast"
FROZEN="$HOME/viral-exe-backup/torch-large-builds-17989395/viral_baseline"
FAST="$HOME/viral-exe-backup/fast-builds-18517761/viral_fast"
TAG="${SLURM_JOB_ID:-manual}"
cd "$REPO" || exit 1
mkdir -p "$OUT/logs"
echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   cpus ${SLURM_CPUS_PER_TASK:-?}   mode $MODE   job $TAG"
echo "viral_tree_fast2.c sha256 $(sha256sum viral_tree_fast2.c | cut -c1-16)"
module purge 2>/dev/null
command -v gcc >/dev/null 2>&1 && [ -e /usr/include/stdio.h ] || { echo "FATAL: no usable gcc" >&2; exit 1; }
echo "gcc: $(gcc --version | head -1)"
export OMP_PROC_BIND=close OMP_PLACES=cores

case "$MODE" in
suite) exec python3 tests/test_fast2_equivalence.py suite --reference-exe "$FROZEN" --max-p 24 \
           --max-threads "${SLURM_CPUS_PER_TASK:-8}" --report "$OUT/fast2_suite_$TAG.json" ;;
large) exec python3 tests/test_fast2_equivalence.py large --reference-exe "$FROZEN" --fast-exe "$FAST" \
           --large-depths 26,28,30 --max-threads "${SLURM_CPUS_PER_TASK:-8}" \
           --report "$OUT/fast2_large_$TAG.json" ;;
scaling) lscpu | grep -E '^(NUMA node[0-9]|Socket)' ; taskset -cp $$
       exec python3 tests/test_fast2_equivalence.py large --reference-exe "$FROZEN" --fast-exe "$FAST" \
           --large-depths 30 --threads 1,8,16,32,64 --samples 8 --max-threads "${SLURM_CPUS_PER_TASK:-64}" \
           --report "$OUT/fast2_scaling_$TAG.json" ;;
*) echo "usage: $0 suite|large|scaling" >&2; exit 2 ;;
esac
