#!/bin/bash
# ---------------------------------------------------------------------------
#  Validation record, as run on NYU Torch for the published campaigns (tag
#  results-10M).  Cluster-bound: account, /scratch/fm2452 and $HOME/viral-exe-backup
#  paths are Torch's.  Submit from the repo root.  See docs/validation.md.
#
#  Checks R1 / R2 (docs/validation.md).  Builds viral_tree_fast.c (and the protected
#  reference) on a compute node - the login node has no C toolchain - and runs
#  tests/test_fast_equivalence.py against the frozen campaign binary.
#
#      R1 equivalence:  sbatch --job-name=viral_fast_r1 --cpus-per-task=4 --mem=8G \
#                              --time=02:00:00 scripts/validation/fast_validate.slurm suite
#      R2 benchmark:    sbatch --job-name=viral_fast_r2 --cpus-per-task=1 --mem=16G \
#                              --time=02:00:00 scripts/validation/fast_validate.slurm bench
#
#  Reports and logs go to $SCRATCH/viral-fast/.  R1's four CPUs are only for
#  the original check.sh, which has a four-thread test.  R2 needs 16G for the
#  p=28 reference (14.1 GiB); everything else is far smaller.  QoS cpu168 has
#  its own caps, so these never wait on the cpu48 memory the p=30 campaign
#  fills; the cpu_short QoS is not valid for this account.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast
#SBATCH --account=torch_pr_577_general
#SBATCH --qos=cpu168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
MODE="${1:-}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT="${SCRATCH:?SCRATCH must be set}/viral-fast"
FROZEN="$HOME/viral-exe-backup/torch-large-builds-17989395/viral_baseline"
TAG="${SLURM_JOB_ID:-manual}"
cd "$REPO" || exit 1
mkdir -p "$OUT/logs"

echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   mode $MODE   job $TAG"
echo "repo $REPO   viral_tree_fast.c sha256 $(sha256sum viral_tree_fast.c | cut -c1-16)"

module purge 2>/dev/null
if ! command -v gcc >/dev/null 2>&1 || [ ! -e /usr/include/stdio.h ]; then
    echo "FATAL: no usable gcc here; the comparison must use the campaign compiler." >&2
    exit 1
fi
echo "gcc: $(gcc --version | head -1)"
if [ ! -x "$FROZEN" ]; then
    echo "FATAL: frozen reference binary missing: $FROZEN" >&2
    exit 1
fi

case "$MODE" in
suite)
    exec python3 tests/test_fast_equivalence.py --cc gcc --reference-exe "$FROZEN" \
        --max-p 26 --report "$OUT/r1_$TAG.json" --keep "$OUT/builds-$TAG" ;;
bench)
    exec python3 tests/test_fast_equivalence.py --cc gcc --reference-exe "$FROZEN" --no-suite \
        --benchmark --bench-depths 20,22,24,26,28 --fast-only-depths 30 --report "$OUT/r2_$TAG.json" ;;
*)
    echo "usage: $0 suite|bench" >&2
    exit 2 ;;
esac
