#!/bin/bash
# run_viral.slurm for campaigns with the validated fast solver: identical,
# except that the launcher is scripts/fast_campaign.py (p = 32 million-v1).
# See docs/campaigns.md.
# Arguments: depth samples-per-job angles-file executable campaign-directory
# Supply --array, --qos, --mem, --time, --account and --output at submission time.
#SBATCH --job-name=viral_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm_%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail
if [[ $# != 5 ]]; then
    echo "Usage: $0 DEPTH SAMPLES ANGLES EXECUTABLE CAMPAIGN_DIR" >&2
    exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?submit as a SLURM array}"
: "${SLURM_ARRAY_JOB_ID:?submit as a SLURM array}"
if [[ ${SLURM_CPUS_PER_TASK:-1} != 1 ]]; then
    echo "This wrapper requires --cpus-per-task=1." >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset VIRAL_TREE_DUMP VIRAL_FORCE_REF VIRAL_SELFCHECK
# Both exec calls preserve the PID so the batch-shell SIGTERM reaches the solver.
exec python3 scripts/fast_campaign.py run \
    --p "$1" --samples "$2" --angles "$3" --exe "$4" --out "$5"
