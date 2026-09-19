#!/bin/bash
# Submit from the repository root; see CODEX_HPC_HANDOFF.md.
# Arguments: depth samples-per-job angles-file executable campaign-directory
# Supply --array, --cpus-per-task, --mem, --time, --account and --output at submission time.
#SBATCH --job-name=viral_tree
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
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_THREAD_LIMIT=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset VIRAL_TREE_DUMP
# Both exec calls preserve the PID so the batch-shell SIGTERM reaches the solver.
# The helper verifies the frozen manifest and records sidecar provenance first.
exec python3 scripts/viral_campaign.py run \
    --p "$1" --samples "$2" --angles "$3" --exe "$4" --out "$5"
