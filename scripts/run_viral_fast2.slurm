#!/bin/bash
# run_viral_fast.slurm for viral_tree_fast2 campaigns: the job's CPUs are the
# threads inside each sample (OMP_NUM_THREADS = --cpus-per-task), and
# scripts/fast2_campaign.py refuses to launch unless that equals the
# campaign's frozen thread count (extension campaigns).  See docs/campaigns.md.
# Arguments: depth samples-per-job angles-file executable campaign-directory
# Supply --array, --qos, --cpus-per-task, --mem, --time, --account, --output.
#SBATCH --job-name=viral_fast2
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
: "${SLURM_CPUS_PER_TASK:?submit with --cpus-per-task}"
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset VIRAL_TREE_DUMP VIRAL_FORCE_REF VIRAL_SELFCHECK VIRAL_KERNEL VIRAL_PAR_MIN
# Both exec calls preserve the PID so the batch-shell SIGTERM reaches the solver.
exec python3 scripts/fast2_campaign.py run \
    --p "$1" --samples "$2" --angles "$3" --exe "$4" --out "$5"
