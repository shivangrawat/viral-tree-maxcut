#!/bin/bash
# ---------------------------------------------------------------------------
#  SLURM job array for viral_tree_hpc  --  p = 26:  100 jobs x 10000 samples = 1e6 samples
#
#      sbatch submit_p26.slurm            (from inside this folder)
#
#  Self-contained: builds the program if needed, writes results and logs into
#  results/, and resumes automatically if resubmitted after a wall-time stop.
#
#  Resources requested below:  16 threads/job, 60G memory, ~45 min per job.
#  Memory per thread at p = 26 is 3.53 GB, so the one hard rule is
#      --mem  >=  --cpus-per-task x 3.53 GB  +  1 GB
#  Fits on any node with >= 64 GB.
#  Everything else (array size, threads, time limit) only changes how long it
#  takes: jobs resume when resubmitted, and different job ids never overlap.
#  Add --partition / --account lines if the cluster requires them.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_tree_p26
#SBATCH --array=0-99
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_p26_%a.out
##SBATCH --partition=NAME
##SBATCH --account=NAME

P=26
N=10000
OUT=results

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" || exit 1
mkdir -p "$OUT"

# Build if needed.  flock serialises the array tasks so only one compiles.
if [ ! -x ./viral_tree_hpc ]; then
    flock ./.build.lock sh -c '[ -x ./viral_tree_hpc ] || make -s viral_tree_hpc' || exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 'exec' replaces this shell by the program, so the SIGTERM sent by
# --signal=B:TERM@600 reaches the program, which then stops cleanly.
exec ./viral_tree_hpc "$P" "$N" "$SLURM_ARRAY_TASK_ID" "params/angles_p${P}.txt" "$OUT"
