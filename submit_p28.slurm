#!/bin/bash
# ---------------------------------------------------------------------------
#  SLURM job array for viral_tree_hpc  --  p = 28:  200 jobs x 5000 samples = 1e6 samples
#
#      sbatch submit_p28.slurm            (from inside this folder)
#
#  Self-contained: builds the program if needed, writes results and logs into
#  results/, and resumes automatically if resubmitted after a wall-time stop.
#
#  Resources requested below:  8 threads/job, 116G memory, ~3 h per job.
#  Memory per thread at p = 28 is 14.13 GB, so the one hard rule is
#      --mem  >=  --cpus-per-task x 14.13 GB  +  1 GB
#  Needs nodes with >= 128 GB.  On 256 GB nodes use 16 threads / 228G; on 512 GB, 32 / 454G.
#  Everything else (array size, threads, time limit) only changes how long it
#  takes: jobs resume when resubmitted, and different job ids never overlap.
#  Add --partition / --account lines if the cluster requires them.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_tree_p28
#SBATCH --array=0-199
#SBATCH --cpus-per-task=8
#SBATCH --mem=116G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_p28_%a.out
##SBATCH --partition=NAME
##SBATCH --account=NAME

P=28
N=5000
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
