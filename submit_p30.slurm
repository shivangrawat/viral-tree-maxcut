#!/bin/bash
# ---------------------------------------------------------------------------
#  SLURM job array for viral_tree_hpc  --  p = 30:  500 jobs x 2000 samples = 1e6 samples
#
#      sbatch submit_p30.slurm            (from inside this folder)
#
#  Self-contained: builds the program if needed, writes results and logs into
#  results/, and resumes automatically if resubmitted after a wall-time stop.
#
#  Resources requested below:  4 threads/job, 228G memory, ~10 h per job.
#  Memory per thread at p = 30 is 56.5 GB, so the one hard rule is
#      --mem  >=  --cpus-per-task x 56.5 GB  +  1 GB
#  Needs nodes with >= 256 GB (4 threads).  On 512 GB nodes use 8 threads / 454G; on 1 TB, 16 / 906G; on 2 TB, 32 / 1810G.  Memory, not cores, limits p = 30.
#  Everything else (array size, threads, time limit) only changes how long it
#  takes: jobs resume when resubmitted, and different job ids never overlap.
#  Add --partition / --account lines if the cluster requires them.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_tree_p30
#SBATCH --array=0-499
#SBATCH --cpus-per-task=4
#SBATCH --mem=228G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_p30_%a.out
##SBATCH --partition=NAME
##SBATCH --account=NAME

P=30
N=2000
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
