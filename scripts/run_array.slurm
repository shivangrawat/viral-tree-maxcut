#!/bin/bash
# Generic SLURM array wrapper: one VIRAL job per array task, on any cluster.
#
#   mkdir -p OUT
#   sbatch --array=0-$((J-1)) --cpus-per-task=T --mem=M --time=W \
#          [--account=... --partition=...] --output=OUT/slurm_%A_%a.out \
#          scripts/run_array.slurm EXE P N FIRST_ID OUT [ANGLES]
#
#   EXE       solver binary (./viral_tree_fast2 recommended; see README.md)
#   P         depth;  N  samples per job;  OUT  directory for the result files
#   FIRST_ID  job id of array task 0: task t runs job FIRST_ID + t, i.e. the
#             samples S = (FIRST_ID + t) * 2^32 + k, k = 0..N-1.  Different job
#             ids never share a sample; the same id must not be merged twice.
#   ANGLES    default params/angles_p<P>.txt
#
# Threads per job = --cpus-per-task.  viral_tree_fast2 shares one workspace
# (~5*2^P bytes) between its threads; viral_tree_hpc and viral_tree_fast need
# one workspace per thread (56.5*2^P and ~7.5*2^P bytes).  The solver prints
# its exact need at start-up.
#
# --signal=B:TERM@600 stops the solver 10 min before the time limit; it saves
# its state and exits 0, so SLURM reports COMPLETED while the result file says
# "status running".  Resubmit the same command (it resumes) until
# merge_results.py reports "0 still running / incomplete".
#SBATCH --job-name=viral
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm_%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail
if [[ $# != 5 && $# != 6 ]]; then
    echo "Usage: $0 EXE P N FIRST_ID OUT [ANGLES]" >&2
    exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?submit as a SLURM array}"
cd "${SLURM_SUBMIT_DIR:-.}"
EXE=$1 P=$2 N=$3 FIRST_ID=$4 OUT=$5 ANGLES=${6:-params/angles_p$2.txt}
[[ -x $EXE ]] || { echo "not an executable: $EXE" >&2; exit 2; }
[[ -f $ANGLES ]] || { echo "no angles file: $ANGLES" >&2; exit 2; }
mkdir -p "$OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Test and debugging hooks of the solvers; none of them may leak into a run.
unset VIRAL_TREE_DUMP VIRAL_FORCE_REF VIRAL_SELFCHECK VIRAL_KERNEL VIRAL_PAR_MIN
# exec keeps the PID, so the batch-shell SIGTERM reaches the solver itself.
exec "$EXE" "$P" "$N" "$((FIRST_ID + SLURM_ARRAY_TASK_ID))" "$ANGLES" "$OUT"
