#!/bin/bash
# ---------------------------------------------------------------------------
#  R3 of SPEEDUP_PLAN.md: replay one COMMITTED campaign job with the fast
#  solver and require every scientific byte of the result file to equal the
#  committed one (all fields except elapsed_s, core_seconds,
#  sec_per_sample_core, host).  The angles path is taken verbatim from the
#  committed file.  Prints REPLAY PASS or REPLAY FAIL.
#
#      sbatch --mem=8G --time=16:00:00 scripts/fast_replay.slurm EXE P JOB
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast_r3
#SBATCH --account=torch_pr_577_general
#SBATCH --qos=cpu168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
EXE="${1:?usage: fast_replay.slurm EXE P JOB}"
P="${2:?need depth}"
JOB="${3:?need job id}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
COMMITTED="$REPO/results/torch-p$P-million-v1/jobs/viral_tree_p${P}_job$JOB.txt"
RUN="${SCRATCH:?SCRATCH must be set}/viral-fast/replay-p$P-job$JOB-${SLURM_JOB_ID:-manual}"
[ -x "$EXE" ] || { echo "not executable: $EXE" >&2; exit 2; }
[ -f "$COMMITTED" ] || { echo "no committed result: $COMMITTED" >&2; exit 2; }
N=$(awk '$1=="n_done" {print $2}' "$COMMITTED")
ANGLES=$(awk '$1=="angles_file" {print $2}' "$COMMITTED")
STATUS=$(awk '$1=="status" {print $2}' "$COMMITTED")
[ "$STATUS" = complete ] || { echo "committed job is not complete" >&2; exit 2; }
[ -f "$ANGLES" ] || { echo "missing angles: $ANGLES" >&2; exit 2; }
mkdir -p "$RUN"
echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   p=$P job=$JOB n=$N"
echo "exe $EXE (sha $(sha256sum "$EXE" | cut -c1-16))"
echo "angles $ANGLES"
echo "committed $COMMITTED"

cd "$REPO" || exit 1
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE LC_ALL=C \
    "$EXE" "$P" "$N" "$JOB" "$ANGLES" "$RUN" > "$RUN/solver.out" 2>&1
rc=$?
grep -E 'self-check|memory touched|DONE|n_cut|c_VIRAL|elapsed ' "$RUN/solver.out"
[ $rc -eq 0 ] || { echo "solver exited with $rc"; tail -20 "$RUN/solver.out"; echo "REPLAY FAIL"; exit 1; }

python3 - "$COMMITTED" "$RUN/viral_tree_p${P}_job$JOB.txt" <<'PYEOF'
import difflib, sys
sys.path.insert(0, "scripts")
import viral_campaign as vc
old, new = (open(f, "rb").read() for f in sys.argv[1:3])
a, b = vc.scientific_bytes(old), vc.scientific_bytes(new)
if a == b:
    print("REPLAY PASS: identical to the committed result (all fields except the four timing fields)")
    sys.exit(0)
print("REPLAY FAIL")
print("".join(difflib.unified_diff(a.decode().splitlines(True), b.decode().splitlines(True),
                                   "committed", "fast")))
sys.exit(1)
PYEOF
