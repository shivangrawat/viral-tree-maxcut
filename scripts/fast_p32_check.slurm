#!/bin/bash
# ---------------------------------------------------------------------------
#  Direct p=32 check: the frozen viral_tree_hpc binary against the fast solver
#  AT p=32 itself, on the same samples (the first N samples of campaign job
#  JOB), with the R1 contract: the whole per-sample VIRAL_TREE_DUMP (cut and
#  both root z as hex floats) and every result-file byte except the four
#  timing fields must be identical.  Prints P32 CHECK PASS or P32 CHECK FAIL.
#
#  The frozen binary needs 226 GiB at p=32, hence --mem=240G (routed to cl).
#      sbatch --qos=cpu48 --mem=240G --time=01:30:00 scripts/fast_p32_check.slurm [N] [JOB]
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_p32_check
#SBATCH --account=torch_pr_577_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
N="${1:-4}"
JOB="${2:-0}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FROZEN="$HOME/viral-exe-backup/torch-large-builds-17989395/viral_baseline"
FAST="$HOME/viral-exe-backup/fast-builds-18517761/viral_fast"
ANGLES="$REPO/params/angles_p32.txt"
RUN="${SCRATCH:?SCRATCH must be set}/viral-fast/p32check-${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN/fast" "$RUN/frozen"
echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   mem ${SLURM_MEM_PER_NODE:-?} MB   p=32 job=$JOB n=$N"
grep -m1 'model name' /proc/cpuinfo
echo "frozen $(sha256sum "$FROZEN" | cut -c1-16)   fast $(sha256sum "$FAST" | cut -c1-16)   angles $ANGLES"
cd "$REPO" || exit 1
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE LC_ALL=C
unset VIRAL_FORCE_REF VIRAL_SELFCHECK

for kind in fast frozen; do
    exe=$FAST; [ $kind = frozen ] && exe=$FROZEN
    t0=$(date +%s)
    VIRAL_TREE_DUMP="$RUN/$kind/samples.hex" "$exe" 32 "$N" "$JOB" "$ANGLES" "$RUN/$kind" > "$RUN/$kind/solver.out" 2>&1
    rc=$?
    echo "$kind: exit $rc after $(( $(date +%s) - t0 )) s"
    grep -E 'memory  |self-check|DONE|elapsed ' "$RUN/$kind/solver.out"
    [ $rc -eq 0 ] || { tail -20 "$RUN/$kind/solver.out"; echo "P32 CHECK FAIL (solver error)"; exit 1; }
done

python3 - "$RUN" "$JOB" <<'PYEOF'
import difflib, sys
sys.path.insert(0, "scripts")
import viral_campaign as vc
run, job = sys.argv[1], sys.argv[2]
res = {k: vc.scientific_bytes(open("%s/%s/viral_tree_p32_job%s.txt" % (run, k, job), "rb").read()) for k in ("fast", "frozen")}
dump = {k: open("%s/%s/samples.hex" % (run, k), "rb").read() for k in ("fast", "frozen")}
print("frozen per-sample dump:\n" + dump["frozen"].decode())
ok = res["fast"] == res["frozen"] and dump["fast"] == dump["frozen"]
if not ok:
    for name, d in (("result", res), ("dump", dump)):
        if d["fast"] != d["frozen"]:
            print("".join(difflib.unified_diff(d["frozen"].decode().splitlines(True),
                                               d["fast"].decode().splitlines(True), "frozen", "fast")))
print("P32 CHECK PASS: per-sample dumps and result files identical" if ok else "P32 CHECK FAIL")
sys.exit(0 if ok else 1)
PYEOF
