#!/bin/bash
# ---------------------------------------------------------------------------
#  Peak-RSS and rate probe for the validated fast solver at one depth.  Runs
#  a few samples into a FRESH directory (re-running into a complete one does
#  no sampling) and reports peak RSS from getrusage, the solver's own
#  s/sample, and the predicted workspace 7.5*2^p.
#
#      sbatch --mem=32G --time=00:30:00 scripts/fast_probe.slurm EXE P [N]
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast_probe
#SBATCH --account=torch_pr_577_general
#SBATCH --qos=cpu168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
EXE="${1:?usage: fast_probe.slurm EXE P [N]}"
P="${2:?need depth}"
N="${3:-3}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ANGLES="$REPO/params/angles_p$P.txt"
RUN="${SCRATCH:?SCRATCH must be set}/viral-fast/probe-p$P-${SLURM_JOB_ID:-manual}"
[ -x "$EXE" ] || { echo "not executable: $EXE" >&2; exit 2; }
[ -f "$ANGLES" ] || { echo "missing angles: $ANGLES" >&2; exit 2; }
mkdir -p "$RUN"
echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   mem ${SLURM_MEM_PER_NODE:-?} MB   exe sha $(sha256sum "$EXE" | cut -c1-16)"
grep -m1 'model name' /proc/cpuinfo
python3 -c "print('predicted workspace %.3f GiB' % ((24*(2**($P-2)-2) + 8*((4*2**$P-2+63)//64+1) + 16*max(2**($P-4),2)) / 2**30))"
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE python3 - "$EXE" "$P" "$N" "$ANGLES" "$RUN" <<'PYEOF'
import os, resource, subprocess, sys, time
exe, p, n, angles, out = sys.argv[1:6]
t0 = time.time()
with open(os.path.join(out, "solver.out"), "w") as log:
    rc = subprocess.call([exe, p, n, "999999", angles, out], stdout=log, stderr=subprocess.STDOUT)
wall = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print("exit_code        %d" % rc)
print("wall_seconds     %.1f" % wall)
print("max_rss_gib      %.3f" % (ru.ru_maxrss / 1048576))
sys.exit(rc)
PYEOF
rc=$?
echo "--- solver output"
cat "$RUN/solver.out"
echo "--- result file"
cat "$RUN/viral_tree_p${P}_job999999.txt" 2>/dev/null
exit $rc
