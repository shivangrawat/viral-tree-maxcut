#!/bin/bash
# ---------------------------------------------------------------------------
#  Validation record, as run on NYU Torch for the published campaigns (tag
#  results-10M).  Cluster-bound: account, /scratch/fm2452 and $HOME/viral-exe-backup
#  paths are Torch's.  Submit from the repo root.  See docs/validation.md.
#
#  Step 0 gate: build and validate the controlled VIRAL executable on a Torch
#  compute node.  The login node has no C toolchain at all, so this job is the
#  only way to obtain a usable executable with the .build.json sidecar that
#  scripts/viral_campaign.py requires.
#
#      sbatch scripts/validation/torch_build_validate.slurm            (from the repo root)
#
#  Exports validated builds plus the .build.json sidecars that
#  scripts/viral_campaign.py requires.  Nothing downstream may run until this
#  prints ALL CHECKS PASSED.
#
#  Four CPUs are requested only because the embedded, unchanged check.sh has an
#  explicit 1-thread-vs-4-thread independence test.  The campaign itself is
#  single-threaded by contract.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_build
#SBATCH --account=torch_pr_577_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/fm2452/viral-controlled/logs/build_%j.out
#SBATCH --open-mode=append

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
WORK="${SCRATCH:?SCRATCH must be set}/viral-controlled"
TAG="${SLURM_JOB_ID:-manual}"
REPORT="$WORK/torch-small-$TAG.json"
BUILDS="$WORK/torch-small-builds-$TAG"

cd "$REPO" || exit 1
mkdir -p "$WORK/logs"

echo "=================================================================="
echo " host            : $(hostname)"
echo " repo            : $REPO"
echo " commit          : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo " work            : $WORK"
echo "=================================================================="

# ---- 1. what does this compute node actually have? -----------------------
echo "--- toolchain probe ---"
module purge 2>/dev/null
for c in gcc cc clang icx; do
    printf '  %-6s : %s\n' "$c" "$(command -v $c || echo 'NOT FOUND')"
done
printf '  %-6s : %s\n' headers "$([ -e /usr/include/stdio.h ] && echo present || echo 'MISSING /usr/include/stdio.h')"
printf '  %-6s : %s\n' python3 "$(python3 --version 2>&1)"

# ---- 2. choose a compiler ------------------------------------------------
CC=""
if command -v gcc >/dev/null 2>&1 && [ -e /usr/include/stdio.h ]; then
    CC=gcc
else
    echo "--- no usable gcc; trying intel/2025.2 (icx) ---"
    module load intel/2025.2 2>&1
    if command -v icx >/dev/null 2>&1 && [ -e /usr/include/stdio.h ]; then
        CC=icx
    fi
fi

if [ -z "$CC" ]; then
    echo
    echo "FATAL: this compute node has no usable C toolchain."
    echo "       Fall back to Apptainer (docker://gcc) per the plan; do NOT use"
    echo "       make native or the committed Mac binary."
    exit 1
fi
echo "--- using CC=$CC : $($CC --version 2>&1 | head -1) ---"

# ---- 3. build + validate + benchmark ------------------------------------
# Compiles reference AND candidates with this same compiler, runs the 64
# equivalence cases, the SIGTERM/SIGKILL resume tests, the campaign checks and
# the unchanged check.sh, then exports only passing builds with their sidecars.
echo "--- tests/test_build_equivalence.py (report: $REPORT) ---"
python3 tests/test_build_equivalence.py \
    --cc "$CC" \
    --benchmark \
    --report "$REPORT" \
    --export-builds "$BUILDS"
rc=$?

echo
echo "=================================================================="
if [ $rc -ne 0 ]; then
    echo " VALIDATION FAILED (exit $rc).  Do not submit any campaign."
    exit $rc
fi
SELECTED=$(python3 - "$REPORT" <<'PY' 2>/dev/null
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
def find(o):
    if isinstance(o, dict):
        if "selected" in o and isinstance(o["selected"], str):
            return o["selected"]
        for v in o.values():
            f = find(v)
            if f: return f
    elif isinstance(o, list):
        for v in o:
            f = find(v)
            if f: return f
    return None
print(find(r) or "baseline")
PY
)
SELECTED=${SELECTED:-baseline}
echo " VALIDATION PASSED"
echo " selected profile : $SELECTED"
echo " report           : $REPORT"
echo " builds           : $BUILDS"
echo
echo " Use this as EXE for every campaign:"
echo "   $BUILDS/viral_$SELECTED"
echo "=================================================================="
ls -la "$BUILDS" 2>/dev/null
