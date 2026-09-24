#!/bin/bash
# ---------------------------------------------------------------------------
#  Single-job probes for the large depths (p = 28, 30).  These are NOT
#  campaigns: they measure what the p>=28 estimates are weakest on, namely real
#  peak RSS and a real Torch sample rate.  Memory and time come from the sbatch
#  line because they differ per depth.
#
#  Equivalence spot-check (needs the p=30 workspace, so 60G):
#    sbatch --account=torch_pr_577_general --mem=60G --time=04:00:00 \
#      --output=/scratch/fm2452/viral-controlled/logs/spot_%j.out \
#      scripts/probe_large_p.slurm spot gcc
#
#  Peak-RSS + rate probe (one per depth; p=28 -> 16G, p=30 -> 60G):
#    sbatch --account=torch_pr_577_general --mem=16G --time=01:00:00 \
#      --output=/scratch/fm2452/viral-controlled/logs/rss28_%j.out \
#      scripts/probe_large_p.slurm rss "$VIRAL_EXE" 28
#    sbatch --account=torch_pr_577_general --mem=60G --time=04:00:00 \
#      --output=/scratch/fm2452/viral-controlled/logs/rss30_%j.out \
#      scripts/probe_large_p.slurm rss "$VIRAL_EXE" 30
#
#  Each RSS probe writes into a FRESH directory on purpose: re-running into a
#  complete directory does no sampling and would report a misleading number.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-controlled/logs/probe_%j.out
#SBATCH --open-mode=append

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${SCRATCH:?SCRATCH must be set}/viral-controlled"
TAG="${SLURM_JOB_ID:-manual}"
MODE="${1:-}"
if [ -z "$MODE" ]; then echo "usage: $0 spot CC | rss EXE P" >&2; exit 2; fi
cd "$REPO" || exit 1
mkdir -p "$WORK/logs"

echo "host: $(hostname)   mode: $MODE   job: $TAG"

case "$MODE" in
spot)
    CC="${2:-gcc}"
    module purge 2>/dev/null
    command -v "$CC" >/dev/null 2>&1 || { module load intel/2025.2 2>&1; }
    echo "--- spot-check depths 24,26,28,30 with CC=$CC ---"
    # Large depths run sequentially, so the allocation only needs the single
    # largest worker (p=30, 56.5 GiB), not one workspace per profile.
    python3 tests/test_build_equivalence.py \
        --cc "$CC" \
        --profiles baseline \
        --skip-existing-checks \
        --spot-depths 24,26,28,30 \
        --spot-samples 4 \
        --report "$WORK/torch-large-$TAG.json" \
        --export-builds "$WORK/torch-large-builds-$TAG"
    rc=$?
    echo "spot-check exit: $rc   report: $WORK/torch-large-$TAG.json"
    exit $rc
    ;;
rss)
    EXE="${2:-}"; [ -n "$EXE" ] || { echo "need executable" >&2; exit 2; }
    P="${3:-}";  [ -n "$P" ]   || { echo "need depth" >&2; exit 2; }
    [ -x "$EXE" ] || { echo "not executable: $EXE" >&2; exit 2; }
    ANGLES="$REPO/params/angles_p$P.txt"
    [ -f "$ANGLES" ] || { echo "missing angles: $ANGLES" >&2; exit 2; }
    RSS_OUT="$WORK/rss-p$P-$TAG"
    mkdir -p "$RSS_OUT/jobs" "$RSS_OUT/logs"
    echo "--- p=$P peak-RSS probe, 4 samples, fresh dir $RSS_OUT ---"
    echo "    predicted workspace: $(python3 -c "print(f'{(56.5*2**$P-48)/2**30:.3f} GiB')")"
    # GNU /usr/bin/time is absent from the Torch compute image, so measure peak
    # RSS with resource.getrusage(RUSAGE_CHILDREN) instead (ru_maxrss is KiB).
    OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE python3 - "$EXE" "$P" "$ANGLES" "$RSS_OUT" \
        >"$RSS_OUT/logs/time.txt" 2>&1 <<'PYEOF'
import os, resource, subprocess, sys, time
exe, p, angles, out = sys.argv[1:5]
t0 = time.time()
with open(os.path.join(out, "logs", "solver.out"), "w") as log:
    rc = subprocess.call([exe, p, "4", "0", angles, os.path.join(out, "jobs")],
                         stdout=log, stderr=subprocess.STDOUT)
wall = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"exit_code        {rc}")
print(f"wall_seconds     {wall:.3f}")
print(f"user_seconds     {ru.ru_utime:.3f}")
print(f"system_seconds   {ru.ru_stime:.3f}")
print(f"max_rss_kib      {ru.ru_maxrss}")
print(f"max_rss_gib      {ru.ru_maxrss/1048576:.3f}")
sys.exit(rc)
PYEOF
    rc=$?
    echo "--- solver output ---";       cat "$RSS_OUT/logs/solver.out"
    echo "--- measured ---";            cat "$RSS_OUT/logs/time.txt"
    python3 -c "print(f'    predicted {(56.5*2**$P-48)/2**30:.3f} GiB workspace (excludes program/allocator/OS overhead)')"
    echo "--- result file ---";         cat "$RSS_OUT/jobs/viral_tree_p${P}_job0.txt" 2>/dev/null
    exit $rc
    ;;
*)
    echo "usage: $0 spot CC | rss EXE P" >&2; exit 2 ;;
esac
