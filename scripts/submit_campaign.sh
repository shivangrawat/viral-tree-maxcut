#!/bin/bash
# ---------------------------------------------------------------------------
#  Resource table + launcher for the VIRAL p = 2..30 campaigns on NYU Torch
#  (results/torch-pP-million-v1; record: docs/campaigns.md).
#
#      export VIRAL_EXE=/scratch/fm2452/viral-controlled/torch-small-builds-<id>/viral_baseline
#      scripts/submit_campaign.sh plan                  # print the table, submit nothing
#      scripts/submit_campaign.sh init   2 4 6 8 10 12 14 16
#      scripts/submit_campaign.sh submit 2 4 6 8 10 12 14 16
#      scripts/submit_campaign.sh status 2 4 6 8 10 12 14 16
#      scripts/submit_campaign.sh verify 24
#      scripts/submit_campaign.sh resume 26            # resubmit only unfinished ids
#
#  With no depths listed, every depth in the table is used.
#
#  Every run is ONE CPU and ONE OpenMP thread: that is the controlled-build
#  contract enforced by scripts/run_viral.slurm and scripts/viral_campaign.py.
#  Concurrency comes from array tasks, never from threads.
#
#  Memory is exact: workspace = 56.5 * 2^p - 48 bytes PER PROCESS.
#  Time model, MEASURED on Torch (job 17985393, gcc 14.3.1, cs611, 1 thread):
#    t(16) = 2.40e-3 s/sample/core, scaling 1.90-1.97x per unit p.  Walltimes
#    below use the SLOWER Intel i9 rates of the original hand-over README at
#    p>=22 as a safety margin.
#  Wall time is soft: the solver checkpoints every 60 s and resubmitting resumes.
#
#  p=28 walltime/memory come from a MEASURED Torch probe (job 17989534, cs601,
#  1 thread): 15.039 s/sample/core and 14.127 GiB peak RSS against a 14.125 GiB
#  predicted workspace.  5000 samples/job is ~20.9 h on a typical node and
#  ~25 h on the slowest observed; 48:00:00 is the cpu48 QoS maximum and leaves
#  ~2x margin.  Depth scaling measured across p=16..30 is ~3.9x per +2 in p.
#
#  p=30 ran 2026-09-20 .. 10-05 (its submission is recorded in docs/campaigns.md).
#  Measured probe (job 17989535, cs614, 1 thread): 59.422 s/sample/core and
#  56.502 GiB peak RSS against a 56.500 GiB predicted workspace.  CONC=100 is a
#  hard ceiling, not a preference: the cpu48 QoS allows 6000G per user and each
#  job needs 60G.  ~14% of jobs are expected to exceed the 48 h wall and need
#  one 'resume' round; that is normal and costs time, not samples.
#
#  p=24, p=26, p=28 and p=30 partitions are FROZEN (docs/campaigns.md):
#  sample k of job j is S=(j<<32)|k, so re-partitioning picks a different
#  sample set.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ACCOUNT="${ACCOUNT:-torch_pr_577_general}"
ALL_P="2 4 6 8 10 12 14 16 18 20 22 24 26 28 30"

#      P   JOBS   SAMPLES/JOB   MEM   WALL       CONC   (frozen?)
TABLE="\
2     1   1000000   2G   01:00:00     1   -
4     1   1000000   2G   01:00:00     1   -
6     1   1000000   2G   01:00:00     1   -
8     1   1000000   2G   01:00:00     1   -
10    1   1000000   2G   01:00:00     1   -
12    1   1000000   2G   01:00:00     1   -
14    1   1000000   2G   02:00:00     1   -
16    1   1000000   2G   03:00:00     1   -
18    4    250000   2G   03:00:00     4   -
20   16     62500   2G   03:00:00    16   -
22   64     15625   2G   08:00:00    64   -
24   50     20000   2G   16:00:00    50   FROZEN
26  100     10000   5G   24:00:00   100   FROZEN
28  200      5000  16G   48:00:00   100   FROZEN
30  500      2000  60G   48:00:00   100   FROZEN"

row() { echo "$TABLE" | awk -v p="$1" '$1==p {print; found=1} END {exit !found}'; }

out_dir()  { echo "$REPO/results/torch-p$1-million-v1"; }
angles()   { echo "$REPO/params/angles_p$1.txt"; }

need_exe() {
    if [ -z "${VIRAL_EXE:-}" ]; then
        echo "ERROR: set VIRAL_EXE to the validated executable from Step 0." >&2
        echo "       It must have a .build.json sidecar beside it." >&2
        exit 2
    fi
    if [ ! -x "$VIRAL_EXE" ]; then
        echo "ERROR: VIRAL_EXE is not executable: $VIRAL_EXE" >&2; exit 2
    fi
    if [ ! -f "$VIRAL_EXE.build.json" ]; then
        echo "ERROR: missing sidecar $VIRAL_EXE.build.json" >&2
        echo "       Only tests/test_build_equivalence.py --export-builds creates it." >&2
        exit 2
    fi
}

cmd_plan() {
    printf '%-4s %6s %12s %6s %10s %6s  %-8s %s\n' P JOBS SAMPLES MEM WALL CONC FROZEN CAMPAIGN
    for p in $PLIST; do
        set -- $(row "$p") || { echo "no table row for p=$p" >&2; continue; }
        printf '%-4s %6s %12s %6s %10s %6s  %-8s %s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$(out_dir "$p")"
    done
}

cmd_init() {
    need_exe
    for p in $PLIST; do
        set -- $(row "$p") || { echo "no table row for p=$p" >&2; exit 1; }
        local P=$1 JOBS=$2 N=$3
        local OUT; OUT=$(out_dir "$P")
        if [ -f "$OUT/manifest.json" ]; then
            echo "p=$P  already initialised ($OUT) - skipping"
            continue
        fi
        echo "p=$P  init: $JOBS jobs x $N samples -> $OUT"
        python3 "$REPO/scripts/viral_campaign.py" init \
            --p "$P" --samples "$N" --jobs "$JOBS" \
            --angles "$(angles "$P")" --exe "$VIRAL_EXE" --out "$OUT" || exit 1
    done
}

cmd_submit() {
    need_exe
    for p in $PLIST; do
        set -- $(row "$p") || { echo "no table row for p=$p" >&2; exit 1; }
        local P=$1 JOBS=$2 N=$3 MEM=$4 WALL=$5 CONC=$6
        local OUT; OUT=$(out_dir "$P")
        if [ ! -f "$OUT/manifest.json" ]; then
            echo "p=$P  NOT initialised - run 'init' first" >&2; exit 1
        fi
        local ARRAY="0-$((JOBS-1))%$CONC"
        echo "p=$P  sbatch --array=$ARRAY --mem=$MEM --time=$WALL"
        sbatch --account="$ACCOUNT" --array="$ARRAY" --mem="$MEM" --time="$WALL" \
            --output="$OUT/logs/slurm_%A_%a.out" \
            "$REPO/scripts/run_viral.slurm" "$P" "$N" "$(angles "$P")" "$VIRAL_EXE" "$OUT" || exit 1
    done
}

cmd_resume() {
    need_exe
    for p in $PLIST; do
        set -- $(row "$p") || exit 1
        local P=$1 N=$3 MEM=$4 WALL=$5 CONC=$6
        local OUT; OUT=$(out_dir "$P")
        if squeue -u "$USER" -h -o '%o' 2>/dev/null | grep -qF "$OUT"; then
            echo "p=$P  still has tasks in the queue - wait before resuming" >&2
            continue
        fi
        local PENDING
        PENDING=$(python3 "$REPO/scripts/viral_campaign.py" pending "$OUT") || exit 1
        if [ -z "$PENDING" ]; then
            echo "p=$P  complete - nothing to resume"
            continue
        fi
        echo "p=$P  resubmitting ids: $PENDING"
        sbatch --account="$ACCOUNT" --array="$PENDING%$CONC" --mem="$MEM" --time="$WALL" \
            --output="$OUT/logs/slurm_%A_%a.out" \
            "$REPO/scripts/run_viral.slurm" "$P" "$N" "$(angles "$P")" "$VIRAL_EXE" "$OUT" || exit 1
    done
}

cmd_status() { for p in $PLIST; do echo "=== p=$p ==="; python3 "$REPO/scripts/viral_campaign.py" status "$(out_dir "$p")"; done; }

cmd_verify() {
    for p in $PLIST; do
        local OUT; OUT=$(out_dir "$p")
        echo "=== p=$p ==="
        if python3 "$REPO/scripts/viral_campaign.py" verify "$OUT"; then
            python3 "$REPO/merge_results.py" "$OUT/jobs" -p "$p" -v > "$OUT/summary.txt.tmp" \
                && mv "$OUT/summary.txt.tmp" "$OUT/summary.txt" && cat "$OUT/summary.txt"
        else
            echo "p=$p INCOMPLETE - not writing summary.txt"
        fi
    done
}

CMD="${1:-plan}"; shift || true
PLIST="${*:-$ALL_P}"

case "$CMD" in
    plan)   cmd_plan   ;;
    init)   cmd_init   ;;
    submit) cmd_submit ;;
    resume) cmd_resume ;;
    status) cmd_status ;;
    verify) cmd_verify ;;
    *) echo "usage: $0 {plan|init|submit|resume|status|verify} [p ...]" >&2; exit 2 ;;
esac
