#!/bin/bash
# ---------------------------------------------------------------------------
#  Validation record, as run on NYU Torch for the published campaigns (tag
#  results-10M).  Cluster-bound: account, /scratch/fm2452 and $HOME/viral-exe-backup
#  paths are Torch's.  Submit from the repo root.  See docs/validation.md.
#
#  Gate G2 (docs/validation.md): build the campaign's viral_tree_fast2 binary,
#  require it to be byte-for-byte the executable the tests validated, export
#  it with its .build.json sidecar to $HOME (not flushed), and compare it AT
#  p=32 with 8 threads against the validated viral_tree_fast binary on the
#  same samples (per-sample dump and result bytes; threads line normalized).
#  Prints GATE PASS or GATE FAIL.
#
#      sbatch --qos=cpu48 --cpus-per-task=8 --mem=32G --time=01:00:00 scripts/validation/fast2_gate.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast2_gate
#SBATCH --account=torch_pr_577_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
TAG="${SLURM_JOB_ID:-manual}"
VALIDATED_EXE_SHA=e18230cf77b905472a985cf9665e2781278a0a98894d0b0acae8b786418cad29
SOURCE_SHA=8b12711a2b6f15e652a7677c43b80aa6715570ffd28b38d9ec4afdab97251988
FAST="$HOME/viral-exe-backup/fast-builds-18517761/viral_fast"
EXPORT="$HOME/viral-exe-backup/fast2-builds-$TAG"
RUN="${SCRATCH:?}/viral-fast/fast2gate-$TAG"
cd "$REPO" || exit 1
mkdir -p "$RUN"
module purge 2>/dev/null
echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   cpus ${SLURM_CPUS_PER_TASK:-?}   job $TAG"
fail() { echo "GATE FAIL: $*"; exit 1; }

[ "$(sha256sum viral_tree_fast2.c | cut -d' ' -f1)" = "$SOURCE_SHA" ] || fail "source is not the validated viral_tree_fast2.c"
CMD=(/usr/bin/gcc -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 -ffp-contract=off -fopenmp
     "$REPO/viral_tree_fast2.c" -lm -o "$RUN/viral_fast2")
"${CMD[@]}" || fail "compile failed"
SHA=$(sha256sum "$RUN/viral_fast2" | cut -d' ' -f1)
echo "built $SHA"
[ "$SHA" = "$VALIDATED_EXE_SHA" ] || fail "build differs from the validated executable"

mkdir -p "$EXPORT" && cp -p "$RUN/viral_fast2" "$EXPORT/viral_fast2" || fail "export failed"
python3 - "$EXPORT/viral_fast2" "$SHA" "$SOURCE_SHA" "$(gcc --version | head -1)" "$(hostname)" "${CMD[*]}" <<'PYEOF' || fail "sidecar"
import json, platform, sys
from datetime import datetime, timezone
exe, sha, src, ver, host, cmd = sys.argv[1:7]
info = dict(profile="fast2", source="viral_tree_fast2.c", source_sha256=src, executable_sha256=sha,
            compile_command=cmd.split(), compiler_version=ver, system=platform.system(),
            machine=platform.machine(), batch_max_per_thread=1, equivalence_passed=True,
            validation=dict(
                suite="job 18986377: 199 cases p=1..24, T=1/4/8, byte-identical to frozen 0f3f8d44; check.sh",
                large="job 18986379: p=26/28/30 identical to fast bc4230d1 at T=1,2,4,8",
                scaling="jobs 19053518/19053519: p=30 identical at T=1..32",
                gate_p32="this job: p=32 T=8 vs fast, see log"),
            built_on=host, created_utc=datetime.now(timezone.utc).isoformat())
open(exe + ".build.json.tmp", "w").write(json.dumps(info, indent=2) + "\n")
print("sidecar prepared (finalized only if the p=32 comparison passes)")
PYEOF

export OMP_PROC_BIND=close OMP_PLACES=cores LC_ALL=C
unset VIRAL_FORCE_REF VIRAL_SELFCHECK VIRAL_KERNEL VIRAL_PAR_MIN
ANGLES="$REPO/params/angles_p32.txt"
mkdir -p "$RUN/fast" "$RUN/fast2"
OMP_NUM_THREADS=1 VIRAL_TREE_DUMP="$RUN/fast/samples.hex" "$FAST" 32 2 0 "$ANGLES" "$RUN/fast" > "$RUN/fast/out" 2>&1 \
    || { tail "$RUN/fast/out"; fail "fast run"; }
OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" VIRAL_TREE_DUMP="$RUN/fast2/samples.hex" "$EXPORT/viral_fast2" 32 2 0 "$ANGLES" "$RUN/fast2" > "$RUN/fast2/out" 2>&1 \
    || { tail "$RUN/fast2/out"; fail "fast2 run"; }
grep -E 'kernel|threads  |memory  |self-check|elapsed ' "$RUN/fast/out" "$RUN/fast2/out"
python3 - "$RUN" <<'PYEOF' || fail "p=32 comparison"
import re, sys
sys.path.insert(0, "scripts")
import viral_campaign as vc
run = sys.argv[1]
norm = lambda b: re.sub(rb"(?m)^threads( +)\d+$", rb"threads\g<1>1", b)
r = {k: norm(vc.scientific_bytes(open("%s/%s/viral_tree_p32_job0.txt" % (run, k), "rb").read())) for k in ("fast", "fast2")}
d = {k: open("%s/%s/samples.hex" % (run, k), "rb").read() for k in ("fast", "fast2")}
print(d["fast"].decode(), end="")
sys.exit(0 if r["fast"] == r["fast2"] and d["fast"] == d["fast2"] else 1)
PYEOF
mv "$EXPORT/viral_fast2.build.json.tmp" "$EXPORT/viral_fast2.build.json" || fail "sidecar finalize"
echo "GATE PASS: built binary = validated executable; p=32 with ${SLURM_CPUS_PER_TASK:-8} threads identical to viral_tree_fast"
echo "exported $EXPORT/viral_fast2"
