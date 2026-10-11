#!/bin/bash
# ---------------------------------------------------------------------------
#  Validation record, as run on NYU Torch for the published campaigns (tag
#  results-10M).  Cluster-bound: account, /scratch/fm2452 and $HOME/viral-exe-backup
#  paths are Torch's.  Submit from the repo root.  See docs/validation.md.
#
#  Gate G1 (docs/validation.md): prove that the 3M new samples of every depth are
#  new.  (1) Static: the extension's job ids must not overlap the 1M campaign's
#  ids, nor any result file of that depth anywhere under results/.  (2) For
#  every sample of both campaigns of every depth (64M in all) compute the
#  solver's initial RNG state and require all of them to be distinct.
#
#      sbatch --qos=cpu48 --mem=4G --time=01:00:00 scripts/validation/seed_audit.slurm   (from the repo root)
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_seed_audit
#SBATCH --account=torch_pr_577_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
RUN="${SCRATCH:?}/viral-fast/seed-audit-${SLURM_JOB_ID:-manual}"
cd "$REPO" || exit 1
mkdir -p "$RUN"
echo "host $(hostname)   repo $REPO"
python3 - "$RUN/spec.txt" <<'PYEOF' || { echo "SEED AUDIT FAIL (static check)"; exit 1; }
import json, sys
from pathlib import Path
spec = open(sys.argv[1], "w")
ok = True
for p in range(2, 33, 2):
    old = json.loads(Path("results/torch-p%d-million-v1/manifest.json" % p).read_text())
    new = json.loads(Path("results/torch-p%d-extension-v1/manifest.json" % p).read_text())
    oi, ni = set(old["expected_job_ids"]), set(new["expected_job_ids"])
    same_angles = old["angles_sha256"] == new["angles_sha256"] and old["angles_hash"] == new["angles_hash"]
    clash = oi & ni
    # any result file of this depth, in any results directory, with a new id
    seen = set()
    for f in Path("results").glob("*/**/viral_tree_p%d_job*.txt" % p):
        if "extension" not in str(f):
            seen.add(int(f.stem.rsplit("job", 1)[1]))
    for f in Path("results").glob("*/viral_tree_p%d_job*.txt" % p):
        seen.add(int(f.stem.rsplit("job", 1)[1]))
    reused = ni & seen
    good = same_angles and not clash and not reused and new["target_samples"] == 3_000_000 \
        and old["target_samples"] == 1_000_000
    ok &= good
    print("p=%-2d old ids %d..%d (%d x %d)  new ids %d..%d (%d x %d)  same angles %s  overlap %d  reused %d  %s" % (
        p, min(oi), max(oi), len(oi), old["samples_per_job"], min(ni), max(ni), len(ni),
        new["samples_per_job"], same_angles, len(clash), len(reused), "ok" if good else "PROBLEM"))
    for m in (old, new):
        ids = sorted(m["expected_job_ids"])
        if ids != list(range(ids[0], ids[-1] + 1)):
            print("non-contiguous ids"); ok = False
        spec.write("%d %d %d %d\n" % (p, ids[0], len(ids), m["samples_per_job"]))
spec.close()
sys.exit(0 if ok else 1)
PYEOF
gcc -O2 -o "$RUN/seed_audit" tests/seed_audit.c || { echo "SEED AUDIT FAIL (compile)"; exit 1; }
"$RUN/seed_audit" < "$RUN/spec.txt"
