#!/bin/bash
# ---------------------------------------------------------------------------
#  Gate G1' of SUBMIT_10M.md: prove that the 6M new samples of every depth
#  (torch-pP-extension-v2) are new.  (1) Static: for every even depth the three
#  campaigns million-v1 (1M), extension-v1 (3M) and extension-v2 (6M) use the
#  same angles, pairwise disjoint contiguous job ids and the planned targets;
#  no result file of that depth anywhere under results/ outside extension-v2
#  carries a v2 id, and extension-v2 itself has no result yet.  (2) For every
#  sample of all three campaigns of every depth (160M in all) compute the
#  solver's initial RNG state and require all of them to be distinct
#  (tests/seed_audit.c, unchanged from G1).
#
#      sbatch --qos=cpu48 --mem=16G --time=02:00:00 scripts/seed_audit_10m.slurm   (from the repo root)
#  (160M states x 32 B = 5.1 GB, and glibc qsort adds a same-size merge buffer: 8G is OOM-killed)
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_seed_audit_10m
#SBATCH --account=torch_pr_577_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN="${SCRATCH:?}/viral-fast/seed-audit-10m-${SLURM_JOB_ID:-manual}"
cd "$REPO" || exit 1
mkdir -p "$RUN"
echo "host $(hostname)   repo $REPO"
python3 - "$RUN/spec.txt" <<'PYEOF' || { echo "SEED AUDIT FAIL (static check)"; exit 1; }
import json, sys
from pathlib import Path
CAMPAIGNS = [("million-v1", 1_000_000), ("extension-v1", 3_000_000), ("extension-v2", 6_000_000)]
SCRATCH_TEST_IDS = {0, 3, 5, 7, 13, 19, 499, 9000, 9001, 999999, 2147483647}
spec = open(sys.argv[1], "w")
ok = True
for p in range(2, 33, 2):
    ms = [json.loads(Path("results/torch-p%d-%s/manifest.json" % (p, c)).read_text()) for c, _ in CAMPAIGNS]
    ids = [set(m["expected_job_ids"]) for m in ms]
    v2 = ids[2]
    same_angles = all(m["angles_sha256"] == ms[0]["angles_sha256"] and m["angles_hash"] == ms[0]["angles_hash"]
                      for m in ms)
    targets = all(m["target_samples"] == t for m, (_, t) in zip(ms, CAMPAIGNS))
    overlap = len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])
    seen, v2_files = set(), 0
    for f in Path("results").glob("**/viral_tree_p%d_job*.txt" % p):
        if "torch-p%d-extension-v2" % p in f.parts:
            v2_files += 1
        else:
            seen.add(int(f.stem.rsplit("job", 1)[1]))
    reused = v2 & seen
    test_ids = v2 & SCRATCH_TEST_IDS
    contiguous = all(sorted(s) == list(range(min(s), max(s) + 1)) for s in ids)
    good = same_angles and targets and overlap == 0 and not reused and not test_ids \
        and v2_files == 0 and contiguous and sum(m["target_samples"] for m in ms) == 10_000_000
    ok &= good
    print("p=%-2d %s  same angles %s  pairwise overlap %d  v2 ids reused %d  v2 results present %d  %s" % (
        p, "  ".join("%s %d..%d (%d x %d)" % (c, min(s), max(s), len(s), m["samples_per_job"])
                     for (c, _), s, m in zip(CAMPAIGNS, ids, ms)),
        same_angles, overlap, len(reused), v2_files, "ok" if good else "PROBLEM"))
    for m in ms:
        s = sorted(m["expected_job_ids"])
        spec.write("%d %d %d %d\n" % (p, s[0], len(s), m["samples_per_job"]))
spec.close()
sys.exit(0 if ok else 1)
PYEOF
gcc -O2 -o "$RUN/seed_audit" tests/seed_audit.c || { echo "SEED AUDIT FAIL (compile)"; exit 1; }
"$RUN/seed_audit" < "$RUN/spec.txt"
