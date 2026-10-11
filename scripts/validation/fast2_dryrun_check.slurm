#!/bin/bash
# ---------------------------------------------------------------------------
#  Validation record, as run on NYU Torch for the published campaigns (tag
#  results-10M).  Cluster-bound: account, /scratch/fm2452 and $HOME/viral-exe-backup
#  paths are Torch's.  Submit from the repo root.  See docs/validation.md.
#
#  Gate G4 (docs/validation.md): after a tiny campaign has run through the real
#  launch path (fast2_campaign.py init + an array of run_viral_fast2.slurm
#  with threads), verify it and require every result file to equal the
#  frozen viral_tree_hpc binary's output for the same job ids (all bytes but
#  the timing fields; the threads line normalized).  Prints DRYRUN PASS/FAIL.
#
#      sbatch --dependency=afterok:<array> --qos=cpu48 --mem=1G --time=00:30:00 \
#             scripts/validation/fast2_dryrun_check.slurm CAMPAIGN_DIR
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast2_dryrun_check
#SBATCH --account=torch_pr_577_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
OUT="${1:?usage: fast2_dryrun_check.slurm CAMPAIGN_DIR}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
FROZEN="$HOME/viral-exe-backup/torch-large-builds-17989395/viral_baseline"
cd "$REPO" || exit 1
python3 scripts/fast2_campaign.py verify "$OUT" || { echo "DRYRUN FAIL (verify)"; exit 1; }
python3 - "$OUT" "$FROZEN" <<'PYEOF'
import json, os, re, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts")
import viral_campaign as vc
out, frozen = Path(sys.argv[1]), sys.argv[2]
m = json.loads((out / "manifest.json").read_text())
norm = lambda b: re.sub(rb"(?m)^threads( +)\d+$", rb"threads\g<1>1", b)
env = dict(os.environ, OMP_NUM_THREADS="1", LC_ALL="C")
env.pop("VIRAL_TREE_DUMP", None)
ok = True
with tempfile.TemporaryDirectory() as d:
    for job in m["expected_job_ids"]:
        subprocess.run([frozen, str(m["p"]), str(m["samples_per_job"]), str(job), m["angles_file"], d],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        name = "viral_tree_p%d_job%d.txt" % (m["p"], job)
        a = norm(vc.scientific_bytes((Path(d) / name).read_bytes()))
        b = norm(vc.scientific_bytes((out / "jobs" / name).read_bytes()))
        print("job %d: %s" % (job, "identical to the frozen binary" if a == b else "DIFFERENT"))
        ok &= a == b
print("DRYRUN PASS" if ok else "DRYRUN FAIL")
sys.exit(0 if ok else 1)
PYEOF
