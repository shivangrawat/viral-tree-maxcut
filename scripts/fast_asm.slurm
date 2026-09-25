#!/bin/bash
# ---------------------------------------------------------------------------
#  Diagnostic for SPEEDUP_PLAN.md: disassemble the frozen binary and a -g
#  rebuild of the protected source (same code, plus source lines), and the
#  fast solver, so the FMA contraction of rz/rx_cs can be read site by site.
#  Writes text files to $SCRATCH/viral-fast/asm/ (the login node has no
#  objdump).
#
#      sbatch scripts/fast_asm.slurm                   (from the repo root)
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast_asm
#SBATCH --account=torch_pr_577_general
#SBATCH --qos=cpu168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${SCRATCH:?SCRATCH must be set}/viral-fast/asm"
FROZEN="$HOME/viral-exe-backup/torch-large-builds-17989395/viral_baseline"
FLAGS=(-Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 -fopenmp)
cd "$REPO" || exit 1
mkdir -p "$OUT"
module purge 2>/dev/null
echo "host $(hostname)   gcc $(gcc --version | head -1)"

objdump -d --no-show-raw-insn "$FROZEN" > "$OUT/frozen.asm"
gcc "${FLAGS[@]}" -g viral_tree_hpc.c -lm -o "$OUT/ref_g"
objdump -d --no-show-raw-insn "$OUT/ref_g" > "$OUT/ref_g.asm"
objdump -dl --no-show-raw-insn "$OUT/ref_g" > "$OUT/ref_g_lines.asm"
# -g must not change the code: compare the two disassemblies without the file header
if diff <(tail -n +4 "$OUT/frozen.asm") <(tail -n +4 "$OUT/ref_g.asm") > "$OUT/frozen_vs_ref_g.diff"; then
    echo "code of the -g rebuild is identical to the frozen binary"
else
    echo "code of the -g rebuild DIFFERS from the frozen binary ($(wc -l < "$OUT/frozen_vs_ref_g.diff") diff lines)"
fi
gcc "${FLAGS[@]}" -ffp-contract=off -g viral_tree_fast.c -lm -o "$OUT/fast_g"
objdump -dl --no-show-raw-insn "$OUT/fast_g" > "$OUT/fast_g_lines.asm"
for f in frozen ref_g fast_g_lines; do
    printf '%-14s FMA instructions: %s\n' "$f" "$(grep -c -E $'\tv(f|fn)m(add|sub)' "$OUT/$f.asm")"
done
ls -la "$OUT"
