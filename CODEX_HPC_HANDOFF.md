# VIRAL: controlled builds and NYU Torch campaigns

This is the handoff for the additive build/launch work on branch `faster`.
No solver algorithm, arithmetic, workspace, RNG, trained parameter, original
test, merger, Makefile, README, or existing SLURM script was changed. The new
workflow supports the supplied even depths **2 through 30**, and explicit
angles for other solver-valid depths **1 through 40**. Supporting an input does
not establish that it fits in memory or has been tested at that depth.

## 1. Protected baseline and exactness contract

- Baseline commit: `164c5e09986174ff38e0aa8c3ea4d8a695fc1cd6`.
- Protected source: `viral_tree_hpc.c`.
- Source SHA-256: `b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51`.
- **Every one of the 74 files tracked at that commit is protected**, including
  existing `build/` artifacts and `results/local_mac/` files. Do not run a build
  or `check.sh` over those artifacts. The new runner uses temporary copies.
- Additions: `tests/test_build_equivalence.py`, `scripts/run_viral.slurm`,
  `scripts/viral_campaign.py`, and this document.

Reference and candidate must use the same unchanged source, compiler, OpenMP
runtime, platform, and angles **path and bytes**. Both builds define
`BATCH_MAX_PER_THREAD=1`; both run with `OMP_NUM_THREADS=1`. The fixed batch
size fixes aggregate summation order across execution speeds and checkpoints.
Independent processes provide concurrency. Neither the original default build
nor archived four-thread Mac results is this controlled comparison baseline.

Compare the entire per-sample `VIRAL_TREE_DUMP` file as bytes (including its
hexadecimal floats). Compare every byte of the result file after removing only
lines whose field name is one of:

```text
elapsed_s
core_seconds
sec_per_sample_core
host
```

Do not parse/reformat floats or apply tolerances. `soft_sum`, `x_soft`, counts,
estimates, error estimates, parameters, thread count, angles metadata, status,
comments, and whitespace all remain in the comparison. The comparator also
rejects missing or duplicate fields. Stop/resume tests compare **final complete
results**; intermediate checkpoints can have different progress counts.

Permitted tuning is the fixed list `baseline` (`-O3`), `O2` (`-O2`),
`native_tune` (`-O3 -mtune=native`), and `native_arch` (`-O3 -march=native` on
x86, `-O3 -mcpu=native` on ARM). No fast-math, approximate math library,
precision reduction, source edits, lookup tables, or shared workspaces. An
unsupported flag is skipped; a scientific-byte mismatch rejects that profile.
Passing finite tests is evidence for the tested inputs, not proof of equivalence
for every possible sample. Mac success cannot certify Linux/libm behavior.

## 2. Local validation and measurements

The runner needs only Python's standard library, a C compiler, and OpenMP. It
does not import or run PyTorch and does not connect to the Torch cluster.
On this Mac, Apple clang needs an existing LLVM OpenMP include/library prefix.
The locally available prefix happens to be inside an existing package:

```bash
python3 tests/test_build_equivalence.py \
  --cc /usr/bin/clang \
  --omp-root /Users/shivangrawat/miniforge3/lib/python3.12/site-packages/torch \
  --benchmark \
  --report /tmp/viral-local-validation.json \
  --export-builds /tmp/viral-local-builds
```

For another Mac, pass its actual LLVM OpenMP prefix (for example an installed
Homebrew `libomp` prefix). The runner adjusts and ad-hoc signs **only temporary
executables** for the installed dylib path. It installs nothing. Output report
files and export directories must be new; use another name for a new run.
Without `--export-builds`, all executables/results disappear with the temporary
directory. Without `--benchmark`, no speed-based selection occurs.

Default coverage:

- 64 cases per profile: p=1 explicitly supplied one-layer fixture; every even
  depth 2..16; job IDs 0, 7, 499; sample counts 1 and 513 at p<=8, 1 and 65
  at p>=10. Nine extra cases use the first 2, 4, or 8 layers of the **whole
  p=30 angles file**, 129 samples each. One p=2 case uses job ID 2147483647.
- p=12, job 13, 4096 samples: repeated fresh runs, 31-to-4096 extension,
  already-complete rerun, SIGTERM save/resume, SIGKILL rollback/resume. Forced
  stop starts from a real saved 31-sample prefix; no checkpoint is fabricated.
  Signal timing varies, but final result and scientific dump bytes must match.
- SLURM wrapper exercised locally with simulated array environment variables;
  its results match direct execution, force one thread despite an inherited
  setting of eight, and pass the **unchanged** merger. No scheduler is invoked.
  Missing jobs, duplicate files, wrong targets, and multiple-CPU launches are
  rejected by the added campaign checks.
- The original `check.sh` is copied unchanged with its fixtures and run using
  four threads. It tests historical cut counts/statistics, independently of
  the new one-thread byte contract. Its p=12 200,000-sample check is **not** a
  200,000-sample reference-versus-candidate byte comparison.
- Five comparator/source-guard regression tests ensure the test harness itself
  catches float spelling changes, tiny scientific differences, signed-zero
  dump changes, malformed fields, and source edits.

Benchmarking uses p=8, 12, 16, job 19, supplied angles, five repetitions per
profile, no dump I/O, and fresh output directories. Calibration chooses fixed
sample counts per depth targeting at least 1.5 seconds per baseline trial.
Timing includes process startup, allocation, and result writes. Execution order is
rotated between repetitions. Every benchmark result is also checked for exact
scientific bytes. A candidate qualifies only when the geometric mean of its
three depth-median time ratios to baseline is <=0.95, with no depth ratio
>1.05. Among qualifiers choose the lowest ratio; otherwise retain baseline.

**Measured on this Mac, 2026-09-18:** Apple clang 21.0.0
(`clang-2100.1.1.101`), ARM64, Python 3.9.6, existing LLVM OpenMP. All four
profiles passed the 64 cases and lifecycle/launcher tests. Original self-tests
reported `ALL CHECKS PASSED`: p=12 cut count 178239; p=8 single/four-thread cut
counts both 43647; extended/fresh p=8 job 5 counts both 26301.

Median wall seconds (five repetitions each; lower is faster):

| p | Samples/trial | Baseline `-O3` | `-O2` | `-O3 -mtune=native` | `-O3 -mcpu=native` |
|---|---:|---:|---:|---:|---:|
| 8 | 524,288 | 2.553073 | 2.571665 | 2.572567 | 2.575759 |
| 12 | 32,768 | 2.396388 | 2.355225 | 2.351551 | 2.360404 |
| 16 | 2,048 | 2.311453 | 2.302678 | 2.277604 | 2.302050 |

Across these depth medians, geometric mean time reductions were **0.46%** for
`-O2`, **0.86%** for native tuning, and **0.34%** for native architecture. All
were slightly slower at p=8. **Selected profile: baseline `-O3`.** No candidate
met the 5% threshold; these small differences do not establish an improvement.
The controlled fixed-batch baseline was not benchmarked against the original
adaptive/multithread build, so this is not a claim that the new launch contract
is faster than the original execution mode.

Raw wall seconds, in repetition order, are preserved here for later sessions:

```text
p=8, n=524288
baseline     2.573735 2.540945 2.553073 2.552656 2.593761
O2           2.556292 2.571665 2.644278 2.568103 2.643289
native_tune  2.532325 2.515429 2.572567 2.582578 2.612851
native_arch  2.540112 2.558017 2.590256 2.575759 2.653609
p=12, n=32768
baseline     2.419086 2.396388 2.326340 2.333647 2.435514
O2           2.355225 2.338122 2.350284 2.407733 2.371218
native_tune  2.351012 2.373320 2.351551 2.442755 2.340200
native_arch  2.430611 2.357616 2.360404 2.418058 2.347883
p=16, n=2048
baseline     2.326566 2.311453 2.269041 2.311343 2.333877
O2           2.301330 2.302678 2.313959 2.308374 2.298919
native_tune  2.323110 2.263106 2.277604 2.262443 2.310811
native_arch  2.302050 2.272021 2.309131 2.280844 2.320184
```

The initial controlled baseline executable SHA-256 was
`c94916eaf581aa6acd4a1f86f02149b9cb2cce3cec21219d6efeea974f8146f3` on
`macOS-26.6-arm64-arm-64bit`. Rebuilt binaries may differ; use each build's
own sidecar identity. In these signal tests, graceful stops saved 216, 219,
215, and 222 samples for baseline, O2, native tuning, and native architecture
respectively; each SIGKILL test resumed the saved 31-sample prefix. All final
4096-sample outputs matched byte-for-byte under the stated contract.

The machine-readable report records all cases, angles hashes, exact compiler
commands, compiler/runtime identity, binary SHA-256, individual trial times,
medians, rejection reasons, and selection. Local temporary artifacts are not
committed and may later be deleted; this document preserves the measured
conclusions. `--export-builds` also writes `validation.json` and a
`viral_<profile>.build.json` beside each validated executable.

**Locally verified depth ceiling: p=16.** The p=30 prefix tests allocate shallow
trees; they are not p=30 executions. There are no observed p=18..30 timings or
memory measurements from this Mac, and no verified p=30 speedup. Linux
compiler/runtime equivalence, large-depth arithmetic, actual Torch RSS,
scheduler behavior, wall time, and the million-sample campaigns await your
Torch validation.

## 3. Memory and single-thread campaign costs

For one worker, the unchanged source allocates:

```text
NST = 2^(p+1) - 2                   stored Vec3 states (24 bytes each)
NW  = ceil((4*2^p - 2) / 64)       seed words (8 bytes each)
ZS  = 2^(p-1) for p>=2, else 2    two double scratch arrays
workspace_bytes = 24*NST + 8*NW + 16*ZS
```

For p>=4 this is `56.5*2^p - 48` bytes. Program, OpenMP, allocator, geometry,
and operating-system overhead are additional. **p=30 is still approximately
56.5 GiB per process**. A conservative initial request is `--mem=60G`, subject
to measured RSS and the site's limits. This work does not reduce memory.

The supplied million-sample partitions must remain fixed when reproducing
those campaigns:

| p | Expected job IDs | Samples/job | Total | Workspace/process | Initial memory request | Estimated single-thread time/job |
|---|---|---:|---:|---:|---:|---:|
| 24 | 0..49 | 20,000 | 1,000,000 | 0.883 GiB | 2G | 7.3 h |
| 26 | 0..99 | 10,000 | 1,000,000 | 3.531 GiB | 5G | 11.9 h |
| 28 | 0..199 | 5,000 | 1,000,000 | 14.125 GiB | 16G | 20.8 h |
| 30 | 0..499 | 2,000 | 1,000,000 | 56.500 GiB | 60G | 33.3 h |

Times are **estimates**, obtained by multiplying the old README's Intel
seconds/sample/core by samples/job. Its p=28 and p=30 rates were themselves
extrapolated. These are not Torch or fixed-batch measurements. Individual
one-thread jobs take longer than the old multi-thread jobs. In particular,
p=30 may require resubmission with a 24-hour wall limit. Use the pilot to set
memory and time; request limits/concurrency according to your allocation.

Sample `k` in job `j` uses `S=(j<<32)|k`, then the unchanged deterministic RNG.
Splitting 500x2000 into another partition selects different random samples,
even when the total is still a million. Freeze the partition and angles before
starting. Changing how many existing IDs run concurrently is fine. Never run
the same campaign/job ID concurrently in two submissions: the original solver
has no per-job writer lock.

## 4. Build and validate on Torch (user-run only)

Use an allocation's compute node for tests and benchmarks. As checked on
2026-09-18, [NYU's Torch submission instructions](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)
require `--account` and say not to select a partition manually for ordinary
jobs. Use `my_slurm_accounts` to find your actual account; these CPU jobs need
no GPU. Confirm your current account/resource limits before choosing requests.

First set paths from the repository root on shared storage:

```bash
my_slurm_accounts
export ACCOUNT=torch_pr_REPLACE_WITH_YOUR_ACCOUNT
export REPO="$PWD"
: "${SCRATCH:?set SCRATCH to your shared scratch directory}"
export WORK="$SCRATCH/viral-controlled"
mkdir -p "$WORK"
# Four CPUs are only for the original self-test's explicit multi-thread check.
srun --account="$ACCOUNT" --ntasks=1 --cpus-per-task=4 \
  --mem=2G --time=01:00:00 --pty bash
```

Inside that compute allocation, load an available GCC module if needed
(`module avail gcc`, then `module load <the actual module name>`), and run:

```bash
cd "$REPO"
gcc --version
python3 tests/test_build_equivalence.py --cc gcc --benchmark \
  --report "$WORK/torch-small.json" --export-builds "$WORK/torch-small-builds"
```

This compiles **both** reference and candidates on Torch with that same GCC
and its OpenMP runtime, validates them, and exports only passing builds. The
baseline compile is equivalent to:

```bash
gcc -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 -fopenmp \
  viral_tree_hpc.c -lm -o /a/new/path/viral_baseline
# Example candidate (not automatically selected or approved):
gcc -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O2 -fopenmp \
  viral_tree_hpc.c -lm -o /a/new/path/viral_O2
```

Use runner-exported executables for campaigns: manual compiler commands alone
do not create the validated build sidecars required by the launcher. Do not
use `make native`, substitute a Mac binary, or copy the Mac selection into a
Torch decision. Native flags describe the CPU on which compilation happens;
build/test on the intended compute CPU class and keep production on a tested,
compatible class. If the cluster has heterogeneous CPUs, validate each intended
class or retain the portable baseline. Keep the compiler module/runtime
environment consistent at submission and during resumed runs.

Next do larger-depth spot checks in increasing order. The runner always keeps
its small suite and adds the requested depths with job IDs 0 and 499. The
example below is a **p=30 allocation**, not something to run on the Mac or a
login node. Earlier spot checks can request less memory using the table above.

```bash
# After leaving the small-test allocation:
srun --account="$ACCOUNT" --ntasks=1 --cpus-per-task=1 \
  --mem=60G --time=04:00:00 --pty bash
cd "$REPO"
# Choose profiles from your Torch results. baseline is always included.
# Replace O2 below if appropriate; using only baseline checks its own lifecycle.
python3 tests/test_build_equivalence.py --cc gcc --profiles baseline,O2 \
  --skip-existing-checks --spot-depths 24,26,28,30 --spot-samples 4 \
  --report "$WORK/torch-large.json" --export-builds "$WORK/torch-large-builds"
```

Large tests run sequentially, so allocation memory need is the largest single
worker, not one workspace for every profile at once. Increase time if pilots
show this is insufficient. A candidate rejected here must not be used even if
it passed/speeded up small tests. The large pass does not benchmark/select a
new winner; review the earlier benchmark plus large-check status. If the final
binary differs from the measured/validated one, repeat validation before use.

For an actual peak-RSS measurement on Linux, use a fresh directory and a
passing exported binary (the example deliberately uses the baseline):

```bash
export EXE="$WORK/torch-large-builds/viral_baseline"
export ANGLES="$REPO/params/angles_p30.txt"
export RSS_OUT="$WORK/rss-p30-reference"
mkdir -p "$RSS_OUT/jobs" "$RSS_OUT/logs"
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE /usr/bin/time -v \
  "$EXE" 30 4 0 "$ANGLES" "$RSS_OUT/jobs" \
  >"$RSS_OUT/logs/solver.out" 2>"$RSS_OUT/logs/time.txt"
```

Run in the allocation, once for a reference and once for a passing candidate,
using **different fresh directories** and the same angles path. Linux GNU
time's maximum RSS is in KiB. A second invocation into a complete directory
would do no sampling and would give a misleading memory/time measurement.
Retain these measurements and the validation JSON alongside campaign records.

## 5. Output layout and campaign manifest

Each campaign has its own directory. Keep reference, candidate, RSS, pilot,
and production outputs separate; do not merge overlapping job IDs across them.

```text
results/<campaign>/
  manifest.json
  provenance.txt
  jobs/viral_tree_p<P>_job<ID>.txt
  logs/slurm_<array-id>_<task-id>.out
  logs/provenance_<array-id>_<task-id>.jsonl
  summary.txt
```

The extra JSONL sidecars record each launch/requeue's build information, runtime
linkage, host, array/task IDs, angles hash, and settings. Solver output bytes and
checkpoint names are untouched. The frozen manifest records depth, complete
angles-file SHA-256 and solver FNV hash, absolute angles path, executable
SHA-256 and build identity, explicit expected IDs, samples per job, total, and
the one-thread/fixed-batch contract. Keep angles and executable at those paths
until the campaign is finished. Do not hand-edit a manifest to adopt another
build, parameter file, partition, or old result directory.

The `init` command below creates **both output and log directories before
submission**. SLURM opens its log before the wrapper executes, so creation
inside a running batch script would be too late. Supply the shown `--output`
option: the wrapper's fallback log pattern is in the submission directory.

## 6. Pilot, arbitrary depth, then full campaigns

For a pilot, keep production's samples/job and use just two of its job IDs in a
separate campaign. This measures progress/resume behavior with the real target.
The following is a p=30 example; set `EXE` to your Torch-validated selection:

```bash
cd "$REPO"
export EXE="$WORK/torch-large-builds/viral_baseline"
export OUT="$REPO/results/torch-p30-pilot-reference"
export ANGLES="$REPO/params/angles_p30.txt"
python3 scripts/viral_campaign.py init --p 30 --samples 2000 --jobs 2 \
  --angles "$ANGLES" --exe "$EXE" --out "$OUT"
sbatch --account="$ACCOUNT" --array=0-1%2 --mem=60G --time=02:00:00 \
  --output="$OUT/logs/slurm_%A_%a.out" scripts/run_viral.slurm \
  30 2000 "$ANGLES" "$EXE" "$OUT"
```

The pilot will probably be partial. Inspect its saved counts, scheduler MaxRSS,
and elapsed/core seconds, then resubmit the same arguments to observe resume.
No production submission is warranted solely from a small-depth speed test.

A generic submission for any supplied even depth 2..30 is:

```bash
cd "$REPO"
P=8
JOBS=10
N=1000
MEM=2G
WALL=01:00:00
CONCURRENCY=4
ANGLES="$REPO/params/angles_p${P}.txt"
OUT="$REPO/results/torch-p${P}-production-v1"
python3 scripts/viral_campaign.py init --p "$P" --samples "$N" --jobs "$JOBS" \
  --angles "$ANGLES" --exe "$EXE" --out "$OUT"
sbatch --account="$ACCOUNT" --array="0-$((JOBS-1))%${CONCURRENCY}" \
  --mem="$MEM" --time="$WALL" --output="$OUT/logs/slurm_%A_%a.out" \
  scripts/run_viral.slurm "$P" "$N" "$ANGLES" "$EXE" "$OUT"
```

For another valid depth, set `P` and explicitly set `ANGLES` to a file with at
least that many indexed layers. Extra layers are permitted: the solver reads
only the first P but hashes the **entire** file. An odd-depth run with a prefix
is not a claim that those are independently trained optimal angles. The helper
requires explicit angles outside the supplied even 2..30 set. The unchanged
solver's limits are p<=40, job ID<=2147483647, samples/job<=4294967295. Angles
paths must contain no whitespace and fit the solver's 512-byte path buffer.

For each supplied million-sample campaign, choose **one** row's assignments
below, then run the common `init`/`sbatch` block once. These commands preserve
the original sample partitions; they do not submit anything automatically.

```bash
# Choose one row:
P=24; JOBS=50;  N=20000; MEM=2G
# P=26; JOBS=100; N=10000; MEM=5G
# P=28; JOBS=200; N=5000;  MEM=16G
# P=30; JOBS=500; N=2000;  MEM=60G

WALL=24:00:00                 # adapt to measured rates and allocation limits
CONCURRENCY=4                # concurrent existing IDs; adjust to your allocation
ANGLES="$REPO/params/angles_p${P}.txt"
OUT="$REPO/results/torch-p${P}-million-v1"
python3 scripts/viral_campaign.py init --p "$P" --samples "$N" --jobs "$JOBS" \
  --angles "$ANGLES" --exe "$EXE" --out "$OUT"
sbatch --account="$ACCOUNT" --array="0-$((JOBS-1))%${CONCURRENCY}" \
  --mem="$MEM" --time="$WALL" --output="$OUT/logs/slurm_%A_%a.out" \
  scripts/run_viral.slurm "$P" "$N" "$ANGLES" "$EXE" "$OUT"
```

The wrapper always requests one task and one CPU, sets OpenMP threads to one,
clears inherited dump output, validates the manifest/build and task ID, writes
provenance, then `exec`s the unchanged solver. The batch SIGTERM at 600 seconds
before the wall limit reaches the solver, which finishes its current sample
and checkpoints. That lead time is a starting value, not a guarantee if a
sample takes over ten minutes; adjust `--signal=B:TERM@SECONDS` after pilots.
A SIGKILL/OOM/node failure loses work since the last successful checkpoint
(normally at 60-second intervals, checked at sample boundaries); resumption
replays those deterministic samples. Do not use per-sample dumps in production:
after a forced kill, an append-only dump can contain uncheckpointed duplicates.

## 7. Progress, safe resubmission, exact completion, and merge

Save the submitted array ID and inspect both scheduler and result progress:

```bash
squeue -u "$USER"
ARRAY_ID=123456               # replace with sbatch's returned array ID
sacct -j "$ARRAY_ID" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
python3 scripts/viral_campaign.py status "$OUT"
python3 merge_results.py "$OUT/jobs" -p "$P"    # explicitly partial preview
```

The existing C program can exit zero after a graceful partial stop. It also
warns about some late checkpoint write failures without necessarily returning
failure. Therefore a SLURM `COMPLETED` state/zero exit code alone does not prove
scientific completion. Inspect logs and run the manifest verifier.

Wait for all previous tasks for this campaign to leave the queue before
resubmitting. Keep the same angles, executable, output directory, job IDs,
and samples/job; **do not re-run `init`**. Derive the unfinished/missing IDs:

```bash
PENDING=$(python3 scripts/viral_campaign.py pending "$OUT") || exit 1
if [ -n "$PENDING" ]; then
  sbatch --account="$ACCOUNT" --array="${PENDING}%${CONCURRENCY}" \
    --mem="$MEM" --time="$WALL" --output="$OUT/logs/slurm_%A_%a.out" \
    scripts/run_viral.slurm "$P" "$N" "$ANGLES" "$EXE" "$OUT"
fi
```

Use the campaign's saved `P`, `N`, `MEM`, `WALL`, `CONCURRENCY`, `ANGLES`, and
`EXE` values; the launcher refuses parameters that differ from the manifest.
To recreate variables after a new login, inspect `manifest.json` and restore
the compiler module environment. `pending` refuses corrupt/unexpected outputs
instead of silently dropping them. You can instead resubmit the complete
original array: already-complete IDs return without adding samples. Separate
submission IDs give distinct logs; a requeue under the same ID appends.

Final verification and unchanged merger, saving its report only after the
expected IDs/counts check succeeds:

```bash
python3 scripts/viral_campaign.py verify "$OUT" && \
  python3 merge_results.py "$OUT/jobs" -p "$P" -v > "$OUT/summary.txt.tmp" && \
  mv "$OUT/summary.txt.tmp" "$OUT/summary.txt"
cat "$OUT/summary.txt"
```

Completion requires every expected ID exactly once, matching depth/angles/path
and one-thread setting, `status complete`, `n_target = n_done = samples/job`,
valid counts, no unexpected result files, and exactly the manifest's total
samples. For the supplied campaigns that is **1,000,000**. The merger's
`0 still running / incomplete` only describes files present and cannot replace
this verification. Preserve the whole campaign directory, validated executable
and sidecar, validation reports, source commit, and angles. Share `summary.txt`
plus the manifest/provenance and job files for a reproducible result.

## 8. Instructions for the next Codex session

1. Read this handoff and the new runner/wrapper before operating the campaign.
2. Recheck the protected commit/source and `git status`; preserve existing work.
3. Treat only the local checks/measurements recorded above as established.
   Do not claim that p=30 ran locally, memory fell, or Torch flags were selected.
4. Obtain actual Torch validation/pilot evidence before choosing a production
   build and resource requests. This implementation made no remote connection
   and submitted no jobs.
5. Freeze build/angles/partition and use separate output directories. Never
   combine the same IDs from reference, candidate, pilot, and production.
6. Verify the manifest's complete expected set before publishing the merger's
   report as a completed million-sample result.
