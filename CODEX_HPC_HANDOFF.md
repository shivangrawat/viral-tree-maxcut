# VIRAL CPU shared-workspace handoff

This is the active guide for branch **`cpu-shared-workspace`**, created from
`faster` at **`463f380927fb7ef21b80af9ba7146f849dc9bd62`**. The numerical
algorithm, supplied angles, random samples, and result-file schema are retained.
The solver now shares one tree between cooperating CPU threads and provides an
exact compact-storage kernel. No GPU or PyTorch execution is involved.

The original README and old `submit_p*.slurm` scripts describe the earlier
sample-parallel implementation. Use the validation runner and generic wrapper
below for this branch. Do not overwrite the tracked `build/` binaries or saved
`results/local_mac/` data. The runner builds and tests in temporary directories.

The implementation changes are confined to the solver, validation runner,
campaign helper, generic wrapper, this handoff, and a README pointer. All other
previously tracked files remain byte-identical, including supplied parameters,
saved results/binaries, `Makefile`, `check.sh`, the merger, and old SLURM scripts.

## 1. Reference, precision, and exactness

The immutable reference is `viral_tree_hpc.c` extracted with `git show` from the
parent commit above. Its SHA-256 is:

```text
b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51
```

The runner checks this hash before compiling the reference. The candidate has
its own source and binary hashes; replacing the reference hash with the edited
source hash is not permitted. Keep this commit available when copying/cloning
the repository onto Torch; a source-only archive cannot supply `git show`.

Reference and candidate use the same compiler, OpenMP runtime, `-O3`, and
`BATCH_MAX_PER_THREAD=1`. The reference runs with one thread. Candidate tests
use both kernels and multiple thread counts. All numeric states, intermediate
rotations, and table values are **binary64 doubles**; seeds/indices remain
integers. The C build rejects fast-math and unsupported/excess-precision double
formats. No approximate trigonometry, reduced precision, algebraically
reassociated neighbor sums, or floating-point parallel reduction is used.

Compare **whole hexadecimal sample dumps as bytes** and **whole result files
as bytes**, removing only these execution-metadata lines:

```text
threads
elapsed_s
core_seconds
sec_per_sample_core
host
```

Everything else, including `soft_sum`, `x_soft`, errors, estimates, counts,
status, angles metadata, formatting, and comments, must match exactly. The
thread field is checked separately against the actual requested test team.
Never parse and reformat scientific floating-point values for comparison.
Byte equivalence is within a validated platform/CPU class/toolchain, not a
promise of identical Mac and Linux libm results.

## 2. Kernels, memory, and execution

The positional command line and checkpoint names are unchanged:

```bash
OMP_NUM_THREADS=4 VIRAL_TREE_KERNEL=compact \
  /path/to/validated/viral_cpu P SAMPLES JOB_ID ANGLES_FILE OUTPUT_DIRECTORY
```

`OMP_NUM_THREADS` defaults to **one** when unset. It now specifies threads
cooperating on one sample, even when only one sample remains. A dump does not
force candidate computation to one thread. `VIRAL_TREE_KERNEL` selects:

- **`shared`**: full original tree storage, one shared workspace per job.
- **`compact`**: exact first-layer table and omitted outer stored level,
  one smaller shared workspace per job.

A persistent OpenMP team generates the original RNG words serially, then
updates independent vertices in contiguous static partitions. The roots and
levels below 1,024 vertices use thread 0. Barriers preserve the old neighbor
values needed by the next level. Consecutive small levels run without redundant
waits; other threads wait while thread 0 finishes the remaining small layers.
Samples, their zero-initialized additions,
dump rows, and checkpoint updates remain in increasing sample-index order.
Parallel first-touch initializes the arrays. Extra threads add runtime/stacks,
not additional complete trees.

The 16-entry first-layer table distinguishes all ordered combinations of the
vertex's seed bit and its three neighbors. It uses the existing rotation
expressions. It does not replace distinct ordered patterns by a simplified
sum. In compact mode the outer stored level is omitted: only its first-layer
`z` is subsequently needed, recovered from the table and original seed bits.

For p>=4, main workspace bytes are:

```text
shared:  24*(2^(p+1)-2) + 8*ceil((4*2^p-2)/64) + 16*2^(p-1)
         = 56.5*2^p - 48
compact: 24*(2^p-2)     + 8*ceil((4*2^p-2)/64) + 16*2^(p-2)
         = 28.5*2^p - 48
```

p=1 and p=2 have explicit small-workspace handling. Geometry, the 384-byte
table, allocator, OpenMP, stacks, and executable overhead are additional.

| p | Shared workspace/job | Compact workspace/job | Initial compact memory request |
|---|---:|---:|---:|
| 24 | 0.883 GiB | 0.445 GiB | 2G |
| 26 | 3.531 GiB | 1.781 GiB | 3G |
| 28 | 14.125 GiB | 7.125 GiB | 9G |
| 30 | 56.500 GiB | **28.500 GiB** | **32G** |

These requests are starting estimates for pilots, not measured Torch RSS.
Reference/shared p=30 validation still needs about **60G**, regardless of the
candidate's smaller workspace. Always use that larger requirement when one
validation invocation tests both kernels/reference sequentially.

SIGTERM, SIGINT, and SIGUSR1 finish the current sample and save progress.
SIGKILL/OOM/node failure rolls back to the last successfully saved checkpoint.
The usual checkpoint interval is 60 seconds, checked at sample boundaries.
A zero exit after a graceful stop can still be incomplete. Failed checkpoint
creation/rename returns nonzero in the new driver. Also inspect flush/fsync
warnings and verify counts; a scheduler exit code alone cannot prove completion.
Do not enable append-only sample dumps in production: forced-stop replay can
append duplicates from uncheckpointed samples.

## 3. Local validation and measured results

On this Mac, the existing LLVM OpenMP headers/runtime reside inside an installed
package; no PyTorch import, dependency installation, or Torch connection occurs:

```bash
python3 -B tests/test_build_equivalence.py --cc /usr/bin/clang \
  --omp-root /Users/shivangrawat/miniforge3/lib/python3.12/site-packages/torch \
  --threads 1,2,3,4,8 --benchmark \
  --report /tmp/viral-cpu-validation.json --export-builds /tmp/viral-cpu-builds
```

Use new report/export paths for another run. A different Mac can supply its
actual installed LLVM OpenMP prefix. The runner fixes dylib paths and signs
only its temporary binaries. It exports `viral_reference`, `viral_cpu`, their
`.build.json` sidecars, and `validation.json`. Production setup requires those
validated sidecars. The report includes compiler/runtime identity, source and
binary hashes, all test configurations, individual timing/RSS measurements,
and selection criteria.

Default correctness coverage is **92 cases per configuration**: the previous
64 cases through p=16; bounded p=18/20 cases at job IDs 0 and 499, with one and
four samples; and zero, signed-zero, zero-field, and large-field fixtures at
p=1..5. Both kernels are tested at 1, 2, 3, 4, and 8 threads: **920 candidate
case comparisons**, in addition to the reference cases and lifecycle tests.

Also checked:

- All 16 table entries against direct reference arithmetic for 15 supplied
  angles files p=2..30 and four artificial fixtures: 304 table-entry checks.
- Exact allocation-size formulas for both kernels at every p=1..40, with no
  large tree allocation during that check.
- Repetition, extension, completed reruns, and SIGTERM/SIGKILL recovery at
  p=12/job 13/4096 samples for every configuration.
- A continuation chain from the reference, through every kernel/thread
  configuration, and back to the reference, matching a fresh run and dump.
- Legacy schema-1 and new schema-2 manifests, simulated local array launches,
  the unchanged merger, missing/duplicate results, wrong targets/threads, and
  rejection of a scheduler CPU count that differs from the manifest.
- 30 AddressSanitizer/UndefinedBehaviorSanitizer cases, including p=12 where
  real multi-thread vertex partitions execute. Leak detection is disabled;
  the checks cover invalid accesses and undefined behavior.
- The unchanged original `check.sh` in an isolated copy, with one thread for
  its two-million-sample shallow statistical check. The new matrix separately
  supplies the multi-thread exactness coverage.

**Locally verified tree depth ceiling: p=20.** Reading p=30 angles for a
first-layer table or shallow prefix is not a full p=30 tree run.

**Measured on this Mac, 2026-09-18:** Apple clang 21.0.0
(`clang-2100.1.1.101`), ARM64, macOS 26.6, Python 3.9.6, and the existing
LLVM OpenMP runtime. All ten configurations passed. **Selected default:
`compact`, ordinary `-O3`.** Its one-thread geometric mean time ratio to the
reference was **0.613890**, or **38.61% less time** (1.63x throughput).
Every measured depth improved; the smallest time reduction was 35.51%.

Median wall timing and allocated CPU cost below use five repetitions.
RSS is the largest observed peak among those five trials. `reference` is
single-threaded; suffixes indicate cooperating threads.

| p | Configuration | Samples/trial | Wall s | Microseconds/sample | Samples/s | CPU microseconds/sample | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 8 | `reference` | 524,288 | 2.580094 | 4.921 | 203204.97 | 4.921 | 2.000 |
| 8 | `shared_t1` | 524,288 | 2.536313 | 4.838 | 206712.64 | 4.838 | 2.000 |
| 8 | `shared_t2` | 147,456 | 1.506316 | 10.215 | 97891.80 | 20.431 | 2.062 |
| 8 | `shared_t4` | 81,920 | 1.590897 | 19.420 | 51492.96 | 77.681 | 2.109 |
| 8 | `shared_t8` | 49,152 | 2.127002 | 43.274 | 23108.58 | 346.192 | 2.281 |
| 8 | `compact_t1` | 524,288 | 1.663826 | 3.173 | 315109.89 | 3.173 | 1.984 |
| 8 | `compact_t2` | 147,456 | 1.569831 | 10.646 | 93931.11 | 21.292 | 2.047 |
| 8 | `compact_t4` | 81,920 | 1.546937 | 18.884 | 52956.27 | 75.534 | 2.125 |
| 8 | `compact_t8` | 49,152 | 2.150502 | 43.752 | 22856.06 | 350.017 | 2.312 |
| 12 | `reference` | 32,768 | 2.367700 | 72.256 | 13839.59 | 72.256 | 2.219 |
| 12 | `shared_t1` | 32,768 | 2.336643 | 71.309 | 14023.54 | 71.309 | 2.219 |
| 12 | `shared_t2` | 14,336 | 1.546287 | 107.860 | 9271.24 | 215.721 | 2.281 |
| 12 | `shared_t4` | 20,480 | 2.948330 | 143.961 | 6946.31 | 575.846 | 2.328 |
| 12 | `shared_t8` | 6,144 | 1.713196 | 278.841 | 3586.28 | 2230.724 | 2.500 |
| 12 | `compact_t1` | 49,152 | 2.160844 | 43.962 | 22746.67 | 43.962 | 2.109 |
| 12 | `compact_t2` | 32,768 | 2.225436 | 67.915 | 14724.30 | 135.830 | 2.188 |
| 12 | `compact_t4` | 15,360 | 1.504908 | 97.976 | 10206.61 | 391.903 | 2.219 |
| 12 | `compact_t8` | 8,192 | 1.505506 | 183.778 | 5441.36 | 1470.221 | 2.406 |
| 16 | `reference` | 2,048 | 2.300077 | 1123.085 | 890.40 | 1123.085 | 5.578 |
| 16 | `shared_t1` | 2,048 | 2.265710 | 1106.304 | 903.91 | 1106.304 | 5.578 |
| 16 | `shared_t2` | 2,048 | 1.766235 | 862.419 | 1159.53 | 1724.839 | 5.641 |
| 16 | `shared_t4` | 3,072 | 2.227165 | 724.989 | 1379.33 | 2899.955 | 5.688 |
| 16 | `shared_t8` | 2,048 | 2.325968 | 1135.727 | 880.49 | 9085.812 | 5.859 |
| 16 | `compact_t1` | 3,072 | 2.056001 | 669.271 | 1494.16 | 669.271 | 3.797 |
| 16 | `compact_t2` | 3,072 | 1.746305 | 568.459 | 1759.14 | 1136.918 | 3.859 |
| 16 | `compact_t4` | 3,072 | 1.590620 | 517.780 | 1931.32 | 2071.120 | 3.906 |
| 16 | `compact_t8` | 2,048 | 1.732530 | 845.962 | 1182.09 | 6767.696 | 4.078 |
| 20 | `reference` | 128 | 2.248717 | 17568.103 | 56.92 | 17568.103 | 58.578 |
| 20 | `shared_t1` | 128 | 2.247022 | 17554.858 | 56.96 | 17554.858 | 58.562 |
| 20 | `shared_t2` | 192 | 1.872490 | 9752.553 | 102.54 | 19505.105 | 58.625 |
| 20 | `shared_t4` | 256 | 1.500596 | 5861.702 | 170.60 | 23446.807 | 58.656 |
| 20 | `shared_t8` | 320 | 1.610490 | 5032.782 | 198.70 | 40262.254 | 58.844 |
| 20 | `compact_t1` | 192 | 2.048902 | 10671.363 | 93.71 | 10671.363 | 30.562 |
| 20 | `compact_t2` | 256 | 1.541794 | 6022.632 | 166.04 | 12045.265 | 30.625 |
| 20 | `compact_t4` | 448 | 1.653912 | 3691.767 | 270.87 | 14767.068 | 30.672 |
| 20 | `compact_t8` | 448 | 1.538138 | 3433.343 | 291.26 | 27466.747 | 30.844 |

At p=20, compact/8 takes **3.433 ms/sample**, 5.12x faster than the
reference and 3.11x faster than compact/1. It costs 27.467 allocated CPU
ms/sample, **2.57x the CPU allocation per sample of compact/1**. Compact/4
takes 3.692 ms/sample at 14.767 allocated CPU ms/sample. At p=8 and p=12,
one thread is fastest; at p=16, compact/4 is fastest. These are local
measurements, not production thread recommendations for Torch.

The separate four-sample p=20 RSS check on the final source measured:

| Threads | Shared peak RSS MiB | Compact peak RSS MiB |
|---:|---:|---:|
| 1 | 58.547 | 30.547 |
| 2 | 58.609 | 30.625 |
| 3 | 58.641 | 30.625 |
| 4 | 58.672 | 30.672 |
| 8 | 58.844 | 30.844 |

Reference RSS was 58.547 MiB. Compact RSS stayed near 30.6--30.8 MiB
as the team grew, consistent with one 28.5 MiB workspace plus runtime
overhead. This checks sharing at p=20; **28.5 GiB at p=30 remains a formula,
not a measured total RSS**.

Recorded identities:

```text
Candidate source SHA-256:
da486d69df64e7a8c15bf547e2a2884d8a10ed2526d42233d5be4c94afc24a52
Reference executable SHA-256:
c0d44cea2a358183e1787c7c7a2fd80cd89855b5b13f0e98d6ceb50463333f47
Candidate executable SHA-256:
2123bf690772b11df260d7d5068af3a5a97f4609ce7acbf41ddbaa5168771868
```

The final source passed the full suite in
`/tmp/viral-cpu-release-validation.json`; exported binaries and sidecars are
in `/tmp/viral-cpu-release-builds/`. The five-repetition benchmark report is
`/tmp/viral-cpu-final-validation.json`. Both runs produced **identical reference
and candidate binary hashes** above. Preserve these temporary artifacts if
needed; the measurements and raw wall trials below remain in this handoff.

<details>
<summary>Raw benchmark wall seconds, in repetition order</summary>

`n` is samples per trial. CPU-seconds = wall-seconds times the thread count;
samples/second = n / wall-seconds.

```text
p=8  reference   n=524288  2.569827458 2.593598500 2.556986041 2.597359208 2.580094417
p=8  shared_t1   n=524288  2.536313167 2.531464167 2.524417875 2.539338500 2.540045458
p=8  shared_t2   n=147456  1.508728250 1.504551334 1.493687083 1.506316166 1.507149208
p=8  shared_t4   n=81920   1.592854792 1.598967458 1.584405959 1.590897000 1.588072667
p=8  shared_t8   n=49152   2.123937208 2.198774833 2.126819292 2.127002333 2.135695625
p=8  compact_t1  n=524288  1.661931250 1.663825916 1.662628958 1.669502083 1.665130333
p=8  compact_t2  n=147456  1.573080042 1.555450625 1.582231125 1.569831333 1.568683583
p=8  compact_t4  n=81920   1.544478958 1.545279375 1.587534583 1.546936750 1.550441791
p=8  compact_t8  n=49152   2.146405125 2.150502250 2.140459292 2.214678209 2.152636875
p=12 reference   n=32768   2.333978750 2.432375500 2.353541875 2.367699958 2.438833916
p=12 shared_t1   n=32768   2.366635083 2.326198375 2.338731708 2.336642625 2.322279333
p=12 shared_t2   n=14336   1.548946042 1.535182541 1.540209333 1.546287417 1.548529250
p=12 shared_t4   n=20480   2.933656750 2.980058500 2.948330042 2.941873334 2.967687417
p=12 shared_t8   n=6144    1.713196375 1.690946250 1.734514625 1.703472666 1.762821083
p=12 compact_t1  n=49152   2.181299209 2.172924541 2.160844000 2.141466458 2.160638083
p=12 compact_t2  n=32768   2.208147375 2.215087666 2.232145125 2.239054917 2.225436208
p=12 compact_t4  n=15360   1.516642833 1.505743584 1.504907833 1.500507250 1.501920750
p=12 compact_t8  n=8192    1.505505833 1.545530500 1.489405375 1.502788416 1.531652125
p=16 reference   n=2048    2.283908916 2.304370334 2.300077209 2.298565791 2.372087667
p=16 shared_t1   n=2048    2.265652083 2.265709750 2.297461792 2.264503875 2.318786750
p=16 shared_t2   n=2048    1.760866083 1.762132167 1.766234958 1.776897083 1.801701417
p=16 shared_t4   n=3072    2.216699875 2.250501167 2.227165416 2.213566375 2.430980125
p=16 shared_t8   n=2048    2.324025750 2.356942500 2.349871292 2.325967875 2.322720584
p=16 compact_t1  n=3072    2.056001041 2.052082625 2.060970583 2.055592667 2.085751417
p=16 compact_t2  n=3072    1.748024541 1.742854833 1.746305333 1.742920875 1.761910292
p=16 compact_t4  n=3072    1.590619792 1.585530916 1.571232917 1.602982459 1.685844666
p=16 compact_t8  n=2048    1.722716833 1.707680959 1.732530083 1.782720208 1.965808833
p=20 reference   n=128     2.339670041 2.248717166 2.239670500 2.249457959 2.241582416
p=20 shared_t1   n=128     2.289503666 2.323343417 2.244379666 2.247021833 2.241612917
p=20 shared_t2   n=192     1.955826666 1.927421917 1.869973084 1.868788250 1.872490083
p=20 shared_t4   n=256     1.546500791 1.587627583 1.484330125 1.500595666 1.484511625
p=20 shared_t8   n=320     1.610490167 1.667576875 1.612802583 1.583947458 1.594185375
p=20 compact_t1  n=192     2.090268250 2.087306792 2.048901666 2.045295500 2.037212417
p=20 compact_t2  n=256     1.590453333 1.575403583 1.540699375 1.541793875 1.538812042
p=20 compact_t4  n=448     1.717097375 1.694653583 1.631840417 1.653911625 1.637549250
p=20 compact_t8  n=448     1.587503709 1.526484209 1.562472125 1.525310584 1.538137834
```

</details>

Benchmarking uses p=8/12/16/20 and five interleaved repetitions, fresh output
directories, and no sample dumps. Each configuration has its own calibrated
sample count targeting at least 1.5 seconds, avoiding excessively long trials
for shallow multi-thread runs. Compare **median seconds per sample**, not raw
wall seconds between different counts. Every timed result is checked against
the frozen reference with exactly the same sample count and job ID 19.

Peak RSS is measured through a fresh standard-library Python parent with
exactly one solver child, using `RUSAGE_CHILDREN`; Mac byte/Linux KiB units are
normalized. Solver wall time includes startup, allocation, and result writes,
but excludes the measurement parent's startup. Allocated CPU-seconds are wall
seconds times the actual tested team size; they are not a measurement of how
long each worker was busy.

Compact becomes the default only after its one-thread geometric mean time
ratio to reference is <=0.95 and no depth ratio exceeds 1.05, with all exactness
checks passing. Multi-thread gains and CPU costs are reported separately.
There is no measured p=30 speedup, runtime, or RSS from this Mac.

## 4. Torch build, validation, and pilot (user-run only)

As checked on 2026-09-18, [NYU's Torch submission instructions](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)
require an account and advise against manually selecting a partition for
ordinary jobs. Use `my_slurm_accounts` and your allocation's actual limits.
These are CPU jobs; request no GPU.

From the repository root on shared storage:

```bash
my_slurm_accounts
export ACCOUNT=torch_pr_REPLACE_WITH_YOUR_ACCOUNT
export REPO="$PWD"
: "${SCRATCH:?set SCRATCH to your shared scratch directory}"
export WORK="$SCRATCH/viral-cpu"
mkdir -p "$WORK"
srun --account="$ACCOUNT" --ntasks=1 --cpus-per-task=8 \
  --mem=2G --time=02:00:00 --pty bash
```

Inside the compute allocation, load an available GCC module if needed
(`module avail gcc`, then the actual module name). Keep that compiler/runtime
module environment for subsequent jobs. Run:

```bash
cd "$REPO"
gcc --version
python3 -B tests/test_build_equivalence.py --cc gcc --threads 1,2,3,4,8 \
  --benchmark --report "$WORK/torch-small.json" \
  --export-builds "$WORK/torch-small-builds"
```

The runner compiles both source versions with the same GCC/OpenMP runtime;
ordinary compilation is `gcc -Wall -Wextra -O3 -DBATCH_MAX_PER_THREAD=1
-fopenmp SOURCE.c -lm -o EXECUTABLE`. Use runner-exported binaries for campaigns,
not an independently rebuilt executable without validation provenance. Do not
copy Mac binaries or infer Linux equivalence from the Mac results. Test the
intended CPU class and do not add native/fast-math flags to bypass failures.
A rejected check aborts validation and exports no candidate.

Then use a sufficiently large compute allocation for increasing-depth spot
checks, leaving the smaller allocation first:

```bash
srun --account="$ACCOUNT" --ntasks=1 --cpus-per-task=8 \
  --mem=60G --time=04:00:00 --pty bash
cd "$REPO"
# Load the same GCC module used for the small-depth validation above.
python3 -B tests/test_build_equivalence.py --cc gcc --threads 1,2,3,4,8 \
  --skip-existing-checks --skip-sanitizers \
  --spot-depths 24,26,28,30 --spot-samples 2 --timeout 1200 \
  --report "$WORK/torch-large.json" --export-builds "$WORK/torch-large-builds"
```

This explicitly skips two checks already run for the same toolchain; skips are
recorded. Other small-depth, lifecycle, wrapper, and memory checks still run.
Large cases use job IDs 0/499 and counts 1/2. Increase wall time or reduce the
thread/depth matrix if pilot timings require it; never lower the memory request
below the reference/shared requirement. Add a thread count to `--threads` and
validate it before requesting that configuration for a campaign.

Measure actual Linux RSS/timing with fresh, separated reference and candidate
outputs, inside an adequate compute allocation:

```bash
mkdir -p "$WORK/rss-reference/jobs" "$WORK/rss-reference/logs"
mkdir -p "$WORK/rss-compact/jobs" "$WORK/rss-compact/logs"
OMP_NUM_THREADS=1 /usr/bin/time -v "$WORK/torch-large-builds/viral_reference" \
  30 4 0 "$REPO/params/angles_p30.txt" "$WORK/rss-reference/jobs" \
  >"$WORK/rss-reference/logs/solver.out" 2>"$WORK/rss-reference/logs/time.txt"
OMP_NUM_THREADS=8 VIRAL_TREE_KERNEL=compact /usr/bin/time -v \
  "$WORK/torch-large-builds/viral_cpu" \
  30 4 0 "$REPO/params/angles_p30.txt" "$WORK/rss-compact/jobs" \
  >"$WORK/rss-compact/logs/solver.out" 2>"$WORK/rss-compact/logs/time.txt"
```

GNU time reports maximum RSS in KiB. A repeated command into an already
complete directory does no sampling and is not a valid timing/memory trial.
Use distinct directories for thread-count comparisons. Select kernel/threads
from Torch's measured wall time, allocated CPU cost, memory, and available
resources; do not assume eight threads is always faster or cheaper.
If Torch's benchmark reports `selected_default: shared`, explicitly select
`--kernel shared` for campaigns; the compiled default was accepted on this Mac.
Any scientific mismatch must be resolved or the affected configuration rejected,
regardless of timing. Never change expected values to obtain a passing run.

## 5. Frozen campaigns, outputs, and submission

A new optimized campaign uses schema 2 and freezes the candidate/reference
identity, angles hashes/path, kernel, thread count, expected IDs, and exact
samples per job. Existing schema-1 campaigns retain the old source/one-thread
contract and are never silently migrated. Direct solver tests demonstrate
cross-thread/backend checkpoint compatibility, but a production manifest keeps
one configuration fixed. Never run the same campaign/job ID concurrently.

```text
results/<campaign>/
  manifest.json
  provenance.txt
  jobs/viral_tree_p<P>_job<ID>.txt
  logs/slurm_<array-id>_<task-id>.out
  logs/provenance_<array-id>_<task-id>.jsonl
  summary.txt
```

Sidecars contain build/runtime/host/array provenance without changing solver
result files. Retain the angles, executable, and sidecar at their recorded
absolute paths. Keep reference, candidate, pilot, and production directories
separate; their overlapping job IDs must not be merged together.

Initialize a p=30 pilot after the Torch checks and RSS measurement:

```bash
cd "$REPO"
EXE="$WORK/torch-large-builds/viral_cpu"
P=30; N=2000; JOBS=2
THREADS=8; KERNEL=compact
MEM=32G; WALL=02:00:00; CONCURRENCY=2
ANGLES="$REPO/params/angles_p${P}.txt"
OUT="$REPO/results/torch-p30-compact-pilot-v1"
python3 scripts/viral_campaign.py init --p "$P" --samples "$N" --jobs "$JOBS" \
  --threads "$THREADS" --kernel "$KERNEL" --angles "$ANGLES" --exe "$EXE" --out "$OUT"
sbatch --account="$ACCOUNT" --array="0-$((JOBS-1))%${CONCURRENCY}" \
  --cpus-per-task="$THREADS" --mem="$MEM" --time="$WALL" \
  --output="$OUT/logs/slurm_%A_%a.out" scripts/run_viral.slurm \
  "$P" "$N" "$ANGLES" "$EXE" "$OUT"
```

The 32G request is an estimate; replace it using measured RSS. For `shared`,
start from 60G at p=30. The pilot may be partial. Verify its memory, progress,
signal handling, and resume before starting production.

`init` creates both job and log directories **before** `sbatch`: SLURM opens
logs before executing the wrapper. Always supply the shown `--output` option;
the wrapper's fallback log pattern is in the submission directory. It derives
OpenMP threads from `--cpus-per-task`, validates against the manifest, sets the
manifest's kernel, clears inherited sample dumps, records provenance, and
executes the solver. The signal lead time defaults to `B:TERM@600`; adjust it
if a measured sample can take longer than ten minutes.

For an arbitrary supplied even depth 2..30, change `P`, `N`, `JOBS`, resource
requests, and the output campaign name, then use the same init/submission block.
Other solver-valid depths 1..40 require an explicit angles file with enough
layers. The whole file is hashed, even when only a prefix is read. Angles paths
must fit the original 512-byte buffer and contain no whitespace.

For each existing million-sample configuration choose one row, use a fresh
production campaign name, and apply the same init/submission block:

```bash
# Choose one row; memory values are initial compact estimates:
P=24; JOBS=50;  N=20000; MEM=2G
# P=26; JOBS=100; N=10000; MEM=3G
# P=28; JOBS=200; N=5000;  MEM=9G
# P=30; JOBS=500; N=2000;  MEM=32G
WALL=24:00:00              # set from pilot results and allocation limits
CONCURRENCY=4             # simultaneous existing IDs, not samples within one job
ANGLES="$REPO/params/angles_p${P}.txt"
OUT="$REPO/results/torch-p${P}-compact-million-v1"
```

In particular, preserve **500 IDs (0..499) × 2,000 samples** for p=30. Changing
that partition selects different random samples because `S=(job_id<<32)|k`.
Changing how many of the frozen IDs run concurrently does not change samples.
No p=30 runtime is inferred from these Mac benchmarks. The old single-thread
planning estimate was 33.3 hours/job; it is historical and must not be presented
as a measured optimized runtime. Resubmit as necessary for the wall-time limit.

## 6. Progress, resume, exact completion, and archival

```bash
squeue -u "$USER"
ARRAY_ID=123456            # replace with the actual returned array ID
sacct -j "$ARRAY_ID" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
python3 scripts/viral_campaign.py status "$OUT"
python3 merge_results.py "$OUT/jobs" -p "$P"   # partial preview only
```

Wait until all earlier tasks for this campaign have left the queue before
resubmission. Restore the same compiler module and campaign variables after a
new login. Keep the same executable, kernel, CPUs, angles, output, IDs, and
sample targets. Do not run `init` again:

```bash
PENDING=$(python3 scripts/viral_campaign.py pending "$OUT") || exit 1
if [ -n "$PENDING" ]; then
  sbatch --account="$ACCOUNT" --array="${PENDING}%${CONCURRENCY}" \
    --cpus-per-task="$THREADS" --mem="$MEM" --time="$WALL" \
    --output="$OUT/logs/slurm_%A_%a.out" scripts/run_viral.slurm \
    "$P" "$N" "$ANGLES" "$EXE" "$OUT"
fi
```

Each new submission has distinct array/task logs; requeues under the same ID
append logs/provenance. Complete IDs return without adding samples. The helper
refuses malformed/unexpected results rather than silently omitting them.

Verify and save the unchanged merger's final report:

```bash
python3 scripts/viral_campaign.py verify "$OUT" && \
  python3 merge_results.py "$OUT/jobs" -p "$P" -v > "$OUT/summary.txt.tmp" && \
  mv "$OUT/summary.txt.tmp" "$OUT/summary.txt"
cat "$OUT/summary.txt"
```

Completion requires every expected ID once, matching depth/angles/thread
metadata, valid counts, complete status, and `n_done=n_target=samples_per_job`,
with exactly the manifest's target total. The supplied campaigns require
**1,000,000 samples**. A scheduler success or the merger's `0 incomplete` among
files present does not detect a missing job and cannot replace verification.

Archive the campaign, validation JSON, binaries and build sidecars, angles,
source commit and final source hashes. No remote connection or submission was
made during this implementation. Torch equivalence, full p=24..30 runs, actual
large-depth RSS/scaling, and production campaigns await your validation.

## 7. Historical compiler-only results from faster

The following measurements are preserved from the preceding additive-only
phase. They used the unchanged source and one-thread/fixed-batch reference;
they are not measurements of the new shared/compact kernels. Their original
baseline commit was `164c5e09986174ff38e0aa8c3ea4d8a695fc1cd6`, with the same
protected solver SHA-256. New commands and policies above supersede the old
one-CPU wrapper restrictions.

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
