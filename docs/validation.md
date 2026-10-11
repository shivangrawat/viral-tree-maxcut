# Validation

The published numbers rest on two kinds of evidence:

1. **The tree value is computed correctly.** An exact enumeration at p ≤ 2
   checks the dynamics independently, and the three solvers produce the same
   samples bit for bit.
2. **The campaigns are what they claim to be.** Every job is complete, the
   samples are disjoint, and the merge reproduces the published table.

This page lists the checks, the cluster jobs that ran them, and what anyone can
rerun locally. The cluster scripts are in [`scripts/validation/`](../scripts/validation/);
their logs stayed on the cluster's scratch space, and the job ids are recorded
in the manifests' `build.validation` entries.

## What "identical" means

Two runs of the same job are identical when

- the whole per-sample dump (`VIRAL_TREE_DUMP`: sample index, cut bit, and
  s_u^z, s_v^z as hex floats) is byte-identical, and
- every byte of the result file is identical except the lines `elapsed_s`,
  `core_seconds`, `sec_per_sample_core` and `host` (and, when a fast2 run uses
  T > 1 threads, the `threads` line).

No tolerances, no reparsing of floats: `soft_sum`, `x_soft`, counts,
estimates, angles metadata, comments and whitespace are all compared. The
comparator is `scientific_bytes` in `scripts/viral_campaign.py`, with its own
unit tests (`ComparatorTests` in `tests/test_build_equivalence.py`).

## Exact enumeration at p = 1, 2

`exact_small_p.py` builds the depth-p tree explicitly, runs the full,
unpruned dynamics on every vertex for all 2^6 (p = 1) or 2^14 (p = 2) seed
configurations, and shares no code with the C solvers. With
`params/angles_p2.txt`:

```
$ python3 exact_small_p.py params/angles_p2.txt 2
exact c_VIRAL(2) = 0.7556152344    (= 12380 / 16384)
```

The published Monte-Carlo value, 0.755779 ± 0.000136 from 10^7 samples, is
1.2 standard errors away. `check.sh` tests every solver against this value.

## Equivalence of the three solvers

| check | job | compared | result |
|---|---|---|---|
| build of `viral_baseline` | 17985393 | `tests/test_build_equivalence.py` on Torch with the campaign build (profile `baseline`): 64 strict cases (p = 1..16; jobs 0, 7, 499, 2^31−1) with per-sample dumps, repeat, extension, SIGTERM and SIGKILL resume, the campaign wrapper, merge and `check.sh` | passed; exported with its sidecar |
| large-p spot check | 17989395 | the same, plus spot cases at p = 24, 26, 28, 30 (72 cases) | passed; same sha256, exported for p = 28, 30 |
| R1 | 18517761 | `viral_fast` vs `viral_baseline`: 235 cases, p = 1..26, plus forced reference kernel, a 5000-sample self-check, the resume protocol and `check.sh` | all identical |
| R2 | 18517764 | timing and peak memory at p = 20..30, results compared too | identical; 3.2–3.7× faster, 7.5× less memory |
| R3 | 18527853, 18527855–57 | `viral_fast` replays of four committed million-v1 jobs, all scientific bytes of the result file | REPLAY PASS (all four) |
| G3 | 18980679 | `viral_baseline` vs `viral_fast` at p = 32 itself (240 GB job) | P32 CHECK PASS |
| fast2 suite | 18986377 | `viral_fast2` vs `viral_baseline`: 199 cases, p = 1..24, at 1, 4 and 8 threads (every level split), forced kernels, threaded self-check and extension, resume protocol, `check.sh` | all identical |
| fast2 large | 18986379 | `viral_fast2` vs `viral_fast` at p = 26, 28, 30 with 1, 2, 4, 8 threads | identical |
| fast2 scaling | 19053518, 19053519 | p = 30 at 1..32 threads | identical |
| G2 | 19055757 | the campaign build of `viral_fast2` is byte-for-byte the validated executable; at p = 32 with 8 threads it equals `viral_fast` | GATE PASS |
| G4 | 19056978, 19056979 | a small campaign through the real launcher (`fast2_campaign.py` + `run_viral_fast2.slurm`, threads) equals `viral_baseline` job for job | DRYRUN PASS |

So `viral_fast2` equals `viral_baseline` directly for p ≤ 24, and through
`viral_fast` at p = 26..32; `viral_fast` equals `viral_baseline` at p ≤ 26, in
four replays of committed jobs and at p = 32.

Why the arithmetic agrees, and what it depends on (glibc's `sincos`, the FMA
pattern of the campaign build, the association order), is explained in
[solvers.md](solvers.md#exactness). The disassembly behind the FMA pattern was
read with `scripts/fast_asm.slurm` (in tag `results-10M`).

### The self-check in every launch

At start-up `viral_tree_fast` and `viral_tree_fast2` run 64 samples at depth
min(p, 12) through their kernel and through the reference kernel and stop on
any differing bit. In the campaigns, all **4,238** launches at p ≥ 4 logged
`self-check … identical` (for example `64 samples at p=12, fused kernel
(8 threads) vs reference kernel: identical`); none failed. This checked every
node a fast binary ran on. (The SLURM logs with these lines are in tag
`results-10M`.)

## Seeds

Gate G1 (job 19056976, before extension-v1) and G1′ (job 19233795, before
extension-v2) computed the initial generator state of every sample of every
campaign, 64M and then **160M** states, and found **no two equal**
(`tests/seed_audit.c`, run by `scripts/validation/seed_audit*.slurm`). The
static part of the gates also checked that the campaigns of a depth use the
same angles, disjoint contiguous job ids and the planned targets. Two
samples with different initial states draw different seeds, so the 160M samples
are 160M different samples. (`seed_audit_10m.slurm` requires that extension-v2
has no results yet, so it fails by design when rerun on the final tree; the C
program itself can be rerun on any campaign list.)

## Campaign results

- **Completeness.** All 48 campaigns have every expected job id exactly once,
  with `status complete`, `n_done = n_target = samples per job`, the manifest's
  angles hash and thread count: 5,179 job files, 160,000,000 samples.
- **Merge.** One call of `merge_results.py` over all 48 `jobs/` directories
  reproduces every line of `results/merge_10M.txt`, and
  `merge_results.py --table` reproduces both columns of Table I of the paper for
  all 16 depths (`make table`).
- **Consistency of the campaigns.** At each depth the three campaigns are
  independent estimates of the same c_VIRAL(p). A χ² test of homogeneity
  (2 degrees of freedom per depth) gives χ² = 45.7 on 32 degrees of freedom
  over all depths (p-value 0.05); the largest contributions are p = 26 (9.5)
  and p = 30 (8.5). Comparing million-v1 with extension-v1 alone, as was done
  before the 10M extension, gives χ² = 13.9 on 16 degrees of freedom (p-value
  0.6). To recompute:

  ```python
  import glob
  def counts(d):
      n = k = 0
      for f in glob.glob(d + "/viral_tree_p*_job*.txt"):
          kv = dict(l.split(None, 1) for l in open(f) if l.strip() and not l.startswith("#"))
          n, k = n + int(kv["n_done"]), k + int(kv["n_cut"])
      return n, k
  chi2 = 0.0
  for p in range(2, 33, 2):
      c = [counts(f"results/torch-p{p}-{name}/jobs") for name in ("million-v1", "extension-v1", "extension-v2")]
      x = sum(k for _, k in c) / sum(n for n, _ in c)
      chi2 += sum((k - n * x) ** 2 / (n * x * (1 - x)) for n, k in c)
  print(chi2)          # 45.7 on 32 degrees of freedom
  ```

## Checks you can run locally

```bash
make                    # macOS: make LIBOMP=<libomp prefix>
make check              # check.sh on all three solvers
make check-sources      # the documented sources compile to the code that ran
make table              # Table I from the committed job files
python3 exact_small_p.py params/angles_p2.txt 2          # needs numpy
python3 tests/test_build_equivalence.py [--omp-root <libomp prefix>]
```

`check.sh` runs four tests on one binary:

1. p = 2, 2·10^6 samples, within 5σ of the exact 12380/16384;
2. p = 12, 200,000 samples, n_cut against the reference value 178239 (a
   difference of a few units means last-bit libm or FMA differences on that
   machine);
3. n_cut with 1 thread equals n_cut with 4 threads (p = 8);
4. a job extended from 10,000 to 30,000 samples equals a fresh 30,000-sample
   job.

On an Apple M-series Mac (clang 21, PyTorch's libomp) all three solvers give
n_cut = 178239 exactly in test 2. The equivalence suites for the fast solvers
(`tests/test_fast_equivalence.py`, `tests/test_fast2_equivalence.py`) need gcc
and, for `test_fast2_equivalence.py`, the campaign binaries, which they
identify by sha256; they are the record of what ran on Torch.

### Across platforms

The 64-bit arithmetic is deterministic, but the last bits of `sin`/`cos` and of
fused multiply-adds can differ between libms, compilers and CPUs. That can flip
the sign of a sample whose s^z is within about 10^-15 of zero, which moves
n_cut by O(1) per 10^6 samples. In practice: job 0 at p = 2 and p = 4 (10^6
samples each), run on a Mac with 4 threads during development, gave the same
n_cut as the cluster (755,814 and 828,260); `soft_sum` differed in the last
digits, because the order of additions and the libm differ
(`results/local_mac/` in tag `results-10M`).
