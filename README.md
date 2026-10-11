# VIRAL on the 3-regular tree: Max-Cut bounds for large-girth graphs

This repository computes **c_VIRAL(p)**, the probability that the depth-p VIRAL
algorithm cuts the root edge of the 3-regular tree. VIRAL is a classical local
algorithm (every vertex carries a unit vector that is rotated p times by angles
set by its neighbours). Because it is local, every 3-regular graph of girth at
least 2p + 2 has a cut containing at least a fraction c_VIRAL(p) of its edges.
With 10^7 Monte-Carlo samples for every even depth p = 2..32, the bound at
p = 32 is **M_66 ≥ 0.91271 at 99.99999% confidence**, above the previous
asymptotic lower bound 0.912.

It contains the solvers, the trained angles, the raw results of all 160M
samples, and the scripts and records needed to check or reproduce every number.

> F. Morone, S. Rawat, D. Sels, *Whither quantum advantage: new classical
> bounds for Max-Cut on large-girth 3-regular graphs* (2026).

## The parameters: `params/`

**`params/angles_p<p>.txt`** holds the fixed VIRAL angles for depth p: the
angle α of the initial state and the angles β_n, γ_n of the p layers,

```
s_i(0)   = (cos α, 0, ε_i sin α),          ε_i = ±1 a fair random seed per vertex
s_i(n+1) = R_x(β_n) R_z(γ_n h_i(n)) s_i(n),  h_i(n) = Σ_{j neighbour of i} s_j^z(n)
σ_i      = sign(s_i^z(p))                   the side of the cut
```

(Eqs. (3)–(5) of the paper). There is one file for every even p = 2..100; the
results use p = 2..32, the other files are for larger depths. The angles were
trained in Ref. [15] (Morone, Kent, Sels, arXiv:2604.01512) by gradient
descent on random 3-regular graphs with 10^6 vertices and are used unchanged.
Any fixed angles give a valid bound; these are not optimised for the tree, so
the bounds are conservative.

Format, shown for `params/angles_p2.txt`:

```
# alpha  (best H/N = -0.45315545, best Ht/N = -0.76681060)    <- comment: training result
0.4556340411                                                  <- alpha
# layer  beta  gamma                                          <- comment
   1  1.0782368501  -0.8633872203                             <- layer 1: beta_0, gamma_0
   2  0.7602795404  -1.2884226254                             <- layer 2: beta_1, gamma_1
```

- Angles are in radians. Line n + 3 of the file is layer n of the file's
  numbering (1..p), i.e. (β_{n−1}, γ_{n−1}) in the paper's.
- Line 1 records the training result for reference: the lowest energy per
  vertex reached on the training graphs, with the continuous spins s^z (H/N)
  and after rounding the spins to ±1 (Ht/N). The solvers skip it.
- The solvers read `params/angles_p<p>.txt` by default (relative to the working
  directory) and use only the first p layers, so a file of a larger depth can
  serve a smaller one.
- Every result file records the angles file's 64-bit FNV-1a hash
  (`angles_hash`) and every campaign manifest its sha256. A resumed job and
  `merge_results.py` refuse to mix angles. Do not edit these files.

## Results

`make table` prints Table I of the paper from the committed result files:

| p | girth g ≥ | c_VIRAL(p) | L (99.99999%) | jobs | samples | core-hours |
|---:|---:|:---:|:---:|---:|---:|---:|
| 2 | 6 | 0.755779(136) | 0.75507 | 4 | 10000000 | 0.0 |
| 4 | 10 | 0.827983(119) | 0.82736 | 4 | 10000000 | 0.0 |
| 6 | 14 | 0.856546(111) | 0.85596 | 4 | 10000000 | 0.0 |
| 8 | 18 | 0.873223(105) | 0.87267 | 4 | 10000000 | 0.0 |
| 10 | 22 | 0.884378(101) | 0.88385 | 4 | 10000000 | 0.1 |
| 12 | 26 | 0.891760(98) | 0.89124 | 4 | 10000000 | 0.3 |
| 14 | 30 | 0.896941(96) | 0.89644 | 4 | 10000000 | 1.0 |
| 16 | 34 | 0.900672(95) | 0.90017 | 4 | 10000000 | 3.8 |
| 18 | 38 | 0.903873(93) | 0.90338 | 7 | 10000000 | 14.7 |
| 20 | 42 | 0.906053(92) | 0.90557 | 25 | 10000000 | 59.2 |
| 22 | 46 | 0.907825(91) | 0.90734 | 100 | 10000000 | 239.3 |
| 24 | 50 | 0.909322(91) | 0.90884 | 140 | 10000000 | 977.4 |
| 26 | 54 | 0.910545(90) | 0.91007 | 325 | 10000000 | 3763.9 |
| 28 | 58 | 0.911469(90) | 0.91100 | 650 | 10000000 | 14467.6 |
| 30 | 62 | 0.912373(89) | 0.91190 | 1400 | 10000000 | 64166.8 |
| 32 | 66 | 0.913183(89) | 0.91271 | 2500 | 10000000 | 205045.8 |
| total | | | | 5179 | 160000000 | 288739.9 |

- **c_VIRAL(p)** is the fraction of samples whose root edge is cut, x = n_cut/n,
  with its standard error sqrt(x(1 − x)/n) in units of the last digit.
- **L** is the exact one-sided Clopper–Pearson lower confidence bound on
  c_VIRAL(p) at confidence 1 − 10^−7, rounded **down** to 5 decimals. Since
  M_g ≥ c_VIRAL(p) for every g ≥ 2p + 2 is a theorem, M_g ≥ L holds for those g
  at that confidence. (Rounding the normal approximation c − 5.2σ down instead
  gives the same numbers except at p = 16 and 24, where it is 10^−5 higher.)
- `results/merge_10M.txt` has the full merge for every depth, including the
  bounds at 95–99.99% confidence.

## Repository layout

```
README.md, Makefile, check.sh
viral_tree_hpc.c       reference solver; its header explains the problem, the algorithm,
                       why the tree value is a bound, the code and the file formats
viral_tree_fast.c      same samples with exact lookup tables: 4x fewer sincos calls
viral_tree_fast2.c     viral_tree_fast plus threads inside a sample and a smaller workspace
merge_results.py       merges result files: estimates, exact confidence bounds, --table
exact_small_p.py       exact enumeration of all seeds at p = 1, 2 (independent check; numpy)
params/                the trained angles (above)
results/               the raw results: 48 campaigns, 5,179 result files (results/README.md)
scripts/               campaign launchers and SLURM wrappers as used, run_array.slurm
scripts/validation/    the build, equivalence and seed-audit jobs run on the cluster
tests/                 equivalence suites, seed_audit.c, check_sources.sh
docs/                  solvers.md, campaigns.md, validation.md
```

## Quick start

You need a C compiler with OpenMP, `make` and Python 3 (standard library only;
`exact_small_p.py` also needs numpy).

```bash
make                     # builds viral_tree_hpc, viral_tree_fast, viral_tree_fast2
make check               # check.sh on each solver: exact p = 2 value, threads, resume (minutes)
mkdir -p run
OMP_NUM_THREADS=1 ./viral_tree_fast2 16 20000 0 params/angles_p16.txt run
python3 merge_results.py run
python3 exact_small_p.py params/angles_p2.txt 2      # exact c_VIRAL(2) = 12380/16384
```

On macOS, Apple clang needs an OpenMP runtime: `brew install libomp` (the
Makefile finds it), or `make LIBOMP=<prefix>` with any prefix that has
`include/omp.h` and `lib/libomp.dylib`, e.g. the one bundled with PyTorch
(`make LIBOMP=$HOME/miniforge3/lib/python3.12/site-packages/torch`).

The command line of all three solvers:

```
./viral_tree_fast2 <p> <n_samples> [job_id=0] [angles_file=params/angles_p<p>.txt] [out_dir=.]
```

A job writes `<out_dir>/viral_tree_p<p>_job<job_id>.txt`, rewritten at least
once a minute; rerunning the same command resumes it. Different job ids are
disjoint sets of samples whose results add; never merge the same (p, job_id)
twice. Threads: `OMP_NUM_THREADS` (inside each sample for `viral_tree_fast2`,
over samples for the other two). Memory, printed at start-up: about 5·2^p bytes
for `viral_tree_fast2` (20 GB at p = 32), 7.5·2^p per thread for
`viral_tree_fast`, 56.5·2^p per thread for `viral_tree_hpc`. Details:
[docs/solvers.md](docs/solvers.md).

## Reproducing the published numbers

### From the committed result files

```bash
make table                                     # Table I, all depths
python3 merge_results.py results/torch-p32-million-v1/jobs results/torch-p32-extension-v1/jobs \
    results/torch-p32-extension-v2/jobs -p 32  # one depth in full, as in results/merge_10M.txt
```

### Re-running the Monte Carlo

Every published sample is fixed by (p, job id, sample index), so rerunning a
job regenerates its result file. The job ids and samples per job of all 48
campaigns are in [docs/campaigns.md](docs/campaigns.md#per-campaign-layout);
the three campaigns of a depth start at job ids 0, 1000 and 2000. For example,
p = 24, extension-v2 (60 jobs of 100,000 samples, ids 2000–2059), on any SLURM
cluster:

```bash
mkdir -p out/p24-extension-v2
sbatch --array=0-59 --cpus-per-task=1 --mem=1G --time=12:00:00 \
    --output=out/p24-extension-v2/slurm_%A_%a.out \
    scripts/run_array.slurm ./viral_tree_fast2 24 100000 2000 out/p24-extension-v2
python3 merge_results.py out/p24-extension-v2 -p 24
```

Resubmit the same command for jobs stopped by the time limit until
`merge_results.py` reports `0 still running / incomplete`. Without SLURM, run
the solver once per job id, on as many machines as you like.

**Identical bytes.** The regenerated files equal the committed ones in every
line except `elapsed_s`, `core_seconds`, `sec_per_sample_core`, `host`,
`angles_file` (an absolute path) and, for `viral_tree_fast2` with a different
thread count, `threads`, provided

- the solver is built with the campaign flags (`make` uses them), and
- `viral_tree_fast` and `viral_tree_fast2` call the same `sincos` as the
  campaigns: glibc 2.39 (they spell out every fused multiply-add themselves);
- `viral_tree_hpc` is also compiled like the campaign build, gcc 14.3.1
  targeting x86-64-v3 (add `make CFLAGS_EXTRA=-march=x86-64-v3` where that is
  not the default), and runs one thread per job (`soft_sum` depends on the
  order of the additions).

On other platforms the last bits of a few samples can differ, which moves
n_cut by O(1) per 10^6 samples and leaves the estimates statistically
unchanged. To compare a regenerated file with the committed one:

```bash
skip='^(elapsed_s|core_seconds|sec_per_sample_core|host|angles_file) '
diff <(grep -vE "$skip" results/torch-p24-extension-v2/jobs/viral_tree_p24_job2000.txt) \
     <(grep -vE "$skip" out/p24-extension-v2/viral_tree_p24_job2000.txt)
```

The campaigns themselves were run with the launchers in `scripts/`, which
freeze a manifest, record every launch and verify completeness; see
[docs/campaigns.md](docs/campaigns.md#how-the-campaigns-were-run).

## Compute resources

All results were computed on the NYU HPC cluster **Torch** (SLURM) between
2026-09-19 and 2026-10-09:

- **Hardware:** mostly `cs` nodes, Intel Xeon Platinum 8592+ (128 cores and
  513 GB per node): 180 of the 221 nodes used. p = 30 million-v1 (60 GB jobs)
  ran on 7 `cl` nodes, and 104 p = 32 jobs on 34 GPU-partition nodes (CPU only);
  the CPU models of those nodes are not recorded.
- **Software:** RHEL 10 (Linux 6.12), glibc 2.39, gcc 14.3.1, libgomp; flags
  `-Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 -fopenmp` (+ `-ffp-contract=off`
  for the fast solvers).
- **Totals:** 5,179 jobs, 160,000,000 samples, **288,739.9 core-hours**
  (about 33 core-years); 71% of it for p = 32 and 22% for p = 30.

| p | jobs (1M + 3M + 6M) | threads per job | memory per job | core-hours | share |
|---:|---|---|---|---:|---:|
| 2–16 | 4 per depth (1 + 1 + 2) | 1 | ≤ 2 GB | 5.2 in all | 0.0% |
| 18 | 7 (4 + 1 + 2) | 1 | ≤ 2 GB | 14.7 | 0.0% |
| 20 | 25 (16 + 3 + 6) | 1 | ≤ 2 GB | 59.2 | 0.0% |
| 22 | 100 (64 + 12 + 24) | 1 | ≤ 2 GB | 239.3 | 0.1% |
| 24 | 140 (50 + 30 + 60) | 1 | ≤ 2 GB | 977.4 | 0.3% |
| 26 | 325 (100 + 75 + 150) | 1 | 5 GB / 1 GB | 3,763.9 | 1.3% |
| 28 | 650 (200 + 150 + 300) | 1 | 16 GB / 2 GB | 14,467.6 | 5.0% |
| 30 | 1,400 (500 + 300 + 600) | 1 / 4 | 60 GB / 6 GB | 64,166.8 | 22.2% |
| 32 | 2,500 (1,000 + 500 + 1,000) | 1 / 8 | 32 GB / 22 GB | 205,045.8 | 71.0% |
| **all** | **5,179** | | | **288,739.9** | 100% |

Threads and memory are given as million-v1 / extensions. Core-hours are the
solvers' own count, threads × wall time of the sampling loop summed over saved
run segments (from the result files); they exclude start-up, work lost when a
job was killed between checkpoints, queue time, and the build, validation and
probe jobs. Per campaign (SLURM arrays, nodes, dates, wall times, requested
resources): [docs/campaigns.md](docs/campaigns.md).

## Validation

- An exact enumeration of all 2^14 seed configurations at p = 2 gives
  c_VIRAL(2) = 12380/16384 = 0.7556152; the 10^7-sample estimate is 1.2
  standard errors away.
- The three solvers produce the same samples bit for bit, checked on the
  cluster with whole per-sample dumps for p = 1..26 and at p = 32, at 1 to 32
  threads, and by a self-check against the reference kernel at the start of
  every one of the 4,238 fast-solver launches.
- The 160M samples are distinct: all initial generator states were computed
  and compared.
- Every job of every campaign is complete, the merge reproduces
  `results/merge_10M.txt` and Table I, and the three independent campaigns of
  each depth are statistically consistent (χ² = 45.7 on 32 degrees of freedom
  over all depths, p-value 0.05).

Details, job ids and the checks you can run yourself:
[docs/validation.md](docs/validation.md).
