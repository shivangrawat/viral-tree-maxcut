# The three solvers

Three C programs compute the same thing: Monte-Carlo samples of VIRAL on the
depth-p tree, written to the same result files. They differ only in how fast
and in how much memory. For the same (p, job id, sample index) they produce the
same sample to the last bit ([validation](validation.md)), so their result
files can be merged.

The problem, the algorithm, the proof that the tree value is a bound, the tree
layout and the output format are explained in the header comment of
[`viral_tree_hpc.c`](../viral_tree_hpc.c) (sections 1–7). This page covers what
the three programs share, how the two fast kernels work, why they are exact,
and where each function is.

## At a glance

| | `viral_tree_hpc.c` | `viral_tree_fast.c` | `viral_tree_fast2.c` |
|---|---|---|---|
| role | reference, the most readable | 4× fewer `sincos` calls | `fast` + threads inside one sample + smaller workspace |
| used for | million-v1, p = 2..30 | million-v1, p = 32 | extension-v1 and -v2, every p |
| kernels | reference | reference (p < 4), tables (p ≥ 4) | reference (p < 4), tables (p = 4, 5), fused (p ≥ 6) |
| `OMP_NUM_THREADS` | threads over samples, one workspace each | threads over samples, one workspace each | threads inside each sample, one shared workspace (fused kernel only) |
| memory | 56.5·2^p bytes per thread | ≈ 7.5·2^p per thread | ≈ 5·2^p in total |
| `sincos` calls per sample | 4·2^p − 2p − 4 | 2^p − 2p | 2^p − 2p |
| p = 28, one thread (core-s per sample) | 17.6 | 4.8 | 4.1 |
| p = 30 | 59–73, 56.5 GB | 19.1, 7.5 GB | 17.5 at 4 threads (4.4 s per sample), 5 GB |
| p = 32 | (226 GB per thread) | 72, 30 GB | 74 at 8 threads (9.2 s per sample), 20 GB |

Timings: Xeon Platinum 8592+ (`cs` nodes) except `viral_tree_hpc` at p = 30,
which ran at 72 core-s per sample on the `cl` nodes (59 on a `cs` node); from
the R2 benchmark and the campaign job files.

## What all three share

**Command line.**

```
<solver> <p> <n_samples> [job_id=0] [angles_file=params/angles_p<p>.txt] [out_dir=.]
```

p is 1..40, n_samples 1..2^32−1, job_id 0..2^31−1; out_dir must exist. Exit
status 0 when the job is complete, was already complete, or was stopped by a
signal with its state saved; 1 on any error.

**Samples.** Sample k of job j at depth p has the global index
S = j·2^32 + k. Its generator is xoshiro256** seeded with
`mix64(0x5DEECE66D ^ (p << 32) ^ mix64(S))` (`sample_rng`). It draws
⌈(4·2^p − 2)/64⌉ words, one seed bit per vertex in flat-index order (level by
level, section 4.1 of the hpc header). Different job ids never share a sample,
and the same (p, j) always gives the same samples, whichever solver, thread
count or restart.

**Angles file.** `params/angles_p<p>.txt` (see the [README](../README.md#the-parameters-params)).
Only the first p layers are read. Its 64-bit FNV-1a hash is stored in the
result file and checked on resume.

**Result file** `<out_dir>/viral_tree_p<p>_job<job>.txt`: one `key value` per
line (hpc header, section 6). It is written atomically (temporary file, fsync,
rename) at start, at least once a minute (`CKPT_SEC`), at the end and on a
signal. If it exists at start the run resumes after checking p, job and the
angles hash; a larger n_samples extends a finished job, a smaller one is
refused. All three write the same first line (`# viral_tree_hpc result/checkpoint
file`), so their files are interchangeable.

**Signals.** SIGTERM, SIGINT and SIGUSR1 finish the current batch, save the
state with `status running`, and exit 0. Rerunning the same command resumes.

**Environment variables.**

| variable | solvers | effect |
|---|---|---|
| `OMP_NUM_THREADS` | all | threads: over samples (hpc, fast) or inside each sample (fast2, fused kernel only) |
| `VIRAL_TREE_DUMP=<file>` | all | append `k cut s_u^z s_v^z` (hex floats) per sample to `<file>`; forces one thread in hpc and fast. Used by the tests. |
| `VIRAL_SELFCHECK=<n>` | fast, fast2 | samples in the start-up self-check (default 64, 0 = off) |
| `VIRAL_FORCE_REF=1` | fast, fast2 | use the reference kernel at every p |
| `VIRAL_KERNEL=ref\|tables\|fused` | fast2 | force a kernel (where it applies) |
| `VIRAL_PAR_MIN=<n>` | fast2 | levels with fewer vertices run on one thread (default 2048) |
| `SLURM_MEM_PER_NODE` | all | only a warning if the workspace exceeds the allocation |

**Compile-time constants** (`-D…`): `CKPT_SEC` (60 s), `REPORT_SEC` (60 s),
`BATCH_TARGET_SEC` (20 s) and `BATCH_MAX_PER_THREAD` (64). Batches only affect
the order in which `soft_sum` is added up, never n_cut; the campaign builds use
`-DBATCH_MAX_PER_THREAD=1` and one thread per job (hpc, fast), which fixes the
order, so result files reproduce byte for byte.

## The reference kernel

`one_sample` in `viral_tree_hpc.c` (`one_sample_ref` in the other two):

- **Light cone.** At layer n only levels 0..p−n−1 can still influence the roots
  at time p, so only those are updated: 4·2^p − 2p − 4 vertex updates per sample
  instead of p·4·2^p. The leaves (half of all vertices) are never updated;
  their seed bits are read once.
- **In place, top down.** Level d is updated after level d−1 and before level
  d+1, so its children are still old; its own old z is saved into a scratch
  array just before it is overwritten, for its children's update. Two scratch
  arrays alternate.
- **Memory:** 24·(2^(p+1) − 2) bytes of spins (levels 0..p−1) + 8·⌈(4·2^p −
  2)/64⌉ of seed bits + 2 × 8·2^(p−1) of scratch = 56.5·2^p bytes (minus 48).
- **Cost:** one `sincos` per vertex update dominates (13–19 ns per update from
  p = 16 in cache to p = 30 in RAM on the `cs` nodes), so half the work is
  layer 0 and a quarter is layer 1.

## The table kernel (`viral_tree_fast.c`, `viral_tree_fast2.c`)

Layers 0 and 1 have very few distinct inputs.

- **T1: layer 0 in 8 values.** At time 0 every z is +s or −s (s = sin α). The
  layer-0 field (a + b) + c of three such values depends only on how many are
  +s: 2s, s and −s are exact, and only ±3s rounds, the same way in any order.
  So a vertex's time-1 spin is `T1[own seed bit][number of + neighbours]`.
  `build_tables` checks this order-independence bit for bit for the actual α
  at start-up and refuses to run otherwise.
- **T2: layer 1 in 1024 values.** A vertex's time-2 spin depends only on the 10
  seed bits of its radius-2 ball (self, parent, grandparent, sibling, two
  children, four grandchildren): `T2[1024]`, 24 KB. For the roots the other
  root plays the parent and its children the grandparent and sibling; for
  level 1 the other root plays the grandparent. `t2_idx` gathers the 10 bits;
  `t2_child_idx` derives a child's index from its parent's.
- **No pass before time 3.** No initialisation and no layer-0 or layer-1 sweep:
  layer 2 is computed straight from lookups for levels 0..p−4,
  S3(x) = U2(T2[x], (T2[parent].z + T2[child1].z) + T2[child2].z).
- **Level p−3 is never stored.** Its time-3 z is needed only by level p−4 at
  layer 3, so it is computed there, with the same one `sincos` per vertex the
  reference spends on it at layer 2 (`lazy_z3` for p = 4).
- **Layers 3..p−1** are the reference update restricted to levels 0..p−4.
- **Memory:** 24·(2^(p−2) − 2) + 8·(⌈(4·2^p − 2)/64⌉ + 1) + 2 × 8·2^(p−4) ≈
  7.5·2^p bytes (30 GB at p = 32 instead of 226 GB).

Measured (R2, one `cs` core, identical results at every depth):

| p | reference s/sample | table kernel s/sample | speedup | reference peak RSS | table kernel peak RSS |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.0633 | 0.0199 | 3.2× | 0.057 GiB | 0.022 GiB |
| 22 | 0.2747 | 0.0764 | 3.6× | 0.223 GiB | 0.031 GiB |
| 24 | 1.072 | 0.298 | 3.6× | 0.885 GiB | 0.119 GiB |
| 26 | 4.446 | 1.210 | 3.7× | 3.533 GiB | 0.471 GiB |
| 28 | 17.60 | 4.817 | 3.7× | 14.13 GiB | 1.877 GiB |
| 30 | (59, probe) | 19.07 | ≈ 3.1× | (56.5 GiB) | 7.502 GiB |

The speedup is below 4× because the bit gathering and lookups take back part
of the saving.

## The fused kernel and threads (`viral_tree_fast2.c`)

- **One more level on the fly.** Let L = p−4 and M = p−5. Level L's time-3
  spin is a function of seed bits only, and it is used only at layer 3. The
  fused kernel stores spins of levels 0..M only. At layer 3 it processes each
  vertex y of level M together with its two children x (level L) and their
  children (level L+1, time-3 z only), and keeps of level L just its time-4 z,
  which level M reads at layer 4. Every vertex is still updated exactly once,
  with the reference's operations. Memory ≈ 5·2^p bytes (20 GB at p = 32).
- **Threads inside one sample.** Within one level every vertex reads only the
  previous level's saved z, its own old state, its not-yet-updated children,
  the tables and the seed bits, and writes only its own slots. So each level's
  loop is split statically over the threads (`omp parallel for`, with the
  loop's barrier between levels); levels below `VIRAL_PAR_MIN` vertices run on
  one thread, and the seed bits are drawn by one thread. The arithmetic of
  each vertex does not depend on the thread, so the results do not depend on
  the number of threads (checked at 1..32 threads).
- **Samples one after another**, in one workspace shared by all threads; the
  `threads` field of the result file is the number of threads per sample. The
  reference and table kernels in this file are serial (`nt = 1`).

## Exactness

The fast kernels reproduce the reference bit for bit only if every operation
is literally the same. Four details make it so:

1. **`sincos()`.** The campaign build of `viral_tree_hpc.c` calls glibc's
   `sincos`: gcc fuses the `cos`/`sin` pair of `rz` into one call, and the
   binary imports only `sincos`. glibc's `cos()` and `sincos()` can differ in
   the last bit, so the fast solvers call `sincos` explicitly (on glibc;
   elsewhere all three use `sin`/`cos`).
2. **The same fused multiply-adds.** RHEL 10's gcc targets x86-64-v3 (FMA,
   AVX2) by default, so the campaign build of `viral_tree_hpc.c` fuses the
   multiply-adds of `rz` and `rx_cs`. Its disassembly (read with
   `scripts/fast_asm.slurm`, in tag `results-10M`) shows one pattern at every
   site: `a·b ± c·d` is computed as `fma(a, b, ±round(c·d))`. The fast solvers
   write exactly that with explicit `fma()` calls and are compiled with
   `-ffp-contract=off`, so these are the only fused operations:

   ```
   rz:     x' = fma(c,  x,  -(sn*y))     y' = fma(sn, x,  c*y)
   rx_cs:  y" = fma(cb, y', -(sb*z))     z" = fma(sb, y', cb*z)
   ```

   Two earlier attempts failed and show why: without any fusion the last bits
   changed already at p = 1; with the compiler's own fusion they changed at
   p = 4, because gcc fuses differently at different inlined sites (the
   start-up self-check caught a 1-ulp difference).
3. **The same association order** for every field: (parent + child1) + child2,
   and for the roots (other root + child1) + child2.
4. **The same random bits**, laid out one per vertex in the same order.

Safety nets in the binaries: `build_tables` verifies the T1 symmetry for the
given α; at start-up the fast solvers run `VIRAL_SELFCHECK` samples (default
64) at depth min(p, 12) through their kernel and the reference kernel and stop
on any differing bit; the reference kernel is kept for p < 4 and
`VIRAL_FORCE_REF=1`.

What this means for reproduction: the fast solvers depend only on the libm
(`sincos`, `fma`); `viral_tree_hpc.c` also depends on how the compiler fuses
multiply-adds, i.e. on the compiler and the target ISA. With gcc 14.3.1, glibc
2.39 and x86-64-v3 all three reproduce the campaign bytes. On Apple clang /
arm64 all three agree with each other in `check.sh` (p = 12: n_cut = 178239,
the cluster's value).

## Function reference

`x` marks the files that define the function. The shared functions are
identical in all files where they appear, except that the fast solvers' `rz`
and `rx_cs` spell out the fused multiply-adds.

| function | hpc | fast | fast2 | what it does |
|---|:-:|:-:|:-:|---|
| `rz`, `rx_cs` | x | x | x | rotations about z (by γ·h) and x (by β, cos/sin precomputed) |
| `rotl`, `rng_next` | x | x | x | xoshiro256** generator |
| `mix64`, `rng_seed` | x | x | x | splitmix64 hash; generator state from a 64-bit seed |
| `sample_rng` | x | x | x | generator of sample S = job·2^32 + k at depth p |
| `build_geometry` | x | x | x | level offsets and sizes of the depth-p tree |
| `bytes_per_thread` | x | x | | workspace size (per thread) |
| `fast_nst`, `fast_zs` | | x | | table-kernel array sizes |
| `kernel_nst`, `kernel_nzs`, `kernel_nzl`, `kernel_nw` | | | x | array sizes of each kernel |
| `bytes_per_sample` | | | x | workspace size (shared) |
| `load_angles`, `file_hash` | x | x | x | read the angles file; its FNV-1a hash |
| `alloc_work`, `free_work`, `touch_work` | | x | x | workspace allocation, release, first touch |
| `touch` | | | x | parallel first touch of one array |
| `seed_z` | x | x | x | time-0 z of a vertex from its seed bit |
| `one_sample` / `one_sample_ref` | x | x | x | the reference kernel: one sample |
| `build_tables` | | x | x | the exact tables T1 and T2, with the symmetry check |
| `bit1`, `bitsn` | | x | x | read 1 or up to 32 seed bits |
| `t2_idx`, `t2_child_idx` | | x | x | T2 index of a vertex; of a child from its parent's |
| `lazy_z3` | | x | x | time-3 z of an unstored vertex (p = 4) |
| `one_sample_fast` | | x | x | the table kernel: one sample |
| `one_sample_fused` | | | x | the fused, threaded kernel: one sample |
| `run_kernel` | | | x | dispatch one sample to the chosen kernel |
| `selfcheck` | | x | x | start-up self-check against the reference kernel |
| `state_path`, `write_state`, `read_state` | x | x | x | result file: path, atomic write, validated read |
| `on_signal`, `install_signals` | x | x | x | clean stop on SIGTERM / SIGINT / SIGUSR1 |
| `usage`, `main` | x | x | x | command line; one job from start or resume to the summary |

## Sources and provenance

The campaigns ran the sources of tag `results-10M` (commit `d578d35`). Since
then only their comments have changed; `make check-sources`
([`tests/check_sources.sh`](../tests/check_sources.sh)) proves it: with the
comments removed every file is identical, the assembly is identical with the
campaign flags, and `scripts/viral_campaign.py` maps each documented file's
sha256 to its campaign sha256.

| file | sha256 at `results-10M` (in the manifests) | sha256 now |
|---|---|---|
| `viral_tree_hpc.c` | `b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51` | `b18dab868846b23b43662aed1b2c462ac20557e9aa95fb3168652f31494280f2` |
| `viral_tree_fast.c` | `f1efdb7dd443bef0a043791a5232678c9659bb2c3be6fab6c959698541b034f8` | `178539f113c5a0fecc59c8711732ed45385097028f0ae3bdbca27809a6d5eed8` |
| `viral_tree_fast2.c` | `8b12711a2b6f15e652a7677c43b80aa6715570ffd28b38d9ec4afdab97251988` | `7abb4e08eae5413fb524556ef2ee9a182aaa842677520e71c50e36020f07d803` |
