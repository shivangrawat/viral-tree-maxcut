# Plan: a faster, leaner VIRAL solver with bit-identical results

Status (2026-09-25): Ideas A and B are implemented in `viral_tree_fast.c` and
validated. This lives on branch `faster-wt`; the frozen `viral_tree_hpc.c` and
the running p=30 campaign are not touched.

- **R1 (job 18517761):** all 235 cases for p = 1..26 are byte-identical to the
  frozen campaign binary: whole per-sample dumps and result files, stop/resume,
  and `check.sh`.
- **R2 (job 18517764, Xeon Platinum 8592+):**

| p | reference s/sample | fast s/sample | speedup | reference peak RSS | fast peak RSS |
|---|---|---|---|---|---|
| 20 | 0.0633 | 0.0199 | 3.18× | 0.057 GiB | 0.022 GiB |
| 22 | 0.2747 | 0.0764 | 3.60× | 0.223 GiB | 0.031 GiB |
| 24 | 1.072 | 0.298 | 3.60× | 0.885 GiB | 0.119 GiB |
| 26 | 4.446 | 1.210 | 3.67× | 3.533 GiB | 0.471 GiB |
| 28 | 17.60 | 4.817 | 3.65× | 14.13 GiB | 1.877 GiB |
| 30 | — | 19.07 | — | (56.5 GiB) | 7.502 GiB |

Results were identical at every depth, and peak RSS is within 2 MiB of the
formulas.

Two lessons on exactness, both now handled in the code:

- **The frozen build uses FMA.** RHEL 10's gcc targets x86-64-v3, so the frozen
  binary FMA-fuses its rotations. That rules out a blanket
  `-ffp-contract=off` over otherwise unchanged code.
- **Automatic fusion isn't reliable.** gcc fuses differently at different
  inlined sites. The frozen binary's fusion pattern, read from its disassembly,
  is therefore written out with explicit `fma()`, and the file is built with
  `-ffp-contract=off`.

## 1. What the code does now

For each sample, `one_sample` (`viral_tree_hpc.c:519-603`) does four things:

1. **Seeds.** It draws one random bit for each of the 4·2^p − 2 vertices,
   64 per word (xoshiro256**, seeded from the global sample index). [527]
2. **Initial spins.** It writes (cos α, 0, ±sin α) into every stored
   (non-leaf) vertex: 2·2^p − 2 of them. [530-531]
3. **p layers.** Layer n updates levels 0..p−n−1, top-down and in place
   (the light cone). A vertex update computes
   h = z(parent) + z(child1) + z(child2) from the old values, makes one
   `sincos(γ_n·h)` call, and applies a z-rotation followed by an x-rotation by
   the fixed β_n. [535-595]
4. **Result.** The edge is cut if sign(z_u) ≠ sign(z_v), and soft = z_u·z_v.
   [598-602]

**Cost.** A sample needs 4·2^p − 2p − 4 vertex updates, each with one libm
`sincos` call. The frozen binary imports only `sincos` and `sqrt` from libm, so
GCC fused the `cos`/`sin` pair in `rz` into a single call. Layer n does
2^(p−n+1) − 2 updates, so **layer 0 is 50% of all updates and layer 1 is 25%**.

**Memory.** Each process needs 56.5·2^p bytes:

- 48·2^p for 24-byte spins of all non-leaf vertices;
- 8·2^p for two z-scratch arrays of 2^(p−1) doubles each;
- 0.5·2^p for the seed bits.

**Measured, from the committed campaigns.** The cost is roughly 13–19 ns per
vertex update at every depth: p=16 (0.0033 s/sample, fits in cache), p=26
(3.9 s), p=28 (14.5 s), and p=30 (59 s on a `cs` probe, ~68 s on `cl`, with
56 GiB in DRAM). There is no upward trend, so time is dominated by the
~4·2^p `sincos` calls, not by memory traffic. That is why time grows ~4× per
+2 in p: that's how fast the vertex count grows.

## 2. Constraints

- **Byte-identical output.** Every per-sample dump line (the hex floats of z_u
  and z_v) and every byte of the result file must match, except the four
  timing fields. This is the same contract as `CODEX_HPC_HANDOFF.md` §1.
- **Double precision throughout.** No fast-math and no approximate math. The
  rotation arithmetic must be FMA-contracted exactly as in the frozen binary.
  RHEL 10's gcc targets x86-64-v3 by default, so the frozen `-O3` build fuses
  multiply-adds, and the new build must use the same compiler and flags.
  (Learned from the first R1: adding `-ffp-contract=off` alone changed the
  last bits, even at p=1.)
- **New file, same interface.** The new solver is a new file,
  `viral_tree_fast.c`. It keeps the same command line, output format,
  checkpoint/resume, signals and dump hook, so its result files are
  interchangeable with the frozen binary's.
- **Contract change to approve.** The handoff's build-tuning rules banned
  "source edits, lookup tables, or shared workspaces". This plan uses a new
  source file with *exact* lookup tables. Each entry is computed by the same
  arithmetic on the same inputs, not approximated. Later, it shares one
  workspace between threads. Acceptance stays the same: byte-identity against
  the frozen reference built with the same compiler on the same node.

## 3. Idea A — exact tables for layers 0 and 1 (removes 75% of `sincos` calls)

**Layer 0 has only 4 distinct `sincos` arguments.** At time 0 every z is
±s, with s = sin α. For a, b, c ∈ {s, −s}, fl(fl(a+b)+c) is one of fl(3s), s,
−s or −fl(3s), and which one depends only on how many of a, b, c are +s:

- 2s is exact, and so are 2s − s and 0 + s;
- only the all-equal case rounds, and it rounds the same way in any order.

So a vertex's time-1 spin is one of 8 values, `T1[own bit][# of + neighbours]`.
Today layer 0 spends 2^(p+1) `sincos` calls computing 4 distinct values.

**Layer 1 needs a 1,024-entry table.** A vertex's time-2 spin depends on its
own time-1 spin and the time-1 z of its parent and two children. Those in turn
depend on the 10 seed bits of the vertex's radius-2 ball: self, parent,
grandparent, sibling, 2 children, 4 grandchildren.

- `T2[1024]` (24 KB, L1/L2-resident) is built once per run.
- It uses the same `rz`/`rx_cs` arithmetic, an explicit `sincos` call, and the
  same association order, (parent + child1) + child2.
- The two roots and level 1 use the same index, with the other root standing in
  for the missing parent or grandparent.

Each entry comes from exactly the operations the old code performs on exactly
the same inputs, so the results are identical by construction. The tests
confirm that the implementation is right. Start-up also checks the T1 symmetry
claim exhaustively.

## 4. Idea B — never store time-0/1/2 states (memory 56.5 → 7.5·2^p)

With T2, any time-2 spin is a lookup from the seed bits. That allows four
changes:

- **No initialisation pass and no layer-0/1 sweeps.** Today that is ~48·2^p
  bytes written, then two full sweeps over the tree.
- **Layer 2 comes straight from lookups.** For levels 0..p−4:
  S3(x) = U2(T2[x], (T2[parent].z + T2[child1].z) + T2[child2].z).
  That's 4 lookups and 1 `sincos` per vertex, written once.
- **Level p−3 is never stored.** Its time-3 z is read only by level p−4 at
  layer 3, so it is computed there on the fly. It still costs one `sincos` per
  vertex, just at a different point.
- **Layers 3..p−1 run exactly as today** (in place, top-down, with z-scratch),
  on levels 0..p−4.

Memory per process:

| component | now | new |
|---|---|---|
| spins | 48·2^p (levels 0..p−1) | 6·2^p (levels 0..p−4) |
| z-scratch | 8·2^p | 1·2^p (0.5 with one buffer iterated backwards) |
| seed bits | 0.5·2^p | 0.5·2^p |
| **total** | **56.5·2^p** | **7.5·2^p** (7.0) |

| p | now | new |
|---|---|---|
| 30 | 56.5 GiB | 7.5 GiB |
| 32 | 226 GiB | 30 GiB (under `cs`'s 32G limit) |
| 34 | 904 GiB | 120 GiB |

**Work.** A sample needs 2^p − 2p `sincos` calls instead of 4·2^p − 2p − 4,
about 4× fewer. The added work is ~2·2^p table lookups, each a 10-bit gather
from the seed bits plus an L1 load.

**Expected speed.** About 3–4× per sample, because `sincos` dominates and the
lookups and memory traffic give back part of the saving. Run R2 measures it.

**Validity.** This works for p ≥ 4. For p ≤ 3 the original kernel is used; it
stays in the file, and the self-check uses it too.

## 5. Idea C — threads inside one sample (the parked idea)

Within a level pass, vertex k reads only its own old state, its parent's saved
z and its children, and it writes only its own slot. So T threads can split
each level, with a barrier between levels, and the result is race-free and
bit-identical for every T.

After Idea B, campaigns are limited by memory, not cores. At 32G per job the
6000G cap allows ~187 jobs, which is 187 of the 3,000 cores the QoS permits.
Threads use the spare cores without more memory. The expected speedup is about
0.8·T up to 4–8 threads; run R4 measures it.

This needs one follow-up: the `threads` result field would record T. The byte
comparison and `viral_campaign.py`'s `threads == 1` check need a documented
exception for new campaigns.

## 6. Smaller bit-identical improvements

- **SoA spin layout** (separate x/y/z arrays): reading a child's z touches 8 B
  instead of 24 B, and the arithmetic around `sincos` becomes vectorisable.
  Any vectorised version must keep the frozen binary's FMA pattern; writing
  it with explicit `fma()` calls would pin that pattern down.
- **`madvise(MADV_HUGEPAGE)`** on the big arrays, to cut TLB misses.
- **One z-scratch buffer** iterated backwards saves 0.5·2^p.
- **"Fused bottom kernel"** (optional): process level p−5 together with its
  children and grandchildren at layer 3. That brings memory to 5·2^p (p=32:
  20 GiB, p=34: 80 GiB) with no extra `sincos`, at the cost of more code.

## 7. Idea D (optional, riskiest) — vectorised `sincos`, bit-exact

After A+B, the ~2^p remaining `sincos` calls are most of the run time. The only
bit-identical way to speed them up is a SIMD port of glibc 2.39's own
`__sincos`. It would have to mirror, instruction for instruction, the variant
that the compute nodes' ifunc selects (probably the FMA build).

- **Gain:** maybe ~2× more, for high effort.
- **Validation:** compare against libm on ≥10^10 arguments, plus every argument
  that occurs in the p ≤ 24 tests. Drop the idea at the first mismatch.
- **Side benefit:** results would no longer depend on each node's libm version.

## 8. Considered and rejected

- **fast-math, or an approximate/vectorised libm** (libmvec, SLEEF): these
  change the last bits.
- **Correctly-rounded sin/cos** (CORE-MATH): glibc ≥2.28 is not always correctly
  rounded, so there would be rare mismatches.
- **Dropping one spin component** because |s| = 1: rebuilding it isn't
  bit-exact.
- **A radius-3 table** (4M entries, 96 MB): random DRAM lookups cost about as
  much as the `sincos` calls they would replace. Could be measured later.
- **Re-partitioning samples or changing seeds:** this changes the sample set.

## 9. What it could mean for campaigns (to be confirmed by R2)

This assumes a 3.5× speedup per sample on one core.

| | p=30 now | p=30 new | p=32 now (est.) | p=32 new |
|---|---|---|---|---|
| memory per job | 60G (`cl` only) | 8G | 240G (`cl` only) | 32G (`cs` OK) |
| jobs at once (6000G cap) | 100 | 750 | 25 | 187 |
| s/sample/core (`cl`) | ~68 | ~20 | ~275 | ~80 |
| 1M samples, steady state | ~8 days | < 1 day* | ~130 days | ~5 days* |

\*If the partitions actually give that many slots. Four threads per job
(Idea C) would give roughly another 3×.

The samples are byte-identical, so new jobs are exactly the samples the frozen
binary would produce. That means a p=30 extension (job ids 500+) merges with
the frozen campaign's results. The frozen campaign itself keeps its pinned
binary.

## 10. Verification (bit-identity)

**1. New equivalence suite.** `tests/test_fast_equivalence.py` reuses the
existing harness functions. It compiles the frozen reference (sha-guarded) and
the new solver with the same gcc and flags on the same node, then compares
entire dumps and result bytes for:

- every p from 1 to 24 (odd p use truncated angle files, like the existing p=1
  fixture), which covers both kernels and every small-p edge case;
- job ids 0, 7, 499 and 2147483647, with n = 1, 65 and 513 (plus 4096 at
  p ≤ 12);
- `angles_p30.txt` at small p, as the current harness does;
- repeat, extension, and SIGTERM/SIGKILL resume;
- identical bytes for T = 1, 2, 4, 8 threads (Idea C);
- the unchanged `check.sh`.

**2. Built-in self-check** (`VIRAL_SELFCHECK=N`). At start-up the solver runs N
samples at depth min(p, 12), using the first layers of the current angles,
through both kernels. It aborts on any bit difference. This guards every node,
compiler and libm the binary lands on.

**3. Replays of committed campaign jobs.** Rerun jobs with the new binary and
compare every scientific byte with the files in
`results/torch-p*-million-v1/jobs`, using the same absolute angles path:

- p=16 job 0 (1M samples);
- p=18, p=20 and p=22 job 0;
- p=24 job 0 (20k samples), p=26 job 0 (10k), p=28 job 0 (5k);
- finally p=30 job 1: the full committed job of 2,000 samples.

Identical files mean the results are identical to the last digit at the depth
the campaigns actually run.

## 11. Quick runs (R0–R2 submitted 2026-09-24)

- **Compiling:** everything compiles on the compute node, because the login
  node has no C toolchain.
- **QoS:** every run requests QoS `cpu168`, which has its own caps (1000 CPUs,
  2000G per user), so none of them waits on the cpu48 memory that p=30 fills.
  The scheduler routes them to the `cpu_short` partition. The `cpu_short` QoS,
  `interactive` and the `cpu_prem` partition are all rejected for this account.
- **Outputs:** everything goes to `$SCRATCH/viral-fast/`: logs in `logs/`, JSON
  reports `r1_<job>.json` and `r2_<job>.json`, and a passing fast build in
  `builds-<job>/`.
- **Scripts:** `scripts/fast_facts.slurm` (R0) and
  `scripts/fast_validate.slurm suite|bench` (R1/R2).

| run | what | resources | pass criterion |
|---|---|---|---|
| R0 (18466052) | node facts: CPU and vector flags, libm sha256, the frozen binary's libm calls, glibc `sincos` cost for \|x\| ≤ 0.8 / 2.4 / 4.5 | 1 CPU, 4G, 30 min | informs Idea D and the speed estimate |
| R1 (18466053) | equivalence suite (§10.1) for p = 1..26 against the frozen campaign binary | 4 CPUs (for `check.sh`), 8G, 2 h | all bytes identical |
| R2 (18466054) | reference vs fast at p=20, 22, 24, 26, 28 (3 interleaved repetitions, results compared too), plus fast alone at p=30; peak RSS from `wait4` | 1 CPU, 16G, 2 h | measured time ratio; RSS ≈ 7.5·2^p |
| R3 | replays (§10.3), as an array | 1 CPU each, 2–16G, up to ~12 h, cpu168 | identical to the committed results |
| R4 | (after Idea C) 1/2/4/8 threads at p=26 and p=28 | 8 CPUs, 16G, ~2 h | identical bytes; speedup curve |

## 12. Implementation order

1. **`viral_tree_fast.c`, in two stages.** First Idea A alone, still storing
   time-2 states. That's the smallest possible change: 18.5·2^p memory and 4×
   fewer `sincos` calls. Then Idea B (7.5·2^p). Include the self-check, and
   keep the original kernel for p ≤ 3.
2. **Test and job scripts:** `tests/test_fast_equivalence.py`, plus
   `scripts/fast_validate.slurm` (R0–R2) and `scripts/fast_replay.slurm` (R3).
3. **Threads.** Once R1–R3 pass, add Idea C and run R4.
4. **Campaign tooling.** Let `viral_campaign.py` accept the validated fast
   source hash and a thread count in the manifest, without loosening its checks
   for the frozen p=30 campaign.
5. **Optional extras:** the fused bottom kernel, SoA and huge pages, and then
   Idea D.
