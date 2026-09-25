# What makes `viral_tree_fast.c` faster and smaller

`viral_tree_fast.c` computes exactly what `viral_tree_hpc.c` computes, to the
last bit. It has the same command line, result file, checkpoint/resume,
signals and `VIRAL_TREE_DUMP` hook. It does about 4× fewer `sincos()` calls and
needs about 1/7.5 of the memory. The frozen `viral_tree_hpc.c` is unchanged.

Measured on a Torch `cs` node (Xeon Platinum 8592+), one core. "Frozen" is the
campaign binary `viral_baseline`, sha `0f3f8d44…`.

| p | frozen s/sample | fast s/sample | speedup | frozen memory | fast memory | results |
|---|---|---|---|---|---|---|
| 20 | 0.063 | 0.020 | 3.2× | 0.06 GiB | 0.02 GiB | identical |
| 22 | 0.275 | 0.076 | 3.6× | 0.22 GiB | 0.03 GiB | identical |
| 24 | 1.07 | 0.30 | 3.6× | 0.89 GiB | 0.12 GiB | identical |
| 26 | 4.45 | 1.21 | 3.7× | 3.53 GiB | 0.47 GiB | identical |
| 28 | 17.6 | 4.82 | 3.7× | 14.1 GiB | 1.88 GiB | identical |
| 30 | ~59 (probe) | 19.1 | ~3.1× | 56.5 GiB | 7.50 GiB | — |

"Identical" means byte-for-byte equal per-sample dumps (hex floats of z_u and
z_v) and result files. Only the four timing fields are excluded, as in
`CODEX_HPC_HANDOFF.md` §1. The equivalence suite passed all 235 cases for
p = 1..26.

---

## 1. Where the original code spends its time and memory

For each sample, `one_sample` (`viral_tree_hpc.c:519`) does four things:

1. **Seeds.** It draws one random bit per vertex for all 4·2^p − 2 vertices.
2. **Initial spins.** It writes (cos α, 0, ±sin α) into every non-leaf vertex:
   2·2^p − 2 spins of 24 bytes each.
3. **p layers.** Layer n updates levels 0..p−n−1 in place. Each vertex update
   computes h = z(parent) + z(child 1) + z(child 2), makes one `sincos(γ_n·h)`
   call, and applies two small rotations.
4. **Result.** It compares the signs of the two root spins.

That is 4·2^p − 2p − 4 `sincos` calls per sample. The measured cost is 13–19 ns
per vertex update at every depth from p=16 (in cache) to p=30 (56 GiB in RAM),
and one glibc `sincos` alone costs 8–23 ns on these nodes. So the run time is
the `sincos` calls, not memory traffic. Layer n does 2^(p−n+1) − 2 updates,
which puts **half of all the work in layer 0 and a quarter in layer 1.**

Memory is 56.5·2^p bytes per process:

- 48·2^p for the spins of every non-leaf vertex;
- 8·2^p for two z-scratch arrays of 2^(p−1) doubles each;
- 0.5·2^p for the seed bits.

---

## 2. Change 1 — layer 0 becomes an 8-entry table (T1)

**Observation.** At time 0 every z is either +s or −s, with s = sin α. The
layer-0 field of a vertex is h = (a + b) + c, with a, b, c its three neighbours'
z values, all in {+s, −s}. In floating point this sum depends only on **how
many** of a, b, c are +s, not on their order:

- 2s is exact, and so are 2s − s = s and 0 + s = s;
- only the all-equal case, 3s, rounds, and it rounds the same way in any order.

So h takes just 4 values, and a vertex's time-1 spin depends only on its own
seed bit and its number of + neighbours (0..3): 8 possible spins in total. The
original code spends 2^(p+1) `sincos` calls on layer 0, yet it only ever
computes 4 distinct arguments.

**Routine.** `build_tables` (`viral_tree_fast.c:414`) does three things:

- computes the 4 field values `h0[m]` for m = 0..3 + neighbours;
- **checks the order-independence claim bit for bit** for the actual sin α,
  over all 8 sign patterns with `memcmp`, and refuses to run if it fails;
- fills `T1[own bit][m] = rx_cs(rz((cos α, 0, ±sin α), γ₀·h0[m]), cos β₀, sin β₀)`,
  with exactly the same operations the reference applies at layer 0.

The initial y = 0 is read through a `volatile`, so the compiler can't
specialise the arithmetic for a known zero.

---

## 3. Change 2 — layer 1 becomes a 1,024-entry table (T2)

**Observation.** A vertex's time-2 spin comes from its own time-1 spin and the
time-1 z of its parent and two children. By Change 1, each of those depends
only on seed bits. Altogether the time-2 spin depends on the **10 seed bits of
the vertex's radius-2 ball**: self, parent, grandparent, sibling, 2 children and
4 grandchildren. That gives 2^10 = 1,024 possible time-2 spins, each 24 bytes,
so the table is 24 KB and lives in L1/L2 cache.

**Routines:**

- **`build_tables`** (second loop): for each 10-bit pattern it takes the four
  time-1 spins from T1 and forms h = (z_parent + z_child1) + z_child2 **in the
  reference's association order**. It then stores
  `T2[i] = rx_cs(rz(S1, γ₁·h), cos β₁, sin β₁)`.
- **`t2_idx(w, d, k)`** (`viral_tree_fast.c:469`): gathers the 10 bits of vertex
  k of level d straight from the seed-bit array `w`. It handles the two special
  cases:
  - **the roots:** the other root plays the parent, and that root's two
    children play grandparent and sibling;
  - **level 1:** the other root plays the grandparent.

  Bit layout: 0 self, 1 parent, 2 grandparent, 3 sibling, 4–5 children,
  6–9 grandchildren.
- **`t2_child_idx(ix, j, gc4)`** (`viral_tree_fast.c:495`): builds a child's
  index from its parent's index plus the 4 new grandchild bits, with no fresh
  bit gathering. For that child, the parent becomes "parent", the parent's
  parent becomes "grandparent", the other child becomes "sibling", and the
  parent's grandchildren become "children". It works unchanged for roots and
  level 1.
- **`bit1`, `bitsn`** (`viral_tree_fast.c:457`, `:460`): read 1 bit, or up to 32
  consecutive bits, from the seed array. They handle reads that span two 64-bit
  words.

---

## 4. Change 3 — no state before time 3 is ever stored

With T2, **any** time-2 spin is a lookup from the seed bits. So the new kernel
`one_sample_fast` (`viral_tree_fast.c:518`) skips the first three phases of the
original entirely:

- **no initialisation pass**, where the original writes 48·2^p bytes;
- **no layer-0 sweep and no layer-1 sweep** over the tree.

**Layer 2 is computed directly from lookups** (section 2 of `one_sample_fast`,
starting at line 528). For every vertex x of levels 0..p−4 it computes

    S3(x) = U₂( T2[x],  (T2[parent].z + T2[child1].z) + T2[child2].z )

That is 4 table lookups and 1 `sincos` per vertex, written once into the spin
array. Each level passes its vertices' time-2 z down to the next level through
the scratch array `zs`, so a parent's index is not recomputed for each of its
two children.

---

## 5. Change 4 — level p−3 is never stored either

Level p−3 is updated only once, at layer 2, and its result (its time-3 z) is
read only by level p−4 at layer 3. So the new code does not store level p−3 at
all. It computes those values **on the fly inside layer 3**, while updating
level p−4. This is the branch marked `d == L, hence n == 3`
(`viral_tree_fast.c:604`):

- **Indices:** for each vertex of level p−4 it builds the indices of its two
  children and their four children with `t2_child_idx`. The 8 bits of level p−1
  and 16 bits of level p come from two `bitsn` reads.
- **The deferred sincos:** it computes each child's time-3 z with one `sincos`.
  This is the same `sincos` the reference spends on that vertex at layer 2,
  just done later, so the call count doesn't change.
- **The update:** it then updates the level-(p−4) vertex as usual.

`lazy_z3` (`viral_tree_fast.c:507`) does the same for the roots' children in
the smallest case, p = 4.

---

## 6. Change 5 — the smaller workspace

Only levels 0..p−4 hold spins now, and the z-scratch only has to cover levels
up to p−5:

| per process | original | fast |
|---|---|---|
| spins | 48·2^p (levels 0..p−1) | 6·2^p (levels 0..p−4) |
| z-scratch | 8·2^p (2 × 2^(p−1) doubles) | 1·2^p (2 × 2^(p−4) doubles) |
| seed bits | 0.5·2^p | 0.5·2^p (+1 word of padding) |
| **total** | **56.5·2^p** | **7.5·2^p** |
| p = 30 | 56.5 GiB | 7.5 GiB |
| p = 32 | 226 GiB | 30 GiB |

**Routines:**

- **`fast_nst`, `fast_zs`** (`viral_tree_fast.c:214`): the new array sizes.
- **`bytes_per_thread(fast)`** (`viral_tree_fast.c:217`): the memory formula;
  it is printed in the banner.
- **`alloc_work` / `touch_work` / `free_work`** (`viral_tree_fast.c:293`–`309`):
  allocate each kernel's workspace. Every page is still touched at start-up,
  so a job that is short of memory fails in seconds.

**Layers 3..p−1 are the reference's own in-place, top-down update**, the same
code, restricted to levels 0..p−4 (section 3 of `one_sample_fast`, line 559).

---

## 7. What it costs, and why the speedup is 3.6× rather than 4×

| per sample | original | fast |
|---|---|---|
| `sincos` calls | 4·2^p − 2p − 4 | 2^p − 2p (≈ 4× fewer) |
| table lookups | 0 | ≈ 2·2^p (10-bit gather + L1 load) |
| init + layer-0/1 sweeps | ~3 passes over 48·2^p bytes | none |

The lookups and bit gathering take back part of the 4× saving, which leaves
3.6–3.7× from p = 22 up.

---

## 8. What keeps it exact

Each table entry is the double the reference computes for that input, but only
if the arithmetic is literally the same. Four details make it so.

1. **`sincos()` explicitly**, in `rz` (`viral_tree_fast.c:134`). The frozen
   binary calls `sincos` too: GCC fuses its `cos`/`sin` pair, and the binary
   imports only `sincos`. glibc's `cos()` and `sincos()` can differ in the last
   bit, so the new code must call the same function.

2. **Explicit `fma()` in `rz` and `rx_cs`** (`viral_tree_fast.c:134`, `:145`),
   compiled with `-ffp-contract=off`. This took two failed runs to find:
   - **Run 1: no fusion.** RHEL 10's gcc targets x86-64-v3 (FMA, AVX2) by
     default, so the frozen `-O3` binary **fuses** its multiply-adds. Building
     the new file with fusion disabled changed the last bits, even at p = 1.
   - **Run 2: automatic fusion.** Letting gcc fuse on its own fails too,
     because it fuses **differently at different inlined sites**. The
     start-up self-check caught a 1-ulp difference at p = 4.
   - **The fix.** The frozen binary's disassembly (`scripts/fast_asm.slurm`)
     shows one pattern at all of its sites. Every `a·b ± c·d` is computed as
     `fma(a, b, ±round(c·d))`: the first product fused, the second rounded.
     The new code writes exactly that:

         rz:     x' = fma(c,  x,  -(sn*y))     y' = fma(sn, x,  c*y)
         rx_cs:  y" = fma(cb, y', -(sb*z))     z" = fma(sb, y', cb*z)

     `-ffp-contract=off` makes these the only fused operations anywhere in the
     file. The reference kernel, the table construction and the table kernel
     therefore all do identical arithmetic.

3. **The same association order** everywhere:
   (parent + child1) + child2, and for the roots (other root + child1) + child2.

4. **The same random bits.** Seeds come from the same xoshiro256** stream,
   seeded from the global sample index S = job·2^32 + k and laid out one bit
   per vertex in the same order. So every sample is the same sample as before.

---

## 9. Safety nets built into the binary

- **The reference kernel is kept.** `one_sample_ref` (`viral_tree_fast.c:323`)
  is the original `one_sample`, differing only in the explicit
  `sincos`/`fma`.
  - It is used for p < 4 (`FAST_MIN_P`).
  - It is used at every p with `VIRAL_FORCE_REF=1`.
- **Start-up self-check.** `selfcheck` (`viral_tree_fast.c:637`) runs
  `VIRAL_SELFCHECK` samples (default 64) at depth min(p, 12) through **both**
  kernels. It uses the first layers of the given angles and stops the program
  if any bit of any sample differs. The banner reports the result, e.g.
  `self-check 64 samples at p=12, tables vs reference kernel: identical`.
- **Symmetry check.** `build_tables` refuses to run if the time-1 symmetry
  (Change 1) fails for the given α.

---

## 10. What did not change

- the command line and exit codes;
- the result-file format and every field, including the first comment line;
- the RNG, the seeding, and the sample partition (S = job·2^32 + k);
- checkpoint/resume and atomic writes;
- SIGTERM/SIGINT/SIGUSR1 handling;
- the `VIRAL_TREE_DUMP` format;
- the batch logic (`-DBATCH_MAX_PER_THREAD=1` keeps the `soft_sum` summation
  order);
- one workspace per OpenMP thread.

Result files from the new binary are therefore interchangeable with the frozen
binary's and can be merged with them, as long as the job ids differ.

---

## 11. Build and validation

    gcc -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 -ffp-contract=off -fopenmp \
        viral_tree_fast.c -lm -o viral_tree_fast            # on a compute node (gcc 14.3.1)

| file | purpose |
|---|---|
| `viral_tree_fast.c` | the new solver |
| `tests/test_fast_equivalence.py` | byte-for-byte comparison against the frozen binary for p = 1..26, including resume and `check.sh`; `--benchmark` for timing and peak RSS |
| `scripts/fast_validate.slurm` | runs the suite (`suite`) or the benchmark (`bench`) on a compute node |
| `scripts/fast_facts.slurm` | node facts and glibc `sincos` timing |
| `scripts/fast_asm.slurm` | disassembles the frozen binary and the new one, to read the FMA pattern |
| `SPEEDUP_PLAN.md` | the plan and the next steps (threads inside one sample, R3 replays) |

Runs:

- **R1** (job 18517761): 235 cases, all identical.
- **R2** (job 18517764): the timing table at the top. Its results were
  compared as well.
