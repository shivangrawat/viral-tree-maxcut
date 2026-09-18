# viral_tree_hpc — VIRAL on the 3-regular tree (MaxCut lower bound)

This folder is self-contained.  It needs a C compiler with OpenMP (gcc is
fine), `make`, and `python3`.  No libraries, no downloads, no configuration.

**What it computes.**  `c_VIRAL(p)`: the probability that the VIRAL local
algorithm, run for `p` layers on the infinite 3-regular tree, cuts the root
edge.  Because the algorithm is local, this number is a rigorous lower bound on
the max-cut fraction of every 3-regular graph of girth ≥ 2p+2, i.e. on
`M_g = inf{ MaxCut(G)/|E| : G 3-regular, girth ≥ g }` for all `g ≥ 2p+2`.
The best published bound is `lim M_g ≥ 0.912`.  **The plan is to measure
`c_VIRAL(p)` for p = 24, 26, 28, 30 with 10⁶ Monte-Carlo samples each**
(statistical error 2.9·10⁻⁴ per point), which locates where the curve crosses
0.912.  The whole story — problem, algorithm, proof, how the code works — is
the preamble of `viral_tree_hpc.c`.

---

## 1. Steps to run (SLURM cluster)

```bash
cd CODE-for-HPC              # this folder
make                         # builds ./viral_tree_hpc   (a few seconds)
make check                   # ~1 minute; must end with:  ALL CHECKS PASSED
sbatch submit_p24.slurm      # 1e6 samples each; cheapest first, see the cost table below
sbatch submit_p26.slurm
sbatch submit_p28.slurm
sbatch submit_p30.slurm      # needs >= 256 GB nodes, see section 2
```

That is all.  The job scripts create what they need, write everything into
`results/`, and all jobs (of all p) are independent of each other, so the
four arrays can run at the same time or one after the other.

While it runs / when it is done:

```bash
squeue -u $USER                          # progress of the array
python3 merge_results.py results         # current combined estimate (valid at any time)
```

**If jobs hit the wall-time limit** before finishing: `sbatch` the same
script again, unchanged.  Every job resumes from its own file; jobs that were
already complete exit immediately.  Repeat until `merge_results.py` shows
`0 still running / incomplete` for that p.

**What to send back:** the folder `results/` (all `viral_tree_p*_job*.txt`
files and the `slurm_*.out` logs; a few MB) — or just the output of
`python3 merge_results.py results -v`.  Partial results are useful too.

### Adapting the resource requests

If the cluster rejects a request, edit the `#SBATCH` lines of that script
following the rules written above them; the only hard constraint is

```
memory per job  ≥  (threads per job) × (memory per thread)  +  1 GB
memory per thread = 56.5·2^p bytes:  p=24 0.88 GB,  p=26 3.53 GB,  p=28 14.1 GB,  p=30 56.5 GB
```

Anything else (fewer/more tasks, fewer/more threads, shorter time limit) only
changes how long it takes.  The program prints its exact memory need at start
and touches all of it immediately, so a job that does not fit fails within
seconds — it never dies hours in.  For p = 28 and 30 the scripts' headers give
the thread/memory pairs for 128 GB, 256 GB, 512 GB, 1 TB nodes.

### Not SLURM?

Each job is one command; any scheduler or plain shell works:

```bash
export OMP_NUM_THREADS=16
./viral_tree_hpc 26 10000 <job_id> params/angles_p26.txt results
```

Run it for `job_id = 0, 1, 2, …` (100 of them for 10⁶ samples) on as many
machines as you like — different ids never overlap — then
`python3 merge_results.py results`.  If a job is killed, rerun the same
command: it resumes.  Same for the other p with the n/job of the table below.

---

## 2. Cost

Measured on an Intel i9-10900X (3.7 GHz), GCC 9.4, `-O3`:

| p | seconds per sample per core | memory per thread | girth covered |
|---|---|---|---|
| 22 | 0.31 | 0.22 GB | ≥ 46 |
| 24 | 1.31 | 0.88 GB | ≥ 50 |
| 26 | **4.3** | **3.53 GB** | ≥ 54 |
| 28 | ≈ 15 (extrapolated) | 14.1 GB | ≥ 58 |
| 30 | ≈ 60 (extrapolated) | 56.5 GB | ≥ 62 |

Memory per thread is 56.5·2^p bytes exactly.  Statistical error after n
samples ≈ 0.29/√n:  n = 10⁵ → 9·10⁻⁴,  10⁶ → 2.9·10⁻⁴.

**The plan: 10⁶ samples for each p** (the p = 28 and 30 timings are
extrapolated from p = 26, factor 2 per unit of p; they will be measured by the
first jobs and printed in the logs as `s/sample/core`):

| script | p | jobs × samples | threads / job | memory / job | wall / job | core-hours |
|---|---|---|---|---|---|---|
| `submit_p24.slurm` | 24 | 50 × 20 000 | 16 | 16 GB | 0.5 h | 360 |
| `submit_p26.slurm` | 26 | 100 × 10 000 | 16 | 60 GB | 0.75 h | 1 200 |
| `submit_p28.slurm` | 28 | 200 × 5 000 | 8 | 116 GB | ≈ 3 h | ≈ 4 700 |
| `submit_p30.slurm` | 30 | 500 × 2 000 | 4 | 228 GB | ≈ 10 h | ≈ 19 000 |

Total ≈ 25 000 core-hours, three quarters of it p = 30.  **p = 30 is limited by
memory, not cores**: 56.5 GB per thread means 4 threads on a 256 GB node, so
its 500 jobs are ≈ 5 000 node-hours on 256 GB nodes (≈ 2 500 on 512 GB nodes
with 8 threads, ≈ 1 250 on 1 TB nodes with 16).  If the cluster has few
large-memory nodes, run p = 24, 26, 28 first and p = 30 as far as the
allocation allows — every completed job is a usable partial result.

---

## 3. Troubleshooting

| symptom | cause / fix |
|---|---|
| `make`: `gcc: command not found` or `omp.h: No such file` | load a compiler: `module load gcc` (or `module avail gcc` to see the names), then `make` again.  Any gcc ≥ 4.9 or clang works; `CC=icc make` for Intel. |
| `make check` fails | do not run the jobs; send the output of `make check` back. |
| `sbatch`: "invalid partition / account / memory specification" | add `#SBATCH --partition=…` / `--account=…` or lower `--mem` **and** `--cpus-per-task` together (keep mem ≥ cpus × 3.6 GB + 1 GB). |
| job log says `allocation failed` or job is killed for memory | the node gave less memory than requested; reduce `--cpus-per-task` (memory scales with threads). |
| jobs end with `STOPPED BY SIGNAL` in the log | normal: wall-time limit reached, state saved.  `sbatch` the same script again. |
| `merge_results.py`: "appears twice" | the same (p, job_id) file exists in two places; delete one copy — they are the same samples. |
| `Illegal instruction` | only possible if someone built with `make native` on a different CPU type; `make clean && make`. |

---

## 4. Files

| file | purpose |
|---|---|
| `viral_tree_hpc.c` | the program (single C file, OpenMP + libm), with a full description in its preamble |
| `Makefile` | `make`, `make check`, `make clean` |
| `check.sh` | self-tests, run by `make check` |
| `submit_p24.slurm` … `submit_p30.slurm` | one SLURM job array per p, 10⁶ samples each (build if needed, resume if resubmitted) |
| `params/angles_p<p>.txt` | the trained angles, even p = 2 … 100; `angles_p26.txt` is the one used |
| `merge_results.py` | adds up all job files, prints the estimate and exact confidence bounds (pure Python 3) |
| `results/` | output directory (shipped with a README so it exists) |
| `exact_small_p.py` | independent exact enumeration for p = 1, 2 (needs numpy); used for validation, not needed to run |

Command line of the program:

```
./viral_tree_hpc <p> <n_samples> [job_id=0] [angles_file=params/angles_p<p>.txt] [out_dir=.]
```

* threads = `OMP_NUM_THREADS` (the job script sets it from `--cpus-per-task`);
* `job_id`: different ids use disjoint, independent samples — results add;
  the same id must never be counted twice (`merge_results.py` checks);
* output `<out_dir>/viral_tree_p<p>_job<job_id>.txt`, rewritten every minute;
  if it exists at start the run resumes from it (after checking that p, job
  and the angles file are the same).

---

## 5. What the output means

```
x_thresh   0.91xxxx +- 0.00029     <-- c_VIRAL(p) estimate: THIS is the bound
x_soft     ...                     diagnostic only, not a bound
```

`merge_results.py` also prints exact one-sided Clopper–Pearson lower confidence
bounds.  The final statement has the form "`M_g ≥ L` for all `g ≥ 2p+2`, at
99.9 % confidence": the implication `c_VIRAL(p) ≥ L ⟹ M_g ≥ L` is a theorem;
`L` being a lower confidence bound for `c_VIRAL(p)` is the statistical part.

---

## 6. Validation done before hand-over

* Per-sample output **bit-identical** to the original code used for all
  results so far, for p = 1 … 20 (~10⁵ samples compared as hex floats).
* p = 1, 2: Monte Carlo agrees with an independent exact enumeration of all
  2⁶ / 2¹⁴ seed configurations (`exact_small_p.py`); 1.6·10⁷ samples at p = 2
  give 0.75566 ± 0.00011 vs exact 0.7556152.
* `n_cut` identical for 1, 2, 4 threads; identical after SIGTERM-stop/resume
  and after SIGKILL/resume; identical when a job is extended versus run
  fresh; wrong p/job/angles/shrink/missing-directory are refused cleanly.
* Memory at p = 26 measured: 3.53 GiB RSS, exactly the formula.
* `-O3` vs `-O3 -march=native`: identical counts, identical speed.
* `merge_results.py` bounds agree with `scipy.stats.beta.ppf` to 10⁻¹³.

Reproducibility: same binary, same machine → bit-identical.  Across
CPUs/compilers the last bit of `sin`/`cos` can differ, which can flip the sign
decision of a sample whose `s^z` is within ~10⁻¹⁵ of zero: `n_cut` changes by
O(1) per million samples.  Harmless.
