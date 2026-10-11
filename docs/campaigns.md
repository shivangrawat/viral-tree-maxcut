# The campaigns behind `results/`

This page records how the 10M samples per depth in `results/` were produced:
which binary ran, with which job ids, on which nodes, with what resources, and
with which commands. The numbers in the tables come from the committed records
(`results/*/manifest.json`, `logs/provenance_*.jsonl` and the job files), except
the requested memory and time limits, which come from the submission notes.
The as-run SLURM stdout logs and the original notes are in tag
[`results-10M`](../README.md#provenance) (commit `d578d35`).

## Summary

Every even depth p = 2..32 has exactly 10,000,000 samples, from three campaigns
with disjoint job ids:

| campaign | directory | job ids | samples per depth | solver | dates (UTC, first launch .. last launch) | core-hours |
|---|---|---|---:|---|---|---:|
| million-v1 | `results/torch-pP-million-v1` | 0 .. | 1,000,000 | `viral_tree_hpc` (p ≤ 30), `viral_tree_fast` (p = 32) | 2026-09-19 .. 2026-10-05 | 46,085.9 |
| extension-v1 | `results/torch-pP-extension-v1` | 1000 .. | 3,000,000 | `viral_tree_fast2` | 2026-10-03 .. 2026-10-04 | 80,422.8 |
| extension-v2 | `results/torch-pP-extension-v2` | 2000 .. | 6,000,000 | `viral_tree_fast2` | 2026-10-05 .. 2026-10-08 | 162,231.2 |
| **total** | 48 campaigns | | **10,000,000** | | 2026-09-19 .. about 2026-10-09 | **288,739.9** |

- 5,179 jobs, 5,276 launches (97 resumes, see [Events](#events)), 160M samples.
- Sample k of job j at depth p is the global index S = j·2^32 + k, seeded by
  `sample_rng(p, S)` (`viral_tree_hpc.c`, section 4.5), so the three campaigns
  draw disjoint samples (checked: [validation](validation.md#seeds)).
- Core-hours are the solvers' own `core_seconds`, i.e. threads × wall time of the
  sampling loop, summed over saved run segments. They exclude start-up, the
  work lost when a job was killed between checkpoints, and all build,
  validation and probe jobs.

## Cluster and software

- **Cluster:** NYU HPC "Torch", SLURM, account `torch_pr_577_general`.
- **Software** (identical in all 5,276 launch records): RHEL 10, Linux
  6.12.0-211.47.1.el10_2.x86_64, glibc 2.39, gcc 14.3.1 20251022 (Red Hat
  14.3.1-4), libgomp.
- **Nodes** (from the launch records' host names):
  - `cs`: Intel Xeon Platinum 8592+, 128 cores and 513 GB per node, 184 nodes.
    Jobs of at most 32 GB were routed here; 180 distinct `cs` nodes ran jobs.
  - `cl`: the nodes SLURM used for jobs above 32 GB (p = 30 million-v1 at
    60 GB); 7 nodes, CPU model not recorded.
  - `ga`, `gh`, `gl`, `gr`: 34 GPU-partition nodes that ran 104 p = 32
    million-v1 jobs, CPU only, after SLURM overflowed them to QoS `cpuplus`;
    CPU models not recorded.
- **QoS:** `cpu48` (48 h limit, 6000 GB and 3000 CPUs per user) for most jobs;
  `cpu168` (7 days, 2000 GB, 1000 CPUs, 500 submitted jobs) for p = 32
  million-v1 and part of p = 32 extension-v2.

## Builds

All three binaries were built on Torch compute nodes with `/usr/bin/gcc` and
these flags, validated there (see [validation](validation.md)), and pinned by
sha256 in every manifest:

```
gcc -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 [-ffp-contract=off] -fopenmp <source> -lm -o <binary>
```

| binary | source (campaign sha256) | executable sha256 | built by | ran |
|---|---|---|---|---|
| `viral_baseline` | `viral_tree_hpc.c` `b3af7c65…` | `0f3f8d44…a1d0` | job 17985393 (`scripts/validation/torch_build_validate.slurm`), and identically by 17989395 (`probe_large_p.slurm spot`) for p = 28, 30 | million-v1, p = 2..30 |
| `viral_fast` | `viral_tree_fast.c` `f1efdb7d…` | `bc4230d1…64b5` | job 18517761 (`fast_validate.slurm suite`, check R1) | million-v1, p = 32 |
| `viral_fast2` | `viral_tree_fast2.c` `8b12711a…` | `e18230cf…cad29` | job 19055757 (`fast2_gate.slurm`, gate G2) | extension-v1 and -v2 |

`-ffp-contract=off` is used for the two fast solvers only. Each binary had a
`.build.json` sidecar next to it that the launchers check
(`scripts/viral_campaign.py`, `build_info`). The `viral_fast` sidecar was
completed by hand from the R1 report (`tests/test_fast_equivalence.py --keep`
does not write `batch_max_per_thread`); the manifest of
`torch-p32-million-v1` records it.

## Per-campaign layout

Columns: requested memory and time limit per job; SLURM array ids (from the
launch records); launches (more than jobs when jobs were resumed); nodes by
name prefix; core-hours and per-job wall time (accumulated over restarts) from
the job files.

### million-v1: 1M samples per depth, job ids from 0

| p | job ids | jobs × samples | solver | threads | mem / time limit | SLURM arrays | launches | nodes | first launch (UTC) | core-hours | wall per job, median / max (h) |
|---:|---|---|---|---:|---|---|---:|---|---|---:|---|
| 2 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 1 h | 17987157 | 1 | cs 1 | 2026-09-19 02:48 | 0.0 | 0.00 / 0.00 |
| 4 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 1 h | 17987158 | 1 | cs 1 | 2026-09-19 02:48 | 0.0 | 0.00 / 0.00 |
| 6 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 1 h | 17987159 | 1 | cs 1 | 2026-09-19 02:48 | 0.0 | 0.00 / 0.00 |
| 8 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 1 h | 17987160 | 1 | cs 1 | 2026-09-19 02:48 | 0.0 | 0.00 / 0.00 |
| 10 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 1 h | 17987161 | 1 | cs 1 | 2026-09-19 02:48 | 0.0 | 0.01 / 0.01 |
| 12 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 1 h | 17987162 | 1 | cs 1 | 2026-09-19 02:48 | 0.1 | 0.06 / 0.06 |
| 14 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 2 h | 17987164 | 1 | cs 1 | 2026-09-19 02:48 | 0.2 | 0.23 / 0.23 |
| 16 | 0–0 | 1 × 1,000,000 | hpc | 1 | 2G / 3 h | 17987254 | 1 | cs 1 | 2026-09-19 02:48 | 0.9 | 0.92 / 0.92 |
| 18 | 0–3 | 4 × 250,000 | hpc | 1 | 2G / 3 h | 17987846 | 4 | cs 4 | 2026-09-19 02:50 | 3.7 | 0.98 / 0.98 |
| 20 | 0–15 | 16 × 62,500 | hpc | 1 | 2G / 3 h | 17987847 | 16 | cs 16 | 2026-09-19 02:50 | 14.6 | 0.92 / 1.00 |
| 22 | 0–63 | 64 × 15,625 | hpc | 1 | 2G / 8 h | 17987848 | 64 | cs 64 | 2026-09-19 02:50 | 73.7 | 1.18 / 1.26 |
| 24 | 0–49 | 50 × 20,000 | hpc | 1 | 2G / 16 h | 17988832 | 50 | cs 50 | 2026-09-19 02:54 | 314.8 | 7.15 / 7.15 |
| 26 | 0–99 | 100 × 10,000 | hpc | 1 | 5G / 24 h | 17988833 | 100 | cs 100 | 2026-09-19 02:54 | 1,072.6 | 10.84 / 13.09 |
| 28 | 0–199 | 200 × 5,000 | hpc | 1 | 16G / 48 h | 18029290 | 200 | cs 200 | 2026-09-19 19:29 | 4,188.1 | 20.08 / 31.46 |
| 30 | 0–499 | 500 × 2,000 | hpc | 1 | 60G / 48 h | 18096904, 18104270, 18717623, 18813300, 18979319, 19233370 | 593 | cl 591, cs 2 | 2026-09-20 20:32 | 20,392.9 | 40.07 / 53.74 |
| 32 | 0–999 | 1000 × 1,000 | fast | 1 | 32G / 48 h | 18528980, 18717138, 18813152, 18979320 | 1004 | cs 898, ga 37, gl 28, gh 25, gr 14, cl 2 | 2026-09-25 17:09 | 20,024.2 | 19.76 / 32.91 |
| **all** | | | | | | | | | | **46,085.9** | |

### extension-v1: 3M samples per depth, job ids from 1000

| p | job ids | jobs × samples | solver | threads | mem / time limit | SLURM arrays | launches | nodes | first launch (UTC) | core-hours | wall per job, median / max (h) |
|---:|---|---|---|---:|---|---|---:|---|---|---:|---|
| 2 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058457 | 1 | cs 1 | 2026-10-03 01:53 | 0.0 | 0.00 / 0.00 |
| 4 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058459 | 1 | cs 1 | 2026-10-03 01:53 | 0.0 | 0.00 / 0.00 |
| 6 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058460 | 1 | cs 1 | 2026-10-03 01:53 | 0.0 | 0.00 / 0.00 |
| 8 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058462 | 1 | cs 1 | 2026-10-03 01:53 | 0.0 | 0.01 / 0.01 |
| 10 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058463 | 1 | cs 1 | 2026-10-03 01:53 | 0.0 | 0.02 / 0.02 |
| 12 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058464 | 1 | cs 1 | 2026-10-03 01:53 | 0.1 | 0.08 / 0.08 |
| 14 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058467 | 1 | cs 1 | 2026-10-03 01:53 | 0.3 | 0.28 / 0.28 |
| 16 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058469 | 1 | cs 1 | 2026-10-03 01:53 | 1.0 | 0.96 / 0.96 |
| 18 | 1000–1000 | 1 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19058501 | 1 | cs 1 | 2026-10-03 01:53 | 3.6 | 3.63 / 3.63 |
| 20 | 1000–1002 | 3 × 1,000,000 | fast2 | 1 | 1G / 12 h | 19058502 | 3 | cs 3 | 2026-10-03 01:53 | 15.2 | 5.06 / 5.07 |
| 22 | 1000–1011 | 12 × 250,000 | fast2 | 1 | 1G / 12 h | 19058503 | 12 | cs 12 | 2026-10-03 01:53 | 53.7 | 4.69 / 5.00 |
| 24 | 1000–1029 | 30 × 100,000 | fast2 | 1 | 1G / 12 h | 19058507 | 30 | cs 30 | 2026-10-03 01:53 | 226.9 | 7.76 / 8.10 |
| 26 | 1000–1074 | 75 × 40,000 | fast2 | 1 | 1G / 24 h | 19058509 | 75 | cs 75 | 2026-10-03 01:53 | 901.6 | 12.10 / 15.38 |
| 28 | 1000–1149 | 150 × 20,000 | fast2 | 1 | 2G / 36 h | 19058511 | 150 | cs 150 | 2026-10-03 01:54 | 3,392.0 | 22.61 / 26.10 |
| 30 | 1000–1299 | 300 × 10,000 | fast2 | 4 | 6G / 24 h | 19058512 | 300 | cs 300 | 2026-10-03 01:51 | 14,555.7 | 12.10 / 18.09 |
| 32 | 1000–1499 | 500 × 6,000 | fast2 | 8 | 22G / 36 h | 19058517 | 500 | cs 500 | 2026-10-03 01:52 | 61,272.7 | 15.26 / 22.17 |
| **all** | | | | | | | | | | **80,422.8** | |

### extension-v2: 6M samples per depth, job ids from 2000

| p | job ids | jobs × samples | solver | threads | mem / time limit | SLURM arrays | launches | nodes | first launch (UTC) | core-hours | wall per job, median / max (h) |
|---:|---|---|---|---:|---|---|---:|---|---|---:|---|
| 2 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234122 | 2 | cs 2 | 2026-10-05 17:31 | 0.0 | 0.00 / 0.00 |
| 4 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234128 | 2 | cs 2 | 2026-10-05 17:31 | 0.0 | 0.00 / 0.00 |
| 6 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234129 | 2 | cs 2 | 2026-10-05 17:31 | 0.0 | 0.00 / 0.00 |
| 8 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234134 | 2 | cs 2 | 2026-10-05 17:31 | 0.0 | 0.01 / 0.01 |
| 10 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234135 | 2 | cs 2 | 2026-10-05 17:31 | 0.0 | 0.02 / 0.02 |
| 12 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234138 | 2 | cs 2 | 2026-10-05 17:31 | 0.1 | 0.07 / 0.07 |
| 14 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234141 | 2 | cs 2 | 2026-10-05 17:31 | 0.5 | 0.24 / 0.24 |
| 16 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234143 | 2 | cs 2 | 2026-10-05 17:31 | 1.9 | 0.96 / 0.99 |
| 18 | 2000–2001 | 2 × 3,000,000 | fast2 | 1 | 1G / 12 h | 19234144 | 2 | cs 2 | 2026-10-05 17:31 | 7.3 | 3.67 / 3.75 |
| 20 | 2000–2005 | 6 × 1,000,000 | fast2 | 1 | 1G / 12 h | 19234146 | 6 | cs 6 | 2026-10-05 17:31 | 29.4 | 4.88 / 4.95 |
| 22 | 2000–2023 | 24 × 250,000 | fast2 | 1 | 1G / 12 h | 19234147 | 24 | cs 24 | 2026-10-05 17:31 | 111.9 | 4.58 / 5.55 |
| 24 | 2000–2059 | 60 × 100,000 | fast2 | 1 | 1G / 12 h | 19234148 | 60 | cs 60 | 2026-10-05 17:31 | 435.7 | 7.20 / 8.68 |
| 26 | 2000–2149 | 150 × 40,000 | fast2 | 1 | 1G / 24 h | 19234149 | 150 | cs 150 | 2026-10-05 17:32 | 1,789.8 | 11.91 / 13.95 |
| 28 | 2000–2299 | 300 × 20,000 | fast2 | 1 | 2G / 36 h | 19234153 | 300 | cs 300 | 2026-10-05 17:35 | 6,887.5 | 22.91 / 29.73 |
| 30 | 2000–2599 | 600 × 10,000 | fast2 | 4 | 6G / 24 h | 19234154 | 600 | cs 600 | 2026-10-05 17:37 | 29,218.2 | 12.04 / 18.88 |
| 32 | 2000–2999 | 1000 × 6,000 | fast2 | 8 | 22G / 36 h | 19234156, 19234157, 19352933 | 1000 | cs 1000 | 2026-10-05 17:32 | 123,748.9 | 15.24 / 24.39 |
| **all** | | | | | | | | | | **162,231.2** | |

Measured peak memory of single jobs (probe jobs, `getrusage`): 14.127 GiB for
`viral_baseline` at p = 28 (job 17989534), 56.502 GiB at p = 30 (17989535),
30.002 GiB for `viral_fast` at p = 32 (18524316). Each is within 2 MiB of the
workspace the solver predicts at start-up (`viral_tree_fast2`: 1.25 GB at
p = 28, 5 GB at p = 30, 20 GB at p = 32; its peak RSS was not measured
separately).

## How the campaigns were run

Every campaign followed the same steps; the launchers enforce them.

1. **Build and validate** the binary on a compute node and keep it with its
   `.build.json` sidecar (table above; [validation](validation.md)).
2. **Freeze** the campaign: `init` writes `manifest.json` (depth, samples per
   job, expected job ids, angles file and hashes, executable and build) and
   `provenance.txt`, and creates `jobs/` and `logs/`. It refuses a non-empty
   directory; the fast2 launcher also refuses job ids that overlap earlier
   campaigns (`--avoid`).
3. **Submit** a SLURM array of the wrapper (`scripts/run_viral*.slurm`). Each
   task appends a launch record to `logs/provenance_<array>_<task>.jsonl`,
   checks its arguments, CPUs and threads against the manifest, and `exec`s the
   solver, which writes `jobs/viral_tree_p<P>_job<ID>.txt`.
4. **Resume** jobs stopped by the time limit: `pending` lists the ids whose
   result file is missing or not complete; resubmitting them continues from the
   last checkpoint (`--signal=B:TERM@600` gives the solver 10 minutes to save).
   A SLURM `COMPLETED` state is not proof of completion: a job stopped by
   signal exits 0 with `status running`.
5. **Verify** (`verify`: every expected id exactly once, complete, consistent
   with the manifest) and **merge** with `merge_results.py`.

### Commands

million-v1, p = 2..28, through `scripts/submit_campaign.sh` (which holds the
resource table above):

```bash
export VIRAL_EXE=/scratch/fm2452/viral-controlled/torch-small-builds-17985393/viral_baseline
scripts/submit_campaign.sh init   2 4 6 8 10 12 14 16 18 20 22 24 26    # manifests 2026-09-19 02:47
scripts/submit_campaign.sh submit 2 4 6 8 10 12 14 16 18 20 22 24 26
# p = 28 and 30: the same, with VIRAL_EXE=…/torch-large-builds-17989395/viral_baseline (same sha256)
scripts/submit_campaign.sh status 28 ; scripts/submit_campaign.sh resume 30 ; scripts/submit_campaign.sh verify 30
```

`submit` runs, per depth, `sbatch --account=… --array=0-(J-1)%CONC --mem=M
--time=W --output=results/torch-pP-million-v1/logs/slurm_%A_%a.out
scripts/run_viral.slurm P N params/angles_pP.txt $VIRAL_EXE results/torch-pP-million-v1`.
p = 30 was submitted as array 18096904 (task 0) and 18104270 (tasks 1–499)
rather than one array.

million-v1, p = 32, with `viral_fast` (as recorded):

```bash
EXE=~/viral-exe-backup/fast-builds-18517761/viral_fast
python3 scripts/fast_campaign.py init --p 32 --samples 1000 --jobs 1000 \
    --angles params/angles_p32.txt --exe $EXE --out results/torch-p32-million-v1
sbatch --qos=cpu168 --array=0-495%62 --mem=32G --time=48:00:00 \
    --output=results/torch-p32-million-v1/logs/slurm_%A_%a.out \
    scripts/run_viral_fast.slurm 32 1000 params/angles_p32.txt $EXE results/torch-p32-million-v1
# then ids 496-995 (array 18717138, --qos=cpu168 --partition=cs --array=496-995%500),
# ids 996-999 (18813152), and 4 resumed ids (18979320), with the same wrapper arguments
```

extension-v1 and -v2 with `viral_fast2`. These commands are **reconstructed**
from the manifests (depth, ids, samples, threads, `--avoid`) and the submission
notes (resources); the literal command lines were not recorded:

```bash
EXE=~/viral-exe-backup/fast2-builds-19055757/viral_fast2
# extension-v1: J jobs of N samples, T threads, from the extension-v1 table above
python3 scripts/fast2_campaign.py init --p $P --samples $N --jobs $J --first-id 1000 --threads $T \
    --angles params/angles_p$P.txt --exe $EXE --out results/torch-p$P-extension-v1 \
    --avoid results/torch-p$P-million-v1
sbatch --account=torch_pr_577_general --qos=cpu48 --dependency=afterany:18104270 \
    --array=1000-$((999 + J))[%C] --cpus-per-task=$T --mem=$MEM --time=$TIME \
    --output=results/torch-p$P-extension-v1/logs/slurm_%A_%a.out \
    scripts/run_viral_fast2.slurm $P $N params/angles_p$P.txt $EXE results/torch-p$P-extension-v1
# extension-v2: the same with --first-id 2000, --avoid <million-v1> <extension-v1>,
# --array=2000-$((1999 + 2*J)) and no dependency; p = 32 was split over three
# arrays: cpu48 2000-2689%200, cpu168 2690-2959%90, cpu48 2960-2999%40
```

Finally, per depth (the output is `results/merge_10M.txt`):

```bash
python3 merge_results.py results/torch-pP-million-v1/jobs results/torch-pP-extension-v1/jobs \
    results/torch-pP-extension-v2/jobs -p P
```

These launchers only work on the original cluster as is: the manifests freeze
absolute paths under `/home/fm2452` and the binaries under
`$HOME/viral-exe-backup`. To run the same samples elsewhere use
`scripts/run_array.slurm` ([README](../README.md#reproducing-the-published-numbers)).

## Events

- **p = 30 million-v1** (`viral_baseline`, 60 GB jobs on `cl` nodes, about 40 h
  per job): 93 tasks reached the 48 h limit, saved their state and were resumed
  (arrays 18717623, 18813300, 18979319); job 431 was resubmitted alone
  (19233370_[431]). One more task (job 397) received the time-limit signal while
  finishing its last sample and completed.
- **p = 32 million-v1:** jobs 666, 669, 685 and 700 of array 18717138 stopped
  without a final checkpoint (no stop message in their logs) and were resumed
  in array 18979320 from their last checkpoint (820, 863, 246 and 153 samples
  done). The first submission of ids 496–999 (array 18528991, waiting on p = 30)
  was cancelled before it started and replaced by 18717138 and 18813152. SLURM
  moved 385 tasks of 18528980 to QoS `cpuplus` (partition `all`); 104 of them
  ran on the GPU-partition nodes.
- **extension-v1:** a first submission (19057066–19057205) was cancelled before
  any task started. **extension-v2:** ids 2960–2999 of p = 32 moved from
  19234157 to 19352933 before they started.
- No result was lost: every job ended `complete` with exactly its target
  ([validation](validation.md#campaign-results)).

## Sample ids and seeds

- million-v1 uses job ids 0 .. J−1 (at most 999), extension-v1 1000 .. (at most
  1499), extension-v2 2000 .. (at most 2999). Test runs on the cluster used ids
  0, 3, 5, 7, 13, 19, 499, 9000–9001, 999999 and 2147483647; their outputs were
  never merged.
- Four guards keep the samples disjoint: `fast2_campaign.py init --avoid`
  refuses overlapping ids; the seed audits G1 and G1′ checked that all 64M and
  then all 160M initial generator states are distinct
  ([validation](validation.md#seeds)); `merge_results.py` refuses a repeated
  (p, job); and every depth uses one angles file, whose hash every manifest
  and result file records.

## Values specific to Torch

To reuse the launchers or the validation scripts on another cluster, change:
the account `torch_pr_577_general` (`#SBATCH` lines of `scripts/validation/*`,
default of `submit_campaign.sh`); QoS names `cpu48` / `cpu168`; log paths under
`/scratch/fm2452`; binaries under `$HOME/viral-exe-backup`; and `module purge`
/ `module load intel/2025.2` in the build scripts.
