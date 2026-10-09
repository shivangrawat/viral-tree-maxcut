# 10M samples per depth: a second, 6M-sample extension of p = 2, 4, …, 32 (plan of 2026-10-05)

**SUBMITTED 2026-10-05.**
- p=30 job 431 was resubmitted as 19233370_[431].
- G1′ seed audit 19233795: PASS (160M states, 0 duplicates).
- The 17 arrays were submitted with no dependency:

| p | array | ids | QoS |
|---|---|---|---|
| 2, 4, 6, 8 | 19234122, 19234128, 19234129, 19234134 | 2000-2001 | cpu48 |
| 10, 12, 14, 16, 18 | 19234135, 19234138, 19234141, 19234143, 19234144 | 2000-2001 | cpu48 |
| 20 | 19234146 | 2000-2005 | cpu48 |
| 22 | 19234147 | 2000-2023 | cpu48 |
| 24 | 19234148 | 2000-2059 | cpu48 |
| 26 | 19234149 | 2000-2149 | cpu48 |
| 28 | 19234153 | 2000-2299 %150 | cpu48 |
| 30 | 19234154 | 2000-2599 %150 | cpu48 |
| 32 | 19234156 | 2000-2689 %200 | cpu48 |
| 32 | 19234157 | 2690-2959 %90 | cpu168 |
| 32 | 19352933 | 2960-2999 %40 (moved from 19234157 on 2026-10-07, never started there) | cpu48 |

Goal: bring every even depth from 2 to 32 from 4M to **10M samples** before
the paper is submitted. That shrinks the standard error by a factor
√(4/10) = 0.63 (p=32: 0.000141 → about 0.000089). The two existing
campaigns per depth (`torch-pP-million-v1`, 1M, and `torch-pP-extension-v1`,
3M) stay exactly as they are. The new **6M samples per depth** go in new
directories `results/torch-pP-extension-v2`. At the end, `merge_results.py`
combines all three.

## 1. Why the new samples are new

- **How seeds work.** A sample's seed is `sample_rng(p, S)` with
  `S = job·2^32 + k`, a bijective mix of p and S. Two samples can coincide
  only with the same p, job and k.
- **Ids already used, per depth:**
  - `million-v1`: 0 … J−1, at most 999 (p=32).
  - `extension-v1`: 1000 … at most 1499 (p=32).
  - Test runs used 0, 3, 5, 7, 13, 19, 499, 9000–9001, 999999 and 2147483647.
    Their outputs live on `/scratch` and are never merged.
- **New ids:** **2000 … at most 2999** (p=32) at every depth. That leaves a gap
  of 500 unused ids after `extension-v1`.
- **Four guards** (the same as for the 4M extension):
  1. `fast2_campaign.py init --first-id 2000 --avoid results/torch-pP-million-v1
     results/torch-pP-extension-v1` refuses any id overlap with either earlier
     campaign's expected ids, and with any result file already in their
     directories.
  2. **Gate G1′, the seed audit:** a copy of `scripts/seed_audit.slurm` with a
     three-campaign static check. The C program `tests/seed_audit.c` is
     unchanged; it already reads any list of campaigns. It computes the initial
     RNG state of all **160M** samples (10M × 16 depths) and requires them all
     to be distinct. It also checks for id overlap against every result file
     under `results/`, the same angles in all three manifests, contiguous ids,
     and targets of 1M / 3M / 6M.
  3. `merge_results.py` refuses any repeated (p, job), and any mix of angle
     files within a depth.
  4. Same angles files: `params/angles_pP.txt`, whose hash is checked in every
     manifest.

## 2. Binary: no new code

- **Same executable as the 3M extension:**
  `~/viral-exe-backup/fast2-builds-19055757/viral_fast2`, sha `e18230cf…cad29`,
  from `viral_tree_fast2.c` sha `8b12711a…`. It lives in `$HOME`, which is not
  flushed. Re-check its sha before submitting.
- **Same launcher:** `scripts/fast2_campaign.py` + `scripts/run_viral_fast2.slurm`,
  unchanged.
- **Already validated:**
  - byte-identical to the frozen binary for p = 1–24 at 1, 4 and 8 threads;
  - identical to fast at p = 26–30 at 1–32 threads, and at p=32 with 8 threads
    (G2);
  - fast identical to frozen at p=32 (G3, `P32 CHECK PASS`);
  - 48M production samples, statistically consistent with the 1M campaigns at
    every depth (χ² ≈ 13.8 on 16 degrees of freedom).
- **Thread counts:** every depth keeps the thread count it ran with in the 3M
  extension (8 at p=32, 4 at p=30, 1 below).

## 3. Job layout

- **Shape:** every job has exactly the shape of a measured 3M-extension job
  (same samples per job, threads, memory and time limit), with twice as many
  jobs. So every time limit already has a measured margin.
- **QoS:** p=32 is split across both QoS so it uses the cpu168 quota as well.
  Everything else is on cpu48.
- **Partition:** all jobs are ≤ 22G, so they run on `cs`, never `cl`.

| p | jobs × samples (ids from 2000) | threads | mem | time limit | measured per job (v1, median / max) | QoS, throttle |
|---|---|---|---|---|---|---|
| 2–16 | 2 × 3,000,000 each | 1 | 1G | 12 h | ≤ 1 h | cpu48 |
| 18 | 2 × 3,000,000 | 1 | 1G | 12 h | 3.6 h | cpu48 |
| 20 | 6 × 1,000,000 | 1 | 1G | 12 h | 5.1 h | cpu48 |
| 22 | 24 × 250,000 | 1 | 1G | 12 h | 4.7 / 5.0 h | cpu48 |
| 24 | 60 × 100,000 | 1 | 1G | 12 h | 7.8 / 8.1 h | cpu48 |
| 26 | 150 × 40,000 | 1 | 1G | 24 h | 12.1 / 15.4 h | cpu48 |
| 28 | 300 × 20,000 | 1 | 2G | 36 h | 22.6 / 26.1 h | cpu48 %150 |
| 30 | 600 × 10,000 | 4 | 6G | 24 h | 12.1 / 18.1 h | cpu48 %150 |
| 32 | 690 × 6,000 (ids 2000–2689) | 8 | 22G | 36 h | 15.3 / 22.2 h | cpu48 %200 |
| 32 | 310 × 6,000 (ids 2690–2999) | 8 | 22G | 36 h | 15.3 / 22.2 h | cpu168 %90 |

**Totals:** 2,158 jobs, 17 arrays, about 161k core-hours (2 × the 80.4k measured
for 3M).

**Peak quota use:**

| QoS | memory | CPUs | headroom |
|---|---|---|---|
| cpu48 | about 5,860G of 6,000G | about 2,600 of 3,000 | ≥ 60G left for p=30 job 431 |
| cpu168 | 1,980G of 2,000G | 720 of 1,000 | 310 submitted jobs, under the 500 limit |

**Wall time:**
- **Memory is the binding cap:** about 397k GB-hours against roughly 7,800G
  usable, so at least 51 h with perfect packing.
- **p=32 is the critical path:** 3.45 waves of about 15.3 h on each QoS,
  about 53 h.
- **The 3M extension's ramp-up and tail** stretched its waves by about 25%.
- **So expect about 3 days** (2.5–4) after the first jobs start.

## 4. Non-interference with the running jobs

- **Separate files:** new directories, the same unchanged scripts and binary,
  and no edits to any file a running job reads.
- **No preemption:** pending jobs can't affect running ones. The p=32 1M
  stragglers already hold their cpu168 slots; the cpu168 part of this campaign
  only takes what is free and fills up once they finish (tonight).
- **p=30 job 431** (cpu48, 60G, on `cl`): submit it first, so it is older and
  ranks ahead. The cpu48 peak above leaves room for it. This campaign never
  uses `cl`.
- **Alternative:** `--dependency=afterany` on every current job. That's the
  strongest guarantee, but job 431 can sit days in the `cl` queue (earlier
  resumes did), and the whole campaign would wait with it. Not recommended.

## 5. Storage

- **/home now:** 15.5k of 30k files.
- **This campaign adds** about 3 files per job (result, Slurm log,
  provenance), about 6.5k in total, reaching about 22k (73%).
- **If results are committed to git,** run `git gc` afterwards so loose
  objects don't count against the file quota.

## 6. Order of operations

1. Submit p=30 job 431, the last 229 samples of the 1M campaign, once the Slurm
   controller answers.
2. **Init** the 16 `torch-pP-extension-v2` campaigns in the main checkout:
   `--first-id 2000 --avoid <million-v1> <extension-v1>`, with the threads
   from the table.
3. **G1′ seed audit:** all 160M states distinct, plus the static checks
   (cpu48, 8G, about 1 h).
4. **Dry run:** `sbatch --test-only` for all 17 arrays; re-check the binary's
   sha.
5. **Submit** the 17 arrays (no dependency, per §4).
6. **Afterwards:** run `fast2_campaign.py status` / `pending`, resubmit
   stopped or node-failed ids, and finally run
   `merge_results.py results/torch-pP-million-v1/jobs results/torch-pP-extension-v1/jobs results/torch-pP-extension-v2/jobs -p P`.
