# 4M samples per depth: the 3M-sample extension of p = 2, 4, …, 32 (plan of 2026-10-02)

Goal: bring every even depth from 2 to 32 from 1M to **4M samples**, which
halves the standard error. The existing 1M campaigns
(`results/torch-pP-million-v1`) are left exactly as they are. The extension
adds **3M new samples per depth**, in new directories
`results/torch-pP-extension-v1`. At the end, `merge_results.py` combines old
and new.

## 1. Why the new samples are new

- **How seeds work.** A sample's seed is `sample_rng(p, S)` with
  `S = job·2^32 + k`. It is made by bijective mixing of p and S, so two
  samples can coincide only with the same p, job and k.
- **Old ids.** Every 1M campaign uses job ids `0 … J−1`; the most is p=32 with
  0 … 999. The `local_mac` runs are job 0 at p=2 and p=4, a subset of those.
  Test runs used ids 0, 3, 5, 7, 13, 19, 499, 999999 and 2147483647; their
  outputs are on `/scratch` and are never merged.
- **New ids.** The extension uses **job ids 1000 …** at every depth.
- **Four guards:**
  1. `fast2_campaign.py init --avoid <the 1M campaign>` refuses any id overlap
     with that campaign's expected ids, and with any result file already in
     its directory.
  2. Gate G1 (`scripts/seed_audit.slurm`) re-checks the ids against every
     result file under `results/`. It then computes the initial RNG state of
     every old and new sample, 64M in total, and requires them all to be
     distinct.
  3. `merge_results.py` refuses any repeated (p, job), and any mix of angle
     files within a depth.
  4. Same angles files: `params/angles_pP.txt`, whose hash is checked in the
     manifests.

## 2. Binary and validation

- **Binary:** `viral_tree_fast2.c` (source sha `8b12711a…`), built
  deterministically as executable `e18230cf…cad29`, and exported with its
  sidecar to `~/viral-exe-backup/fast2-builds-<gate job>/`.
- **Validation:**
  - byte-identical to the frozen binary for p = 1–24 at 1, 4 and 8 threads (suite
    18986377);
  - identical to the validated fast binary at p = 26, 28, 30 with 1–32 threads
    (18986379, 19053518, 19053519);
  - at p=32 itself: G2 (fast2 against fast, 8 threads) and G3 (fast against
    frozen, 18980679).

## 3. Job layout

All jobs use QoS `cpu48`, partition `cs` (≤32G jobs cannot go to `cl`), and
time limits over 6 h so they don't land in `cpu_short`.

| p | jobs (ids from 1000) × samples | threads | mem | time limit | est. per job | throttle |
|---|---|---|---|---|---|---|
| 2–16 | 1 × 3,000,000 each | 1 | 1G | 12 h | ≤ 1 h | – |
| 18 | 1 × 3,000,000 | 1 | 1G | 12 h | ~3.6 h | – |
| 20 | 3 × 1,000,000 | 1 | 1G | 12 h | ~4 h | – |
| 22 | 12 × 250,000 | 1 | 1G | 12 h | ~4 h | – |
| 24 | 30 × 100,000 | 1 | 1G | 12 h | ~6 h | %30 |
| 26 | 75 × 40,000 | 1 | 1G | 24 h | ~10 h | %75 |
| 28 | 150 × 20,000 | 1 | 2G | 36 h | ~19 h | %150 |
| 30 | 300 × 10,000 | 4 | 6G | 24 h | ~10 h | %150 |
| 32 | 500 × 6,000 | 8 | 22G | 36 h | ~14 h | %200 |

- **Totals:** 1,079 jobs; about 71,000 core-hours.
- **Peak usage:** about 5,700G of 6,000G and about 2,450 of 3,000 CPUs.

## 4. Order of operations

1. **G2:** build, export and compare at p=32 (`fast2_gate.slurm`).
2. **Commit and fast-forward.** Commit on `faster-wt` and fast-forward `faster`
   (new files only; nothing the running campaigns execute changes).
3. **Init** the 16 extension campaigns in the main checkout, with `--avoid`
   pointing at each 1M campaign.
4. **G1** seed audit, and **G4** a launcher dry run on a scratch campaign,
   compared with the frozen binary.
5. **Submit** the 16 arrays with `--dependency=afterok:18980679`. That is G3,
   which itself waits for every p=30 array, so nothing starts until p=30 is
   finished **and** G3 passes. If G3 fails, the arrays never start.
6. **Afterwards:** `fast2_campaign.py status`/`verify` on each extension;
   resubmit any job that stops at the limit (`pending`); finally
   `merge_results.py results/torch-pP-million-v1/jobs results/torch-pP-extension-v1/jobs -p P`.
