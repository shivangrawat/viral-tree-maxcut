# p = 32 campaign with the fast solver — SUBMITTED 2026-09-25

**As submitted:** campaign `results/torch-p32-million-v1` (1000 × 1000, manifest
2026-09-25).

- **Array A 18528980:** `--qos=cpu168 --array=0-495%62 --mem=32G --time=48:00:00`.
  `cpu168` allows at most 500 submitted jobs per user; 4 slots were taken by
  the R3 replays.
- **Array B 18528991:** `--qos=cpu48 --array=496-999%187 --mem=32G --time=48:00:00`
  with `--dependency=after:18104270_499`, so it becomes eligible only once
  p=30's last task has started, i.e. once p=30 has no pending tasks.
- **Safety net:** R3 replays 18527853/55/56/57 are running as confirmation. If
  any reports REPLAY FAIL, cancel both arrays.

**Update 2026-09-28:**
- Array A finished (496 tasks in 54 h; Slurm overflowed 385 of them to QoS
  `cpuplus` / partition `all`), and all four R3 replays passed.
- B's tasks 496–995 were cancelled and resubmitted as **array C 18717138**:
  `--qos=cpu168 --partition=cs --array=496-995%500`. That keeps them off
  p=30's `cpu48` quota and `cl` nodes; `cpu168` allows at most 500 submitted
  jobs per user.
- Tasks 996–999 stay in 18528991 on `cpu48`, still waiting on p=30's last
  task.
- `--exclude` is rejected by the site plugin, so overflow could still place a
  few tasks on `cl`.

**Update 2026-09-29:** 18528991 (ids 996–999, never started) was cancelled and
resubmitted as **array 18813152**: `--qos=cpu168 --partition=cs
--array=996-999%4`, 32G, 48 h. No p=32 task depends on p=30 any more.

1,000,000 samples at p=32 with `viral_tree_fast` (byte-identical to the frozen
solver; see `FAST_SOLVER_CHANGES.md`). The running p=30 campaign is left
exactly as it is.

## Measured (probe job 18524316, cs623, Xeon Platinum 8592+, one core)

| quantity | value |
|---|---|
| peak RSS | **30.002 GiB** (getrusage; predicted 30.000). sacct MaxRSS 30.06 GiB |
| rate | **73.0 s/sample/core** (71.4–73.4 over 3 samples) |
| start-up | memory touched in 21.8 s; self-check 0.02 s |
| routing | `--mem=32G` goes to partition `cs`; 34G would go to `cl` (queue estimate Oct 2) |

The frozen solver would need 226 GiB per job at p=32, so it could only run on
`cl`, about 25 jobs at a time.

## Memory breakdown per job

| part | size |
|---|---|
| spins of levels 0..28 (2^30 − 2 vertices × 24 B) | 24.000 GiB |
| z-scratch (2 × 2^28 doubles) | 4.000 GiB |
| seed bits (17,179,869,182 bits = 2^28 + 1 words) | 2.000 GiB |
| tables T1/T2, program, libraries, stack | ≈ 2–60 MiB |
| **request** | **32G** (headroom ≈ 1.9 GiB) |

A `cs` node has 128 cores and 513 GB, so up to 15 of these jobs fit on one
node by memory.

## Job shape

| | |
|---|---|
| jobs | **1,000 × 1,000 samples** = 1,000,000 (job ids 0..999, sample k of job j is S = j·2^32 + k) |
| per job | 1 CPU, 1 OpenMP thread, `--mem=32G`, `--time=40:00:00` |
| time per job | median ≈ 21 h at 73–76 s/sample; slow nodes up to ≈ 33 h (the p=28 spread on `cs`: p90/median 1.23, max/median 1.57) |
| total | ≈ 20,300–21,100 core-hours (the frozen solver: ≈ 76,000) |
| binary | `~/viral-exe-backup/fast-builds-18517761/viral_fast` (in `$HOME`, not flushed; sha `bc4230d1…`) |
| angles | `params/angles_p32.txt` (hash `ade916c0630c4c01`) |
| partition | `cs` (184 nodes), routed automatically at 32G |

The solver checkpoints every 60 s, so a job that hits the wall loses at most a
minute and continues with `resume`.

## Decision 2026-09-25: one array, all 1000 jobs on QoS cpu168

The user chose `cpu168` for every job, because its quota is separate from
p=30's. That means:

- **One array:** `--qos=cpu168 --array=0-999%62 --mem=32G --time=48:00:00`.
- **Limits:** `cpu168` allows at most 7 days per job, 1000 CPUs and 2000G per
  user, so 62 jobs run at once.
- **Job size doesn't change the finish date.** The 2000G cap fixes the rate
  at ≈ 62 × 1,180 ≈ 73k samples/day at 73 s/sample, so 1M samples take
  ≈ 14 days whatever the job length. 1,000 × 1,000 (≈ 21 h each) is kept
  because shorter jobs backfill more easily. Pending tasks can still be
  cancelled and resubmitted on `cpu48` later, once p=30 is done, to finish
  faster.
- **Gate:** the R3 replays at p=24/26/28 must pass before the array is
  submitted (≈ 7 h). The p=30 replay (≈ 11 h) finishes afterwards as a
  confirmation; if it fails, the array is cancelled.

Earlier two-array alternative, kept for reference:

## Two arrays, so p=30 is never slowed down

- **QoS caps:** `cpu48` allows 6000G per user, and p=30 uses about 5.6–6.0 TB
  of it until it finishes. `cpu168` has its own 2000G cap.
- **Array A:** QoS `cpu168`, ids **0–399**, `%62` (2000G / 32G). Submit when
  the validation gate passes. At 62 jobs it produces ≈ 70k samples/day and
  finishes in ≈ 5.7 days.
- **Array B:** QoS `cpu48`, ids **400–999**, `%187` (6000G / 32G). Submit only
  after the p=30 array has **no pending tasks left** (≈ Sep 29). From then on
  it only picks up memory that finishing p=30 tasks release, so it can never
  take a slot a p=30 task would have used. At full width it produces
  ≈ 210k samples/day and finishes in ≈ 2.8 days.

**Expected finish:** about Oct 4–5 if the slots are granted; using `cpu168`
alone, about Oct 10. `cs` is ~92% allocated right now, so the slot counts are
ceilings, not promises.

## Validation gate (before array A)

1. **R3 replays** with the exact binary above, on QoS `cpu168` and `cs`. Each
   result file must equal the committed campaign file byte for byte, apart from
   the timing fields:
   - p=24 job 0 (20k samples, ≈ 1.7 h)
   - p=26 job 0 (10k, ≈ 3.4 h)
   - p=28 job 0 (5k, ≈ 6.7 h)
   - p=30 job 1 (2k, ≈ 10.6 h, 8G)
2. **Optional p=32 cross-check**, not gating: 2 samples of the frozen binary
   against the fast one, at p=32 itself. That needs 240G on `cl` under
   `cpu168`, ≈ 15 min of run time, but `cl` queues are days long.

## Tooling needed (new files only; nothing the p=30 tasks execute changes)

- `scripts/viral_campaign.py` pins the frozen source hash and is run by every
  p=30 task at start-up, so it stays untouched. A new
  `scripts/fast_campaign.py` would do the same `init / run / status / verify /
  pending` with the fast source hash, plus `scripts/run_viral_fast.slurm`.
- A build sidecar for the fast binary recording `batch_max_per_thread = 1`,
  the source sha `f1efdb7d…`, and the R1/R3 results.
- Commit on `faster-wt` and fast-forward `faster` to it. This only adds files,
  so the p=30 tree and its uncommitted results are unaffected. The campaign
  then lives in `results/torch-p32-million-v1` of the main checkout, like the
  others.

## Commands (once the above exists)

    python3 scripts/fast_campaign.py init --p 32 --samples 1000 --jobs 1000 \
        --angles params/angles_p32.txt \
        --exe ~/viral-exe-backup/fast-builds-18517761/viral_fast \
        --out results/torch-p32-million-v1
    # array A (now)
    sbatch --account=torch_pr_577_general --qos=cpu168 --array=0-399%62 --mem=32G --time=40:00:00 \
        --output=results/torch-p32-million-v1/logs/slurm_%A_%a.out \
        scripts/run_viral_fast.slurm 32 1000 params/angles_p32.txt EXE results/torch-p32-million-v1
    # array B (after the p=30 array has no pending tasks)
    sbatch ... --qos=cpu48 --array=400-999%187 ...   (same otherwise)
