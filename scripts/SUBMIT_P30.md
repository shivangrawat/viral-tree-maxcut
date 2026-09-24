# p = 30 campaign — drafted 2026-09-19, NOT SUBMITTED

Everything needed to launch p=30 is in place. Nothing has been submitted and no
campaign directory exists yet. Run the commands in "Launch" when you want it.

Before launching, read "Should you run this at all?" at the bottom — for the same
compute, extending p=28 to ~5M samples gives a *better* bound, faster.

## Preconditions

1. **p=28 must be finished.** The `cpu48` QoS allows 6000G of memory per user.
   p=28 holds 200 x 16G = 3200G while it runs, leaving room for only ~46 of the
   100 p=30 jobs. Launching early silently halves concurrency.

       scripts/submit_campaign.sh verify 28      # must report complete

2. **Executable must still be present** (it lives on `/scratch`, which is flushed
   periodically — it is NOT backed up):

       ls -l /scratch/fm2452/viral-controlled/torch-large-builds-17989395/viral_baseline

   Expected sha256 `0f3f8d448e90101120d41008f4de7dc1d541870b127f865f0d48a103bf93a1d0`.
   If it has been flushed, rebuild with
   `sbatch scripts/probe_large_p.slurm spot gcc` and use the new export directory;
   the binary is reproducible but its path changes with the job id.

## Launch

    cd /home/fm2452/projects/flaviano_quantum
    export VIRAL_EXE=/scratch/fm2452/viral-controlled/torch-large-builds-17989395/viral_baseline

    scripts/submit_campaign.sh plan   30     # confirm the row below
    scripts/submit_campaign.sh init   30     # writes results/torch-p30-million-v1/manifest.json
    scripts/submit_campaign.sh submit 30     # sbatch --array=0-499%100 --mem=60G --time=48:00:00

Then, roughly daily:

    scripts/submit_campaign.sh status 30
    scripts/submit_campaign.sh resume 30     # only after the queue drains; see "Stragglers"
    scripts/submit_campaign.sh verify 30     # writes summary.txt when all 500 are complete

## What is being submitted

| | |
|---|---|
| jobs | 500 x 2,000 samples = 1,000,000 (ids 0..499, partition FROZEN) |
| resources | 1 core, 1 OpenMP thread, `--mem=60G`, `--time=48:00:00` |
| account / QoS | `torch_pr_577_general`, QoS `cpu48` auto-assigned, no `--partition` |
| angles | `params/angles_p30.txt`, hash `6f62b5044d2a2a66` |
| binary | `viral_baseline`, sha `0f3f8d44...`, `tested_depths` includes 30 |

Measured, not extrapolated (probe job 17989535, cs614, 1 thread):

| quantity | value | note |
|---|---|---|
| rate | 59.422 s/sample/core | campaign estimate 57.9 after the probe->campaign factor measured at p=28 |
| peak RSS | 56.502 GiB | vs 56.500 GiB predicted workspace; overhead is ~2 MiB, same as p=28 |
| per job | ~32.2 h typical | 2,000 x 57.9 s |
| total | ~16,090 core-hours | 4x the p=28 campaign |
| wall-clock | ~6.7 days | 5 rounds of 100, plus a resume round -> 7-8 days |

`CONC=100` is a ceiling, not a preference: 6000G QoS cap / 60G per job. Do not
raise it. Unlike p=28 (where lifting the throttle to 200 cost only 2.5% in
throughput), there is no headroom here.

## Stragglers — expect them

Projecting p=30 through the p=28 rate distribution measured live on 2026-09-19:

| percentile | p=30 job time |
|---|---|
| 50th | 32.2 h |
| 75th | 37.3 h |
| 90th | 50.5 h  <- past the 48 h wall |
| 100th | 71.0 h |

**~14% of jobs (~70 of 500) are expected to hit the wall.** This is normal and
costs time, not samples: the solver checkpoints every 60 s, so a job killed at
the wall resumes where it stopped. Wait for the queue to drain, then run
`resume 30`, which resubmits only the unfinished ids. Budget one or two resume
rounds.

## Rejected alternatives (do not re-litigate)

- **`cpu168` QoS** (7-day wall, would fix the stragglers): capped at 2000G, so
  only 33 concurrent jobs -> ~20 days. Worse overall.
- **`cpu_prem` partition** (no memory cap, would allow all 500 at once):
  `PreemptMode=REQUEUE` at `PriorityTier=800` vs `cs`'s 1000, so jobs get
  requeued when higher-priority work lands. Checkpointing should make that cost
  ~a minute per event, but that path has never been exercised in this campaign.
  If you want it, pilot it with a handful of jobs first.
- **Smaller samples/job to dodge the wall**: the 500x2000 partition is frozen.
  Sample k of job j is `S=(j<<32)|k`, so re-partitioning draws a different
  sample set and breaks reproducibility against the handoff table.

## Should you run this at all?

For an identical 16,090 core-hours:

| | estimate | 99.99% lower bound |
|---|---|---|
| p=30, 1M samples | x ~ 0.910320 | 0.909256 |
| p=28, 4.95M samples | x ~ 0.910302 | **0.909824** |

Extending p=28 wins by +0.00057 on the number that goes in the paper, and also:

- **girth**: p=28's bound holds for g >= 58, p=30's only for g >= 62 — the deeper
  bound applies to strictly fewer graphs *and* is numerically weaker;
- **wall-clock**: at 16G a job you get 375 concurrent slots instead of 100, so
  ~790 extra jobs finish in ~1.8 days versus 6.7.

Why: depth 28->30 buys about +0.0001 in the estimate, but the Clopper-Pearson
penalty for having only 1M samples is 0.00106 — ten times larger. Every earlier
step was worth more than that penalty; 28->30 is the first that is not. The
campaign has crossed from depth-limited to statistics-limited.

To extend p=28 instead (job ids 200+ cannot collide with the frozen 0..199):

    python3 scripts/viral_campaign.py init --p 28 --samples 5000 --jobs 790 \
        --first-id 200 --angles params/angles_p28.txt --exe "$VIRAL_EXE" \
        --out results/torch-p28-extended-v1
    # then sbatch --array=200-989%375 --mem=16G --time=48:00:00 ... scripts/run_viral.slurm
    # merge_results.py combines both directories: ids are disjoint and angles_hash matches

The counter-argument is presentational: if the paper needs "we ran at depth 30"
as a claim in itself, or a referee will ask why you stopped at 28, that is a real
reason to run it. A cheaper version of that claim is a partial p=30 (e.g. 100k
samples, ~1,600 core-hours) that demonstrates the depth without pretending to a
competitive bound.
