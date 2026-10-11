# results/

The raw results behind the paper: 10,000,000 samples of c_VIRAL(p) for every
even depth p = 2..32, in 48 campaign directories (three per depth), plus the
merged numbers.

| path | contents |
|---|---|
| `merge_10M.txt` | the merged result of every depth: the output of `merge_results.py` over the three campaigns of each p, under a short header |
| `torch-pP-million-v1/` | 1,000,000 samples of depth P, job ids 0, 1, … |
| `torch-pP-extension-v1/` | 3,000,000 samples, job ids 1000, 1001, … |
| `torch-pP-extension-v2/` | 6,000,000 samples, job ids 2000, 2001, … |

How each campaign was run, on which nodes and at what cost:
[docs/campaigns.md](../docs/campaigns.md). How the results were checked:
[docs/validation.md](../docs/validation.md).

## Reproduce the merged numbers

```bash
# one depth, in full (as in merge_10M.txt)
python3 merge_results.py results/torch-p32-million-v1/jobs results/torch-p32-extension-v1/jobs \
    results/torch-p32-extension-v2/jobs -p 32
# every depth, as Table I of the paper
python3 merge_results.py --table results/torch-p*-*/jobs          # or: make table
```

`merge_results.py` reads the job files of the directories it is given (not
recursively). It refuses a job id that appears twice for the same depth and a
mix of angles files.

## A campaign directory

```
torch-pP-<campaign>/
  manifest.json                         frozen at init: what the campaign must contain
  provenance.txt                        the build record of the binary (same as manifest "build")
  jobs/viral_tree_pP_jobJ.txt           one result file per job
  logs/provenance_<array>_<task>.jsonl  one record per launch of a job
```

- **`manifest.json`**: `p`, `samples_per_job`, `expected_job_ids`,
  `target_samples`, the angles file (absolute path on the cluster), its sha256
  and the solvers' FNV hash, the executable and its sha256, `build` (source and
  its sha256, compile command, compiler version, validation record),
  `omp_num_threads`, `batch_max_per_thread`, and for the extensions
  `avoided_campaigns` (the earlier campaigns whose job ids it may not reuse).
- **`jobs/`**: the solvers' result files, one `key value` per line:
  `status`, `p`, `job`, `n_target`, `n_done`, `n_cut`, `soft_sum` (Σ s_u^z s_v^z,
  17 digits), the derived `x_thresh`, `stderr`, `x_soft`, then `elapsed_s` and
  `core_seconds` (the cost, accumulated over restarts), `threads`,
  `sec_per_sample_core`, `host`, `angles_file`, `angles_hash`, `alpha`.
  Only `p`, `job`, `n_done`, `n_cut`, `soft_sum`, `status` and `angles_hash` are
  needed to merge. Format details: section 6 of the header of
  `viral_tree_hpc.c`.
- **`logs/provenance_*.jsonl`**: one JSON record per launch, appended by the
  launcher before it starts the solver: time, host, platform, SLURM array and
  task id, the build, angles sha256, p, samples, threads, and the runtime
  linkage (`ldd`). A job that was resumed has more than one record.

The absolute paths in these files (`/home/fm2452/...`, `/scratch/fm2452/...`)
are those of the cluster and are kept as provenance; nothing here should be
edited by hand.

The SLURM stdout logs of the campaigns (`logs/slurm_<array>_<task>.out`,
614 MB: banners, progress lines, self-check results, stop and resume messages)
are not in the current tree. They are in tag `results-10M`
(commit `d578d35`), together with an early local test run (`local_mac/`).
