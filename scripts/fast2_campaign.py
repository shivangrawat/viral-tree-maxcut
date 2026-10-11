#!/usr/bin/env python3
"""Freeze, inspect and launch VIRAL campaigns run with viral_tree_fast2 (threads per job).

Same commands, manifest and checks as viral_campaign.py (init / run / status /
verify / pending), with two differences:

  * the build sidecar must name viral_tree_fast2.c as its source (byte-identical
    to viral_tree_hpc.c: docs/validation.md);
  * a campaign has a fixed number of threads per job, recorded in the manifest
    (--threads at init); every task must run with exactly that many CPUs and
    OpenMP threads, and every result file must say so.

`init --avoid DIR ...` refuses to create a campaign whose job ids overlap the
expected ids of any listed campaign of the same depth and angles: that is
what guarantees new samples (sample k of job j is S = j*2^32 + k).

Used for all extension campaigns, results/torch-pP-extension-v1 and -v2
(docs/campaigns.md).  viral_campaign.py is loaded as a module for its helpers,
with its SOURCE_SHA256 replaced; init, inspection and launch are redefined
here for the thread count.  Linux only (the launch record runs ldd).
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

_spec = importlib.util.spec_from_file_location("viral_campaign", Path(__file__).with_name("viral_campaign.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

# sha256 of the validated viral_tree_fast2.c (suite 18986377, large 18986379, scaling 19053518/9)
FAST2_SOURCE_SHA256 = "8b12711a2b6f15e652a7677c43b80aa6715570ffd28b38d9ec4afdab97251988"
vc.SOURCE_SHA256 = FAST2_SOURCE_SHA256


def threads_of(manifest):
    """Thread count frozen in a fast2 manifest; ValueError if it is missing or invalid."""
    t = manifest.get("omp_num_threads")
    if not isinstance(t, int) or t < 1:
        raise ValueError("manifest has no valid thread count")
    return t


def check_avoid(p, angles_sha, ids, avoid_dirs):
    """Refuse overlap with the expected ids of earlier campaigns of the same p and angles."""
    new = set(ids)
    for d in avoid_dirs:
        m = json.loads((Path(d) / "manifest.json").read_text())
        if m["p"] != p:
            raise ValueError("--avoid %s is a p=%d campaign, not p=%d" % (d, m["p"], p))
        if m["angles_sha256"] != angles_sha:
            raise ValueError("--avoid %s used different angles" % d)
        clash = new & set(m["expected_job_ids"])
        if clash:
            raise ValueError("job ids overlap %s: %s" % (d, sorted(clash)[:10]))
        # also any result file actually present there (whatever its id)
        for path in (Path(d) / "jobs").glob("viral_tree_p%d_job*.txt" % p):
            job = int(path.stem.rsplit("job", 1)[1])
            if job in new:
                raise ValueError("job id %d already has a result in %s" % (job, d))


def initialize(args):
    """init: as viral_campaign.initialize, plus --threads (frozen) and the --avoid id-overlap guard."""
    angles, executable, out = vc.paths(args)
    info = vc.build_info(executable)
    if not 1 <= args.p <= 40 or not 1 <= args.samples <= 4294967295:
        raise ValueError("p must be 1..40 and samples must be 1..4294967295")
    if args.first_id < 0 or args.jobs < 1 or args.first_id + args.jobs - 1 > 2147483647:
        raise ValueError("invalid job ID range")
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    if out.exists() and any(out.iterdir()):
        raise ValueError("campaign directory must be new or empty")
    ids = list(range(args.first_id, args.first_id + args.jobs))
    angles_sha = vc.sha256(angles)
    if not args.avoid:
        raise ValueError("give --avoid with every earlier campaign of this depth (or --avoid none)")
    if args.avoid != ["none"]:
        check_avoid(args.p, angles_sha, ids, args.avoid)
    manifest = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    p=args.p, samples_per_job=args.samples, expected_job_ids=ids,
                    target_samples=args.jobs * args.samples,
                    angles_file=str(angles), angles_sha256=angles_sha,
                    angles_hash=vc.angles_hash(angles), executable=str(executable),
                    executable_sha256=vc.sha256(executable), build=info,
                    omp_num_threads=args.threads, batch_max_per_thread=1,
                    avoided_campaigns=[str(Path(d).resolve()) for d in args.avoid if d != "none"])
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "provenance.txt").write_text(
        "Frozen fast2 campaign build; per-invocation provenance is in logs/provenance_*.jsonl.\n" +
        json.dumps(info, indent=2) + "\n")
    print("Initialized %s: p=%d, %d jobs (ids %d..%d) x %d = %d samples, %d threads/job" %
          (out, args.p, args.jobs, ids[0], ids[-1], args.samples, manifest["target_samples"], args.threads))


def inspect_campaign(out):
    """status / verify / pending: as viral_campaign.inspect_campaign, with the frozen thread count."""
    manifest = vc.load_campaign(out)
    p, n, t = manifest["p"], manifest["samples_per_job"], threads_of(manifest)
    expected = set(manifest["expected_job_ids"])
    present, incomplete, total, errors = set(), [], 0, []
    for path in sorted((out / "jobs").glob("*.txt")):
        try:
            f = vc.read_fields(path.read_bytes())
            job, depth = int(f[b"job"]), int(f[b"p"])
            if job not in expected or depth != p or job in present:
                raise ValueError("unexpected or duplicate job/depth")
            if path.name != "viral_tree_p%d_job%d.txt" % (p, job):
                raise ValueError("filename does not match job/depth")
            if f[b"angles_hash"].decode() != manifest["angles_hash"]:
                raise ValueError("different angles hash")
            if f[b"angles_file"].decode() != manifest["angles_file"] or f[b"threads"] != str(t).encode():
                raise ValueError("different angles path or thread count")
            done, cut = int(f[b"n_done"]), int(f[b"n_cut"])
            if int(f[b"n_target"]) != n or not 0 <= cut <= done <= n:
                raise ValueError("wrong target or invalid sample counts")
            if f[b"status"] not in (b"complete", b"running"):
                raise ValueError("invalid status")
            if f[b"status"] == b"complete" and done != n:
                raise ValueError("complete status with wrong sample count")
            present.add(job)
            total += done
            if done != n or f[b"status"] != b"complete":
                incomplete.append(job)
        except (ValueError, KeyError) as error:
            errors.append("%s: %s" % (path.name, error))
    missing = sorted(expected - present)
    return dict(p=p, threads=t, expected_jobs=len(expected), present_jobs=len(present),
                missing_ids=missing, incomplete_ids=sorted(incomplete),
                samples_done=total, target_samples=manifest["target_samples"], errors=errors,
                complete=not missing and not incomplete and not errors and total == manifest["target_samples"])


def launch(args):
    """run: as viral_campaign.launch, with exactly the frozen number of CPUs and OpenMP threads."""
    angles, executable, out = vc.paths(args)
    m = vc.load_campaign(out)
    t = threads_of(m)
    job = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if (args.p, args.samples, str(angles), str(executable)) != (
            m["p"], m["samples_per_job"], m["angles_file"], m["executable"]):
        raise ValueError("launch arguments differ from frozen manifest")
    if job not in m["expected_job_ids"]:
        raise ValueError("array task ID is outside the campaign")
    if os.environ.get("OMP_NUM_THREADS") != str(t) or os.environ.get("SLURM_CPUS_PER_TASK") != str(t):
        raise ValueError("this campaign requires exactly %d CPUs and %d OpenMP threads per task" % (t, t))
    array = os.environ["SLURM_ARRAY_JOB_ID"]
    if not array.isdigit():
        raise ValueError("invalid array submission ID")
    record = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  platform=platform.platform(), array_id=array, task_id=job,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), build=m["build"],
                  angles_sha256=m["angles_sha256"], p=args.p, samples=args.samples,
                  omp_num_threads=t)
    record["runtime_linkage"] = subprocess.check_output(["ldd", str(executable)], text=True,
                                                        stderr=subprocess.STDOUT)
    with (out / "logs" / ("provenance_%s_%d.jsonl" % (array, job))).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    os.execv(str(executable), [str(executable), str(args.p), str(args.samples), str(job),
                             str(angles), str(out / "jobs")])


def main():
    """Command-line entry point; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "run"):
        p = sub.add_parser(name)
        p.add_argument("--p", type=int, required=True)
        p.add_argument("--samples", type=int, required=True)
        p.add_argument("--angles")
        p.add_argument("--exe", required=True, help="validated executable, with adjacent .build.json")
        p.add_argument("--out", required=True)
        if name == "init":
            p.add_argument("--jobs", type=int, required=True)
            p.add_argument("--first-id", type=int, required=True)
            p.add_argument("--threads", type=int, required=True)
            p.add_argument("--avoid", nargs="+", default=[],
                           help="earlier campaign directories of this depth whose ids must not be reused, or 'none'")
    for name in ("status", "verify", "pending"):
        sub.add_parser(name).add_argument("out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args)
        elif args.command == "run":
            launch(args)
        else:
            state = inspect_campaign(args.out.resolve())
            if args.command == "pending":
                if state["errors"]:
                    raise ValueError("; ".join(state["errors"]))
                print(",".join(map(str, sorted(state["missing_ids"] + state["incomplete_ids"]))))
            else:
                print(json.dumps(state, indent=2))
            return 1 if state["errors"] or (args.command == "verify" and not state["complete"]) else 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
