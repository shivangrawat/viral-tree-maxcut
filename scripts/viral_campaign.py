#!/usr/bin/env python3
"""Freeze, inspect, and launch one-process VIRAL campaigns (standard library only).

A campaign is a directory OUT with manifest.json (depth, samples per job, the
expected job ids, the angles and the validated executable with its build
sidecar), provenance.txt, jobs/ (the solver's result files) and logs/ (one
provenance record per launch).  Commands:

  init    --p P --samples N --jobs J [--first-id F] --exe EXE [--angles A] --out OUT
          freeze a new campaign; OUT must be new or empty
  run     --p P --samples N --exe EXE [--angles A] --out OUT
          launch SLURM array task $SLURM_ARRAY_TASK_ID as job id: check it against
          the manifest, append a provenance record, exec the solver (1 CPU, 1 thread)
  status  OUT    print the campaign state as JSON
  verify  OUT    the same, and exit 1 unless every job is complete and valid
  pending OUT    print the job ids still to run, comma-separated (for resubmission)

EXE needs a .build.json sidecar from a validated build (tests/test_build_equivalence.py
--export-builds, scripts/validation/).  load_campaign() re-reads the absolute paths
frozen in the manifest, so status/verify/pending of the published campaigns work on
the original cluster only; merge_results.py works anywhere.  fast_campaign.py,
fast2_campaign.py, the tests and some validation gates import this module.  The
published campaigns and their commands: docs/campaigns.md.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


SOURCE_SHA256 = "b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51"
# The solver sources of the campaigns (tag results-10M) were later documented
# with comment-only edits.  Each documented file maps to the campaign source
# it compiles like; tests/check_sources.sh proves the equivalence.
DOCUMENTED_SOURCE_SHA256 = {
    "b18dab868846b23b43662aed1b2c462ac20557e9aa95fb3168652f31494280f2":    # viral_tree_hpc.c
        "b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51",
    "178539f113c5a0fecc59c8711732ed45385097028f0ae3bdbca27809a6d5eed8":    # viral_tree_fast.c
        "f1efdb7dd443bef0a043791a5232678c9659bb2c3be6fab6c959698541b034f8",
    "7abb4e08eae5413fb524556ef2ee9a182aaa842677520e71c50e36020f07d803":    # viral_tree_fast2.c
        "8b12711a2b6f15e652a7677c43b80aa6715570ffd28b38d9ec4afdab97251988",
}
VOLATILE = {b"elapsed_s", b"core_seconds", b"sec_per_sample_core", b"host"}
FIELDS = set(b"status p job n_target n_done n_cut soft_sum x_thresh stderr x_soft "
             b"elapsed_s core_seconds threads sec_per_sample_core host angles_file "
             b"angles_hash alpha".split())


def sha256(path):
    """Hex sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angles_hash(path):
    """64-bit FNV-1a hash of a file as 16 hex digits: the solvers' angles_hash."""
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return "%016x" % value


def read_fields(data):
    """Validate the schema; values remain bytes, including every float spelling."""
    fields = {}
    for line in data.splitlines():
        if not line or line.startswith(b"#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or parts[0] in fields:
            raise ValueError("malformed or duplicate result field")
        fields[parts[0]] = parts[1]
    if set(fields) != FIELDS:
        raise ValueError("result fields differ from the protected source schema")
    return fields


def scientific_bytes(data):
    """A result file without its VOLATILE lines (timing, host): the bytes that must reproduce."""
    read_fields(data)
    return b"".join(line for line in data.splitlines(keepends=True)
                    if not line.split() or line.split()[0] not in VOLATILE)


def campaign_source_sha(sha):
    """Campaign source hash of a source file hash; documented copies map to their campaign source."""
    return DOCUMENTED_SOURCE_SHA256.get(sha, sha)


def build_info(executable):
    """Load and check the .build.json sidecar of a validated build of the campaign source.

    Raises ValueError unless the sidecar names the campaign source (or its documented
    copy), matches the executable's sha256, records -DBATCH_MAX_PER_THREAD=1 and a passed
    equivalence check, and was built for this OS and machine.  Returns the sidecar dict.
    """
    executable = Path(executable).resolve()
    info = json.loads(Path(str(executable) + ".build.json").read_text())
    if campaign_source_sha(info["source_sha256"]) != SOURCE_SHA256:
        raise ValueError("build used a different protected source")
    if info["executable_sha256"] != sha256(executable):
        raise ValueError("executable differs from its build sidecar")
    if info.get("batch_max_per_thread") != 1 or not info.get("equivalence_passed"):
        raise ValueError("build has not passed the controlled equivalence checks")
    if info["system"] != platform.system() or info["machine"] != platform.machine():
        raise ValueError("build platform differs from this machine")
    return info


def paths(args):
    """Resolve (angles, executable, out); refuse paths the solver's fixed-size buffers cannot hold."""
    angles = (Path(args.angles) if args.angles else
              Path(__file__).resolve().parents[1] / "params" / ("angles_p%d.txt" % args.p))
    if not args.angles and (args.p % 2 or not 2 <= args.p <= 30):
        raise ValueError("this depth requires an explicit --angles file")
    angles, executable, out = angles.resolve(), Path(args.exe).resolve(), Path(args.out).resolve()
    # The unchanged C program uses fixed-size path buffers and whitespace parsing.
    if len(os.fsencode(angles)) >= 512 or any(c.isspace() for c in str(angles)):
        raise ValueError("angles path must be shorter than 512 bytes and contain no whitespace")
    if len(os.fsencode(out / "jobs" / ("viral_tree_p%d_job2147483647.txt" % args.p))) >= 1100:
        raise ValueError("output path exceeds the solver's buffer")
    return angles, executable, out


def initialize(args):
    """init: freeze a new campaign in args.out (manifest.json, provenance.txt, jobs/, logs/)."""
    angles, executable, out = paths(args)
    info = build_info(executable)
    if not 1 <= args.p <= 40 or not 1 <= args.samples <= 4294967295:
        raise ValueError("p must be 1..40 and samples must be 1..4294967295")
    if args.first_id < 0 or args.jobs < 1 or args.first_id + args.jobs - 1 > 2147483647:
        raise ValueError("invalid job ID range")
    # Refuse to adopt old results or silently replace a frozen campaign.
    if out.exists() and any(out.iterdir()):
        raise ValueError("campaign directory must be new or empty")
    manifest = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    p=args.p, samples_per_job=args.samples,
                    expected_job_ids=list(range(args.first_id, args.first_id + args.jobs)),
                    target_samples=args.jobs * args.samples,
                    angles_file=str(angles), angles_sha256=sha256(angles),
                    angles_hash=angles_hash(angles), executable=str(executable),
                    executable_sha256=sha256(executable), build=info,
                    omp_num_threads=1, batch_max_per_thread=1)
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "provenance.txt").write_text(
        "Frozen campaign build; per-invocation provenance is in logs/provenance_*.jsonl.\n" +
        json.dumps(info, indent=2) + "\n")
    print("Initialized %s: p=%d, %d jobs x %d = %d samples" %
          (out, args.p, args.jobs, args.samples, manifest["target_samples"]))


def load_campaign(out):
    """Read out/manifest.json and check that its angles file and build are still the frozen ones."""
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != 1:
        raise ValueError("unknown campaign schema")
    if sha256(manifest["angles_file"]) != manifest["angles_sha256"]:
        raise ValueError("campaign angles have changed")
    info = build_info(manifest["executable"])
    if info != manifest["build"] or info["executable_sha256"] != manifest["executable_sha256"]:
        raise ValueError("campaign build identity has changed")
    return manifest


def inspect_campaign(out):
    """status / verify / pending: check every file in out/jobs against the manifest.

    Returns a dict with the expected and present job counts, the missing and
    incomplete ids, samples done and targeted, per-file errors and `complete`.
    """
    manifest = load_campaign(out)
    p, n = manifest["p"], manifest["samples_per_job"]
    expected = set(manifest["expected_job_ids"])
    present, incomplete, total, errors = set(), [], 0, []
    for path in sorted((out / "jobs").glob("*.txt")):
        try:
            f = read_fields(path.read_bytes())
            job, depth = int(f[b"job"]), int(f[b"p"])
            if job not in expected or depth != p or job in present:
                raise ValueError("unexpected or duplicate job/depth")
            if path.name != "viral_tree_p%d_job%d.txt" % (p, job):
                raise ValueError("filename does not match job/depth")
            if f[b"angles_hash"].decode() != manifest["angles_hash"]:
                raise ValueError("different angles hash")
            if f[b"angles_file"].decode() != manifest["angles_file"] or f[b"threads"] != b"1":
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
    return dict(p=p, expected_jobs=len(expected), present_jobs=len(present),
                missing_ids=missing, incomplete_ids=sorted(incomplete),
                samples_done=total, target_samples=manifest["target_samples"], errors=errors,
                complete=not missing and not incomplete and not errors and total == manifest["target_samples"])


def launch(args):
    """run: check this array task against the manifest, log provenance and exec the solver (no return)."""
    angles, executable, out = paths(args)
    m = load_campaign(out)
    job = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if (args.p, args.samples, str(angles), str(executable)) != (
            m["p"], m["samples_per_job"], m["angles_file"], m["executable"]):
        raise ValueError("launch arguments differ from frozen manifest")
    if job not in m["expected_job_ids"]:
        raise ValueError("array task ID is outside the campaign")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("SLURM_CPUS_PER_TASK", "1") != "1":
        raise ValueError("the controlled build requires one CPU and one OpenMP thread")
    array = os.environ["SLURM_ARRAY_JOB_ID"]
    if not array.isdigit():
        raise ValueError("invalid array submission ID")
    # Append rather than replace: automatic requeues can reuse the same array ID.
    record = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  platform=platform.platform(), array_id=array, task_id=job,
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), build=m["build"],
                  angles_sha256=m["angles_sha256"], p=args.p, samples=args.samples,
                  omp_num_threads=1)
    linkage = "otool" if sys.platform == "darwin" else "ldd"
    command = [linkage] + (["-L"] if linkage == "otool" else []) + [str(executable)]
    record["runtime_linkage"] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
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
            p.add_argument("--first-id", type=int, default=0)
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
