#!/usr/bin/env python3
"""Check viral_tree_fast2.c (threads inside a sample + fused kernel) byte for byte.

suite : against the frozen campaign binary for every p in 1..max_p, at 1, 4 and
        8 threads (VIRAL_PAR_MIN=1 splits every level, the hardest case for a
        race), plus the forced kernels, threaded resume/extension, the
        protocol checks and check.sh.
large : against the validated viral_tree_fast binary at p=26, 28, 30 with 1, 2,
        4 and 8 threads, with timing and peak RSS.

Every comparison covers the whole per-sample VIRAL_TREE_DUMP and every byte of
the result file except the four timing fields.  The one exception: with T > 1
threads the `threads` line says T, so that line alone is normalized.
Standard library only; needs gcc with -fopenmp and the two reference binaries,
which are identified by their sha256, so in practice it runs on the cluster
that built them (scripts/validation/fast2_validate.slurm, docs/validation.md).
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def _load(name, path):
    """Import a Python file as a module under the given name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fasteq = _load("fast_equivalence", ROOT / "tests" / "test_fast_equivalence.py")
equiv, campaign = fasteq.equiv, fasteq.campaign

SOURCE = ROOT / "viral_tree_fast2.c"
FLAGS = fasteq.FAST_FLAGS                     # frozen flags + -ffp-contract=off
FROZEN_SHA = fasteq.FROZEN_EXE_SHA256
FAST_SHA = "bc4230d144b94ac10fd9c22b38ae588a1d6d222056d9aa6cdb0a7caa30ed64b5"
JOBS = (0, 7, 499, 2147483647)
GIB = 1024.0 ** 3


def compile_fast2(cc, work):
    """Compile viral_tree_fast2.c with FLAGS; returns (executable, build info)."""
    exe = work / "viral_fast2"
    command = [cc] + FLAGS + [str(SOURCE), "-lm", "-o", str(exe)]
    built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if built.returncode:
        raise RuntimeError("fast2 build failed:\n" + built.stdout)
    if built.stdout.strip():
        print("fast2 build diagnostics:\n" + built.stdout, flush=True)
    return exe, dict(source=SOURCE.name, source_sha256=campaign.sha256(SOURCE),
                     executable_sha256=campaign.sha256(exe), compile_command=command,
                     fma_instructions=fasteq.fma_count(exe), diagnostics=built.stdout)


def normalized(result_bytes):
    """Scientific bytes with the threads line set to 1 (the only field that may differ)."""
    return re.sub(rb"(?m)^threads( +)\d+$", rb"threads\g<1>1", result_bytes)


def run(exe, p, n, job, angles, out, threads=1, dump=True, extra_env=None, timeout=7200):
    """Run one job with `threads` threads and require a complete result.

    Returns (scientific bytes, dump bytes, seconds, log, parsed fields).
    """
    out.mkdir(parents=True, exist_ok=True)
    env = equiv.environment()
    for key in ("VIRAL_FORCE_REF", "VIRAL_SELFCHECK", "VIRAL_KERNEL", "VIRAL_PAR_MIN"):
        env.pop(key, None)
    env["OMP_NUM_THREADS"] = str(threads)
    env.update(extra_env or {})
    if dump:
        env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
    begin = time.perf_counter()
    log = equiv.capture([exe, p, n, job, angles, out], cwd=ROOT, env=env, timeout=timeout)
    duration = time.perf_counter() - begin
    data = equiv.result_path(out, p, job).read_bytes()
    fields = campaign.read_fields(data)
    if fields[b"status"] != b"complete" or fields[b"n_done"] != str(n).encode():
        raise AssertionError("%s p=%d did not complete" % (Path(exe).name, p))
    dumped = (out / "samples.hex").read_bytes() if dump else b""
    return campaign.scientific_bytes(data), dumped, duration, log, fields


def check_same(ref, cand, label):
    """Require identical dumps and results (threads line normalized) of two run() outputs."""
    equiv.equal_bytes(normalized(ref[0]), normalized(cand[0]), label + " result")
    equiv.equal_bytes(ref[1], cand[1], label + " dump")


def suite_cases(max_p):
    """Cases (p, n, job, angles) for p = 1..max_p, plus many samples at p = 4..8 and a
    longer angles file read only in part."""
    cases = []
    for p in range(1, max_p + 1):
        if p <= 12:
            counts, jobs = (1, 65, 257), JOBS
        elif p <= 18:
            counts, jobs = (65,), JOBS
        else:
            counts, jobs = (8,), (0, 499)
        cases += [(p, n, job, fasteq.angles_for(p)) for job in jobs for n in counts]
    cases += [(p, 2048, 3, fasteq.angles_for(p)) for p in (4, 5, 6, 7, 8) if p <= max_p]
    cases += [(p, 129, job, ROOT / "params/angles_p30.txt") for p in (2, 5, 6, 8, 12) for job in (0, 7)]
    return cases


def run_suite(ref, fast2, args, work):
    """suite mode: every case at 1, 4 and 8 threads against the frozen binary, the forced
    kernels, a threaded self-check, threaded extension, the stop/resume protocol and
    check.sh.  Returns a summary dict."""
    threads = [t for t in (1, 4, 8) if t <= args.max_threads]
    summary = dict(cases=0, threads=threads, depths={})
    for i, (p, n, job, angles) in enumerate(suite_cases(args.max_p)):
        r = run(ref, p, n, job, angles, work / "ref")
        times = {}
        for t in threads:
            extra = {"VIRAL_PAR_MIN": "1"} if t > 1 else None
            c = run(fast2, p, n, job, angles, work / ("t%d" % t), threads=t, extra_env=extra)
            label = "p=%d n=%d job=%d %s T=%d" % (p, n, job, angles.name, t)
            if p >= 4 and "identical" not in c[3]:
                raise AssertionError(label + ": no passing self-check reported")
            check_same(r, c, label)
            times[t] = c[2]
            shutil.rmtree(work / ("t%d" % t))
        shutil.rmtree(work / "ref")
        row = summary["depths"].setdefault(str(p), dict(cases=0, reference_s=0.0, fast2_s={}))
        row["cases"] += 1
        row["reference_s"] += r[2]
        for t, s in times.items():
            row["fast2_s"][str(t)] = row["fast2_s"].get(str(t), 0.0) + s
        summary["cases"] += 1
        print("PASS p=%-2d n=%-4d job=%-10d %-15s ref %6.2f s  %s" % (
            p, n, job, angles.name, r[2], "  ".join("T=%d %6.2f s" % (t, s) for t, s in times.items())),
            flush=True)

    # forced kernels inside fast2
    for p, kernel in ((9, "ref"), (8, "tables"), (14, "tables"), (14, "fused")):
        angles = fasteq.angles_for(p)
        r = run(ref, p, 65, 7, angles, work / "ref")
        c = run(fast2, p, 65, 7, angles, work / "forced", threads=4,
                extra_env={"VIRAL_KERNEL": kernel, "VIRAL_PAR_MIN": "1"})
        check_same(r, c, "VIRAL_KERNEL=%s p=%d" % (kernel, p))
        shutil.rmtree(work / "ref"); shutil.rmtree(work / "forced")
        print("PASS VIRAL_KERNEL=%s p=%d" % (kernel, p), flush=True)
        summary["cases"] += 1

    # long threaded self-check (fused kernel at p=12, every level split)
    _, _, t, log, _ = run(fast2, 14, 1, 0, fasteq.angles_for(14), work / "sc", threads=8, dump=False,
                          extra_env={"VIRAL_SELFCHECK": "2000", "VIRAL_PAR_MIN": "1"})
    if "2000 samples at p=12, fused kernel (8 threads)" not in log or "identical" not in log:
        raise AssertionError("threaded self-check did not report success:\n" + log[-2000:])
    print("PASS threaded self-check, 2000 samples at p=12 (%.1f s)" % t, flush=True)

    # threaded extension: 31 samples, then extended to 4096 with 4 threads
    p, job, n, angles = 14, 13, 4096, fasteq.angles_for(14)
    fresh = run(ref, p, n, job, angles, work / "ext_ref")
    run(fast2, p, 31, job, angles, work / "ext", threads=4, extra_env={"VIRAL_PAR_MIN": "1"})
    ext = run(fast2, p, n, job, angles, work / "ext", threads=4, extra_env={"VIRAL_PAR_MIN": "1"})
    if "RESUMING" not in ext[3]:
        raise AssertionError("threaded extension did not resume")
    check_same(fresh, ext, "threaded extension p=14")
    print("PASS threaded extension 31 -> 4096 at p=14, identical to the reference", flush=True)

    # the single-thread protocol: repeat, extension, SIGTERM and SIGKILL resume
    r_fresh, r_dump, _ = equiv.resume_checks(ref, work / "resume_ref")
    f_fresh, f_dump, stops = equiv.resume_checks(fast2, work / "resume_fast2")
    equiv.equal_bytes(r_fresh, f_fresh, "resume protocol result")
    equiv.equal_bytes(r_dump, f_dump, "resume protocol dump")
    summary["stop_resume"] = stops
    print("PASS repeat / extension / SIGTERM / SIGKILL resume, identical to the reference", flush=True)

    if not args.skip_check_sh:
        summary["check_sh"] = equiv.original_checks(fast2, work)   # its threads are now in-sample
        print(summary["check_sh"], flush=True)
    return summary


def timed(exe, p, n, job, angles, out, threads):
    """One run with dump: result and dump bytes, wall time, s/sample and the child's peak RSS."""
    out.mkdir(parents=True, exist_ok=True)
    env = equiv.environment()
    env["OMP_NUM_THREADS"] = str(threads)
    env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
    for key in ("VIRAL_FORCE_REF", "VIRAL_SELFCHECK", "VIRAL_KERNEL", "VIRAL_PAR_MIN"):
        env.pop(key, None)
    with (out / "log.txt").open("w") as log:
        begin = time.perf_counter()
        proc = subprocess.Popen([str(exe), str(p), str(n), str(job), str(angles), str(out)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        wall = time.perf_counter() - begin
    if proc.returncode:
        raise RuntimeError("%s p=%d T=%d exited %d:\n%s" % (Path(exe).name, p, threads, proc.returncode,
                                                           (out / "log.txt").read_text()[-3000:]))
    data = equiv.result_path(out, p, job).read_bytes()
    fields = campaign.read_fields(data)
    return dict(result=campaign.scientific_bytes(data), dump=(out / "samples.hex").read_bytes(),
                wall_s=wall, sec_per_sample_wall=float(fields[b"elapsed_s"]) / n,
                peak_rss_bytes=usage.ru_maxrss * 1024)


def run_large(fast, fast2, args, work):
    """large mode: fast2 at each thread count against the validated fast binary at p = 26..30,
    requiring identical results and dumps; returns timings, peak RSS and speedups."""
    depths = [int(x) for x in args.large_depths.split(",") if x]
    samples = {26: 16, 28: 8, 30: 4, 32: 2}
    wanted = [int(x) for x in args.threads.split(",") if x]
    threads = [t for t in wanted if t <= args.max_threads]
    results = {}
    for p in depths:
        n = args.samples or samples.get(p, 2)
        angles = fasteq.angles_for(p)
        base = timed(fast, p, n, 19, angles, work / ("p%d_fast" % p), 1)
        row = dict(samples=n, fast=dict(sec_per_sample=base["sec_per_sample_wall"],
                                        peak_rss_gib=base["peak_rss_bytes"] / GIB), fast2={})
        shutil.rmtree(work / ("p%d_fast" % p))
        for t in threads:
            c = timed(fast2, p, n, 19, angles, work / ("p%d_t%d" % (p, t)), t)
            equiv.equal_bytes(normalized(base["result"]), normalized(c["result"]), "p=%d T=%d result" % (p, t))
            equiv.equal_bytes(base["dump"], c["dump"], "p=%d T=%d dump" % (p, t))
            row["fast2"][str(t)] = dict(sec_per_sample=c["sec_per_sample_wall"],
                                        peak_rss_gib=c["peak_rss_bytes"] / GIB,
                                        speedup_vs_fast=base["sec_per_sample_wall"] / c["sec_per_sample_wall"])
            shutil.rmtree(work / ("p%d_t%d" % (p, t)))
        results[str(p)] = row
        print("p=%d n=%d  fast %.3f s/sample %.3f GiB | %s | identical results and dumps" % (
            p, n, row["fast"]["sec_per_sample"], row["fast"]["peak_rss_gib"],
            "  ".join("T=%s %.3f s (%.2fx) %.3f GiB" % (t, v["sec_per_sample"], v["speedup_vs_fast"],
                                                      v["peak_rss_gib"]) for t, v in row["fast2"].items())),
            flush=True)
    return results


def main():
    """Build fast2, run the chosen mode against the pinned reference binary, write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("suite", "large"))
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--reference-exe", type=Path, required=True, help="frozen campaign binary (suite)")
    parser.add_argument("--fast-exe", type=Path, help="validated viral_tree_fast binary (large)")
    parser.add_argument("--max-p", type=int, default=24)
    parser.add_argument("--max-threads", type=int, default=8)
    parser.add_argument("--large-depths", default="26,28,30")
    parser.add_argument("--threads", default="1,2,4,8", help="thread counts for large mode")
    parser.add_argument("--samples", type=int, default=0, help="samples per run in large mode (0: per-depth default)")
    parser.add_argument("--skip-check-sh", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cc = shutil.which(args.cc)
    if not cc:
        parser.error("compiler not found")
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  cpu=fasteq.cpu_model(), mode=args.mode, status="running")
    status = 0
    with tempfile.TemporaryDirectory(prefix="viral-fast2-") as directory:
        work = Path(directory)
        try:
            fast2, report["fast2_build"] = compile_fast2(cc, work)
            ref = args.reference_exe.resolve()
            report["reference"] = dict(executable=str(ref), sha256=campaign.sha256(ref),
                                       frozen=campaign.sha256(ref) == FROZEN_SHA)
            print("host %s (%s); fast2 sha %s from source %s; FMA instructions %s" % (
                report["host"], report["cpu"], report["fast2_build"]["executable_sha256"][:16],
                report["fast2_build"]["source_sha256"][:16], report["fast2_build"]["fma_instructions"]), flush=True)
            if args.mode == "suite":
                if not report["reference"]["frozen"]:
                    raise RuntimeError("the reference is not the frozen campaign binary")
                report["suite"] = run_suite(ref, fast2, args, work / "suite")
                print("ALL FAST2 EQUIVALENCE CHECKS PASSED (%d cases)" % report["suite"]["cases"], flush=True)
            else:
                fast = args.fast_exe.resolve()
                if campaign.sha256(fast) != FAST_SHA:
                    raise RuntimeError("the fast binary is not the validated build")
                report["large"] = run_large(fast, fast2, args, work / "large")
                print("ALL FAST2 LARGE-p CHECKS PASSED", flush=True)
            report["status"] = "passed"
        except subprocess.CalledProcessError as error:
            report.update(status="failed", error="%s\n%s" % (error, error.output))
            print("FAILED: %s\n%s" % (error, error.output), flush=True)
            status = 1
        except (AssertionError, RuntimeError, ValueError, OSError) as error:
            report.update(status="failed", error=str(error))
            print("FAILED: %s" % error, flush=True)
            status = 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
