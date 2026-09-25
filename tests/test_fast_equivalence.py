#!/usr/bin/env python3
"""Check viral_tree_fast.c byte for byte against the frozen solver, and time both.

Reference: the frozen campaign executable (--reference-exe) or, by default, a
fresh build of the protected viral_tree_hpc.c with the campaign flags.
Candidate: viral_tree_fast.c built with the same compiler.  Every case compares
the whole per-sample VIRAL_TREE_DUMP file and every byte of the result file
except elapsed_s, core_seconds, sec_per_sample_core and host - the contract of
CODEX_HPC_HANDOFF.md section 1.  Standard library only; run it on a compute
node (the login node has no C toolchain).

    R1:  python3 tests/test_fast_equivalence.py --reference-exe FROZEN_EXE
    R2:  python3 tests/test_fast_equivalence.py --reference-exe FROZEN_EXE --no-suite \\
             --benchmark --bench-depths 20,22,24,26,28 --fast-only-depths 30
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("build_equivalence", ROOT / "tests" / "test_build_equivalence.py")
equiv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(equiv)
campaign = equiv.campaign

FAST_SOURCE = ROOT / "viral_tree_fast.c"
FROZEN_EXE_SHA256 = "0f3f8d448e90101120d41008f4de7dc1d541870b127f865f0d48a103bf93a1d0"
# The frozen build's flags plus -ffp-contract=off.  The frozen binary's FMA
# contraction of rz/rx_cs is written out with explicit fma() calls in
# viral_tree_fast.c; switching the compiler's own contraction off keeps it from
# fusing differently at different sites.
FAST_FLAGS = ["-Wall", "-Wextra", "-DBATCH_MAX_PER_THREAD=1", "-O3", "-ffp-contract=off", "-fopenmp"]
JOBS = (0, 7, 499, 2147483647)
# Samples per benchmark run: roughly a minute of reference time per run on Torch.
BENCH_SAMPLES = {16: 16384, 18: 4096, 20: 1024, 22: 256, 24: 48, 26: 16, 28: 4, 30: 2, 32: 1}
GIB = 1024.0 ** 3


def angles_for(p):
    """The params file whose first p layers are used (odd p read the next even file)."""
    return ROOT / ("params/angles_p%d.txt" % (p + p % 2))


def predicted_bytes(p):
    """Workspace per thread of the reference and the table kernel (formulas of the C files)."""
    nw = (4 * 2 ** p - 2 + 63) // 64
    reference = 24 * (2 * 2 ** p - 2) + 8 * nw + 16 * (2 ** (p - 1) if p >= 2 else 2)
    if p < 4:
        return reference, reference
    return reference, 24 * (2 ** (p - 2) - 2) + 8 * (nw + 1) + 16 * max(2 ** (p - 4), 2)


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def fma_count(exe):
    """Number of FMA instructions in an executable (None without objdump)."""
    if not shutil.which("objdump"):
        return None
    text = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(exe)], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True).stdout
    return sum(1 for line in text.splitlines() if "\tvfmadd" in line or "\tvfmsub" in line
               or "\tvfnmadd" in line or "\tvfnmsub" in line)


def compile_fast(cc, work):
    exe = work / "viral_fast"
    command = [cc] + FAST_FLAGS + [str(FAST_SOURCE), "-lm", "-o", str(exe)]
    built = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if built.returncode:
        raise RuntimeError("fast build failed:\n" + built.stdout)
    if built.stdout.strip():
        print("fast build diagnostics:\n" + built.stdout, flush=True)
    return exe, dict(source=FAST_SOURCE.name, source_sha256=campaign.sha256(FAST_SOURCE),
                     executable_sha256=campaign.sha256(exe), compile_command=command,
                     compiler_version=equiv.capture([cc, "--version"]).strip(),
                     diagnostics=built.stdout, system=platform.system(), machine=platform.machine())


def fresh_reference(cc, work):
    """Build the protected source with the campaign flags; report whether it is the frozen binary."""
    equiv.source_guard()
    exe, info = equiv.compile_build(argparse.Namespace(cc=cc, omp_root=None), work, "baseline")
    info["reproduces_frozen_executable"] = info["executable_sha256"] == FROZEN_EXE_SHA256
    return exe, info


def run(exe, p, n, job, angles, out, dump=False, extra_env=None, timeout=3600):
    """test_build_equivalence.run_solver plus environment overrides; returns the log too."""
    out.mkdir(parents=True, exist_ok=True)
    env = equiv.environment()
    for key in ("VIRAL_FORCE_REF", "VIRAL_SELFCHECK"):
        env.pop(key, None)
    env.update(extra_env or {})
    if dump:
        env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
    begin = time.perf_counter()
    log = equiv.capture([exe, p, n, job, angles, out], cwd=ROOT, env=env, timeout=timeout)
    duration = time.perf_counter() - begin
    data = equiv.result_path(out, p, job).read_bytes()
    fields = campaign.read_fields(data)
    if (fields[b"n_done"], fields[b"n_target"], fields[b"status"], fields[b"threads"]) != (
            str(n).encode(), str(n).encode(), b"complete", b"1"):
        raise AssertionError("%s did not write a complete single-thread result" % Path(exe).name)
    dumped = (out / "samples.hex").read_bytes() if dump else b""
    return campaign.scientific_bytes(data), dumped, duration, log


def suite_cases(max_p):
    cases = []
    for p in range(1, max_p + 1):
        if p <= 12:
            counts, jobs = (1, 65, 513), JOBS
        elif p <= 20:
            counts, jobs = (1, 65), JOBS
        elif p <= 24:
            counts, jobs = (8,), (0, 499)
        else:
            counts, jobs = (4,), (0,)
        cases += [(p, n, job, angles_for(p)) for job in jobs for n in counts]
    # many samples at the smallest table-kernel depths, where every edge case lives
    cases += [(p, 4096, 3, angles_for(p)) for p in (4, 5, 6, 7) if p <= max_p]
    # a longer angles file read only in part, as the existing harness does
    cases += [(p, 129, job, ROOT / "params/angles_p30.txt") for p in (2, 4, 5, 8, 12) for job in (0, 7)]
    return cases


def compare(ref, fast, case, work, tag, fast_env=None):
    p, n, job, angles = case
    label = "p=%d n=%d job=%d %s%s" % (p, n, job, angles.name, " " + tag if tag else "")
    r_res, r_dump, r_t, _ = run(ref, p, n, job, angles, work / "ref", dump=True)
    f_res, f_dump, f_t, f_log = run(fast, p, n, job, angles, work / "fast", dump=True, extra_env=fast_env)
    equiv.equal_bytes(r_res, f_res, label + " result")
    equiv.equal_bytes(r_dump, f_dump, label + " dump")
    shutil.rmtree(work / "ref")
    shutil.rmtree(work / "fast")
    return label, r_t, f_t, f_log


def run_suite(ref, fast, args, work):
    summary = dict(cases=0, depths={})
    for case in suite_cases(args.max_p):
        label, r_t, f_t, log = compare(ref, fast, case, work, "")
        p = case[0]
        if p >= 4 and "identical" not in log:
            raise AssertionError(label + ": the fast binary did not report a passing self-check")
        row = summary["depths"].setdefault(str(p), dict(cases=0, samples=0, reference_s=0.0, fast_s=0.0))
        row["cases"] += 1
        row["samples"] += case[1]
        row["reference_s"] += r_t
        row["fast_s"] += f_t
        summary["cases"] += 1
        print("PASS %-40s reference %7.2f s   fast %7.2f s" % (label, r_t, f_t), flush=True)

    # the reference kernel inside the fast binary
    for p in (4, 9, 14):
        label, _, _, log = compare(ref, fast, (p, 65, 7, angles_for(p)), work, "VIRAL_FORCE_REF=1",
                                   fast_env={"VIRAL_FORCE_REF": "1"})
        if "kernel          reference" not in log:
            raise AssertionError(label + ": VIRAL_FORCE_REF=1 did not select the reference kernel")
        summary["cases"] += 1
        print("PASS " + label, flush=True)

    # a long start-up self-check (its failure would exit non-zero and raise)
    _, _, t, log = run(fast, 14, 1, 0, angles_for(14), work / "selfcheck",
                       extra_env={"VIRAL_SELFCHECK": "5000"})
    if "5000 samples at p=12" not in log or "identical" not in log:
        raise AssertionError("long self-check did not report success:\n" + log[-2000:])
    print("PASS self-check of 5000 samples at p=12 (%.1f s)" % t, flush=True)

    # repeat, extension, SIGTERM and SIGKILL resume: each binary must be
    # self-consistent, and the two protocol results must be identical
    r_fresh, r_dump, _ = equiv.resume_checks(ref, work / "resume_ref")
    f_fresh, f_dump, stops = equiv.resume_checks(fast, work / "resume_fast")
    equiv.equal_bytes(r_fresh, f_fresh, "resume protocol result")
    equiv.equal_bytes(r_dump, f_dump, "resume protocol dump")
    summary["stop_resume"] = stops
    print("PASS repeat / extension / SIGTERM / SIGKILL resume, identical to the reference", flush=True)

    if args.skip_check_sh:
        summary["check_sh"] = "SKIPPED by explicit option"
    else:
        print("Running the unchanged check.sh with the fast binary (4 threads) ...", flush=True)
        summary["check_sh"] = equiv.original_checks(fast, work)
        print(summary["check_sh"], flush=True)
    return summary


def timed(exe, p, n, job, angles, out):
    """One run with wall time, the solver's own s/sample, and the child's peak RSS."""
    out.mkdir(parents=True, exist_ok=True)
    env = equiv.environment()
    for key in ("VIRAL_FORCE_REF", "VIRAL_SELFCHECK"):
        env.pop(key, None)
    log_path = out / "log.txt"
    with log_path.open("w") as log:
        begin = time.perf_counter()
        proc = subprocess.Popen([str(exe), str(p), str(n), str(job), str(angles), str(out)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        wall = time.perf_counter() - begin
    if proc.returncode != 0:
        raise RuntimeError("%s p=%d exited with %d:\n%s" % (Path(exe).name, p, proc.returncode,
                                                           log_path.read_text()[-4000:]))
    data = equiv.result_path(out, p, job).read_bytes()
    fields = campaign.read_fields(data)
    if fields[b"status"] != b"complete" or fields[b"n_done"] != str(n).encode():
        raise AssertionError("%s p=%d did not complete" % (Path(exe).name, p))
    return dict(result=campaign.scientific_bytes(data), wall_s=wall,
                sec_per_sample=float(fields[b"sec_per_sample_core"]),
                peak_rss_bytes=usage.ru_maxrss * 1024)


def run_benchmark(ref, fast, depths, fast_only, reps, work):
    results = {}
    for p in depths + [q for q in fast_only if q not in depths]:
        n = BENCH_SAMPLES.get(p, 1)
        names = ["reference", "fast"] if p in depths else ["fast"]
        runs = {name: [] for name in names}
        expected = None
        for rep in range(reps if p in depths else 1):
            for name in (names if rep % 2 == 0 else names[::-1]):
                out = work / ("p%d_%d_%s" % (p, rep, name))
                r = timed(ref if name == "reference" else fast, p, n, 19, angles_for(p), out)
                if expected is None:
                    expected = r["result"]
                equiv.equal_bytes(expected, r["result"], "benchmark p=%d %s" % (p, name))
                runs[name].append(r)
                shutil.rmtree(out)
        predicted = dict(zip(("reference", "fast"), predicted_bytes(p)))
        row = dict(samples=n, job=19, predicted_workspace_bytes=predicted)
        for name, rs in runs.items():
            row[name] = dict(sec_per_sample=[r["sec_per_sample"] for r in rs],
                             wall_s=[r["wall_s"] for r in rs],
                             median_sec_per_sample=statistics.median(r["sec_per_sample"] for r in rs),
                             median_wall_s=statistics.median(r["wall_s"] for r in rs),
                             peak_rss_bytes=max(r["peak_rss_bytes"] for r in rs))
        text = "p=%d n=%d" % (p, n)
        for name in names:
            text += "   %s %.4f s/sample, peak RSS %.3f GiB (predicted %.3f)" % (
                name, row[name]["median_sec_per_sample"], row[name]["peak_rss_bytes"] / GIB,
                predicted[name] / GIB)
        if "reference" in row:
            row["speedup_sec_per_sample"] = (row["reference"]["median_sec_per_sample"] /
                                             row["fast"]["median_sec_per_sample"])
            row["speedup_wall"] = row["reference"]["median_wall_s"] / row["fast"]["median_wall_s"]
            row["peak_rss_ratio"] = row["reference"]["peak_rss_bytes"] / row["fast"]["peak_rss_bytes"]
            text += "   => %.2fx faster, %.1fx less memory, identical results" % (
                row["speedup_sec_per_sample"], row["peak_rss_ratio"])
        results[str(p)] = row
        print(text, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--reference-exe", type=Path,
                        help="frozen executable to compare against (default: fresh build of the protected source)")
    parser.add_argument("--max-p", type=int, default=26, help="deepest suite depth (reference memory 56.5*2^p)")
    parser.add_argument("--no-suite", action="store_true", help="skip the equivalence suite")
    parser.add_argument("--skip-check-sh", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-depths", default="20,22,24,26,28")
    parser.add_argument("--fast-only-depths", default="", help="depths timed with the fast binary alone")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--report", type=Path, help="new JSON report path")
    parser.add_argument("--keep", type=Path, help="new directory for the fast executable after a passing suite")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("do not run with python -O")
    cc = shutil.which(args.cc)
    if not cc:
        parser.error("compiler not found: " + args.cc)
    if args.report and args.report.exists():
        parser.error("report path already exists")
    if args.keep and args.keep.exists():
        parser.error("keep directory already exists")
    depths = [int(x) for x in args.bench_depths.split(",") if x]
    fast_only = [int(x) for x in args.fast_only_depths.split(",") if x]

    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  cpu=cpu_model(), platform=platform.platform(), status="running",
                  contract="entire dumps and result bytes except elapsed_s, core_seconds, "
                           "sec_per_sample_core, host")
    status = 0
    with tempfile.TemporaryDirectory(prefix="viral-fast-") as directory:
        work = Path(directory)
        try:
            fast, report["fast_build"] = compile_fast(cc, work)
            fresh, report["fresh_reference_build"] = fresh_reference(cc, work)
            if args.reference_exe:
                ref = args.reference_exe.resolve()
                if not os.access(ref, os.X_OK):
                    raise RuntimeError("reference executable not found: %s" % ref)
            else:
                ref = fresh
            ref_sha = campaign.sha256(ref)
            report["reference"] = dict(executable=str(ref), executable_sha256=ref_sha,
                                       is_frozen_campaign_binary=ref_sha == FROZEN_EXE_SHA256,
                                       fma_instructions=fma_count(ref))
            report["fast_build"]["fma_instructions"] = fma_count(fast)
            print("FMA instructions: reference %s, fast %s" % (
                report["reference"]["fma_instructions"], report["fast_build"]["fma_instructions"]), flush=True)
            print("host %s (%s)" % (report["host"], report["cpu"]), flush=True)
            print("reference %s  sha %s  frozen campaign binary: %s" % (
                ref, ref_sha[:16], ref_sha == FROZEN_EXE_SHA256), flush=True)
            print("fresh build of the protected source reproduces the frozen binary: %s" %
                  report["fresh_reference_build"]["reproduces_frozen_executable"], flush=True)
            print("fast build sha %s from source sha %s" % (
                report["fast_build"]["executable_sha256"][:16], report["fast_build"]["source_sha256"][:16]),
                flush=True)
            if not args.no_suite:
                report["suite"] = run_suite(ref, fast, args, work / "suite")
                print("ALL FAST EQUIVALENCE CHECKS PASSED (%d cases)" % report["suite"]["cases"], flush=True)
                if args.keep:
                    args.keep.mkdir(parents=True)
                    shutil.copy2(fast, args.keep / "viral_fast")
                    info = dict(report["fast_build"], equivalence_passed=True,
                                reference=report["reference"], host=report["host"],
                                created_utc=report["created_utc"])
                    (args.keep / "viral_fast.build.json").write_text(json.dumps(info, indent=2) + "\n")
            if args.benchmark:
                report["benchmark"] = run_benchmark(ref, fast, depths, fast_only, args.reps, work / "bench")
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
