#!/usr/bin/env python3
"""Build the unchanged solver, require exact bytes, and optionally benchmark.

Compiles viral_tree_hpc.c with the campaign flags under each tuning profile
and requires every build to reproduce the baseline build byte for byte: whole
per-sample dumps and result files for p = 1..16, repeats, extension,
SIGTERM/SIGKILL resume, the campaign wrapper (scripts/run_viral.slurm with fake
SLURM variables), merge_results.py and check.sh.  This is the suite behind the
campaign build of viral_tree_hpc (scripts/validation/torch_build_validate.slurm).

    python3 tests/test_build_equivalence.py                         (Linux, gcc)
    python3 tests/test_build_equivalence.py --omp-root <libomp prefix>   (macOS)

No third-party Python packages. Default tests stop at p=16. All solver builds,
results, dumps, and original self-test fixtures live in a temporary directory.
Use --export-builds NEW_DIR to retain validated binaries and build sidecars.
"""

import argparse
from datetime import datetime, timezone
import difflib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("campaign", ROOT / "scripts" / "viral_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
SOURCE = ROOT / "viral_tree_hpc.c"
SOURCE_SHA256 = "b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51"
PROFILES = {
    "baseline": ["-O3"],
    "O2": ["-O2"],
    "native_tune": ["-O3", "-mtune=native"],
    "native_arch": ["-O3", "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"],
}


def source_guard(path=SOURCE):
    """Refuse to test a viral_tree_hpc.c that is neither the campaign source nor its documented copy."""
    actual = campaign.sha256(path)
    if campaign.campaign_source_sha(actual) != SOURCE_SHA256:
        raise ValueError("Protected source changed: expected %s, found %s" % (SOURCE_SHA256, actual))


def equal_bytes(expected, actual, label):
    """Raise AssertionError, with a unified diff, unless the two byte strings are identical."""
    if expected != actual:
        diff = "".join(difflib.unified_diff(expected.decode().splitlines(True),
                                          actual.decode().splitlines(True),
                                          fromfile="reference", tofile="candidate"))
        raise AssertionError("%s: byte mismatch\n%s" % (label, diff[:4000]))


def environment():
    """Environment of a strict run: one OpenMP thread, no dump file, C locale."""
    env = os.environ.copy()
    env.pop("VIRAL_TREE_DUMP", None)
    env.update(OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OMP_PROC_BIND="FALSE", LC_ALL="C")
    return env


def capture(command, **kwargs):
    """Run a command; return its combined stdout and stderr (raise if it fails)."""
    return subprocess.check_output([str(x) for x in command], stderr=subprocess.STDOUT,
                                   text=True, **kwargs)


def compile_build(args, work, profile):
    """Compile viral_tree_hpc.c with the campaign flags plus one tuning profile.

    Returns (executable, sidecar dict), or (None, diagnostic) when a non-baseline
    profile is not supported by the compiler.  On macOS the binary is relinked to
    the libomp of --omp-root and re-signed.
    """
    exe = work / ("viral_" + profile)
    flags = ["-Wall", "-Wextra", "-DBATCH_MAX_PER_THREAD=1"] + PROFILES[profile]
    if args.omp_root:
        omp = args.omp_root.resolve()
        flags += ["-Xpreprocessor", "-fopenmp", "-I" + str(omp / "include"),
                  "-L" + str(omp / "lib"), "-lomp", "-Wl,-headerpad_max_install_names"]
    else:
        flags += ["-fopenmp"]
    command = [args.cc] + flags + [str(SOURCE), "-lm", "-o", str(exe)]
    compiled = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    unsupported = re.search(r"unknown (?:argument|option)|unsupported (?:argument|option)|"
                            r"unrecognized command-line option|argument unused|not supported|"
                            r"unknown target CPU|bad value .* switch", compiled.stdout, re.I)
    if compiled.returncode or unsupported:
        if profile != "baseline" and unsupported:
            return None, dict(status="unsupported", flags=flags, diagnostic=compiled.stdout)
        raise RuntimeError("compile failed for %s:\n%s" % (profile, compiled.stdout))
    if args.omp_root:
        dylib = args.omp_root.resolve() / "lib" / "libomp.dylib"
        install_id = capture(["otool", "-D", dylib]).splitlines()[1].strip()
        capture(["install_name_tool", "-change", install_id, dylib, exe])
        capture(["codesign", "--force", "--sign", "-", exe])
    linkage = capture((["otool", "-L"] if sys.platform == "darwin" else ["ldd"]) + [exe])
    info = dict(profile=profile, source_sha256=campaign.sha256(SOURCE), executable_sha256=campaign.sha256(exe),
                compiler=args.cc, compiler_version=capture([args.cc, "--version"]).strip(),
                compile_command=[str(x) for x in command], flags=flags, runtime_linkage=linkage,
                system=platform.system(), machine=platform.machine(), platform=platform.platform(),
                created_utc=datetime.now(timezone.utc).isoformat(), batch_max_per_thread=1,
                omp_num_threads=1, equivalence_passed=False)
    if args.omp_root:
        info["openmp_library_sha256"] = campaign.sha256(dylib)
        info["openmp_library"] = str(dylib)
    return exe, info


def result_path(out, p, job):
    """Path of the result file of (p, job) in out."""
    return out / ("viral_tree_p%d_job%d.txt" % (p, job))


def run_solver(exe, p, samples, job, angles, out, dump=False, timeout=300):
    """Run one job single-threaded and require a complete result.

    Returns (scientific bytes of the result, per-sample dump or b"", seconds, log).
    """
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    if dump:
        env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
    begin = time.perf_counter()
    log = capture([exe, p, samples, job, angles, out], cwd=ROOT, env=env, timeout=timeout)
    duration = time.perf_counter() - begin
    data = result_path(out, p, job).read_bytes()
    fields = campaign.read_fields(data)
    if (fields[b"n_done"], fields[b"n_target"], fields[b"status"], fields[b"threads"]) != (
            str(samples).encode(), str(samples).encode(), b"complete", b"1"):
        raise AssertionError("solver did not write a complete single-thread result")
    return campaign.scientific_bytes(data), (out / "samples.hex").read_bytes() if dump else b"", duration, log


def test_cases(work, args):
    """The strict cases (p, n, job, angles): p = 1 and even p <= 16, jobs 0, 7, 499 and
    2^31-1, the p=30 angles file as a prefix for small p, and any opt-in spot depths."""
    p1 = work / "angles_p1.txt"
    # An explicit one-layer fixture using the supplied p=2 alpha and first row.
    p1.write_bytes(b"\n".join((ROOT / "params/angles_p2.txt").read_bytes().splitlines()[:4]) + b"\n")
    cases = []
    for p in [1] + list(range(2, 17, 2)):
        angles = p1 if p == 1 else ROOT / ("params/angles_p%d.txt" % p)
        for job in (0, 7, 499):
            for n in (1, 513 if p <= 8 else 65):
                cases.append((p, n, job, angles))
    for p in (2, 4, 8):
        for job in (0, 7, 499):
            cases.append((p, 129, job, ROOT / "params/angles_p30.txt"))
    cases.append((2, 129, 2147483647, ROOT / "params/angles_p2.txt"))
    for p in args.spot_depths:
        for job in (0, 499):
            cases.append((p, args.spot_samples, job, ROOT / ("params/angles_p%d.txt" % p)))
    return cases


def strict_cases(exe, work, cases, expected=None):
    """Run every case with the per-sample dump; with `expected`, require identical bytes.

    Returns the observed (result, dump) pairs.
    """
    observed = []
    for i, (p, n, job, angles) in enumerate(cases):
        result, dump, _, _ = run_solver(exe, p, n, job, angles, work / str(i), dump=True)
        rows = dump.splitlines()
        if len(rows) != n or [int(row.split()[0]) for row in rows] != list(range(n)):
            raise AssertionError("incomplete or out-of-order per-sample dump")
        if expected is not None:
            equal_bytes(expected[i][0], result, "p=%d job=%d n=%d result" % (p, job, n))
            equal_bytes(expected[i][1], dump, "p=%d job=%d n=%d dump" % (p, job, n))
        observed.append((result, dump))
    return observed


def resume_checks(exe, work):
    """Repeat, extend, rerun-complete, SIGTERM and SIGKILL resume at p=12.

    Every path must end byte-identical to a fresh run.  Returns (fresh result,
    fresh dump, interruption records).
    """
    p, job, n = 12, 13, 4096
    angles = ROOT / "params/angles_p12.txt"
    fresh, dump, _, _ = run_solver(exe, p, n, job, angles, work / "fresh", dump=True)
    repeat, repeat_dump, _, _ = run_solver(exe, p, n, job, angles, work / "repeat", dump=True)
    equal_bytes(fresh, repeat, "repeated result")
    equal_bytes(dump, repeat_dump, "repeated dump")
    run_solver(exe, p, 31, job, angles, work / "extend", dump=True)
    extended, extended_dump, _, log = run_solver(exe, p, n, job, angles, work / "extend", dump=True)
    assert "RESUMING" in log
    equal_bytes(fresh, extended, "extended result")
    equal_bytes(dump, extended_dump, "extended dump")
    before = result_path(work / "extend", p, job).read_bytes()
    _, _, _, log = run_solver(exe, p, n, job, angles, work / "extend", dump=True)
    assert "already complete" in log
    equal_bytes(before, result_path(work / "extend", p, job).read_bytes(), "completed rerun")
    records = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        out = work / sig.name
        # A real committed prefix makes forced-stop rollback nontrivial without
        # shortening the production checkpoint interval or editing checkpoint data.
        run_solver(exe, p, 31, job, angles, out, dump=(sig == signal.SIGTERM))
        env = environment()
        if sig == signal.SIGTERM:
            env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
        with (out / "interruption.log").open("w") as log_file:
            process = subprocess.Popen([str(exe), str(p), str(n), str(job), str(angles), str(out)],
                                       env=env, stdout=log_file, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 20
                while True:
                    fields = campaign.read_fields(result_path(out, p, job).read_bytes())
                    if fields[b"status"] == b"running" and fields[b"n_target"] == str(n).encode():
                        break
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise AssertionError("did not observe a running checkpoint before stopping")
                    time.sleep(0.001)
                time.sleep(0.01)
                process.send_signal(sig)
                rc = process.wait(timeout=30)
                assert rc == (0 if sig == signal.SIGTERM else -signal.SIGKILL), rc
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        fields = campaign.read_fields(result_path(out, p, job).read_bytes())
        done = int(fields[b"n_done"])
        assert fields[b"status"] == b"running" and 31 <= done < n
        if sig == signal.SIGTERM:
            assert done > 31 and "STOPPED BY SIGNAL" in (out / "interruption.log").read_text()
        else:
            assert done == 31, "SIGKILL test must roll back to the saved prefix"
        resumed, resumed_dump, _, log = run_solver(exe, p, n, job, angles, out, dump=(sig == signal.SIGTERM))
        assert "RESUMING" in log
        equal_bytes(fresh, resumed, sig.name + " resumed result")
        if sig == signal.SIGTERM:
            equal_bytes(dump, resumed_dump, "SIGTERM resumed dump")
        records.append(dict(signal=sig.name, saved_samples=done, final_samples=n))
    return fresh, dump, records


def campaign_checks(exe, info, work):
    """End-to-end campaign path on a tiny campaign; returns the merge report.

    init, two tasks of scripts/run_viral.slurm with fake SLURM variables (each equal
    to a direct run), verify, merge_results.py, and the verifier's detection of a
    missing, an altered and a duplicate file and of a wrong CPU count.
    """
    sidecar = Path(str(exe) + ".build.json")
    sidecar.write_text(json.dumps(info, indent=2) + "\n")
    out = work / "campaign"
    angles = ROOT / "params/angles_p4.txt"
    helper = ROOT / "scripts/viral_campaign.py"
    capture([sys.executable, helper, "init", "--p", 4, "--samples", 129, "--jobs", 2,
             "--angles", angles, "--exe", exe, "--out", out])
    assert not campaign.inspect_campaign(out)["complete"]
    for job in (0, 1):
        env = environment()
        env.update(SLURM_ARRAY_JOB_ID="123", SLURM_ARRAY_TASK_ID=str(job),
                   SLURM_CPUS_PER_TASK="1", SLURM_SUBMIT_DIR=str(ROOT), OMP_NUM_THREADS="8")
        capture(["bash", ROOT / "scripts/run_viral.slurm", 4, 129, angles, exe, out], env=env)
        reference, dump, _, _ = run_solver(exe, 4, 129, job, angles, work / ("direct%d" % job))
        equal_bytes(reference, campaign.scientific_bytes(result_path(out / "jobs", 4, job).read_bytes()),
                    "wrapper vs direct result")
        assert (out / "logs" / ("provenance_123_%d.jsonl" % job)).exists()
    assert campaign.inspect_campaign(out)["complete"]
    capture([sys.executable, helper, "verify", out])
    report = capture([sys.executable, ROOT / "merge_results.py", out / "jobs", "-p", 4, "-v"])
    assert "258" in report and "0 still running" in report, report
    # The unchanged merger cannot detect a missing expected job; our verifier must.
    path = result_path(out / "jobs", 4, 1)
    saved = path.read_bytes()
    path.unlink()
    assert campaign.inspect_campaign(out)["missing_ids"] == [1]
    path.write_bytes(saved.replace(b"n_target      129", b"n_target      130"))
    assert campaign.inspect_campaign(out)["errors"]
    path.write_bytes(saved)
    extra = out / "jobs" / "duplicate.txt"
    extra.write_bytes(saved)
    assert campaign.inspect_campaign(out)["errors"]
    extra.unlink()
    env["SLURM_CPUS_PER_TASK"] = "2"
    refused = subprocess.run(["bash", str(ROOT / "scripts/run_viral.slurm"), "4", "129",
                              str(angles), str(exe), str(out)], env=env, capture_output=True)
    assert refused.returncode != 0
    return report


class ComparatorTests(unittest.TestCase):
    """Unit tests of the result-file comparator and of the source guard."""

    def setUp(self):
        """A minimal result file with every field set to 1."""
        self.data = b"# comment\n" + b"".join(k + b" 1\n" for k in sorted(campaign.FIELDS))

    def test_only_four_volatile_fields_excluded(self):
        """Changing a field changes the scientific bytes unless the field is volatile."""
        for field in campaign.FIELDS:
            changed = self.data.replace(field + b" 1\n", field + b" 2\n")
            self.assertEqual(campaign.scientific_bytes(self.data) == campaign.scientific_bytes(changed),
                             field in campaign.VOLATILE, field)

    def test_float_spelling_and_whitespace_preserved(self):
        """Any change in how soft_sum is spelled is a difference."""
        for value in (b"1.0", b"1.0000000000000002", b"+1", b" 1"):
            self.assertNotEqual(campaign.scientific_bytes(self.data),
                                campaign.scientific_bytes(self.data.replace(b"soft_sum 1", b"soft_sum " + value)))

    def test_missing_and_duplicate_fields_rejected(self):
        """A result file with a missing or a repeated field is rejected."""
        for bad in (self.data.replace(b"soft_sum 1\n", b""), self.data + b"soft_sum 1\n"):
            with self.assertRaises(ValueError):
                campaign.scientific_bytes(bad)

    def test_source_guard(self):
        """The guard accepts the repository source and refuses any other."""
        source_guard()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "altered.c"
            path.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                source_guard(path)

    def test_hex_dump_comparison_is_exact(self):
        """-0 and +0 in a dump are different bytes."""
        with self.assertRaises(AssertionError):
            equal_bytes(b"0 1 -0x0p+0 0x1p-1\n", b"0 1 0x0p+0 0x1p-1\n", "signed zero")


def original_checks(exe, work):
    """Run check.sh on exe in an isolated copy (as ./viral_tree_hpc, 4 threads); return its log."""
    fixture = work / "original_self_tests"
    fixture.mkdir()
    for name in ("check.sh", "merge_results.py"):
        shutil.copy2(ROOT / name, fixture / name)
    shutil.copytree(ROOT / "params", fixture / "params")
    shutil.copy2(exe, fixture / "viral_tree_hpc")
    env = environment()
    env["OMP_NUM_THREADS"] = "4"  # Original cut-count tests; separate from strict one-thread comparisons.
    log = capture(["bash", fixture / "check.sh"], env=env, timeout=1800)
    if "ALL CHECKS PASSED" not in log:
        raise AssertionError(log)
    return log


def benchmark(builds, work, min_seconds):
    """Fixed sample counts per depth; five interleaved repetitions per build."""
    results = {}
    rejected = {}
    for p in (8, 12, 16):
        angles = ROOT / ("params/angles_p%d.txt" % p)
        samples = 64
        while True:
            out = work / ("calibrate_%d_%d" % (p, samples))
            _, _, duration, _ = run_solver(builds["baseline"], p, samples, 19, angles, out)
            if duration >= min_seconds:
                break
            samples *= max(2, min(16, math.ceil(min_seconds / max(duration, 0.001))))
        values = {name: [] for name in builds}
        expected = None
        names = list(builds)
        for rep in range(5):
            # Change order across trials to reduce consistent warmup/order bias.
            order = names[rep % len(names):] + names[:rep % len(names)]
            for name in order:
                result, _, wall, _ = run_solver(builds[name], p, samples, 19, angles,
                                                work / ("%d_%d_%s" % (p, rep, name)))
                if expected is None:
                    assert name == "baseline"
                    expected = result
                try:
                    equal_bytes(expected, result, "benchmark p=%d %s" % (p, name))
                except AssertionError as error:
                    if name == "baseline":
                        raise
                    rejected[name] = str(error)
                values[name].append(wall)
        medians = {name: statistics.median(times) for name, times in values.items()}
        results[str(p)] = dict(samples=samples, repetitions=5, wall_seconds=values, medians=medians)
        print("Benchmark p=%d n=%d medians: %s" % (p, samples, json.dumps(medians)), flush=True)
    scores = {}
    for name in builds:
        if name in rejected:
            continue
        ratios = [r["medians"][name] / r["medians"]["baseline"] for r in results.values()]
        ratio = math.exp(statistics.mean(math.log(x) for x in ratios))
        scores[name] = dict(geometric_mean_time_ratio=ratio, worst_depth_time_ratio=max(ratios),
                            eligible=ratio <= 0.95 and max(ratios) <= 1.05)
    eligible = [name for name in scores if scores[name]["eligible"]]
    selected = min(eligible, key=lambda name: scores[name]["geometric_mean_time_ratio"]) if eligible else "baseline"
    return dict(depths=results, scores=scores, rejected=rejected, selected=selected,
                rule="geometric mean of depth median time ratios <=0.95; no depth ratio >1.05")


def main():
    """Guard the source, unit-test the comparator, build each profile and run every check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cc", default="clang" if sys.platform == "darwin" else "gcc")
    parser.add_argument("--omp-root", type=Path, help="macOS LLVM OpenMP prefix with include/omp.h and lib/libomp.dylib")
    parser.add_argument("--profiles", default=",".join(PROFILES), help="comma-separated fixed profile names; baseline is always included")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-min-seconds", type=float, default=1.5)
    parser.add_argument("--skip-existing-checks", action="store_true", help="omit the expensive original check.sh; reported explicitly")
    parser.add_argument("--spot-depths", default="", help="EXPLICIT opt-in: extra even depths up to 30; large allocations!")
    parser.add_argument("--spot-samples", type=int, default=4)
    parser.add_argument("--report", type=Path, help="new JSON report file outside tracked artifacts")
    parser.add_argument("--export-builds", type=Path, help="new directory for validated executables and .build.json sidecars")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("do not run this validation workflow with python -O")
    args.cc = shutil.which(args.cc)
    if not args.cc:
        parser.error("compiler not found")
    if args.omp_root and sys.platform != "darwin":
        parser.error("--omp-root is only for macOS Apple clang; use a normal OpenMP compiler on Linux")
    names = list(dict.fromkeys(["baseline"] + args.profiles.split(",")))
    if any(name not in PROFILES for name in names):
        parser.error("profiles must come from " + ",".join(PROFILES))
    args.spot_depths = [int(p) for p in args.spot_depths.split(",") if p]
    if any(p not in range(2, 31, 2) for p in args.spot_depths) or args.spot_samples < 1:
        parser.error("spot depths must be supplied even depths 2..30 and samples must be positive")
    if args.benchmark_min_seconds < 1:
        parser.error("benchmark trials must target at least one second")
    if args.report and args.report.exists():
        parser.error("report path already exists; use a new filename")
    if args.export_builds and args.export_builds.exists():
        parser.error("export directory already exists; use a new directory")
    source_guard()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComparatorTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    report = dict(source_sha256=campaign.sha256(SOURCE), platform=platform.platform(),
                  created_utc=datetime.now(timezone.utc).isoformat(), profiles={},
                  baseline="-O3 -DBATCH_MAX_PER_THREAD=1; OMP_NUM_THREADS=1",
                  compared_bytes="entire dumps and result bytes except elapsed_s, core_seconds, sec_per_sample_core, host")
    with tempfile.TemporaryDirectory(prefix="viral-equivalence-") as directory:
        work = Path(directory)
        cases = test_cases(work, args)
        report["cases"] = [dict(p=p, samples=n, job_id=job, angles=str(angles), angles_sha256=campaign.sha256(angles))
                           for p, n, job, angles in cases]
        builds, expected, resume_reference = {}, None, None
        for name in names:
            print("Building and checking " + name, flush=True)
            exe, info = compile_build(args, work, name)
            if exe is None:
                report["profiles"][name] = info
                print("SKIP %s: %s" % (name, info["diagnostic"].strip()), flush=True)
                continue
            try:
                observed = strict_cases(exe, work / name / "cases", cases, expected)
                resumed, resumed_dump, stops = resume_checks(exe, work / name / "resume")
                if expected is None:
                    expected = observed
                    resume_reference = (resumed, resumed_dump)
                else:
                    equal_bytes(resume_reference[0], resumed, name + " protocol result")
                    equal_bytes(resume_reference[1], resumed_dump, name + " protocol dump")
                info.update(equivalence_passed=True, tested_depths=sorted({case[0] for case in cases}),
                            strict_cases=len(cases), stop_resume=stops)
                merger = campaign_checks(exe, info, work / name / "launch")
                report["profiles"][name] = dict(status="passed", build=info, merger_report=merger)
                builds[name] = exe
                print("PASS %s: %d dump/result cases, repeats, extension, TERM/KILL resume, wrapper, merger" %
                      (name, len(cases)), flush=True)
            except AssertionError as error:
                if name == "baseline":
                    raise
                info["equivalence_passed"] = False
                report["profiles"][name] = dict(status="rejected", build=info, diagnostic=str(error))
                print("REJECT %s: %s" % (name, error), flush=True)
        if not args.skip_existing_checks:
            print("Running unchanged check.sh in an isolated temporary copy (up to several minutes)...", flush=True)
            report["existing_checks"] = original_checks(builds["baseline"], work)
            print(report["existing_checks"], flush=True)
        else:
            report["existing_checks"] = "SKIPPED by explicit option"
        report["selected_profile"] = "baseline"
        if args.benchmark:
            report["benchmark"] = benchmark(builds, work / "bench", args.benchmark_min_seconds)
            report["selected_profile"] = report["benchmark"]["selected"]
            for name, error in report["benchmark"]["rejected"].items():
                report["profiles"][name]["status"] = "rejected"
                report["profiles"][name]["diagnostic"] = error
                report["profiles"][name]["build"]["equivalence_passed"] = False
                del builds[name]
        source_guard()
        if args.export_builds:
            args.export_builds.mkdir(parents=True)
            for name, exe in builds.items():
                target = args.export_builds / exe.name
                info = report["profiles"][name]["build"]
                shutil.copy2(exe, target)
                Path(str(target) + ".build.json").write_text(json.dumps(info, indent=2) + "\n")
            (args.export_builds / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Selected profile: %s%s" % (report["selected_profile"],
          "" if args.benchmark else " (no benchmark requested; baseline retained)"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
