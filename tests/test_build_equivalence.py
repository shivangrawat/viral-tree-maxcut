#!/usr/bin/env python3
"""Exact CPU/reference validation, peak RSS, and bounded benchmarks (stdlib only).

The reference is extracted from the pinned faster commit, never from edited
solver code. Builds and scientific outputs live in temporary directories.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
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
spec = importlib.util.spec_from_file_location("campaign", ROOT / "scripts/viral_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
SOURCE = ROOT / "viral_tree_hpc.c"
REFERENCE_COMMIT = "463f380927fb7ef21b80af9ba7146f849dc9bd62"
SOURCE_SHA256 = "b3af7c651c2271f594d347807175c9bea602e2a65d712914eadf9fa9f3766d51"


@dataclass(frozen=True)
class Execution:
    path: Path
    kernel: str = "reference"
    threads: int = 1

    def __str__(self):
        return str(self.path)

    @property
    def name(self):
        return "reference" if self.kernel == "reference" else "%s_t%d" % (self.kernel, self.threads)


def reference_bytes():
    data = subprocess.check_output(["git", "show", REFERENCE_COMMIT + ":viral_tree_hpc.c"], cwd=ROOT)
    source_guard(data)
    return data


def source_guard(data):
    actual = hashlib.sha256(data).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError("Protected reference changed: expected %s, found %s" % (SOURCE_SHA256, actual))


def equal_bytes(expected, actual, label):
    if expected != actual:
        diff = "".join(difflib.unified_diff(expected.decode().splitlines(True), actual.decode().splitlines(True),
                                          fromfile="reference", tofile="candidate"))
        raise AssertionError("%s: byte mismatch\n%s" % (label, diff[:4000]))


def environment(execution):
    env = os.environ.copy()
    for key in ("VIRAL_TREE_DUMP", "VIRAL_TREE_KERNEL", "OMP_THREAD_LIMIT"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS=str(execution.threads), OMP_DYNAMIC="FALSE",
               OMP_PROC_BIND="close", OMP_PLACES="cores", LC_ALL="C",
               ASAN_OPTIONS="detect_leaks=0:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
    if execution.kernel != "reference":
        env["VIRAL_TREE_KERNEL"] = execution.kernel
    return env


def capture(command, **kwargs):
    return subprocess.check_output([str(x) for x in command], stderr=subprocess.STDOUT, text=True, **kwargs)


def compile_build(args, work, name, source, extra=()):
    exe = work / name
    flags = ["-Wall", "-Wextra", "-O3", "-DBATCH_MAX_PER_THREAD=1"] + list(extra)
    if args.omp_root:
        omp = args.omp_root.resolve()
        flags += ["-Xpreprocessor", "-fopenmp", "-I" + str(omp / "include"),
                  "-L" + str(omp / "lib"), "-lomp", "-Wl,-headerpad_max_install_names"]
    else:
        flags += ["-fopenmp"]
    command = [args.cc] + flags + [str(source), "-lm", "-o", str(exe)]
    log = capture(command)
    if log.strip():
        print(log, flush=True)
    if args.omp_root:
        dylib = args.omp_root.resolve() / "lib/libomp.dylib"
        install_id = capture(["otool", "-D", dylib]).splitlines()[1].strip()
        capture(["install_name_tool", "-change", install_id, dylib, exe])
        capture(["codesign", "--force", "--sign", "-", exe])
    linkage = capture((["otool", "-L"] if sys.platform == "darwin" else ["ldd"]) + [exe])
    info = dict(source_sha256=campaign.sha256(source), executable_sha256=campaign.sha256(exe),
                reference_commit=REFERENCE_COMMIT, reference_source_sha256=SOURCE_SHA256,
                compiler=args.cc, compiler_version=capture([args.cc, "--version"]).strip(),
                compile_command=command, flags=flags, runtime_linkage=linkage,
                system=platform.system(), machine=platform.machine(), platform=platform.platform(),
                created_utc=datetime.now(timezone.utc).isoformat(), batch_max_per_thread=1,
                equivalence_passed=False, validated_configs=[])
    if args.omp_root:
        info.update(openmp_library=str(dylib), openmp_library_sha256=campaign.sha256(dylib))
    return exe, info


def result_path(out, p, job):
    return out / ("viral_tree_p%d_job%d.txt" % (p, job))


def measured_child(command):
    # A fresh Python parent per measurement: exactly one solver in RUSAGE_CHILDREN.
    begin = time.perf_counter()
    child = subprocess.run(command)
    wall = time.perf_counter() - begin
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    print("VIRAL_MEASURE " + json.dumps(dict(wall_seconds=wall, peak_rss_bytes=rss)), flush=True)
    return child.returncode


def run_solver(execution, p, samples, job, angles, out, dump=False, measure=False, timeout=1200):
    out.mkdir(parents=True, exist_ok=True)
    env = environment(execution)
    if dump:
        env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
    command = [str(execution), str(p), str(samples), str(job), str(angles), str(out)]
    if measure:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--measure-child"] + command
    begin = time.perf_counter()
    log = capture(command, cwd=ROOT, env=env, timeout=timeout)
    metrics = dict(wall_seconds=time.perf_counter() - begin)
    if measure:
        metrics = json.loads(next(line[len("VIRAL_MEASURE "):] for line in log.splitlines()
                                  if line.startswith("VIRAL_MEASURE ")))
    metrics["allocated_cpu_seconds"] = metrics["wall_seconds"] * execution.threads
    metrics["samples_per_second"] = samples / metrics["wall_seconds"]
    metrics["seconds_per_sample"] = metrics["wall_seconds"] / samples
    data = result_path(out, p, job).read_bytes()
    fields = campaign.read_fields(data)
    assert (fields[b"n_done"], fields[b"n_target"], fields[b"status"], fields[b"threads"]) == (
        str(samples).encode(), str(samples).encode(), b"complete", str(execution.threads).encode()), fields
    return campaign.scientific_bytes(data), (out / "samples.hex").read_bytes() if dump else b"", metrics, log


def test_cases(work, args):
    p1 = work / "angles_p1.txt"
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
    for p in sorted(set([18, 20] + args.spot_depths)):
        for job in (0, 499):
            for n in (1, args.spot_samples):
                cases.append((p, n, job, ROOT / ("params/angles_p%d.txt" % p)))
    fixtures = []
    for name, alpha, beta, gamma in (("positive_zero", "0.0", "0.0", "0.0"),
                                     ("negative_zero", "-0.0", "-0.0", "-0.0"),
                                     ("zero_field", "-0.7", "0.3", "-0.0"),
                                     ("large_field", "0.7", "-1.2", "10000.0")):
        path = work / (name + ".txt")
        path.write_text("# test fixture\n%s\n# layer beta gamma\n" % alpha +
                        "".join("%d %s %s\n" % (layer, beta, gamma) for layer in range(1, 7)))
        fixtures.append(path)
        for p in (1, 2, 3, 4, 5):
            cases.append((p, 33, 7, path))
    return cases, fixtures


def strict_cases(execution, work, cases, expected=None, timeout=1200):
    observed = []
    for i, (p, n, job, angles) in enumerate(cases):
        result, dump, _, _ = run_solver(execution, p, n, job, angles, work / str(i), dump=True, timeout=timeout)
        rows = dump.splitlines()
        assert len(rows) == n and [int(row.split()[0]) for row in rows] == list(range(n))
        if expected is not None:
            equal_bytes(expected[i][0], result, "%s p=%d job=%d n=%d result" % (execution.name, p, job, n))
            equal_bytes(expected[i][1], dump, "%s p=%d job=%d n=%d dump" % (execution.name, p, job, n))
        observed.append((result, dump))
    return observed


def resume_checks(execution, work, expected=None):
    p, job, n = 12, 13, 4096
    angles = ROOT / "params/angles_p12.txt"
    fresh, dump, _, _ = run_solver(execution, p, n, job, angles, work / "fresh", dump=True)
    if expected is not None:
        equal_bytes(expected[0], fresh, execution.name + " lifecycle result")
        equal_bytes(expected[1], dump, execution.name + " lifecycle dump")
    repeat, repeat_dump, _, _ = run_solver(execution, p, n, job, angles, work / "repeat", dump=True)
    equal_bytes(fresh, repeat, "repeated result")
    equal_bytes(dump, repeat_dump, "repeated dump")
    run_solver(execution, p, 31, job, angles, work / "extend", dump=True)
    extended, extended_dump, _, log = run_solver(execution, p, n, job, angles, work / "extend", dump=True)
    assert "RESUMING" in log
    equal_bytes(fresh, extended, "extended result")
    equal_bytes(dump, extended_dump, "extended dump")
    before = result_path(work / "extend", p, job).read_bytes()
    _, _, _, log = run_solver(execution, p, n, job, angles, work / "extend", dump=True)
    assert "already complete" in log
    equal_bytes(before, result_path(work / "extend", p, job).read_bytes(), "completed rerun")
    equal_bytes(dump, (work / "extend/samples.hex").read_bytes(), "completed dump unchanged")
    records = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        out = work / sig.name
        run_solver(execution, p, 31, job, angles, out, dump=(sig == signal.SIGTERM))
        env = environment(execution)
        if sig == signal.SIGTERM:
            env["VIRAL_TREE_DUMP"] = str(out / "samples.hex")
        with (out / "interruption.log").open("w") as stream:
            child = subprocess.Popen([str(execution), str(p), str(n), str(job), str(angles), str(out)],
                                     env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while True:
                    fields = campaign.read_fields(result_path(out, p, job).read_bytes())
                    if fields[b"status"] == b"running" and fields[b"n_target"] == str(n).encode():
                        break
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise AssertionError("did not observe a running checkpoint before stopping")
                    time.sleep(0.001)
                time.sleep(0.01)
                child.send_signal(sig)
                rc = child.wait(timeout=60)
                assert rc == (0 if sig == signal.SIGTERM else -signal.SIGKILL), rc
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        fields = campaign.read_fields(result_path(out, p, job).read_bytes())
        done = int(fields[b"n_done"])
        assert fields[b"status"] == b"running" and 31 <= done < n
        if sig == signal.SIGTERM:
            assert done > 31 and "STOPPED BY SIGNAL" in (out / "interruption.log").read_text()
        else:
            assert done == 31, "forced stop must roll back to the real saved prefix"
        resumed, resumed_dump, _, log = run_solver(execution, p, n, job, angles, out, dump=(sig == signal.SIGTERM))
        assert "RESUMING" in log
        equal_bytes(fresh, resumed, sig.name + " resumed result")
        if sig == signal.SIGTERM:
            equal_bytes(dump, resumed_dump, "SIGTERM resumed dump")
        records.append(dict(signal=sig.name, saved_samples=done, final_samples=n))
    return (fresh, dump), records


def cross_resume(reference, executions, work):
    chain = [reference] + executions + [reference]
    angles = ROOT / "params/angles_p8.txt"
    for i, execution in enumerate(chain):
        actual, dump, _, _ = run_solver(execution, 8, 31 * (i + 1), 19, angles, work / "mixed", dump=True)
    expected, expected_dump, _, _ = run_solver(reference, 8, 31 * len(chain), 19, angles, work / "fresh", dump=True)
    equal_bytes(expected, actual, "cross-backend/thread/reference continuation")
    equal_bytes(expected_dump, dump, "cross-backend/thread/reference dump")
    return [execution.name for execution in chain]


PROBE_BODY = r'''
#undef main
int main(int argc, char **argv) {
    if (argc == 2) {
#ifdef CANDIDATE_PROBE
        for (int p = 1; p <= 40; p++) for (int compact = 0; compact <= 1; compact++) {
            size_t states, words, scratch;
            work_sizes(p, compact, &states, &words, &scratch);
            printf("%d %d %zu %zu %zu %zu\n", p, compact, states, words, scratch,
                   states*sizeof(Vec3) + words*sizeof(uint64_t) + 2*scratch*sizeof(double));
        }
#endif
        return 0;
    }
    int p = atoi(argv[1]);
    Angles A;
    A.beta=calloc(p,sizeof(double)); A.gamma=calloc(p,sizeof(double));
    A.cb=calloc(p,sizeof(double)); A.sb=calloc(p,sizeof(double));
    if (!load_angles(argv[2],p,&A)) return 1;
#ifdef CANDIDATE_PROBE
    build_first_layer(&A);
#endif
    for (unsigned pattern=0; pattern<16; pattern++) {
#ifdef CANDIDATE_PROBE
        Vec3 s=A.first[pattern];
#else
        double z[4];
        for (unsigned b=0;b<4;b++) z[b]=(pattern&(1u<<b))?A.sa:-A.sa;
        const double h=z[1]+z[2]+z[3];
        Vec3 s=rx_cs(rz((Vec3){A.ca,0.0,z[0]},A.gamma[0]*h),A.cb[0],A.sb[0]);
#endif
        printf("%u %a %a %a\n",pattern,s.x,s.y,s.z);
    }
    free(A.beta);free(A.gamma);free(A.cb);free(A.sb);
    return 0;
}
'''


def table_and_size_checks(args, work, frozen, fixtures):
    probes = []
    for name, source in (("reference", frozen), ("candidate", SOURCE)):
        path = work / (name + "_probe.c")
        path.write_text(("#define CANDIDATE_PROBE\n" if name == "candidate" else "") +
                        "#define main solver_main\n#include " + json.dumps(str(source)) + "\n" + PROBE_BODY)
        binary, _ = compile_build(args, work, name + "_probe", path)
        probes.append(binary)
    inputs = [(p, ROOT / ("params/angles_p%d.txt" % p)) for p in range(2, 31, 2)]
    inputs += [(6, path) for path in fixtures]
    for p, path in inputs:
        expected = capture([probes[0], p, path]).encode()
        equal_bytes(expected, capture([probes[1], p, path]).encode(), "all 16 table patterns: " + str(path))
    sizes = []
    for line in capture([probes[1], "sizes"]).splitlines():
        p, compact, states, words, scratch, total = map(int, line.split())
        expected_states = (2**p - 2) if compact and p >= 2 else 2**(p+1)-2
        expected_words = (2**(p+2)-2+63)//64
        expected_scratch = max(2, 2**(p-2)) if compact else max(2, 2**(p-1))
        assert (states, words, scratch) == (expected_states, expected_words, expected_scratch)
        assert total == 24*states + 8*words + 16*scratch
        sizes.append(dict(p=p, kernel="compact" if compact else "shared", workspace_bytes=total))
    return dict(table_patterns=16, angle_files=len(inputs), memory_sizes=sizes)


def campaign_checks(execution, info, work):
    Path(str(execution.path) + ".build.json").write_text(json.dumps(info, indent=2) + "\n")
    out, angles = work / "campaign", ROOT / "params/angles_p4.txt"
    helper = ROOT / "scripts/viral_campaign.py"
    command = [sys.executable, helper, "init", "--p", 4, "--samples", 129, "--jobs", 2,
               "--angles", angles, "--exe", execution.path, "--out", out, "--threads", execution.threads]
    if execution.kernel != "reference":
        command += ["--kernel", execution.kernel]
    capture(command)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema"] == (1 if execution.kernel == "reference" else 2)
    provenance = json.loads((out / "provenance.txt").read_text().split("\n", 1)[1])
    assert provenance["build"] == info
    assert (provenance["kernel"], provenance["omp_num_threads"]) == (execution.kernel, execution.threads)
    assert provenance["angles_sha256"] == manifest["angles_sha256"]
    assert not campaign.inspect_campaign(out)["complete"]
    for job in (0, 1):
        env = environment(execution)
        env.update(SLURM_ARRAY_JOB_ID="123", SLURM_ARRAY_TASK_ID=str(job),
                   SLURM_CPUS_PER_TASK=str(execution.threads), SLURM_SUBMIT_DIR=str(ROOT),
                   OMP_NUM_THREADS="99", OMP_THREAD_LIMIT="1", VIRAL_TREE_KERNEL="invalid_inherited_value")
        capture(["bash", ROOT / "scripts/run_viral.slurm", 4, 129, angles, execution.path, out], env=env)
        expected, _, _, _ = run_solver(execution, 4, 129, job, angles, work / ("direct%d" % job))
        raw = result_path(out / "jobs", 4, job).read_bytes()
        equal_bytes(expected, campaign.scientific_bytes(raw), "wrapper vs direct")
        assert int(campaign.read_fields(raw)[b"threads"]) == execution.threads
        assert (out / "logs" / ("provenance_123_%d.jsonl" % job)).exists()
    capture([sys.executable, helper, "verify", out])
    merged = capture([sys.executable, ROOT / "merge_results.py", out / "jobs", "-p", 4, "-v"])
    assert "258" in merged and "0 still running" in merged
    path = result_path(out / "jobs", 4, 1)
    saved = path.read_bytes()
    path.unlink()
    assert campaign.inspect_campaign(out)["missing_ids"] == [1]
    path.write_bytes(saved.replace(b"n_target      129", b"n_target      130"))
    assert campaign.inspect_campaign(out)["errors"]
    path.write_bytes(saved.replace(b"threads       " + str(execution.threads).encode(), b"threads       999"))
    assert campaign.inspect_campaign(out)["errors"]
    path.write_bytes(saved)
    extra = out / "jobs/duplicate.txt"
    extra.write_bytes(saved)
    assert campaign.inspect_campaign(out)["errors"]
    extra.unlink()
    env["SLURM_CPUS_PER_TASK"] = str(execution.threads + 1)
    refused = subprocess.run(["bash", str(ROOT / "scripts/run_viral.slurm"), "4", "129", str(angles),
                              str(execution.path), str(out)], env=env, capture_output=True)
    assert refused.returncode != 0
    return merged


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.data = b"# comment\n" + b"".join(k + b" 1\n" for k in sorted(campaign.FIELDS))

    def test_only_five_metadata_fields_excluded(self):
        self.assertEqual(campaign.VOLATILE, {b"threads", b"elapsed_s", b"core_seconds", b"sec_per_sample_core", b"host"})
        for field in campaign.FIELDS:
            changed = self.data.replace(field + b" 1\n", field + b" 2\n")
            self.assertEqual(campaign.scientific_bytes(self.data) == campaign.scientific_bytes(changed),
                             field in campaign.VOLATILE, field)

    def test_float_spelling_and_whitespace_preserved(self):
        for value in (b"1.0", b"1.0000000000000002", b"+1", b" 1"):
            self.assertNotEqual(campaign.scientific_bytes(self.data), campaign.scientific_bytes(
                self.data.replace(b"soft_sum 1", b"soft_sum " + value)))

    def test_missing_and_duplicate_fields_rejected(self):
        for bad in (self.data.replace(b"soft_sum 1\n", b""), self.data + b"soft_sum 1\n"):
            with self.assertRaises(ValueError):
                campaign.scientific_bytes(bad)

    def test_reference_guard(self):
        data = reference_bytes()
        with self.assertRaises(ValueError):
            source_guard(data + b"\n")

    def test_hex_dump_comparison_is_exact(self):
        with self.assertRaises(AssertionError):
            equal_bytes(b"0 1 -0x0p+0 0x1p-1\n", b"0 1 0x0p+0 0x1p-1\n", "signed zero")


def original_checks(execution, work):
    fixture = work / "original_self_tests"
    fixture.mkdir()
    for name in ("check.sh", "merge_results.py"):
        shutil.copy2(ROOT / name, fixture / name)
    shutil.copytree(ROOT / "params", fixture / "params")
    shutil.copy2(execution.path, fixture / "viral_tree_hpc")
    # The new matrix explicitly tests multiple cooperating thread counts; use
    # one thread for the original two-million-sample shallow statistical check.
    log = capture(["bash", fixture / "check.sh"], env=environment(execution), timeout=1800)
    assert "ALL CHECKS PASSED" in log
    return log


def sanitizer_checks(args, work, reference):
    sanitized, _ = compile_build(args, work, "viral_sanitized", SOURCE,
                                ("-fsanitize=address,undefined", "-fno-omit-frame-pointer"))
    count = 0
    for p in (1, 2, 4, 8, 12):
        angles = ROOT / ("params/angles_p%d.txt" % max(2, p))
        expected, dump, _, _ = run_solver(reference, p, 9, 7, angles, work / ("san_ref_%d" % p), dump=True)
        for kernel in ("shared", "compact"):
            for threads in (1, 3, 4):
                run = Execution(sanitized, kernel, threads)
                result, actual_dump, _, _ = run_solver(run, p, 9, 7, angles,
                                                      work / ("san_%d_%s" % (p, run.name)), dump=True)
                equal_bytes(expected, result, "sanitizer result")
                equal_bytes(dump, actual_dump, "sanitizer dump")
                count += 1
    return dict(cases=count, address_sanitizer="passed", undefined_behavior_sanitizer="passed")


def memory_checks(reference, executions, work):
    measurements = {}
    for run in [reference] + executions:
        _, _, metrics, log = run_solver(run, 20, 4, 0, ROOT / "params/angles_p20.txt",
                                        work / run.name, measure=True)
        if run.kernel != "reference":
            expected_bytes = (28.5 if run.kernel == "compact" else 56.5) * 2**20 - 48
            actual_bytes = int(next(line.split()[1] for line in log.splitlines() if line.strip().startswith("workspace_bytes")))
            assert actual_bytes == expected_bytes
            # A generous runtime allowance, still catching an eightfold copy.
            assert metrics["peak_rss_bytes"] < 2 * expected_bytes + 64 * 2**20
        measurements[run.name] = metrics
    return measurements


def benchmark(reference, executions, work, min_seconds):
    runs = [reference] + [r for r in executions if r.threads in (1, 2, 4, 8)]
    results = {}
    for p in (8, 12, 16, 20):
        angles = ROOT / ("params/angles_p%d.txt" % p)
        counts, values, expected = {}, {}, {}
        for run in runs:
            samples = 64
            while True:
                _, _, metrics, _ = run_solver(run, p, samples, 19, angles,
                    work / ("calibrate_%d_%s_%d" % (p, run.name, samples)), measure=True)
                if metrics["wall_seconds"] >= min_seconds:
                    break
                samples *= max(2, min(16, math.ceil(min_seconds / max(metrics["wall_seconds"], 0.001))))
            counts[run.name], values[run.name] = samples, []
            if samples not in expected:
                expected[samples] = run_solver(reference, p, samples, 19, angles,
                                               work / ("expected_%d_%d" % (p, samples)))[0]
            print("Calibrated p=%d %-12s n=%d (%.3fs)" % (p, run.name, samples, metrics["wall_seconds"]), flush=True)
        for repetition in range(5):
            order = runs[repetition % len(runs):] + runs[:repetition % len(runs)]
            for run in order:
                samples = counts[run.name]
                result, _, metrics, _ = run_solver(run, p, samples, 19, angles,
                    work / ("trial_%d_%d_%s" % (p, repetition, run.name)), measure=True)
                equal_bytes(expected[samples], result, "benchmark p=%d %s" % (p, run.name))
                values[run.name].append(metrics)
        medians = {name: statistics.median(m["seconds_per_sample"] for m in trials) for name, trials in values.items()}
        results[str(p)] = dict(samples_per_trial=counts, repetitions=5, trials=values,
                              median_seconds_per_sample=medians)
        print("Benchmark p=%d seconds/sample: %s" % (p, json.dumps(medians)), flush=True)
    ratios = [row["median_seconds_per_sample"]["compact_t1"] / row["median_seconds_per_sample"]["reference"]
              for row in results.values()] if any(r.name == "compact_t1" for r in runs) else [math.inf]
    mean_ratio = math.exp(statistics.mean(math.log(ratio) for ratio in ratios))
    eligible = mean_ratio <= 0.95 and max(ratios) <= 1.05
    return dict(depths=results, compact_one_thread_geometric_time_ratio=mean_ratio,
                compact_one_thread_worst_depth_ratio=max(ratios), compact_eligible=eligible,
                selected_default="compact" if eligible else "shared",
                rule="compact_t1 geometric time ratio <=0.95 and every depth ratio <=1.05")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="clang" if sys.platform == "darwin" else "gcc")
    parser.add_argument("--omp-root", type=Path, help="Apple clang LLVM OpenMP prefix, containing include/ and lib/")
    parser.add_argument("--threads", default="1,2,3,4,8", help="exactness matrix; always includes 1")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-min-seconds", type=float, default=1.5)
    parser.add_argument("--skip-existing-checks", action="store_true")
    parser.add_argument("--skip-sanitizers", action="store_true", help="explicit skip, recorded in report")
    parser.add_argument("--spot-depths", default="", help="opt-in additional even depths up to 30; large allocations!")
    parser.add_argument("--spot-samples", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200, help="seconds per correctness case")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--export-builds", type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("do not disable assertions with python -O")
    args.cc = shutil.which(args.cc)
    if not args.cc:
        parser.error("compiler not found")
    if args.omp_root and sys.platform != "darwin":
        parser.error("--omp-root is only for Apple clang on macOS")
    threads = sorted(set([1] + [int(t) for t in args.threads.split(",")]))
    args.spot_depths = [int(p) for p in args.spot_depths.split(",") if p]
    if min(threads) < 1 or any(p not in range(2, 31, 2) for p in args.spot_depths) or args.spot_samples < 1:
        parser.error("invalid threads, spot depths, or samples")
    if args.benchmark_min_seconds < 1 or args.timeout <= 0:
        parser.error("benchmark target must be >=1 second; timeout must be positive")
    if args.report and args.report.exists() or args.export_builds and args.export_builds.exists():
        parser.error("report/export paths must be new")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComparatorTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        return 1
    source_hash = campaign.sha256(SOURCE)
    report = dict(reference_commit=REFERENCE_COMMIT, reference_source_sha256=SOURCE_SHA256,
                  candidate_source_sha256=source_hash, platform=platform.platform(),
                  created_utc=datetime.now(timezone.utc).isoformat(), configurations={},
                  excluded_metadata=sorted(key.decode() for key in campaign.VOLATILE))
    with tempfile.TemporaryDirectory(prefix="viral-cpu-equivalence-") as directory:
        work = Path(directory)
        frozen = work / "viral_tree_reference.c"
        frozen.write_bytes(reference_bytes())
        ref_path, ref_info = compile_build(args, work, "viral_reference", frozen)
        path, info = compile_build(args, work, "viral_cpu", SOURCE)
        info["build_kind"] = "cpu-shared-v2"
        reference = Execution(ref_path)
        executions = [Execution(path, kernel, nt) for kernel in ("shared", "compact") for nt in threads]
        cases, fixtures = test_cases(work, args)
        report["cases"] = [dict(p=p, samples=n, job_id=job, angles=str(a), angles_sha256=campaign.sha256(a))
                           for p, n, job, a in cases]
        print("Checking all table patterns and allocation sizes...", flush=True)
        report["table_and_sizes"] = table_and_size_checks(args, work, frozen, fixtures)
        print("Running frozen reference (%d cases)..." % len(cases), flush=True)
        expected = strict_cases(reference, work / "reference/cases", cases, timeout=args.timeout)
        lifecycle, stops = resume_checks(reference, work / "reference/resume")
        ref_info.update(equivalence_passed=True, omp_num_threads=1)
        report["reference_build"] = ref_info
        for execution in executions:
            print("Checking " + execution.name, flush=True)
            strict_cases(execution, work / execution.name / "cases", cases, expected, timeout=args.timeout)
            _, stops = resume_checks(execution, work / execution.name / "resume", lifecycle)
            info["validated_configs"].append(dict(kernel=execution.kernel, threads=execution.threads))
            report["configurations"][execution.name] = dict(status="passed", cases=len(cases), stop_resume=stops)
            print("PASS %s: %d dump/result cases, repeat, extension, TERM/KILL resume" %
                  (execution.name, len(cases)), flush=True)
        info.update(equivalence_passed=True, tested_depths=sorted({c[0] for c in cases}))
        # Discover and exercise the actual default, not a label inferred from source.
        probe_env = environment(reference)
        probe_env.pop("OMP_NUM_THREADS")
        default_dir = work / "default"
        default_dir.mkdir()
        log = capture([path, 4, 9, 0, ROOT / "params/angles_p4.txt", default_dir], env=probe_env)
        info["default_kernel"] = next(line.split()[1] for line in log.splitlines() if line.strip().startswith("kernel"))
        wanted = run_solver(reference, 4, 9, 0, ROOT / "params/angles_p4.txt", work / "default_ref")[0]
        equal_bytes(wanted, campaign.scientific_bytes(result_path(default_dir, 4, 0).read_bytes()), "default kernel")
        assert campaign.read_fields(result_path(default_dir, 4, 0).read_bytes())[b"threads"] == b"1"
        report["cross_resume_chain"] = cross_resume(reference, executions, work / "cross_resume")
        report["wrapper_and_merger"] = {}
        for execution in [reference] + executions:
            report["wrapper_and_merger"][execution.name] = campaign_checks(
                execution, ref_info if execution.kernel == "reference" else info, work / execution.name / "launch")
        print("PASS cross-backend/thread continuation, legacy/new manifests, wrapper and unchanged merger", flush=True)
        report["sanitizers"] = "SKIPPED explicitly" if args.skip_sanitizers else sanitizer_checks(args, work, reference)
        print("Sanitizers: " + str(report["sanitizers"]), flush=True)
        report["peak_rss_p20"] = memory_checks(reference, executions, work / "memory")
        print("Peak RSS p=20: " + json.dumps({n: r["peak_rss_bytes"] for n, r in report["peak_rss_p20"].items()}), flush=True)
        report["existing_checks"] = "SKIPPED explicitly" if args.skip_existing_checks else original_checks(Execution(path, "compact"), work)
        print(report["existing_checks"], flush=True)
        if args.benchmark:
            report["benchmark"] = benchmark(reference, executions, work / "benchmark", args.benchmark_min_seconds)
        if campaign.sha256(SOURCE) != source_hash:
            raise AssertionError("candidate source changed during validation")
        source_guard(frozen.read_bytes())
        report["candidate_build"] = info
        if args.export_builds:
            args.export_builds.mkdir(parents=True)
            for executable, metadata in ((ref_path, ref_info), (path, info)):
                target = args.export_builds / executable.name
                shutil.copy2(executable, target)
                Path(str(target) + ".build.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (args.export_builds / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("ALL CPU EQUIVALENCE CHECKS PASSED", flush=True)
    if args.benchmark:
        print("Benchmark-selected default: " + report["benchmark"]["selected_default"], flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--measure-child":
        sys.exit(measured_child(sys.argv[2:]))
    sys.exit(main())
