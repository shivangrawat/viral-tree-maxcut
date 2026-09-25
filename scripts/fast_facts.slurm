#!/bin/bash
# ---------------------------------------------------------------------------
#  R0 of SPEEDUP_PLAN.md: facts about a compute node for the fast solver -
#  CPU and vector flags, libc/libm identity, which libm calls the frozen
#  binary makes, and what one glibc sincos() costs on arguments like the
#  solver's.  Read-only apart from a micro-benchmark built in $TMPDIR.
#
#      sbatch scripts/fast_facts.slurm                 (from the repo root)
#
#  QoS cpu168 has its own caps, so this never waits on the cpu48 memory the
#  p=30 campaign fills; the cpu_short QoS is not valid for this account.
# ---------------------------------------------------------------------------
#SBATCH --job-name=viral_fast_r0
#SBATCH --account=torch_pr_577_general
#SBATCH --qos=cpu168
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/fm2452/viral-fast/logs/%x_%j.out

set -uo pipefail
FROZEN="$HOME/viral-exe-backup/torch-large-builds-17989395/viral_baseline"
echo "host $(hostname)   partition ${SLURM_JOB_PARTITION:-?}   qos ${SLURM_JOB_QOS:-?}   job ${SLURM_JOB_ID:-manual}"

echo "--- cpu"
lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)|L1d|L2|L3|CPU max MHz)'
echo "vector flags: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -x -E 'sse4_2|avx|avx2|fma|bmi2|avx512f|avx512dq|avx512vl' | tr '\n' ' ')"

echo "--- libc / libm / compiler"
ldd --version | head -1
LIBM=$(readlink -f /lib64/libm.so.6)
echo "libm $LIBM   sha256 $(sha256sum "$LIBM" | cut -d' ' -f1)"
module purge 2>/dev/null
echo "gcc: $(command -v gcc) $(gcc --version 2>/dev/null | head -1)"
if command -v objdump >/dev/null 2>&1 && [ -x "$FROZEN" ]; then
    echo "frozen binary calls: $(objdump -d "$FROZEN" | grep -o -E '<[a-z_]+@plt>' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
else
    echo "objdump or the frozen binary is unavailable"
fi

echo "--- glibc sincos() cost on arguments like the solver's"
T="${TMPDIR:-/tmp}/viral_fast_r0_${SLURM_JOB_ID:-$$}"
mkdir -p "$T"
cat > "$T/bench.c" <<'EOF'
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ns per sincos() call over arguments uniform in [-r, r] */
static double bench(double r, double *checksum) {
    enum { N = 1 << 16 };
    static double a[N];
    uint64_t x = 88172645463325252ULL;
    for (int i = 0; i < N; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        a[i] = ((double)(x >> 11) / 9007199254740992.0 * 2.0 - 1.0) * r;
    }
    const int reps = 1500;
    double acc = 0.0, s, c;
    const double t0 = now();
    for (int k = 0; k < reps; k++)
        for (int i = 0; i < N; i++) { sincos(a[i], &s, &c); acc += s + c; }
    const double t = now() - t0;
    *checksum = acc;
    return 1e9 * t / ((double)reps * N);
}

int main(void) {
    const double ranges[] = { 0.8, 2.4, 4.5 };   /* glibc's three code paths */
    for (int i = 0; i < 3; i++) {
        double chk;
        const double ns = bench(ranges[i], &chk);
        printf("sincos: %6.2f ns/call for |x| <= %.1f   (checksum %.6e)\n", ns, ranges[i], chk);
    }
    return 0;
}
EOF
if gcc -O2 -o "$T/bench" "$T/bench.c" -lm; then "$T/bench"; else echo "micro-benchmark did not compile"; fi
rm -rf "$T"
echo "--- done"
