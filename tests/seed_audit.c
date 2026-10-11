/*
 * seed_audit.c - seed-uniqueness audit of the campaigns (gates G1 and G1',
 * docs/validation.md; run by scripts/validation/seed_audit*.slurm).
 *
 * Reads lines "p first_job n_jobs samples_per_job" (one per campaign) from
 * stdin, computes for EVERY sample S = job*2^32 + k of every campaign the
 * initial xoshiro256** state the solver uses (sample_rng below is copied
 * verbatim from viral_tree_hpc.c), and checks that no two samples - of the
 * same depth or of different depths - start from the same state.  Two
 * samples with distinct states draw distinct seed bits, i.e. are different
 * samples.  Exit 0 and "SEED AUDIT PASS" if all states are distinct.
 *
 * Only the first two of the four state words are compared (16 bytes per
 * sample instead of 32).  That is conservative: if no two samples share
 * those 128 bits, no two share the whole state.
 *
 * Memory: 32 bytes per sample (one Entry), plus as much again for glibc
 * qsort's merge buffer: 2 x 5.1 GB for the 160M samples of the 10M campaigns.
 *
 *     gcc -O2 -o seed_audit tests/seed_audit.c && ./seed_audit < spec.txt
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t s[4]; } Rng;     /* xoshiro256** state */

/* ---- verbatim from viral_tree_hpc.c (see the comments there) ---- */

/* mix64 -- splitmix64 finalizer, a bijective hash of 64-bit words. */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* rng_seed -- xoshiro256** state from one 64-bit seed: four consecutive
 * splitmix64 outputs starting at seed.                                     */
static Rng rng_seed(uint64_t seed) {
    Rng r;
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r.s[i] = z ^ (z >> 31);
    }
    return r;
}

/* sample_rng -- initial state of the sample with global index
 * S = job * 2^32 + k at depth p.                                            */
static inline Rng sample_rng(int p, uint64_t S) {
    return rng_seed(mix64(0x5DEECE66DULL ^ ((uint64_t)p << 32) ^ mix64(S)));
}
/* ------------------------------------------------------------------ */

/* one sample: the first two words of its initial state, and where it is from */
typedef struct { uint64_t a, b; uint32_t p, job, k; } Entry;

/* cmp -- qsort order of entries: by state word a, then b. */
static int cmp(const void *x, const void *y) {
    const Entry *u = x, *v = y;
    if (u->a != v->a) return u->a < v->a ? -1 : 1;
    if (u->b != v->b) return u->b < v->b ? -1 : 1;
    return 0;
}

/* main -- read up to 256 spec lines, enumerate every sample state, sort the
 * states and report duplicates (at most 20 printed).  Exit 0 if none, 1 if
 * there are duplicates, 2 on a bad spec line or when out of memory.        */
int main(void) {
    int p; long first, njobs, per;
    size_t cap = 0, n = 0;
    Entry *e = NULL;
    struct { int p; long first, njobs, per; } spec[256];
    int ns = 0;
    while (ns < 256 && scanf("%d %ld %ld %ld", &p, &first, &njobs, &per) == 4) {
        if (p < 1 || p > 40 || first < 0 || njobs < 1 || per < 1 || per > 4294967295L ||
            first + njobs - 1 > 2147483647L) { fprintf(stderr, "bad spec line\n"); return 2; }
        spec[ns].p = p; spec[ns].first = first; spec[ns].njobs = njobs; spec[ns].per = per; ns++;
        cap += (size_t)njobs * (size_t)per;
    }
    e = malloc(cap * sizeof *e);
    if (!e) { fprintf(stderr, "out of memory for %zu entries\n", cap); return 2; }
    for (int i = 0; i < ns; i++) {
        for (long j = spec[i].first; j < spec[i].first + spec[i].njobs; j++)
            for (long k = 0; k < spec[i].per; k++) {
                const Rng r = sample_rng(spec[i].p, ((uint64_t)j << 32) | (uint64_t)k);
                e[n].a = r.s[0];  e[n].b = r.s[1];
                e[n].p = (uint32_t)spec[i].p;  e[n].job = (uint32_t)j;  e[n].k = (uint32_t)k;
                n++;
            }
        printf("p=%-2d jobs %ld..%ld x %ld samples\n", spec[i].p, spec[i].first,
               spec[i].first + spec[i].njobs - 1, spec[i].per);
    }
    qsort(e, n, sizeof *e, cmp);
    size_t dup = 0;
    for (size_t i = 1; i < n; i++)
        if (e[i].a == e[i - 1].a && e[i].b == e[i - 1].b) {
            if (dup < 20)
                printf("DUPLICATE STATE: p=%u job=%u k=%u  and  p=%u job=%u k=%u\n", e[i - 1].p,
                       e[i - 1].job, e[i - 1].k, e[i].p, e[i].job, e[i].k);
            dup++;
        }
    printf("%zu sample states checked, %zu duplicates (first two state words compared)\n", n, dup);
    printf(dup ? "SEED AUDIT FAIL\n" : "SEED AUDIT PASS\n");
    free(e);
    return dup ? 1 : 0;
}
