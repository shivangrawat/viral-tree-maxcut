/*
 * ============================================================================
 *  viral_tree_fast2.c
 *
 *  viral_tree_fast.c plus two changes, with the same results to the last bit:
 *    A. threads inside one sample: OMP_NUM_THREADS threads share ONE tree and
 *       split every level between them; samples run one after another.
 *    B. a smaller workspace, about 5*2^p bytes instead of 7.5*2^p, from one
 *       more level computed on the fly (see THE FUSED KERNEL at the end of
 *       this header).
 *  Same command line, output file, checkpoint/resume, signals and
 *  VIRAL_TREE_DUMP hook as viral_tree_hpc.c (read that file for the problem,
 *  the algorithm and the formats).  The `threads` field of the result file is
 *  the number of threads working on each sample.
 *
 *      compile :  gcc -O3 -ffp-contract=off -fopenmp viral_tree_fast2.c -o viral_tree_fast2 -lm
 *      run     :  OMP_NUM_THREADS=<threads per sample> ./viral_tree_fast2 <p> <n> [job] [angles] [out]
 *
 * ----------------------------------------------------------------------------
 *  WHERE THE TIME GOES
 * ----------------------------------------------------------------------------
 *
 *  viral_tree_hpc makes one libm sincos() call per vertex update,
 *  4*2^p - 2p - 4 of them per sample, and that call dominates its run time at
 *  every depth.  Layer n updates 2^(p-n+1) - 2 vertices, so layer 0 is half of
 *  all updates and layer 1 a quarter.  Both layers have very few distinct
 *  inputs.
 *
 *  Time 1.  Every time-0 z is +s or -s (s = sin alpha).  For a, b, c in
 *      {+s, -s} the double (a + b) + c depends only on how many of a, b, c
 *      are +s: 2s is exact, 2s - s and 0 + s are exact, and only the
 *      all-equal case rounds - the same way in any order.  So the time-1 spin
 *      of a vertex is T1[own seed bit][number of + neighbours]: 8 values.
 *      (Checked for the actual s at start-up.)
 *
 *  Time 2.  The time-2 spin of a vertex depends on its own time-1 spin and
 *      the time-1 z of its parent and two children, hence only on the 10 seed
 *      bits of its radius-2 ball - self, parent, grandparent, sibling, two
 *      children, four grandchildren: T2[1024].  The two roots and level 1 fit
 *      the same index, with the other root standing in for the missing parent
 *      or grandparent.
 *
 *  Both tables are built once per run with the reference's rz/rx_cs
 *  arithmetic, the same sincos() call and the same association order
 *  (parent + child1) + child2, so every entry is exactly the double the
 *  reference computes for that input.  (The reference binary imports sincos,
 *  not sin/cos: GCC fuses the cos/sin pair of its rz.  Here sincos is called
 *  explicitly, in both kernels.)  Nothing is approximated.
 *
 * ----------------------------------------------------------------------------
 *  THE TABLE KERNEL (p >= 4)
 * ----------------------------------------------------------------------------
 *
 *  - No initialisation pass and no layer-0 or layer-1 sweep: every time-2
 *    spin is a T2 lookup from the seed bits.
 *  - Layer 2 is computed straight from lookups for levels 0..p-4 and written
 *    once: S3(x) = U2(T2[x], (T2[parent].z + T2[child1].z) + T2[child2].z).
 *  - Level p-3 is never stored: its time-3 z is read only by level p-4 at
 *    layer 3, so it is computed there - the same one sincos per vertex that
 *    the reference spends on it at layer 2, just later.
 *  - Layers 3..p-1 are the reference's in-place top-down update, restricted
 *    to levels 0..p-4.
 *
 *      sincos calls / sample :  2^p - 2p             (reference 4*2^p - 2p - 4)
 *      memory / thread       :  24 B * (2^(p-2) - 2)          spins, levels 0..p-4
 *                             +  8 B * (ceil((4*2^p-2)/64) + 1) seed bits
 *                             +  8 B * 2 * 2^(p-4)            z-scratch
 *                             ~  7.5 * 2^p bytes             (reference 56.5 * 2^p)
 *           p=26: 0.47 GB   p=28: 1.9 GB   p=30: 7.5 GB   p=32: 30 GB
 *
 *  p < 4 uses the reference kernel, which is kept below unchanged except that
 *  it calls sincos() explicitly.  VIRAL_FORCE_REF=1 uses it at every p.
 *
 *  Self-check.  At start-up the table kernel and the reference kernel run
 *  VIRAL_SELFCHECK samples (default 64, 0 disables) at depth min(p, 12), with
 *  the first layers of the given angles, and the program stops if any bit of
 *  any sample differs.  The banner reports the result.
 *
 *  Exactness also needs the rotation arithmetic of the frozen campaign binary
 *  (gcc 14.3.1 -O3 -fopenmp on RHEL 10, default ISA x86-64-v3), which fuses
 *  its multiply-adds into FMAs.  rz/rx_cs below spell that fusion out with
 *  fma() (see there), and the file is compiled with -ffp-contract=off so that
 *  the compiler adds none of its own: the arithmetic is then the same at
 *  every site - reference kernel, tables and table kernel alike.
 *  Validation: tests/test_fast_equivalence.py compares whole per-sample dumps
 *  and result files with the frozen viral_tree_hpc binary.
 *
 * ----------------------------------------------------------------------------
 *  THE FUSED KERNEL (p >= 6)
 * ----------------------------------------------------------------------------
 *
 *  Let L = p-4 and M = p-5.  The table kernel stores the time-3 spins of
 *  levels 0..L.  Level L's time-3 spin is a pure function of seed bits (its
 *  T2 value, its parent's and its children's), and it is used only at layer 3:
 *  as the children's z of level M, and for level L's own last update.  So
 *  the fused kernel stores spins of levels 0..M only, and at layer 3 it
 *  processes each vertex y of level M together with its two children x:
 *    S3(x)   = U2(T2[x], (T2[y].z + T2[x's child1].z) + T2[x's child2].z)
 *    z3(c)   for the four grandchildren c (level p-3), as the table kernel does
 *    S4(y)   = U3(S3(y), (zpar + S3(x1).z) + S3(x2).z)
 *    S4(x)   = U3(S3(x), (S3(y).z + z3(c1)) + z3(c2))     - only its z is kept
 *  Every sincos and every association order is the one the reference uses,
 *  and every vertex update still happens exactly once.  Level L then holds
 *  just its time-4 z, which level M reads at layer 4.
 *
 *      memory / sample :  24 B * (2^(p-3) - 2)           spins, levels 0..p-5
 *                       +  8 B * 2^(p-3)                 time-4 z of level p-4
 *                       +  8 B * (ceil((4*2^p-2)/64) + 1) seed bits
 *                       +  8 B * 2 * 2^(p-5)             z-scratch
 *                       ~  5 * 2^p bytes
 *           p=28: 1.25 GB   p=30: 5.0 GB   p=32: 20 GB
 *
 *  Threads.  Within one level every vertex reads only the previous level's
 *  saved z, its own old state, its children (not yet touched this layer), the
 *  tables and the seed bits, and writes only its own slots.  So the k-loop of
 *  each level is split statically over the threads, with the implicit
 *  barrier of the OpenMP loop between levels.  The arithmetic of each vertex
 *  is unchanged, so the results do not depend on the number of threads.
 *  Levels with fewer than VIRAL_PAR_MIN vertices (default 2048) run on one
 *  thread; VIRAL_PAR_MIN=1 splits every level (used by the tests).  The seed
 *  bits are drawn by one thread.
 *
 *  Kernels: reference for p < 4, table kernel for p = 4, 5, fused kernel for
 *  p >= 6.  VIRAL_KERNEL=ref|tables|fused forces one (where it applies).
 * ============================================================================
 */

#define _GNU_SOURCE          /* for sincos() on glibc */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <omp.h>

#define P_MAX                 40      /* sanity limit on the depth              */
/* the next four can be overridden with -D at compile time (used by check.sh) */
#ifndef REPORT_SEC
#define REPORT_SEC            60.0    /* progress line to stdout at most so often */
#endif
#ifndef CKPT_SEC
#define CKPT_SEC              60.0    /* output file rewritten at most so often   */
#endif
#ifndef BATCH_TARGET_SEC
#define BATCH_TARGET_SEC      20.0    /* batch size grows until a batch takes this */
#endif
#ifndef BATCH_MAX_PER_THREAD
#define BATCH_MAX_PER_THREAD  64      /* ... but never beyond this many per thread */
#endif
#define N_SAMPLES_MAX         4294967295ULL   /* k must fit in 32 bits */
#define FAST_MIN_P            4       /* smallest depth of the table kernel         */
#define FUSED_MIN_P           6       /* smallest depth of the fused kernel         */
#define PAR_MIN_DEFAULT       2048    /* levels smaller than this run on one thread */
#define SELFCHECK_P           12      /* depth of the start-up self-check (<= p)    */
#define SELFCHECK_DEFAULT     64      /* samples in the start-up self-check         */

/* ───────────────────────────── spins ───────────────────────────────────── */

typedef struct { double x, y, z; } Vec3;

/* R_z by angle th, with the reference binary's exact arithmetic.
 *
 * sincos() is called explicitly: the reference binary calls sincos as well
 * (GCC fuses the cos/sin pair of its rz), and glibc's cos() and sincos() can
 * differ in the last bit.
 *
 * The multiply-adds are written as explicit fma() calls.  The frozen binary
 * (gcc 14.3.1 -O3, whose RHEL 10 default ISA x86-64-v3 has FMA) contracts
 * every "a*b -/+ c*d" of rz and rx_cs as fma(a, b, -/+ round(c*d)) - the
 * first product fused, the second rounded - at all of its sites (read from
 * its disassembly).  Leaving the contraction to the compiler is not enough:
 * GCC fuses differently at different inlined sites, which changes last bits.
 * Compile this file with -ffp-contract=off so that these fma() calls are the
 * only fused operations.                                                    */
static inline Vec3 rz(Vec3 s, double th) {
    double c, sn;
#if defined(__GLIBC__)
    sincos(th, &sn, &c);
#else
    c = cos(th);  sn = sin(th);
#endif
    return (Vec3){ fma(c, s.x, -(sn*s.y)),  fma(sn, s.x, c*s.y),  s.z };
}
/* R_x with cos/sin of the angle precomputed (beta_n is the same for every
 * vertex of a layer).                                                       */
static inline Vec3 rx_cs(Vec3 s, double c, double sn) {
    return (Vec3){ s.x,  fma(c, s.y, -(sn*s.z)),  fma(sn, s.y, c*s.z) };
}

/* ─────────────────────── random numbers: xoshiro256** ──────────────────── */

typedef struct { uint64_t s[4]; } Rng;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(Rng *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;    s[3] = rotl(s[3], 45);
    return result;
}

/* splitmix64 finalizer (see viral_tree_hpc.c) */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

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

/* Seed of the sample with global index S = job * 2^32 + k. */
static inline Rng sample_rng(int p, uint64_t S) {
    return rng_seed(mix64(0x5DEECE66DULL ^ ((uint64_t)p << 32) ^ mix64(S)));
}

/* ───────────────────────────── geometry ────────────────────────────────── */

static int     P;                /* depth                                        */
static size_t  NTOT;             /* 4*2^P - 2 : all vertices, leaves included    */
static size_t  NST;              /* 2*2^P - 2 : vertices of levels 0..P-1        */
static size_t  NW;               /* 64-bit words holding one seed bit per vertex */
static size_t  ZS;               /* reference z-scratch length = 2^(P-1)         */
static size_t  OFF[P_MAX + 1];   /* OFF[d]  = flat index of the first vertex of level d */
static size_t  NLEV[P_MAX + 1];  /* NLEV[d] = number of vertices of level d            */

/* (Re)builds the geometry for depth p.  The self-check switches depth. */
static void build_geometry(int p) {
    P = p;
    NLEV[0] = 2;  OFF[0] = 0;
    for (int d = 1; d <= p; d++) {
        NLEV[d] = (size_t)1 << (d + 1);
        OFF[d]  = OFF[d - 1] + NLEV[d - 1];
    }
    NST  = OFF[p];                  /* = 2 + 4 + ... + 2^p = 2^(p+1) - 2       */
    NTOT = OFF[p] + NLEV[p];        /* + 2^(p+1)            = 2^(p+2) - 2      */
    NW   = (NTOT + 63) / 64;
    ZS   = (p >= 2) ? ((size_t)1 << (p - 1)) : 2;   /* largest saved level is P-2 */
}

/* the three kernels and their workspaces                                    */
enum { K_REF, K_TABLES, K_FUSED };
static const char *KERNEL_NAME[] = { "reference", "tables (exact layers 0-1, layer 2 from lookups)",
                                     "fused (tables, levels below p-5 never stored)" };

/* spins stored: reference levels 0..P-1, tables 0..P-4, fused 0..P-5 */
static size_t kernel_nst(int k) { return k == K_REF ? NST : k == K_TABLES ? OFF[P - 3] : OFF[P - 4]; }
/* z-scratch length: the largest level that saves its z, at least 2 (the roots) */
static size_t kernel_nzs(int k) {
    const size_t z = k == K_REF ? ZS : (size_t)1 << (P - (k == K_TABLES ? 4 : 5));
    return (z < 2) ? 2 : z;
}
/* z-only array: time-4 z of level P-4 (fused kernel only) */
static size_t kernel_nzl(int k) { return k == K_FUSED ? NLEV[P - 4] : 0; }
static size_t kernel_nw(int k)  { return k == K_REF ? NW : NW + 1; }

static size_t bytes_per_sample(int k) {
    return kernel_nst(k) * sizeof(Vec3) + kernel_nw(k) * sizeof(uint64_t)
         + (2 * kernel_nzs(k) + kernel_nzl(k)) * sizeof(double);
}

/* ───────────────────────────── angles ──────────────────────────────────── */

typedef struct {
    double  alpha, ca, sa;          /* alpha, cos(alpha), sin(alpha)          */
    double *beta, *gamma;           /* beta[n], gamma[n], n = 0..P-1          */
    double *cb, *sb;                /* cos(beta[n]), sin(beta[n])             */
} Angles;

/* File format (as written by the training code):
 *     line 1 : "# alpha ..."  comment
 *     line 2 : alpha
 *     line 3 : "# layer beta gamma"  comment
 *     then p lines:  <layer number 1..p>  <beta>  <gamma>                  */
static int load_angles(const char *fname, int p, Angles *A) {
    FILE *f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open angles file '%s'\n", fname); return 0; }
    char line[1024];
    if (!fgets(line, sizeof line, f))               { fclose(f); goto bad; }
    if (fscanf(f, "%lf", &A->alpha) != 1)           { fclose(f); goto bad; }
    if (!fgets(line, sizeof line, f))               { fclose(f); goto bad; }  /* rest of line 2 */
    if (!fgets(line, sizeof line, f))               { fclose(f); goto bad; }  /* line 3 */
    for (int n = 0; n < p; n++) {
        int layer;
        if (fscanf(f, "%d %lf %lf", &layer, &A->beta[n], &A->gamma[n]) != 3) {
            fprintf(stderr, "ERROR: '%s' has fewer than %d layers (stopped at layer %d)\n",
                    fname, p, n);
            fclose(f); return 0;
        }
        if (layer != n + 1) {
            fprintf(stderr, "ERROR: '%s': expected layer %d, found %d\n", fname, n + 1, layer);
            fclose(f); return 0;
        }
    }
    fclose(f);
    /* cos/sin of the fixed angles with sincos(), exactly as the reference. */
#if defined(__GLIBC__)
    sincos(A->alpha, &A->sa, &A->ca);
    for (int n = 0; n < p; n++) sincos(A->beta[n], &A->sb[n], &A->cb[n]);
#else   /* other libms: plain sin/cos (may differ from glibc in the last bit) */
    A->sa = sin(A->alpha);  A->ca = cos(A->alpha);
    for (int n = 0; n < p; n++) { A->sb[n] = sin(A->beta[n]); A->cb[n] = cos(A->beta[n]); }
#endif
    return 1;
bad:
    fprintf(stderr, "ERROR: '%s' is not a valid angles file\n", fname);
    return 0;
}

/* FNV-1a hash of a whole file; used to make sure a resumed run uses the
 * same angles as the run it continues.                                     */
static int file_hash(const char *fname, uint64_t *h) {
    FILE *f = fopen(fname, "rb");
    if (!f) return 0;
    uint64_t x = 0xcbf29ce484222325ULL;
    int c;
    while ((c = fgetc(f)) != EOF) { x ^= (uint64_t)(unsigned char)c; x *= 0x100000001b3ULL; }
    fclose(f);
    *h = x;
    return 1;
}

/* ───────────────────────────── workspace ───────────────────────────────── */

typedef struct {
    Vec3     *st;       /* spins of the stored levels (see kernel_nst)          */
    uint64_t *w;        /* seed bits of ALL vertices (+1 word for the tables)   */
    double   *zs[2];    /* old z of the level above the one being updated       */
    double   *zl;       /* fused kernel: time-4 z of level P-4                  */
    size_t    nst, nw, nzs, nzl;
} Work;

static int alloc_work(Work *W, int kernel) {
    W->nst = kernel_nst(kernel);
    W->nw  = kernel_nw(kernel);
    W->nzs = kernel_nzs(kernel);
    W->nzl = kernel_nzl(kernel);
    W->st    = malloc(W->nst * sizeof(Vec3));
    W->w     = malloc(W->nw  * sizeof(uint64_t));
    W->zs[0] = malloc(W->nzs * sizeof(double));
    W->zs[1] = malloc(W->nzs * sizeof(double));
    W->zl    = W->nzl ? malloc(W->nzl * sizeof(double)) : NULL;
    return W->st && W->w && W->zs[0] && W->zs[1] && (W->zl || !W->nzl);
}

static void free_work(Work *W) {
    free(W->st);  free(W->w);  free(W->zs[0]);  free(W->zs[1]);  free(W->zl);
    memset(W, 0, sizeof *W);
}

/* zero every page with the threads that will use it (first-touch NUMA) */
static void touch(void *p, size_t bytes) {
    char *c = p;
    const size_t chunk = (size_t)1 << 20;
    const long   nch   = (long)((bytes + chunk - 1) / chunk);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < nch; i++) {
        const size_t a = (size_t)i * chunk, b = (a + chunk < bytes) ? a + chunk : bytes;
        memset(c + a, 0, b - a);
    }
}

static void touch_work(Work *W) {
    touch(W->st,    W->nst * sizeof(Vec3));
    touch(W->w,     W->nw  * sizeof(uint64_t));
    touch(W->zs[0], W->nzs * sizeof(double));
    touch(W->zs[1], W->nzs * sizeof(double));
    if (W->zl) touch(W->zl, W->nzl * sizeof(double));
}

/* seed sign of the vertex with flat index i, as a z-component */
static inline double seed_z(const uint64_t *w, size_t i, double sa) {
    return ((w[i >> 6] >> (i & 63)) & 1ULL) ? sa : -sa;
}

/* ──────────────────── reference kernel (viral_tree_hpc.c) ─────────────────
 * Returns 1 if the root edge is cut, 0 otherwise; *soft_out = s_u^z s_v^z. */
static int one_sample_ref(Rng *rng, const Angles *A, Work *W, double *soft_out)
{
    Vec3     *st = W->st;
    uint64_t *w  = W->w;
    const double ca = A->ca, sa = A->sa;

    /* 1. seeds: one random bit per vertex, drawn in flat-index order, 64 per
     *    word.  (Bit i of the sequence = eps of vertex i.)                  */
    for (size_t m = 0; m < NW; m++) w[m] = rng_next(rng);

    /* 2. initial states of the stored levels 0..P-1 */
    for (size_t i = 0; i < NST; i++)
        st[i] = (Vec3){ ca, 0.0, seed_z(w, i, sa) };
    /* leaves (flat indices NST..NTOT-1) are read from w[] directly at layer 0 */

    /* 3. the p layers, in place, top-down, inside the light cone */
    for (int n = 0; n < P; n++) {
        const double g  = A->gamma[n];
        const double cb = A->cb[n], sb = A->sb[n];
        const int    Lmax = P - n - 1;      /* deepest level that must be updated */

        /* level 0: u = st[0], v = st[1].  Neighbours of u: v and level-1
         * vertices 0,1;  of v: u and level-1 vertices 2,3.  Both fields are
         * computed from OLD values before either root is overwritten.      */
        {
            const double zu = st[0].z, zv = st[1].z;
            double c0, c1, c2, c3;
            if (P >= 2) {                       /* level 1 is a stored level  */
                const Vec3 *l1 = st + OFF[1];
                c0 = l1[0].z; c1 = l1[1].z; c2 = l1[2].z; c3 = l1[3].z;
            } else {                            /* P == 1: level 1 = leaves   */
                c0 = seed_z(w, OFF[1] + 0, sa); c1 = seed_z(w, OFF[1] + 1, sa);
                c2 = seed_z(w, OFF[1] + 2, sa); c3 = seed_z(w, OFF[1] + 3, sa);
            }
            const double hu = zv + c0 + c1;     /* same association order as   */
            const double hv = zu + c2 + c3;     /* the reference code          */
            if (Lmax >= 1) { W->zs[0][0] = zu; W->zs[0][1] = zv; }  /* level 1 needs them */
            st[0] = rx_cs(rz(st[0], g * hu), cb, sb);
            st[1] = rx_cs(rz(st[1], g * hv), cb, sb);
        }

        /* levels 1..Lmax (see viral_tree_hpc.c section 4.4) */
        for (int d = 1; d <= Lmax; d++) {
            const size_t  off  = OFF[d], nd = NLEV[d];
            const double *zpar = W->zs[(d - 1) & 1];
            double       *save = (d < Lmax) ? W->zs[d & 1] : NULL;
            Vec3         *lv   = st + off;

            if (d + 1 < P) {                    /* children are stored states  */
                const Vec3 *ch = st + OFF[d + 1];
                if (save) {
                    for (size_t k = 0; k < nd; k++) {
                        const double h = zpar[k >> 1] + ch[2*k].z + ch[2*k + 1].z;
                        save[k] = lv[k].z;
                        lv[k]   = rx_cs(rz(lv[k], g * h), cb, sb);
                    }
                } else {
                    for (size_t k = 0; k < nd; k++) {
                        const double h = zpar[k >> 1] + ch[2*k].z + ch[2*k + 1].z;
                        lv[k] = rx_cs(rz(lv[k], g * h), cb, sb);
                    }
                }
            } else {                            /* d = P-1 (so n = 0): children are leaves */
                const size_t lf = OFF[P];
                for (size_t k = 0; k < nd; k++) {
                    const double h = zpar[k >> 1] + seed_z(w, lf + 2*k, sa)
                                                  + seed_z(w, lf + 2*k + 1, sa);
                    lv[k] = rx_cs(rz(lv[k], g * h), cb, sb);
                }
            }
        }
    }

    /* 4. read the roots */
    const double zu = st[0].z, zv = st[1].z;
    *soft_out = zu * zv;
    const int su = (zu >= 0.0) ? 1 : -1;
    const int sv = (zv >= 0.0) ? 1 : -1;
    return (su != sv) ? 1 : 0;
}

/* ───────────────────── table kernel: the exact tables ──────────────────── */

typedef struct {
    Vec3 t1[2][4];      /* time-1 spin: [own seed bit][number of + neighbours]   */
    Vec3 t2[1024];      /* time-2 spin: [radius-2 ball], bit layout of t2_idx    */
} Tables;

/* T2 index bit layout:  0 self, 1 parent, 2 grandparent, 3 sibling,
 * 4 child 1, 5 child 2, 6-7 the children of child 1, 8-9 those of child 2.
 * Builds both tables; returns 0 if the time-1 symmetry does not hold.       */
static int build_tables(const Angles *A, Tables *T)
{
    volatile double vzero = 0.0;        /* the y = 0 of the initial spin, opaque */
    const double sa = A->sa, ca = A->ca;

    /* layer-0 field: (a + b) + c over the three neighbours, each +sa or -sa;
     * h0[m] for m = number of +sa, and every order must give the same bits */
    double h0[4];
    for (int m = 0; m < 4; m++) {
        const double a = (m > 0) ? sa : -sa, b = (m > 1) ? sa : -sa, c = (m > 2) ? sa : -sa;
        h0[m] = (a + b) + c;
    }
    for (int t = 0; t < 8; t++) {
        const double a = (t & 1) ? sa : -sa, b = (t & 2) ? sa : -sa, c = (t & 4) ? sa : -sa;
        const double h = (a + b) + c;
        const int    m = (t & 1) + ((t >> 1) & 1) + ((t >> 2) & 1);
        if (memcmp(&h, &h0[m], sizeof h) != 0) return 0;
    }

    /* time 1: layer 0 applied to (ca, 0, eps sa), exactly as the reference */
    for (int b = 0; b < 2; b++)
        for (int m = 0; m < 4; m++) {
            const Vec3 s0 = { ca, vzero, b ? sa : -sa };
            T->t1[b][m] = rx_cs(rz(s0, A->gamma[0] * h0[m]), A->cb[0], A->sb[0]);
        }

    /* time 2: layer 1 on the time-1 spins of the radius-2 ball */
    for (unsigned i = 0; i < 1024; i++) {
        const unsigned x  = i & 1u,         pa = (i >> 1) & 1u, gp = (i >> 2) & 1u,
                       si = (i >> 3) & 1u,  c1 = (i >> 4) & 1u, c2 = (i >> 5) & 1u,
                       a1 = (i >> 6) & 1u,  b1 = (i >> 7) & 1u,
                       a2 = (i >> 8) & 1u,  b2 = (i >> 9) & 1u;
        const Vec3   s1 = T->t1[x][pa + c1 + c2];
        const double zp = T->t1[pa][gp + x + si].z;      /* parent: grandparent, self, sibling */
        const double z1 = T->t1[c1][x + a1 + b1].z;      /* child 1: self and its children     */
        const double z2 = T->t1[c2][x + a2 + b2].z;
        const double h  = zp + z1 + z2;                  /* (parent + child1) + child2        */
        T->t2[i] = rx_cs(rz(s1, A->gamma[1] * h), A->cb[1], A->sb[1]);
    }
    return 1;
}

/* seed bit of flat index i, and n <= 32 consecutive seed bits from i */
static inline unsigned bit1(const uint64_t *w, size_t i) {
    return (unsigned)(w[i >> 6] >> (i & 63)) & 1u;
}
static inline unsigned bitsn(const uint64_t *w, size_t i, unsigned n) {
    const size_t   q = i >> 6;
    const unsigned r = (unsigned)(i & 63);
    uint64_t v = w[q] >> r;
    if (r + n > 64) v |= w[q + 1] << (64 - r);
    return (unsigned)(v & ((1ULL << n) - 1));
}

/* T2 index of vertex k of level d (0 <= d <= P-2). */
static inline unsigned t2_idx(const uint64_t *w, int d, size_t k)
{
    unsigned self, par, gpar, sib;
    if (d >= 2) {
        self = bit1(w, OFF[d] + k);
        sib  = bit1(w, OFF[d] + (k ^ 1));
        par  = bit1(w, OFF[d - 1] + (k >> 1));
        gpar = bit1(w, OFF[d - 2] + (k >> 2));
    } else if (d == 1) {                        /* parent is a root              */
        self = bit1(w, OFF[1] + k);
        sib  = bit1(w, OFF[1] + (k ^ 1));
        par  = bit1(w, k >> 1);
        gpar = bit1(w, 1 - (k >> 1));           /* the other root                */
    } else {                                    /* a root: k = 0 (u) or 1 (v)    */
        self = bit1(w, k);
        par  = bit1(w, 1 - k);                  /* the other root ...            */
        gpar = bit1(w, OFF[1] + 2 * (1 - k));   /* ... and its two children      */
        sib  = bit1(w, OFF[1] + 2 * (1 - k) + 1);
    }
    return self | (par << 1) | (gpar << 2) | (sib << 3)
         | (bitsn(w, OFF[d + 1] + 2 * k, 2) << 4)
         | (bitsn(w, OFF[d + 2] + 4 * k, 4) << 6);
}

/* T2 index of child j (0, 1) of a vertex with index ix; gc4 = the four seed
 * bits of that child's grandchildren.  Valid at every level, roots included. */
static inline unsigned t2_child_idx(unsigned ix, unsigned j, unsigned gc4)
{
    return ((ix >> (4 + j)) & 1u)               /* self        = that child        */
         | ((ix & 1u) << 1)                     /* parent      = the vertex        */
         | (((ix >> 1) & 1u) << 2)              /* grandparent = its parent        */
         | (((ix >> (5 - j)) & 1u) << 3)        /* sibling     = the other child   */
         | (((ix >> (6 + 2 * j)) & 3u) << 4)    /* children    = its grandchildren */
         | (gc4 << 6);
}

/* Time-3 z of vertex j of level e = P-3 (never stored), from T2.  Used for
 * the roots' children when P = 4.                                           */
static double lazy_z3(const uint64_t *w, const Angles *A, const Tables *T, int e, size_t j)
{
    const Vec3  *t2 = T->t2;
    const double h  = t2[t2_idx(w, e - 1, j >> 1)].z
                    + t2[t2_idx(w, e + 1, 2 * j)].z
                    + t2[t2_idx(w, e + 1, 2 * j + 1)].z;
    return rx_cs(rz(t2[t2_idx(w, e, j)], A->gamma[2] * h), A->cb[2], A->sb[2]).z;
}

/* ─────────────────────── table kernel: one sample ─────────────────────────
 * Same contract as one_sample_ref.  Requires P >= FAST_MIN_P.               */
static int one_sample_fast(Rng *rng, const Angles *A, const Tables *T, Work *W, double *soft_out)
{
    Vec3       *st = W->st;
    uint64_t   *w  = W->w;
    const Vec3 *t2 = T->t2;
    const int   L  = P - 4;                /* deepest stored level */

    /* 1. seeds: the reference's stream and layout, bit i = eps of vertex i */
    for (size_t m = 0; m < NW; m++) w[m] = rng_next(rng);

    /* 2. layer 2 (time 2 -> 3) for levels 0..L, every time-2 input from T2.
     *    Level d keeps its time-2 z in zs[d&1] for level d+1.                */
    {
        const double g = A->gamma[2], cb = A->cb[2], sb = A->sb[2];
        {
            const unsigned iu = t2_idx(w, 0, 0), iv = t2_idx(w, 0, 1);
            const double zu = t2[iu].z, zv = t2[iv].z;
            const double c0 = t2[t2_idx(w, 1, 0)].z, c1 = t2[t2_idx(w, 1, 1)].z;
            const double c2 = t2[t2_idx(w, 1, 2)].z, c3 = t2[t2_idx(w, 1, 3)].z;
            const double hu = zv + c0 + c1;
            const double hv = zu + c2 + c3;
            if (L >= 1) { W->zs[0][0] = zu; W->zs[0][1] = zv; }
            st[0] = rx_cs(rz(t2[iu], g * hu), cb, sb);
            st[1] = rx_cs(rz(t2[iv], g * hv), cb, sb);
        }
        for (int d = 1; d <= L; d++) {
            const size_t  off  = OFF[d], nd = NLEV[d], off3 = OFF[d + 3];
            const double *zpar = W->zs[(d - 1) & 1];
            double       *save = (d < L) ? W->zs[d & 1] : NULL;
            for (size_t k = 0; k < nd; k++) {
                const unsigned ix = t2_idx(w, d, k);
                const unsigned g8 = bitsn(w, off3 + 8 * k, 8);   /* grandchildren of both children */
                const double   z1 = t2[t2_child_idx(ix, 0, g8 & 15u)].z;
                const double   z2 = t2[t2_child_idx(ix, 1, g8 >> 4)].z;
                const double   h  = zpar[k >> 1] + z1 + z2;
                if (save) save[k] = t2[ix].z;
                st[off + k] = rx_cs(rz(t2[ix], g * h), cb, sb);
            }
        }
    }

    /* 3. layers 3..P-1: the reference's in-place top-down update over levels
     *    0..L.  At layer 3 the children of level L (level P-3, not stored)
     *    get their time-3 z here, from T2.                                   */
    for (int n = 3; n < P; n++) {
        const double g  = A->gamma[n];
        const double cb = A->cb[n], sb = A->sb[n];
        const int    Lmax = P - n - 1;      /* deepest level that must be updated */

        {
            const double zu = st[0].z, zv = st[1].z;
            double c0, c1, c2, c3;
            if (L >= 1) {                       /* level 1 is stored             */
                const Vec3 *l1 = st + OFF[1];
                c0 = l1[0].z; c1 = l1[1].z; c2 = l1[2].z; c3 = l1[3].z;
            } else {                            /* P == 4, n == 3: level 1 = P-3 */
                c0 = lazy_z3(w, A, T, 1, 0); c1 = lazy_z3(w, A, T, 1, 1);
                c2 = lazy_z3(w, A, T, 1, 2); c3 = lazy_z3(w, A, T, 1, 3);
            }
            const double hu = zv + c0 + c1;
            const double hv = zu + c2 + c3;
            if (Lmax >= 1) { W->zs[0][0] = zu; W->zs[0][1] = zv; }
            st[0] = rx_cs(rz(st[0], g * hu), cb, sb);
            st[1] = rx_cs(rz(st[1], g * hv), cb, sb);
        }

        for (int d = 1; d <= Lmax; d++) {
            const size_t  off  = OFF[d], nd = NLEV[d];
            const double *zpar = W->zs[(d - 1) & 1];
            double       *save = (d < Lmax) ? W->zs[d & 1] : NULL;
            Vec3         *lv   = st + off;

            if (d < L) {                        /* children are stored spins     */
                const Vec3 *ch = st + OFF[d + 1];
                if (save) {
                    for (size_t k = 0; k < nd; k++) {
                        const double h = zpar[k >> 1] + ch[2*k].z + ch[2*k + 1].z;
                        save[k] = lv[k].z;
                        lv[k]   = rx_cs(rz(lv[k], g * h), cb, sb);
                    }
                } else {
                    for (size_t k = 0; k < nd; k++) {
                        const double h = zpar[k >> 1] + ch[2*k].z + ch[2*k + 1].z;
                        lv[k] = rx_cs(rz(lv[k], g * h), cb, sb);
                    }
                }
            } else {                            /* d == L, hence n == 3 and no save */
                const double g2 = A->gamma[2], cb2 = A->cb[2], sb2 = A->sb[2];
                const size_t o1 = OFF[P - 1], o0 = OFF[P];
                for (size_t k = 0; k < nd; k++) {
                    const unsigned ix  = t2_idx(w, d, k);
                    const unsigned g8  = bitsn(w, o1 + 8 * k, 8);    /* children's grandchildren       */
                    const unsigned g16 = bitsn(w, o0 + 16 * k, 16);  /* grandchildren's grandchildren  */
                    const double   zx  = t2[ix].z;                   /* this vertex at time 2          */
                    double zc[2];
                    for (unsigned j = 0; j < 2; j++) {
                        const unsigned ic = t2_child_idx(ix, j, (g8 >> (4 * j)) & 15u);
                        const double   za = t2[t2_child_idx(ic, 0, (g16 >> (8 * j))     & 15u)].z;
                        const double   zb = t2[t2_child_idx(ic, 1, (g16 >> (8 * j + 4)) & 15u)].z;
                        const double   hc = zx + za + zb;
                        zc[j] = rx_cs(rz(t2[ic], g2 * hc), cb2, sb2).z;
                    }
                    const double h = zpar[k >> 1] + zc[0] + zc[1];
                    lv[k] = rx_cs(rz(lv[k], g * h), cb, sb);
                }
            }
        }
    }

    /* 4. read the roots */
    const double zu = st[0].z, zv = st[1].z;
    *soft_out = zu * zv;
    const int su = (zu >= 0.0) ? 1 : -1;
    const int sv = (zv >= 0.0) ? 1 : -1;
    return (su != sv) ? 1 : 0;
}

/* ─────────────────────── fused kernel: one sample ─────────────────────────
 * Same contract as one_sample_ref; requires P >= FUSED_MIN_P.  See THE FUSED
 * KERNEL in the header.  Levels with at least g_par_min vertices are split
 * statically over the OpenMP threads; every vertex is computed with exactly
 * the same operations whichever thread does it.                             */
static size_t g_par_min = PAR_MIN_DEFAULT;

static int one_sample_fused(Rng *rng, const Angles *A, const Tables *T, Work *W, double *soft_out)
{
    Vec3        *st = W->st;
    uint64_t    *w  = W->w;
    double      *zl = W->zl;
    const Vec3  *t2 = T->t2;
    const int    L  = P - 4;               /* kept only as its time-4 z, in zl  */
    const int    M  = P - 5;               /* deepest level with stored spins   */
    const size_t par_min = g_par_min;

    /* 1. seeds (one thread): the reference's stream and layout */
    for (size_t m = 0; m < NW; m++) w[m] = rng_next(rng);

    /* 2. layer 2 (time 2 -> 3) for levels 0..M, every time-2 input from T2.
     *    Level d keeps its time-2 z in zs[d&1] for level d+1.                */
    {
        const double g = A->gamma[2], cb = A->cb[2], sb = A->sb[2];
        {
            const unsigned iu = t2_idx(w, 0, 0), iv = t2_idx(w, 0, 1);
            const double zu = t2[iu].z, zv = t2[iv].z;
            const double c0 = t2[t2_idx(w, 1, 0)].z, c1 = t2[t2_idx(w, 1, 1)].z;
            const double c2 = t2[t2_idx(w, 1, 2)].z, c3 = t2[t2_idx(w, 1, 3)].z;
            const double hu = zv + c0 + c1;
            const double hv = zu + c2 + c3;
            W->zs[0][0] = zu;  W->zs[0][1] = zv;          /* M >= 1: level 1 needs them */
            st[0] = rx_cs(rz(t2[iu], g * hu), cb, sb);
            st[1] = rx_cs(rz(t2[iv], g * hv), cb, sb);
        }
        for (int d = 1; d <= M; d++) {
            const size_t  off  = OFF[d], nd = NLEV[d], off3 = OFF[d + 3];
            const double *zpar = W->zs[(d - 1) & 1];
            double       *save = (d < M) ? W->zs[d & 1] : NULL;
            #pragma omp parallel for schedule(static) if(nd >= par_min)
            for (size_t k = 0; k < nd; k++) {
                const unsigned ix = t2_idx(w, d, k);
                const unsigned g8 = bitsn(w, off3 + 8 * k, 8);   /* grandchildren of both children */
                const double   z1 = t2[t2_child_idx(ix, 0, g8 & 15u)].z;
                const double   z2 = t2[t2_child_idx(ix, 1, g8 >> 4)].z;
                const double   h  = zpar[k >> 1] + z1 + z2;
                if (save) save[k] = t2[ix].z;
                st[off + k] = rx_cs(rz(t2[ix], g * h), cb, sb);
            }
        }
    }

    /* 3. layers 3..P-1, in place, top-down.  Layer 3 updates level M together
     *    with its children (level L, computed here, kept as time-4 z in zl)
     *    and grandchildren (level L+1, time-3 z only).  At layer 4 level M
     *    reads its children's z from zl.  Everything else is the reference's
     *    update on stored spins.                                             */
    for (int n = 3; n < P; n++) {
        const double g  = A->gamma[n];
        const double cb = A->cb[n], sb = A->sb[n];
        const int    Lmax = P - n - 1;      /* deepest level that must be updated */
        const int    top  = (Lmax < M) ? Lmax : M;

        {
            const double zu = st[0].z, zv = st[1].z;
            const Vec3  *l1 = st + OFF[1];   /* level 1 is stored (M >= 1)      */
            const double c0 = l1[0].z, c1 = l1[1].z, c2 = l1[2].z, c3 = l1[3].z;
            const double hu = zv + c0 + c1;
            const double hv = zu + c2 + c3;
            if (Lmax >= 1) { W->zs[0][0] = zu; W->zs[0][1] = zv; }
            st[0] = rx_cs(rz(st[0], g * hu), cb, sb);
            st[1] = rx_cs(rz(st[1], g * hv), cb, sb);
        }

        for (int d = 1; d <= top; d++) {
            const size_t  off  = OFF[d], nd = NLEV[d];
            const double *zpar = W->zs[(d - 1) & 1];
            double       *save = (d < Lmax && d < M) ? W->zs[d & 1] : NULL;
            Vec3         *lv   = st + off;

            if (n == 3 && d == M) {             /* fused: level M, L and L+1    */
                const double g2 = A->gamma[2], cb2 = A->cb[2], sb2 = A->sb[2];
                const size_t oL2 = OFF[L + 2], oP1 = OFF[P - 1], oP = OFF[P];
                #pragma omp parallel for schedule(static) if(nd >= par_min)
                for (size_t k = 0; k < nd; k++) {
                    const unsigned iy  = t2_idx(w, d, k);
                    const unsigned g8  = bitsn(w, oL2 + 8 * k, 8);     /* grandchildren of y's children   */
                    const unsigned g16 = bitsn(w, oP1 + 16 * k, 16);   /* ... of y's grandchildren        */
                    const unsigned g32 = bitsn(w, oP + 32 * k, 32);    /* ... of y's great-grandchildren  */
                    const double   zy2 = t2[iy].z;                     /* y at time 2                     */
                    const Vec3     sy3 = lv[k];                        /* y at time 3                     */
                    Vec3   sx3[2];
                    double zc3[2][2];
                    for (unsigned j = 0; j < 2; j++) {
                        const unsigned ix = t2_child_idx(iy, j, (g8 >> (4 * j)) & 15u);
                        unsigned ic[2];
                        for (unsigned i = 0; i < 2; i++)
                            ic[i] = t2_child_idx(ix, i, (g16 >> (8 * j + 4 * i)) & 15u);
                        /* x at time 3, as the table kernel's layer 2 computes it */
                        const double hx = zy2 + t2[ic[0]].z + t2[ic[1]].z;
                        sx3[j] = rx_cs(rz(t2[ix], g2 * hx), cb2, sb2);
                        /* x's children at time 3 (z only), as its lazy branch does */
                        const double zx2 = t2[ix].z;
                        for (unsigned i = 0; i < 2; i++) {
                            const unsigned sh = 16 * j + 8 * i;
                            const double   za = t2[t2_child_idx(ic[i], 0, (g32 >> sh) & 15u)].z;
                            const double   zb = t2[t2_child_idx(ic[i], 1, (g32 >> (sh + 4)) & 15u)].z;
                            const double   hc = zx2 + za + zb;
                            zc3[j][i] = rx_cs(rz(t2[ic[i]], g2 * hc), cb2, sb2).z;
                        }
                    }
                    /* layer 3: y first, then its children, which read y's OLD z */
                    const double hy = zpar[k >> 1] + sx3[0].z + sx3[1].z;
                    lv[k] = rx_cs(rz(sy3, g * hy), cb, sb);
                    for (unsigned j = 0; j < 2; j++) {
                        const double hx3 = sy3.z + zc3[j][0] + zc3[j][1];
                        zl[2 * k + j] = rx_cs(rz(sx3[j], g * hx3), cb, sb).z;
                    }
                }
            } else if (n == 4 && d == M) {      /* children: time-4 z of level L */
                #pragma omp parallel for schedule(static) if(nd >= par_min)
                for (size_t k = 0; k < nd; k++) {
                    const double h = zpar[k >> 1] + zl[2*k] + zl[2*k + 1];
                    lv[k] = rx_cs(rz(lv[k], g * h), cb, sb);
                }
            } else {                            /* children are stored spins     */
                const Vec3 *ch = st + OFF[d + 1];
                if (save) {
                    #pragma omp parallel for schedule(static) if(nd >= par_min)
                    for (size_t k = 0; k < nd; k++) {
                        const double h = zpar[k >> 1] + ch[2*k].z + ch[2*k + 1].z;
                        save[k] = lv[k].z;
                        lv[k]   = rx_cs(rz(lv[k], g * h), cb, sb);
                    }
                } else {
                    #pragma omp parallel for schedule(static) if(nd >= par_min)
                    for (size_t k = 0; k < nd; k++) {
                        const double h = zpar[k >> 1] + ch[2*k].z + ch[2*k + 1].z;
                        lv[k] = rx_cs(rz(lv[k], g * h), cb, sb);
                    }
                }
            }
        }
    }

    /* 4. read the roots */
    const double zu = st[0].z, zv = st[1].z;
    *soft_out = zu * zv;
    const int su = (zu >= 0.0) ? 1 : -1;
    const int sv = (zv >= 0.0) ? 1 : -1;
    return (su != sv) ? 1 : 0;
}

static int run_kernel(int kernel, Rng *rng, const Angles *A, const Tables *T, Work *W, double *soft)
{
    if (kernel == K_FUSED)  return one_sample_fused(rng, A, T, W, soft);
    if (kernel == K_TABLES) return one_sample_fast(rng, A, T, W, soft);
    return one_sample_ref(rng, A, W, soft);
}

/* Start-up self-check: n samples at depth pc through `kernel` (with the
 * current threads) and through the reference kernel, compared bit for bit.
 * Leaves the geometry at depth pc; the caller restores it.                  */
static int selfcheck(int pc, uint64_t n, int kernel, const Angles *A, const Tables *T)
{
    build_geometry(pc);
    Work R, F;
    memset(&R, 0, sizeof R);  memset(&F, 0, sizeof F);
    int ok = alloc_work(&R, K_REF) && alloc_work(&F, kernel);
    if (!ok) fprintf(stderr, "ERROR: self-check: out of memory\n");
    for (uint64_t k = 0; ok && k < n; k++) {
        Rng r1 = sample_rng(pc, k), r2 = sample_rng(pc, k);
        double s1, s2;
        const int c1 = one_sample_ref(&r1, A, &R, &s1);
        const int c2 = run_kernel(kernel, &r2, A, T, &F, &s2);
        if (c1 != c2 || memcmp(&s1, &s2, sizeof s1) != 0
            || memcmp(&R.st[0].z, &F.st[0].z, sizeof(double)) != 0
            || memcmp(&R.st[1].z, &F.st[1].z, sizeof(double)) != 0) {
            fprintf(stderr, "ERROR: self-check failed at p=%d, sample %" PRIu64 ":\n"
                            "       reference  cut %d  z_u %a  z_v %a\n"
                            "       %-10s cut %d  z_u %a  z_v %a\n",
                    pc, k, c1, R.st[0].z, R.st[1].z, kernel == K_FUSED ? "fused" : "tables",
                    c2, F.st[0].z, F.st[1].z);
            ok = 0;
        }
    }
    free_work(&R);  free_work(&F);
    return ok;
}

/* ────────────────────────── result / checkpoint file ───────────────────── */

typedef struct {
    int      p, job;
    uint64_t n_target, n_done, n_cut;
    double   soft_sum, elapsed_s, core_s;    /* core_s = sum of elapsed x threads */
    int      complete;
    char     angles_file[512];
    uint64_t angles_hash;
    double   alpha;
} State;

static void state_path(char *buf, size_t len, const char *outdir, int p, int job) {
    snprintf(buf, len, "%s/viral_tree_p%d_job%d.txt", outdir, p, job);
}

/* atomic write: temp file + fsync + rename */
static int write_state(const char *path, const State *S, int threads, const char *host) {
    char tmp[1200];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) { fprintf(stderr, "WARNING: cannot write %s (%s)\n", tmp, strerror(errno)); return 0; }
    double x  = S->n_done ? (double)S->n_cut / (double)S->n_done : 0.0;
    double se = S->n_done ? sqrt(x * (1.0 - x) / (double)S->n_done) : 0.0;
    double xs = S->n_done ? 0.5 * (1.0 - S->soft_sum / (double)S->n_done) : 0.0;
    fprintf(f, "# viral_tree_hpc result/checkpoint file - machine written, do not edit\n");
    fprintf(f, "status        %s\n", S->complete ? "complete" : "running");
    fprintf(f, "p             %d\n",           S->p);
    fprintf(f, "job           %d\n",           S->job);
    fprintf(f, "n_target      %" PRIu64 "\n",  S->n_target);
    fprintf(f, "n_done        %" PRIu64 "\n",  S->n_done);
    fprintf(f, "n_cut         %" PRIu64 "\n",  S->n_cut);
    fprintf(f, "soft_sum      %.17g\n",        S->soft_sum);
    fprintf(f, "x_thresh      %.8f\n",         x);
    fprintf(f, "stderr        %.8f\n",         se);
    fprintf(f, "x_soft        %.8f\n",         xs);
    fprintf(f, "elapsed_s     %.1f\n",         S->elapsed_s);
    fprintf(f, "core_seconds  %.1f\n",         S->core_s);
    fprintf(f, "threads       %d\n",           threads);
    fprintf(f, "sec_per_sample_core  %.4f\n",
            S->n_done ? S->core_s / (double)S->n_done : 0.0);
    fprintf(f, "host          %s\n",           host);
    fprintf(f, "angles_file   %s\n",           S->angles_file);
    fprintf(f, "angles_hash   %016" PRIx64 "\n", S->angles_hash);
    fprintf(f, "alpha         %.10f\n",        S->alpha);
    if (fflush(f) != 0 || fsync(fileno(f)) != 0) {
        fprintf(stderr, "WARNING: flush/fsync of %s failed (%s)\n", tmp, strerror(errno));
    }
    fclose(f);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "WARNING: cannot rename %s -> %s (%s)\n", tmp, path, strerror(errno));
        return 0;
    }
    return 1;
}

/* returns 1 if a file was read, 0 if there is none; exits on a corrupt file */
static int read_state(const char *path, State *S) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024], key[64], val[512];
    int have = 0;
    memset(S, 0, sizeof *S);
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%63s %511s", key, val) != 2) continue;
        if      (!strcmp(key, "status"))      { S->complete = !strcmp(val, "complete"); have |= 1; }
        else if (!strcmp(key, "p"))           { S->p = atoi(val);                          have |= 2; }
        else if (!strcmp(key, "job"))         { S->job = atoi(val);                        have |= 4; }
        else if (!strcmp(key, "n_target"))    { S->n_target = strtoull(val, NULL, 10);     have |= 8; }
        else if (!strcmp(key, "n_done"))      { S->n_done = strtoull(val, NULL, 10);       have |= 16; }
        else if (!strcmp(key, "n_cut"))       { S->n_cut = strtoull(val, NULL, 10);        have |= 32; }
        else if (!strcmp(key, "soft_sum"))    { S->soft_sum = strtod(val, NULL);           have |= 64; }
        else if (!strcmp(key, "elapsed_s"))   { S->elapsed_s = strtod(val, NULL);          have |= 128; }
        else if (!strcmp(key, "core_seconds")){ S->core_s = strtod(val, NULL);             have |= 512; }
        else if (!strcmp(key, "angles_hash")) { S->angles_hash = strtoull(val, NULL, 16);  have |= 256; }
        else if (!strcmp(key, "angles_file")) { snprintf(S->angles_file, sizeof S->angles_file, "%s", val); }
        else if (!strcmp(key, "alpha"))       { S->alpha = strtod(val, NULL); }
    }
    fclose(f);
    if (have != 1023) {
        fprintf(stderr, "ERROR: %s exists but is incomplete or corrupt (fields 0x%x).\n"
                        "       Move it away to start this job from scratch.\n", path, have);
        exit(1);
    }
    if (S->n_cut > S->n_done) {
        fprintf(stderr, "ERROR: %s is inconsistent (n_cut > n_done)\n", path);
        exit(1);
    }
    return 1;
}

/* ───────────────────────────── signals ─────────────────────────────────── */

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static void install_signals(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGUSR1, &sa, NULL);
}

/* ─────────────────────────────── main ──────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s <p> <n_samples> [job_id=0] [angles_file=params/angles_p<p>.txt] [out_dir=.]\n"
        "       OMP_NUM_THREADS=<t>  threads inside each sample (memory does not grow with t)\n"
        "       VIRAL_SELFCHECK=<n>  start-up self-check samples (default %d, 0 = off)\n"
        "       VIRAL_KERNEL=ref|tables|fused   force a kernel (VIRAL_FORCE_REF=1 = ref)\n"
        "       VIRAL_PAR_MIN=<n>    levels with fewer vertices run on one thread (default %d)\n",
        prog, SELFCHECK_DEFAULT, PAR_MIN_DEFAULT);
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 6) { usage(argv[0]); return 1; }

    /* ---- arguments ---- */
    int p = atoi(argv[1]);
    if (p < 1 || p > P_MAX) { fprintf(stderr, "ERROR: p must be in 1..%d\n", P_MAX); return 1; }

    char *end = NULL;
    unsigned long long nreq = strtoull(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || nreq < 1 || nreq > N_SAMPLES_MAX) {
        fprintf(stderr, "ERROR: n_samples must be an integer in 1..%llu\n", N_SAMPLES_MAX); return 1;
    }
    long job_l = (argc > 3) ? strtol(argv[3], &end, 10) : 0;
    if (argc > 3 && (end == argv[3] || *end != '\0')) { fprintf(stderr, "ERROR: bad job_id\n"); return 1; }
    if (job_l < 0 || job_l > 2147483647L) { fprintf(stderr, "ERROR: job_id must be in 0..2147483647\n"); return 1; }
    int job = (int)job_l;

    char afile[512];
    if (argc > 4) snprintf(afile, sizeof afile, "%s", argv[4]);
    else          snprintf(afile, sizeof afile, "params/angles_p%d.txt", p);
    const char *outdir = (argc > 5) ? argv[5] : ".";

    /* ---- angles ---- */
    Angles A;
    A.beta  = calloc((size_t)p, sizeof(double));  A.gamma = calloc((size_t)p, sizeof(double));
    A.cb    = calloc((size_t)p, sizeof(double));  A.sb    = calloc((size_t)p, sizeof(double));
    if (!A.beta || !A.gamma || !A.cb || !A.sb) { fprintf(stderr, "out of memory\n"); return 1; }
    if (!load_angles(afile, p, &A)) return 1;
    uint64_t ahash;
    if (!file_hash(afile, &ahash)) { fprintf(stderr, "ERROR: cannot hash '%s'\n", afile); return 1; }

    /* ---- geometry, kernel and memory ---- */
    build_geometry(p);
    int kernel = (p >= FUSED_MIN_P) ? K_FUSED : (p >= FAST_MIN_P) ? K_TABLES : K_REF;
    {
        const char *force = getenv("VIRAL_FORCE_REF"), *kname = getenv("VIRAL_KERNEL");
        if (force && atoi(force) != 0) kernel = K_REF;
        if (kname) {
            if      (!strcmp(kname, "ref"))                          kernel = K_REF;
            else if (!strcmp(kname, "tables") && p >= FAST_MIN_P)   kernel = K_TABLES;
            else if (!strcmp(kname, "fused")  && p >= FUSED_MIN_P)  kernel = K_FUSED;
            else { fprintf(stderr, "ERROR: VIRAL_KERNEL=%s does not apply at p=%d\n", kname, p); return 1; }
        }
        const char *pm = getenv("VIRAL_PAR_MIN");
        if (pm && atol(pm) > 0) g_par_min = (size_t)atol(pm);
    }
    const size_t per_sample = bytes_per_sample(kernel);

    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);

    /* ---- previous state (resume) ---- */
    char spath[1100];
    state_path(spath, sizeof spath, outdir, p, job);
    State S;
    int resumed = read_state(spath, &S);
    if (resumed) {
        if (S.p != p || S.job != job) {
            fprintf(stderr, "ERROR: %s belongs to p=%d job=%d, not p=%d job=%d\n",
                    spath, S.p, S.job, p, job);
            return 1;
        }
        if (S.angles_hash != ahash) {
            fprintf(stderr, "ERROR: %s was produced with different angles (hash %016" PRIx64
                            ", now %016" PRIx64 " from '%s').\n"
                            "       Refusing to mix samples with different angles.\n",
                    spath, S.angles_hash, ahash, afile);
            return 1;
        }
        if (S.n_done > nreq) {
            fprintf(stderr, "ERROR: %s already holds %" PRIu64 " samples > requested %llu.\n"
                            "       Request n_samples >= %" PRIu64 " to extend, or move the file away.\n",
                    spath, S.n_done, nreq, S.n_done);
            return 1;
        }
        if (S.n_done == nreq) {
            printf("%s: already complete with %" PRIu64 " samples "
                   "(x_thresh = %.6f +- %.6f).  Nothing to do.\n", spath, S.n_done,
                   (double)S.n_cut / S.n_done,
                   sqrt(((double)S.n_cut / S.n_done) * (1.0 - (double)S.n_cut / S.n_done) / S.n_done));
            return 0;
        }
        S.n_target = nreq;
        S.complete = 0;
    } else {
        memset(&S, 0, sizeof S);
        S.p = p;  S.job = job;  S.n_target = nreq;
    }
    snprintf(S.angles_file, sizeof S.angles_file, "%s", afile);
    S.angles_hash = ahash;
    S.alpha = A.alpha;

    const uint64_t n_remaining = nreq - S.n_done;

    /* ---- threads: all of them work inside each sample (fused kernel only) ---- */
    const char *dump_file = getenv("VIRAL_TREE_DUMP");
    int nt = omp_get_max_threads();
    if (!getenv("OMP_NUM_THREADS"))
        printf("WARNING: OMP_NUM_THREADS is not set; using %d threads (all hardware threads).\n"
               "         Set OMP_NUM_THREADS explicitly on a cluster.\n", nt);
    if (kernel != K_FUSED) nt = 1;           /* the other kernels are serial */
    if (nt < 1) nt = 1;
    omp_set_num_threads(nt);

    /* ---- banner ---- */
    printf("viral_tree_fast2 : VIRAL on the 3-regular tree, MaxCut\n");
    printf("  host            %s\n", host);
    printf("  p               %d\n", p);
    printf("  job_id          %d      (global sample index S = job*2^32 + k)\n", job);
    printf("  n_samples       %llu   (this job)\n", nreq);
    if (resumed) printf("  RESUMING        %" PRIu64 " samples already done, %" PRIu64 " to go\n",
                        S.n_done, n_remaining);
    printf("  angles          %s   (hash %016" PRIx64 ")\n", afile, ahash);
    printf("  alpha           %.10f\n", A.alpha);
    for (int n = 0; n < p; n++)
        printf("    layer %2d      beta = %13.10f   gamma = %13.10f\n", n + 1, A.beta[n], A.gamma[n]);
    printf("  kernel          %s\n", KERNEL_NAME[kernel]);
    if (kernel == K_FUSED)
        printf("  tree            %zu vertices (%zu spins stored: levels 0..%d; level %d as z only)\n",
               NTOT, kernel_nst(kernel), p - 5, p - 4);
    else if (kernel == K_TABLES)
        printf("  tree            %zu vertices (%zu spins stored: levels 0..%d)\n", NTOT, kernel_nst(kernel), p - 4);
    else
        printf("  tree            %zu vertices (%zu stored, %zu leaves as bits)\n", NTOT, NST, NTOT - NST);
    printf("  threads         %d   (inside each sample; levels below %zu vertices on one thread)\n",
           nt, g_par_min);
    printf("  memory          %.3f GB  (one sample, shared by the threads)\n", per_sample / 1073741824.0);
    printf("  output          %s\n", spath);
    {   /* warn if this obviously exceeds a SLURM allocation */
        const char *mpn = getenv("SLURM_MEM_PER_NODE");   /* MB */
        if (mpn) {
            double alloc_gb = atof(mpn) / 1024.0;
            if (alloc_gb > 0.0 && per_sample / 1073741824.0 > alloc_gb - 0.5)
                printf("  WARNING: needs more than the SLURM allocation of %.1f GB. "
                       "Expect an out-of-memory kill.\n", alloc_gb);
        }
    }
    fflush(stdout);

    /* ---- exact tables and the start-up self-check ---- */
    Tables *T = NULL;
    if (kernel != K_REF) {
        T = malloc(sizeof *T);
        if (!T) { fprintf(stderr, "out of memory (tables)\n"); return 1; }
        if (!build_tables(&A, T)) {
            fprintf(stderr, "ERROR: the layer-0 field is not order-independent for alpha = %.17g;\n"
                            "       the table kernel does not apply.  Rerun with VIRAL_FORCE_REF=1.\n",
                    A.alpha);
            return 1;
        }
        long nsc = SELFCHECK_DEFAULT;
        const char *sc = getenv("VIRAL_SELFCHECK");
        if (sc) nsc = atol(sc);
        if (nsc > 0) {
            const int pc = (p < SELFCHECK_P) ? p : SELFCHECK_P;
            const double tsc = omp_get_wtime();
            const int ok = selfcheck(pc, (uint64_t)nsc, kernel, &A, T);
            build_geometry(p);
            if (!ok) return 1;
            printf("  self-check      %ld samples at p=%d, %s kernel (%d threads) vs reference kernel: "
                   "identical (%.2f s)\n", nsc, pc, kernel == K_FUSED ? "fused" : "tables", nt,
                   omp_get_wtime() - tsc);
            fflush(stdout);
        }
    }

    /* ---- workspace: one, shared by the threads; every page is touched now
     *      (fails fast if memory is short; first touch by the using threads) */
    Work W;
    memset(&W, 0, sizeof W);
    if (!alloc_work(&W, kernel)) {
        fprintf(stderr, "ERROR: allocation failed (%.2f GB)\n", per_sample / 1073741824.0);
        return 1;
    }
    double t_alloc = omp_get_wtime();
    touch_work(&W);
    printf("  memory touched  in %.1f s\n\n", omp_get_wtime() - t_alloc);
    fflush(stdout);

    FILE *dump = NULL;
    if (dump_file) {
        dump = fopen(dump_file, "a");
        if (!dump) { fprintf(stderr, "ERROR: cannot open dump file '%s'\n", dump_file); return 1; }
    }

    install_signals();

    /* ---- main loop over samples k in [n_done, n_target) ---- */
    const double t0 = omp_get_wtime();
    const double elapsed_prev = S.elapsed_s, core_prev = S.core_s;
    double t_report = t0, t_ckpt = t0;
    uint64_t batch = 1;                       /* samples run one after another */
    uint64_t done  = S.n_done;
    const uint64_t start_done = S.n_done;     /* samples already in the file */
    const uint64_t jobhi = (uint64_t)job << 32;

    if (!write_state(spath, &S, nt, host)) {  /* claim the file immediately */
        fprintf(stderr, "ERROR: cannot write the output file %s (does out_dir exist?)\n", spath);
        return 1;
    }

    while (done < nreq) {
        uint64_t hi = done + batch;  if (hi > nreq) hi = nreq;
        uint64_t bc = 0;  double bs = 0.0;
        const double tb0 = omp_get_wtime();

        /* Samples in order; with BATCH_MAX_PER_THREAD=1 each batch is one
         * sample, so soft_sum is summed in exactly the frozen binary's order. */
        for (uint64_t k = done; k < hi; k++) {
            Rng r = sample_rng(p, jobhi | k);
            double soft;
            const int cut = run_kernel(kernel, &r, &A, T, &W, &soft);
            bc += (uint64_t)cut;
            bs += soft;
            if (dump)
                fprintf(dump, "%" PRIu64 " %d %a %a\n", k, cut, W.st[0].z, W.st[1].z);
        }
        const double tnow = omp_get_wtime();
        const double tbatch = tnow - tb0;

        done = hi;
        S.n_done = done;  S.n_cut += bc;  S.soft_sum += bs;
        S.elapsed_s = elapsed_prev + (tnow - t0);
        S.core_s    = core_prev + (tnow - t0) * nt;
        S.complete  = (done == nreq);

        /* grow the batch until it takes ~BATCH_TARGET_SEC (result unaffected) */
        if (tbatch < BATCH_TARGET_SEC && batch < (uint64_t)BATCH_MAX_PER_THREAD)
            batch *= 2;

        const int stopping = g_stop && !S.complete;
        if (tnow - t_ckpt >= CKPT_SEC || S.complete || stopping) {
            t_ckpt = tnow;
            write_state(spath, &S, nt, host);
        }
        if (tnow - t_report >= REPORT_SEC || S.complete || stopping) {
            t_report = tnow;
            const double x  = (double)S.n_cut / (double)done;
            const double se = sqrt(x * (1.0 - x) / (double)done);
            const double dt   = (tnow - t0 > 0.0) ? tnow - t0 : 1e-9;
            const double rate = (double)(done - start_done) / dt;          /* samples/s, this run */
            const double eta  = (double)(nreq - done) / rate;
            printf("  [p=%d job=%d] %10" PRIu64 " / %llu   x_thresh = %.6f +- %.6f   "
                   "%.3f s/sample/core   elapsed %.0f s   ETA %.1f h\n",
                   p, job, done, nreq, x, se, dt * nt / (double)(done - start_done),
                   S.elapsed_s, eta / 3600.0);
            fflush(stdout);
        }
        if (stopping) {
            printf("\nSTOPPED BY SIGNAL after %" PRIu64 " samples; state saved in %s.\n"
                   "Rerun the same command to resume.\n", done, spath);
            fflush(stdout);
            if (dump) fclose(dump);
            return 0;
        }
    }

    /* ---- final summary ---- */
    const double x  = (double)S.n_cut / (double)S.n_done;
    const double se = sqrt(x * (1.0 - x) / (double)S.n_done);
    const double xs = 0.5 * (1.0 - S.soft_sum / (double)S.n_done);
    printf("\nDONE  p = %d  job = %d\n", p, job);
    printf("  n_samples   %" PRIu64 "\n", S.n_done);
    printf("  n_cut       %" PRIu64 "\n", S.n_cut);
    printf("  c_VIRAL(%d) = %.6f  +-  %.6f     (thresholded: this is the bound)\n", p, x, se);
    printf("  x_soft      %.6f                 (diagnostic only)\n", xs);
    printf("  elapsed     %.0f s wall (all runs), %.4f s/sample/core\n",
           S.elapsed_s, S.core_s / (double)S.n_done);
    printf("  written to  %s\n", spath);

    if (dump) fclose(dump);
    free_work(&W);
    free(T); free(A.beta); free(A.gamma); free(A.cb); free(A.sb);
    return 0;
}
