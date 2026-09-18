/*
 * ============================================================================
 *  viral_tree_hpc.c
 *
 *  Monte-Carlo evaluation of the VIRAL local algorithm for MAX-CUT on the
 *  infinite 3-regular tree.  The number it produces, c_VIRAL(p), is a
 *  rigorous lower bound on the max-cut fraction of EVERY 3-regular graph of
 *  girth >= 2p+2, and therefore on the quantity M_g defined below.
 *
 *  Self-contained: one C file, OpenMP, libm.  No other dependencies.
 *
 *      compile :  gcc -O3 -fopenmp viral_tree_hpc.c -o viral_tree_hpc -lm      (or: make)
 *      run     :  OMP_NUM_THREADS=16 ./viral_tree_hpc <p> <n_samples> [job_id] [angles_file] [out_dir]
 *
 *  A full description of the problem, the algorithm, the code, the resource
 *  requirements and the output format follows.  Please read at least
 *  sections 5 and 6 before running.
 *
 * ----------------------------------------------------------------------------
 *  1. THE PROBLEM
 * ----------------------------------------------------------------------------
 *
 *  For a graph G = (V,E), MaxCut(G) is the largest number of edges that can
 *  be made to join the two sides of a bipartition of V; equivalently
 *
 *      MaxCut(G) = max over sigma in {-1,+1}^V of  #{ (i,j) in E : sigma_i != sigma_j }.
 *
 *  For 3-regular graphs one normalizes by |E| = 3|V|/2 and asks how small the
 *  max-cut FRACTION can be when the graph has no short cycles:
 *
 *      M_g  =  inf { MaxCut(G)/|E(G)| :  G 3-regular, girth(G) >= g }.
 *
 *  M_g is non-decreasing in g and the limit g -> infinity exists.  The best
 *  known bounds on that limit are
 *
 *      0.912    <=  lim_{g->inf} M_g  <=  0.92386
 *
 *  lower bound: Csoka, Gerencser, Harangi, Virag (2015) and Gamarnik, Li
 *  (2018), obtained with LOCAL ALGORITHMS exactly as described below;
 *  upper bound: the max-cut of random 3-regular graphs, Harangi,
 *  arXiv:2506.21296 (2-RSB bound).  For comparison, QAOA reaches 0.8971 at
 *  depth p = 17 (Farhi, Gutmann, Ranard, Villalonga, arXiv:2503.12789).
 *
 *  Goal of this computation: evaluate c_VIRAL(p) for large p as precisely as
 *  possible.  A value above 0.912 with a statistically decisive error bar
 *  would be a new record lower bound on lim M_g.
 *
 * ----------------------------------------------------------------------------
 *  2. THE ALGORITHM (VIRAL) AND WHY ITS TREE VALUE IS A BOUND
 * ----------------------------------------------------------------------------
 *
 *  VIRAL attaches a unit 3-vector ("classical spin") s_i to every vertex i
 *  and iterates p layers of a purely local rule.  With fixed angles
 *  alpha, beta_0..beta_{p-1}, gamma_0..gamma_{p-1}:
 *
 *      seeds          eps_i  = +1 or -1, independent fair coins, one per vertex
 *      initial state  s_i(0) = ( cos(alpha), 0, eps_i * sin(alpha) )
 *      layer n        h_i(n)   = sum over neighbours j of i  of  s_j^z(n)
 *                     s_i(n+1) = R_x(beta_n) R_z(gamma_n * h_i(n)) s_i(n)
 *      output         sigma_i  = sign( s_i^z(p) )        (sign(0) := +1)
 *
 *  where R_z(t): (x,y,z) -> (x cos t - y sin t,  x sin t + y cos t,  z)
 *        R_x(t): (x,y,z) -> (x,  y cos t - z sin t,  y sin t + z cos t).
 *
 *  Locality.  One layer moves information across exactly one edge, so
 *  s_i(p) is a function of the seeds inside the ball of radius p around i
 *  only.  For an edge (u,v) the pair (sigma_u, sigma_v) is therefore a
 *  function of the seeds inside the ball of radius p around the EDGE, i.e.
 *  around both endpoints.
 *
 *  Girth.  If girth(G) >= 2p+2, that ball contains no cycle: it is a tree,
 *  and it is the SAME tree for every edge of every such graph - the
 *  3-regular tree of depth p rooted at an edge (drawn in section 4).  Hence
 *  the probability that an edge is cut,
 *
 *      c_VIRAL(p)  =  Prob[ sigma_u != sigma_v ]      (root edge of the tree)
 *
 *  does not depend on the edge or on the graph.  By linearity of
 *  expectation the expected cut fraction of G is exactly c_VIRAL(p), and a
 *  random variable is at least its mean with positive probability, so SOME
 *  seed configuration cuts at least a fraction c_VIRAL(p) of the edges:
 *
 *      MaxCut(G)/|E(G)|  >=  c_VIRAL(p)     for every 3-regular G with girth >= 2p+2,
 *
 *      M_g  >=  c_VIRAL(p)                  for every g >= 2p+2.
 *
 *  This is the whole argument.  Nothing about random graphs enters; the
 *  only input is the single number c_VIRAL(p), which this program measures.
 *
 *  The angles.  Any fixed choice of angles gives a valid bound.  The files
 *  in params/ contain angles that were trained (by gradient descent) on
 *  random 3-regular graphs with 10^6 vertices; they are the ones used for
 *  all results so far.  They are not optimal for the tree, so every
 *  c_VIRAL(p) obtained with them is a conservative bound.
 *
 * ----------------------------------------------------------------------------
 *  3. WHAT THE PROGRAM COMPUTES, AND THE ERROR BAR
 * ----------------------------------------------------------------------------
 *
 *  c_VIRAL(p) is estimated by plain Monte Carlo.  One SAMPLE means: draw
 *  fresh seeds eps_i on all 4*2^p - 2 vertices of the depth-p tree, run the
 *  p layers, and record the indicator
 *
 *      X = 1 if sigma_u != sigma_v  (root edge cut),  X = 0 otherwise.
 *
 *  With n independent samples, n_cut = sum of X, the estimate is
 *
 *      x_thresh = n_cut / n,      standard error  sqrt( x (1-x) / n ).
 *
 *  X is a Bernoulli variable, so exact binomial (Clopper-Pearson)
 *  confidence bounds are available; merge_results.py prints them.  The
 *  logical status of the final statement is: "c_VIRAL(p) >= L with
 *  confidence 1 - delta" is a statistical statement about the Monte-Carlo
 *  estimate; the implication  c_VIRAL(p) >= L  ==>  M_g >= L  is a theorem.
 *
 *  The program also records x_soft = ( 1 - < s_u^z s_v^z > ) / 2, the
 *  un-thresholded correlation.  It is a diagnostic only; the bound is
 *  x_thresh.
 *
 *  Statistics is the binding constraint, not depth: the variance of X is
 *  at most 1/4, so the error bar is ~0.5/sqrt(n) whatever p is, while the
 *  cost of one sample grows like 2^p.  Reaching an error bar of 2.4e-4
 *  (what is needed to separate 0.9127 from 0.912 at 99.9%) takes about
 *  1.4 million samples.
 *
 * ----------------------------------------------------------------------------
 *  4. HOW THE CODE WORKS
 * ----------------------------------------------------------------------------
 *
 *  4.1 The tree.  Level 0 holds the two root vertices u (index 0) and v
 *      (index 1), joined by the root edge.  Level d >= 1 holds 2^(d+1)
 *      vertices; vertex k of level d has parent k>>1 in level d-1 (for d=1:
 *      k=0,1 hang below u, k=2,3 below v) and children 2k, 2k+1 in level
 *      d+1.  Every vertex has degree 3 except the leaves (level p).
 *      Total: 2 + 4 + 8 + ... + 2^(p+1) = 4*2^p - 2 vertices.
 *      Vertices are stored level by level: flat index = OFF[d] + k, with
 *      OFF[0]=0, OFF[d+1] = OFF[d] + 2^(d+1).  No adjacency list is needed.
 *
 *  4.2 Why a finite tree gives the EXACT value.  The leaves have degree 1
 *      instead of 3, so their update is wrong.  But wrong information
 *      travels one level per layer and needs p+1 layers to reach the root;
 *      the roots are read after p layers, so they are exact.
 *
 *  4.3 Light-cone pruning.  Conversely, at layer n (producing time n+1)
 *      only levels d <= p-n-1 can still influence the root at time p.  The
 *      code updates only those.  Levels 0..L contain 2^(L+2) - 2 vertices,
 *      so the work per sample is
 *          sum_{n=0}^{p-1} ( 2^(p-n+1) - 2 )  =  4*2^p - 2p - 4  vertex updates,
 *      i.e. about 4*2^p instead of p*4*2^p (a factor p saved).  This is exact, not an
 *      approximation: every value that is read was written at the previous
 *      layer.  In particular the leaves are never updated at all - only
 *      their seeds are ever read, once, at layer 0.
 *
 *  4.4 Memory layout (per thread).  Because a vertex update at level d reads
 *      level d-1, d and d+1 at the OLD time, the straightforward
 *      implementation keeps two full copies of the tree.  This code instead
 *      updates IN PLACE, level by level from the root downwards:
 *        - level d+1 is still old when level d is updated (top-down order);
 *        - level d itself is read before being written, element by element;
 *        - level d-1 has already been overwritten, so its OLD z-components
 *          (the only thing the update needs) are copied to a scratch array
 *          just before it is overwritten.  Two scratch arrays alternate.
 *      The leaves (half of all vertices) are never stored as states: their
 *      seeds are kept as raw random bits (1 bit each) and read at layer 0.
 *      Memory per thread:
 *          24 B * (2^(p+1) - 2)      states of levels 0..p-1 (3 doubles)
 *        +  8 B * ceil((4*2^p - 2)/64)   seed bits, one per vertex
 *        +  8 B * 2 * 2^(p-1)        two z-scratch arrays
 *        =  56.5 * 2^p  bytes (minus 48):
 *           p=22: 0.22 GB   p=24: 0.88 GB   p=26: 3.53 GB   p=28: 14.1 GB   p=30: 56.5 GB
 *      (GB = 2^30 bytes here; measured RSS at p=26: 3.53 GB, as computed.)
 *      Each OpenMP thread owns one such workspace, so total memory is
 *      (threads) x (the above).  Set OMP_NUM_THREADS accordingly.
 *
 *  4.5 Random numbers.  Each sample has its own xoshiro256** stream, seeded
 *      from a splitmix64 hash of its GLOBAL SAMPLE INDEX
 *          S = job_id * 2^32 + k,        k = 0 .. n_samples-1.
 *      Consequences:
 *        - the result is bit-for-bit reproducible and independent of the
 *          number of threads and of how the run was split or restarted;
 *        - different job_ids use disjoint index ranges automatically, so
 *          many jobs (e.g. a SLURM array) can run in parallel and their
 *          counts can simply be added;
 *        - job_id 0 reproduces exactly the samples of the original
 *          single-machine runs.
 *      NEVER run the same (p, job_id) twice and add both results: they are
 *      the same samples.  merge_results.py refuses duplicates.
 *
 *  4.6 Parallelism.  Samples are independent and all cost the same, so
 *      OpenMP parallelizes over samples with a static schedule; nothing is
 *      parallel inside a sample.  Samples are processed in batches; the
 *      batch size adapts so that a batch takes ~20 s (it does not affect
 *      the result).
 *
 *  4.7 Checkpoint / restart.  The output file
 *          <out_dir>/viral_tree_p<p>_job<job>.txt
 *      is rewritten atomically (write temp file, fsync, rename) about once
 *      a minute and at the end.  It holds n_done, n_cut and soft_sum.  If it
 *      exists when the program starts, the run RESUMES from n_done, after
 *      checking that p, job_id and the angles file (by hash) are the same.
 *      A run killed mid-batch redoes that batch: nothing is lost or double
 *      counted.  Running with a larger n_samples than a finished job
 *      extends it; running with a smaller one is refused.
 *
 *  4.8 Signals.  SIGTERM, SIGINT and SIGUSR1 make the program finish the
 *      current batch, write the file with status "running", and exit(0).
 *      With SLURM use  --signal=B:TERM@600  (see section 5) so that the
 *      wall-time limit ends the job cleanly; resubmitting the same command
 *      continues where it stopped.
 *
 * ----------------------------------------------------------------------------
 *  5. COMPILING, RUNNING, RESOURCES
 * ----------------------------------------------------------------------------
 *
 *  Compile (GCC >= 4.9, clang, or any OpenMP-capable C99 compiler):
 *
 *      gcc -O3 -fopenmp viral_tree_hpc.c -o viral_tree_hpc -lm        (or:  make)
 *
 *  No CPU-specific flags are needed: the run time is dominated by sin/cos
 *  from libm, and -march=native was measured to change neither speed nor
 *  results.  A binary built on a login node therefore runs on any compute
 *  node of the same architecture.
 *
 *  Run:
 *
 *      export OMP_NUM_THREADS=<threads>
 *      ./viral_tree_hpc <p> <n_samples> [job_id=0] [angles_file] [out_dir=.]
 *
 *      p            depth, an even number for which params/angles_p<p>.txt exists
 *      n_samples    samples for THIS job (1 .. 4294967295)
 *      job_id       integer >= 0; different jobs are independent (section 4.5)
 *      angles_file  default  params/angles_p<p>.txt   (relative to the cwd)
 *      out_dir      default  .  (must exist)
 *
 *  Resources.  Measured with this code on an Intel i9-10900X (3.7 GHz),
 *  GCC 9.4, -O3:
 *
 *      p     time / sample / core     memory / thread     4*2^p vertices
 *      22          0.31 s  (measured)      0.22 GB          1.7e7
 *      24          1.34 s  (measured)      0.88 GB          6.7e7
 *      26          4.3  s  (measured)      3.53 GB          2.7e8
 *      28        ~15    s  (extrapolated) 14.1  GB          1.1e9
 *      30        ~60    s  (extrapolated) 56.5  GB          4.3e9
 *
 *  Memory doubles per unit of p exactly; time grows by a factor 2 to 3 per
 *  unit of p (memory traffic starts to matter above the cache sizes).
 *  Error bar after n samples ~ 0.29/sqrt(n) (at x ~ 0.91):
 *      n = 1e5 -> 9e-4,   n = 4e5 -> 4.5e-4,   n = 1.4e6 -> 2.4e-4.
 *  Core-hours for 1e6 samples: p=24 ~360, p=26 ~1200, p=28 ~4700, p=30 ~19000.
 *
 *  Choosing threads per job.  Threads x (memory/thread) must fit in the
 *  job's memory allocation with ~1 GB spare.  The program prints its
 *  estimate at start-up and touches all its memory immediately, so a job
 *  that is going to run out of memory fails within seconds, not hours.
 *
 *  SLURM job arrays (ready to use: submit_p24.slurm ... submit_p30.slurm,
 *  one per depth, 1e6 samples each; see README.md).  The essence of each:
 *
 *      #SBATCH --array=0-99               # 100 independent jobs
 *      #SBATCH --cpus-per-task=16
 *      #SBATCH --mem=60G                  # >= 16 x 3.53 GB + 1 GB  (p = 26)
 *      #SBATCH --time=04:00:00
 *      #SBATCH --signal=B:TERM@600        # clean stop 10 min before the limit
 *      #SBATCH --output=results/slurm_p26_%a.out
 *      export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
 *      exec ./viral_tree_hpc 26 10000 $SLURM_ARRAY_TASK_ID params/angles_p26.txt results
 *
 *  100 jobs x 10000 samples = 1e6 samples.  If the wall-time limit stops the
 *  jobs, resubmit the identical script: every task resumes.  Then
 *
 *      python3 merge_results.py results
 *
 *  adds up all jobs (of every p present) and prints the estimates with exact
 *  confidence bounds.
 *
 * ----------------------------------------------------------------------------
 *  6. OUTPUT FILE FORMAT
 * ----------------------------------------------------------------------------
 *
 *  <out_dir>/viral_tree_p<p>_job<job>.txt   -   "key value" per line:
 *
 *      status       running | complete
 *      p, job, n_target, n_done, n_cut       integers
 *      soft_sum     sum over samples of s_u^z * s_v^z   (17 significant digits)
 *      x_thresh, stderr, x_soft              derived, for reading by eye
 *      elapsed_s    wall-clock seconds accumulated over all (re)starts
 *      core_seconds elapsed_s x threads, accumulated (cost of the job)
 *      threads, sec_per_sample_core, host    timing information
 *      angles_file, angles_hash, alpha       provenance
 *
 *  Only n_done, n_cut, soft_sum (and p, job for bookkeeping) are needed to
 *  merge jobs: n and n_cut add.  A file with status "running" is a valid
 *  partial result and can be merged as well.
 *
 * ----------------------------------------------------------------------------
 *  7. REPRODUCIBILITY AND TESTS
 * ----------------------------------------------------------------------------
 *
 *  For a given binary on a given machine the output is deterministic.
 *  Across compilers/CPUs the last bit of sin/cos or fused multiply-add can
 *  differ; this can flip the sign decision of a sample whose s^z is within
 *  ~1e-15 of zero, i.e. changes n_cut by O(1) in a million.  Harmless, but
 *  do not expect bit-identical n_cut between different machines.
 *
 *  Validation performed before release (see README.md and check.sh):
 *    - per-sample output bit-identical to the original two-buffer code
 *      (viral_tree.c, used for all published points) for p = 1..20;
 *    - p = 1, 2: agreement with an independent exact enumeration over all
 *      2^6 and 2^14 seed configurations (exact_small_p.py);
 *    - n_cut independent of thread count, batch size, and of
 *      stop/resume and job splitting.
 *  `make check` reruns the fast subset of these tests.
 *
 *  Test hook: if the environment variable VIRAL_TREE_DUMP=<file> is set,
 *  the run is forced to one thread and every sample appends a line
 *  "k cut s_u^z s_v^z" (hex floats) to <file>.  Used only for testing.
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
#define N_SAMPLES_MAX         4294967295ULL   /* k must fit in 32 bits (sec. 4.5) */

/* ───────────────────────────── spins ───────────────────────────────────── */

typedef struct { double x, y, z; } Vec3;

/* R_z by angle th.  Identical arithmetic to the reference code.            */
static inline Vec3 rz(Vec3 s, double th) {
    double c = cos(th), sn = sin(th);
    return (Vec3){ c*s.x - sn*s.y,  sn*s.x + c*s.y,  s.z };
}
/* R_x with cos/sin of the angle precomputed (the angle beta_n is the same
 * for every vertex of a layer; computing cos/sin once per layer instead of
 * once per vertex gives bit-identical results and saves time).           */
static inline Vec3 rx_cs(Vec3 s, double c, double sn) {
    return (Vec3){ s.x,  c*s.y - sn*s.z,  sn*s.y + c*s.z };
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

/* splitmix64 finalizer: a bijective hash of 64-bit words.  Per-sample seeds
 * MUST pass through it before rng_seed(), because rng_seed() expands its
 * argument by stepping it with the golden-ratio constant G: two seeds that
 * differ by a multiple of G would produce states sharing 3 of 4 words.    */
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

/* Seed of sample with global index S (section 4.5).  Same formula as the
 * reference code, where S was the plain sample counter.                   */
static inline Rng sample_rng(int p, uint64_t S) {
    return rng_seed(mix64(0x5DEECE66DULL ^ ((uint64_t)p << 32) ^ mix64(S)));
}

/* ───────────────────────────── geometry ────────────────────────────────── */

static int     P;        /* depth                                            */
static size_t  NTOT;     /* 4*2^P - 2 : all vertices, leaves included        */
static size_t  NST;      /* 2*2^P - 2 : vertices of levels 0..P-1 (stored)   */
static size_t  NW;       /* 64-bit words holding one seed bit per vertex     */
static size_t  ZS;       /* length of each z-scratch array = 2^(P-1)         */
static size_t *OFF;      /* OFF[d]  = flat index of the first vertex of level d, d = 0..P */
static size_t *NLEV;     /* NLEV[d] = number of vertices of level d               */

static void build_geometry(int p) {
    P    = p;
    OFF  = malloc((size_t)(p + 1) * sizeof *OFF);
    NLEV = malloc((size_t)(p + 1) * sizeof *NLEV);
    if (!OFF || !NLEV) { fprintf(stderr, "out of memory (geometry)\n"); exit(1); }
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

static size_t bytes_per_thread(void) {
    return NST * sizeof(Vec3) + NW * sizeof(uint64_t) + 2 * ZS * sizeof(double);
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
    /* cos/sin of the fixed angles.  sincos() is used explicitly because the
     * reference code evaluated cos(beta) and sin(beta) inside the update
     * loop, where GCC fuses them into one sincos() call, and glibc's cos()
     * and sincos() can differ in the last bit (they do for one of the p=14
     * angles).  Using sincos() here keeps the results bit-identical to the
     * reference on the same machine.  Statistically it makes no difference. */
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

/* ────────────────────────── per-thread workspace ───────────────────────── */

typedef struct {
    Vec3     *st;       /* states of levels 0..P-1,  NST entries              */
    uint64_t *w;        /* seed bits of ALL vertices, NW words                 */
    double   *zs[2];    /* old z of the level above the one being updated, ZS  */
} Work;

/* seed sign of the vertex with flat index i, as a z-component */
static inline double seed_z(const uint64_t *w, size_t i, double sa) {
    return ((w[i >> 6] >> (i & 63)) & 1ULL) ? sa : -sa;
}

/* ─────────────────────────── one Monte-Carlo sample ───────────────────────
 * Returns 1 if the root edge is cut, 0 otherwise; *soft_out = s_u^z s_v^z. */
static int one_sample(Rng *rng, const Angles *A, Work *W, double *soft_out)
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

        /* levels 1..Lmax.  Level d reads: its own old state (in place), the
         * old z of its parent (saved in zs[(d-1)&1] when level d-1 was
         * updated), and the old z of its children (level d+1, not yet
         * touched this layer).  If a deeper level will be updated after it
         * (d < Lmax), it saves its own old z into zs[d&1] first.           */
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
                /* here d == Lmax necessarily, hence save == NULL             */
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
        "       threads: set OMP_NUM_THREADS (memory = threads x 56.5*2^p bytes)\n", prog);
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

    /* ---- geometry and memory ---- */
    build_geometry(p);
    const size_t per_thread = bytes_per_thread();

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

    /* ---- threads ---- */
    const char *dump_file = getenv("VIRAL_TREE_DUMP");
    int nt = omp_get_max_threads();
    if (!getenv("OMP_NUM_THREADS"))
        printf("WARNING: OMP_NUM_THREADS is not set; using %d threads (all hardware threads).\n"
               "         Memory = threads x %.2f GB.  Set OMP_NUM_THREADS explicitly on a cluster.\n",
               nt, per_thread / 1073741824.0);
    if ((uint64_t)nt > n_remaining) nt = (int)n_remaining;
    if (dump_file) nt = 1;
    if (nt < 1) nt = 1;
    omp_set_num_threads(nt);

    /* ---- banner ---- */
    printf("viral_tree_hpc : VIRAL on the 3-regular tree, MaxCut\n");
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
    printf("  tree            %zu vertices (%zu stored, %zu leaves as bits)\n", NTOT, NST, NTOT - NST);
    printf("  threads         %d\n", nt);
    printf("  memory          %.3f GB / thread,  %.3f GB total\n",
           per_thread / 1073741824.0, (double)nt * per_thread / 1073741824.0);
    printf("  output          %s\n", spath);
    {   /* warn if this obviously exceeds a SLURM allocation */
        const char *mpn = getenv("SLURM_MEM_PER_NODE");   /* MB */
        if (mpn) {
            double alloc_gb = atof(mpn) / 1024.0;
            if (alloc_gb > 0.0 && (double)nt * per_thread / 1073741824.0 > alloc_gb - 0.5)
                printf("  WARNING: needs more than the SLURM allocation of %.1f GB. "
                       "Expect an out-of-memory kill; reduce OMP_NUM_THREADS.\n", alloc_gb);
        }
    }
    fflush(stdout);

    /* ---- workspaces: allocate, then touch every page from the owning thread
     *      (fails fast if memory is short; puts pages on the right NUMA node) */
    Work *W = calloc((size_t)nt, sizeof *W);
    if (!W) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int t = 0; t < nt; t++) {
        W[t].st    = malloc(NST * sizeof(Vec3));
        W[t].w     = malloc(NW  * sizeof(uint64_t));
        W[t].zs[0] = malloc(ZS  * sizeof(double));
        W[t].zs[1] = malloc(ZS  * sizeof(double));
        if (!W[t].st || !W[t].w || !W[t].zs[0] || !W[t].zs[1]) {
            fprintf(stderr, "ERROR: allocation failed for thread %d (%.2f GB/thread x %d threads)\n",
                    t, per_thread / 1073741824.0, nt);
            return 1;
        }
    }
    double t_alloc = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        memset(W[t].st,    0, NST * sizeof(Vec3));
        memset(W[t].w,     0, NW  * sizeof(uint64_t));
        memset(W[t].zs[0], 0, ZS  * sizeof(double));
        memset(W[t].zs[1], 0, ZS  * sizeof(double));
    }
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
    uint64_t batch = (uint64_t)nt;
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

        #pragma omp parallel for schedule(static) reduction(+:bc,bs) num_threads(nt)
        for (uint64_t k = done; k < hi; k++) {
            const int tid = omp_get_thread_num();
            Rng r = sample_rng(p, jobhi | k);
            double soft;
            int cut = one_sample(&r, &A, &W[tid], &soft);
            bc += (uint64_t)cut;
            bs += soft;
            if (dump)   /* nt == 1 here, so the order is sequential */
                fprintf(dump, "%" PRIu64 " %d %a %a\n", k, cut, W[tid].st[0].z, W[tid].st[1].z);
        }
        const double tnow = omp_get_wtime();
        const double tbatch = tnow - tb0;

        done = hi;
        S.n_done = done;  S.n_cut += bc;  S.soft_sum += bs;
        S.elapsed_s = elapsed_prev + (tnow - t0);
        S.core_s    = core_prev + (tnow - t0) * nt;
        S.complete  = (done == nreq);

        /* grow the batch until it takes ~BATCH_TARGET_SEC (result unaffected) */
        if (tbatch < BATCH_TARGET_SEC && batch < (uint64_t)BATCH_MAX_PER_THREAD * (uint64_t)nt)
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
    for (int t = 0; t < nt; t++) { free(W[t].st); free(W[t].w); free(W[t].zs[0]); free(W[t].zs[1]); }
    free(W); free(A.beta); free(A.gamma); free(A.cb); free(A.sb); free(OFF); free(NLEV);
    return 0;
}
