#!/usr/bin/env python3
"""
merge_results.py  --  combine the output files of viral_tree_hpc jobs.

usage:  python3 merge_results.py [DIR ...] [-p P] [-v]

Reads every  viral_tree_p<p>_job<job>.txt  in the given directories (default:
current directory), groups them by depth p, adds up n_done and n_cut, and
prints for each p:

    x_thresh = n_cut / n,  its standard error,
    the one-sided lower confidence bounds at 95%, 99%, 99.9%, 99.99%
    (exact Clopper-Pearson, i.e. the smallest c such that a Bernoulli(c)
    variable would produce >= n_cut successes in n trials with probability
    >= delta), and the normal approximation for comparison.

    M_g >= c_VIRAL(p) for all g >= 2p+2 is a theorem; the confidence bound is
    the statistical part of the statement.

Safety checks
  * the same (p, job) must not appear twice (that would double count the same
    samples); the script stops with an error if it does;
  * all files of one p must have the same angles_hash; the script stops if not;
  * files with status "running" are partial results; they are included and
    counted, and the number of unfinished jobs is reported.

Pure Python 3, no dependencies.  If scipy is available (-v) the exact bound is
cross-checked against scipy.stats.beta.ppf.
"""
import sys, os, glob, math, argparse


# ----------------------------------------------------------------------------
# regularized incomplete beta I_x(a,b) by Lentz's continued fraction
# (Numerical Recipes 'betacf'), and its inverse by bisection.
# ----------------------------------------------------------------------------
def _betacf(a, b, x, maxit=100000, eps=1e-15):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny: d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a, b, x):
    """I_x(a, b), regularized incomplete beta."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q, a, b):
    """quantile of Beta(a,b) by bisection on betainc (monotone)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q: lo = mid
        else: hi = mid
        if hi - lo < 1e-15: break
    return 0.5 * (lo + hi)


def clopper_pearson_lower(k, n, delta):
    """one-sided lower bound L: P[Bin(n, L) >= k] = delta  (L = 0 if k = 0)."""
    if k == 0: return 0.0
    return beta_ppf(delta, k, n - k + 1)


def normal_lower(k, n, delta):
    """normal approximation for comparison; z from the inverse error function."""
    x = k / n
    z = -math.sqrt(2.0) * _erfinv(2.0 * delta - 1.0)
    return x - z * math.sqrt(x * (1.0 - x) / n)


def _erfinv(y):
    # bisection on math.erf, plenty accurate for our purposes
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid) < y: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
def read_file(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            w = line.split(None, 1)
            if len(w) == 2:
                d[w[0]] = w[1].strip()
    need = ('p', 'job', 'n_done', 'n_cut', 'soft_sum', 'status', 'angles_hash')
    for k in need:
        if k not in d:
            raise ValueError(f"{path}: missing field '{k}'")
    return {
        'path': path, 'p': int(d['p']), 'job': int(d['job']),
        'n': int(d['n_done']), 'cut': int(d['n_cut']), 'soft': float(d['soft_sum']),
        'complete': d['status'] == 'complete', 'hash': d['angles_hash'],
        'core_s': float(d.get('core_seconds', 'nan')),
        'angles': d.get('angles_file', '?'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*', default=['.'])
    ap.add_argument('-p', type=int, default=None, help='only this depth')
    ap.add_argument('-v', action='store_true', help='list every job; cross-check with scipy if present')
    args = ap.parse_args()

    files = []
    for d in args.dirs:
        files += glob.glob(os.path.join(d, 'viral_tree_p*_job*.txt'))
    if not files:
        sys.exit("no viral_tree_p*_job*.txt files found")

    recs = [read_file(f) for f in sorted(files)]
    if args.p is not None:
        recs = [r for r in recs if r['p'] == args.p]

    by_p = {}
    for r in recs:
        by_p.setdefault(r['p'], []).append(r)

    for p in sorted(by_p):
        rs = by_p[p]
        # --- safety checks
        seen = {}
        for r in rs:
            if r['job'] in seen:
                sys.exit(f"ERROR: p={p} job={r['job']} appears twice:\n  {seen[r['job']]}\n  {r['path']}\n"
                         "These are the SAME samples; remove one of the files.")
            seen[r['job']] = r['path']
        hashes = {r['hash'] for r in rs}
        if len(hashes) > 1:
            sys.exit(f"ERROR: p={p}: files were produced with different angles (hashes {sorted(hashes)}).\n"
                     "Samples with different angles must not be combined.")
        # --- combine
        n = sum(r['n'] for r in rs)
        k = sum(r['cut'] for r in rs)
        soft = sum(r['soft'] for r in rs)
        core = sum(r['core_s'] for r in rs)
        unfinished = sum(1 for r in rs if not r['complete'])
        if n == 0:
            print(f"p = {p}: no samples yet"); continue
        x = k / n
        se = math.sqrt(x * (1.0 - x) / n)
        xs = 0.5 * (1.0 - soft / n)

        print(f"\n=== p = {p}   (bound valid for girth g >= {2*p+2}) ===")
        print(f"  jobs           {len(rs)}   ({unfinished} still running / incomplete)")
        print(f"  angles         {rs[0]['angles']}   hash {rs[0]['hash']}")
        print(f"  n_samples      {n}")
        print(f"  n_cut          {k}")
        print(f"  x_thresh       {x:.6f}  +-  {se:.6f}      <-- c_VIRAL({p}) estimate (the bound)")
        print(f"  x_soft         {xs:.6f}                     (diagnostic only)")
        if core == core:  # not nan
            print(f"  cost           {core/3600:.1f} core-hours, {core/n:.3f} core-seconds per sample")
        print(f"  one-sided lower confidence bounds on c_VIRAL({p}):")
        print(f"     confidence     exact (Clopper-Pearson)     normal approx.")
        for conf in (0.95, 0.99, 0.999, 0.9999):
            delta = 1.0 - conf
            L = clopper_pearson_lower(k, n, delta)
            Ln = normal_lower(k, n, delta)
            flag = "  > 0.912" if L > 0.912 else ""
            print(f"     {100*conf:7.2f}%         {L:.6f}                {Ln:.6f}{flag}")
        if args.v:
            print("  per job:")
            for r in sorted(rs, key=lambda r: r['job']):
                xr = r['cut'] / r['n'] if r['n'] else float('nan')
                print(f"     job {r['job']:5d}  n={r['n']:9d}  n_cut={r['cut']:9d}  x={xr:.6f}"
                      f"  {'complete' if r['complete'] else 'RUNNING '}  {r['path']}")
            try:
                from scipy.stats import beta as _beta
                Ls = _beta.ppf(0.001, k, n - k + 1)
                print(f"  scipy cross-check of the 99.9% bound: {Ls:.9f}  "
                      f"(this script: {clopper_pearson_lower(k, n, 0.001):.9f})")
            except ImportError:
                pass


if __name__ == '__main__':
    main()
