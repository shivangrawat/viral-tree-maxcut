#!/usr/bin/env python3
"""
exact_small_p.py  --  independent EXACT evaluation of c_VIRAL(p) for p <= 2.

For p = 1 the depth-p tree has 6 vertices, for p = 2 it has 14, so all 2^6 or
2^14 seed configurations can be enumerated and c_VIRAL(p) = Prob[root edge cut]
computed exactly.  This is written directly from the equations, with an explicit
adjacency list and the full (unpruned) dynamics on every vertex, and shares no
code with viral_tree_hpc.c.  It is the independent check of the C program.

usage:  python3 exact_small_p.py <angles_file> <p>          (p = 1 or 2)
prints: exact c_VIRAL(p) and the exact soft value, to be compared with a
        Monte-Carlo run of viral_tree_hpc at the same p and angles.

needs:  numpy (the only non-standard dependency in this repository).

With params/angles_p2.txt it prints c_VIRAL(2) = 12380/16384 = 0.7556152344,
the exact value that check.sh tests the solvers against.
"""
import sys, itertools, math
import numpy as np


def read_angles(fname, p):
    """Return (alpha, betas, gammas) for layers 1..p of an angles file (comment lines skipped)."""
    with open(fname) as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith('#')]
    alpha = float(lines[0].split()[0])
    beta, gamma = [], []
    for n in range(p):
        w = lines[1 + n].split()
        assert int(w[0]) == n + 1
        beta.append(float(w[1])); gamma.append(float(w[2]))
    return alpha, beta, gamma


def build_tree(p):
    """3-regular tree of depth p rooted at the edge (0, 1).

    Returns (adj, N): adjacency lists and the number of vertices N = 4*2^p - 2.
    Vertices are numbered breadth first; the leaves have a single neighbour.
    """
    adj = {0: [1], 1: [0]}
    frontier = [0, 1]
    nxt = 2
    for _ in range(p):
        new_frontier = []
        for v in frontier:
            for _ in range(2):          # two children each (u and v have 2 children + each other)
                adj.setdefault(v, []).append(nxt)
                adj[nxt] = [v]
                new_frontier.append(nxt)
                nxt += 1
        frontier = new_frontier
    return adj, nxt


def main():
    """Run all 2^N seed configurations through the full dynamics; print exact c_VIRAL(p) and x_soft."""
    fname, p = sys.argv[1], int(sys.argv[2])
    assert p in (1, 2), "exact enumeration only for p = 1, 2"
    alpha, beta, gamma = read_angles(fname, p)
    adj, N = build_tree(p)
    assert N == 4 * 2**p - 2
    # every configuration of seeds: rows = configurations, columns = vertices
    eps = np.array(list(itertools.product([-1.0, 1.0], repeat=N)))   # (2^N, N)
    M = eps.shape[0]
    x = np.full((M, N), math.cos(alpha))
    y = np.zeros((M, N))
    z = eps * math.sin(alpha)
    for n in range(p):
        # local field from the actual neighbours (leaves have one neighbour)
        h = np.zeros((M, N))
        for i in range(N):
            for j in adj[i]:
                h[:, i] += z[:, j]
        # R_z(gamma h) then R_x(beta)
        c, s = np.cos(gamma[n] * h), np.sin(gamma[n] * h)
        x, y = c * x - s * y, s * x + c * y
        cb, sb = math.cos(beta[n]), math.sin(beta[n])
        y, z = cb * y - sb * z, sb * y + cb * z
    su = np.where(z[:, 0] >= 0, 1, -1)
    sv = np.where(z[:, 1] >= 0, 1, -1)
    cut = np.mean(su != sv)
    soft = 0.5 * (1.0 - np.mean(z[:, 0] * z[:, 1]))
    print(f"p = {p}   vertices = {N}   configurations = {M}")
    print(f"exact c_VIRAL({p}) = {cut:.10f}    (= {int(round(cut*M))} / {M})")
    print(f"exact x_soft       = {soft:.10f}")


if __name__ == "__main__":
    main()
