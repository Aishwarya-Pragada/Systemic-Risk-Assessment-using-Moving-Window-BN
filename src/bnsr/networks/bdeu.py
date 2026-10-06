"""BDeu local scores for binary variables, for *every* subset of a node's candidate parents.

For node i with candidate parents cand_i[0..C-1], a parent set is encoded as a bitmask s over the
candidates (bit c <-> node cand_i[c]).  We tabulate

    score[s]  = local BDeu score of node i given parent set s     (-inf if |s| > M)
    F[mask]   = log sum_{s subset of mask} exp(score[s])           (order-MCMC marginal, Friedman & Koller)
    G[mask]   = max_{s subset of mask} score[s],   A[mask] = argmax (best parent set given an order)

BDeu (Heckerman et al.; Cooper & Herskovits 1992 as cited in the paper, ref. 29) for a binary node with
k binary parents, q = 2^k configurations, equivalent sample size `ess`:
    alpha_ij = ess / q,   alpha_ijk = ess / (2 q)
    score = sum_j [ lgamma(alpha_ij) - lgamma(alpha_ij + n_ij) + sum_k (lgamma(alpha_ijk + n_ijk) - lgamma(alpha_ijk)) ]
Configurations that are not observed contribute 0, so only observed ones are summed.
"""
import math

import numpy as np

from .._jit import njit


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True)
def _logaddexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def node_score_table(X, i, cand_i, M, ess):
    """Local BDeu score of node i for all 2**C subsets of its candidate parents."""
    N = X.shape[0]
    C = cand_i.shape[0]
    S = 1 << C
    scores = np.full(S, -np.inf)
    codes = np.zeros((S, N), dtype=np.int64)
    keys = np.empty(N, dtype=np.int64)
    for s in range(S):
        if s > 0:
            low = s & (-s)
            c = 0
            while (low >> c) != 1:
                c += 1
            prev = s ^ low
            col = cand_i[c]
            for r in range(N):
                codes[s, r] = codes[prev, r] | (X[r, col] << c)
        k = _popcount(s)
        if k > M:
            continue
        for r in range(N):
            keys[r] = codes[s, r] * 2 + X[r, i]
        for a in range(1, N):                      # insertion sort (N is ~30)
            val = keys[a]
            b = a - 1
            while b >= 0 and keys[b] > val:
                keys[b + 1] = keys[b]
                b -= 1
            keys[b + 1] = val
        q = float(1 << k)
        a_j = ess / q
        a_jk = ess / (2.0 * q)
        lg_aj = math.lgamma(a_j)
        lg_ajk = math.lgamma(a_jk)
        total = 0.0
        a = 0
        while a < N:
            code = keys[a] >> 1
            n0 = 0
            n1 = 0
            while a < N and (keys[a] >> 1) == code:
                if (keys[a] & 1) == 1:
                    n1 += 1
                else:
                    n0 += 1
                a += 1
            total += lg_aj - math.lgamma(a_j + n0 + n1)
            if n0 > 0:
                total += math.lgamma(a_jk + n0) - lg_ajk
            if n1 > 0:
                total += math.lgamma(a_jk + n1) - lg_ajk
        scores[s] = total
    return scores


@njit(cache=True)
def subset_transforms(scores, C):
    """Zeta transforms over the subset lattice: log-sum-exp (F) and max / argmax (G, A)."""
    S = scores.shape[0]
    F = scores.copy()
    G = scores.copy()
    A = np.arange(S)
    for b in range(C):
        bit = 1 << b
        for m in range(S):
            if (m & bit) != 0:
                p = m ^ bit
                F[m] = _logaddexp(F[m], F[p])
                if G[p] > G[m]:
                    G[m] = G[p]
                    A[m] = A[p]
    return F, G, A


@njit(cache=True)
def build_tables(X, cand, M, ess):
    """Tables for all nodes. X: (N, v) int64, cand: (v, C) int64. Returns scores, F, G, A of shape (v, 2**C)."""
    v = cand.shape[0]
    C = cand.shape[1]
    S = 1 << C
    scores = np.empty((v, S))
    F = np.empty((v, S))
    G = np.empty((v, S))
    A = np.empty((v, S), dtype=np.int64)
    for i in range(v):
        sc = node_score_table(X, i, cand[i], M, ess)
        f, g, a = subset_transforms(sc, C)
        for s in range(S):
            scores[i, s] = sc[s]
            F[i, s] = f[s]
            G[i, s] = g[s]
            A[i, s] = a[s]
    return scores, F, G, A


def local_score_reference(X, i, parents, ess):
    """Slow, independent pure-Python BDeu local score (used only by the tests)."""
    N = X.shape[0]
    k = len(parents)
    q = 2 ** k
    a_j, a_jk = ess / q, ess / (2.0 * q)
    counts = {}
    for r in range(N):
        cfg = tuple(int(X[r, p]) for p in parents)
        counts.setdefault(cfg, [0, 0])[int(X[r, i])] += 1
    total = 0.0
    for n0, n1 in counts.values():
        total += math.lgamma(a_j) - math.lgamma(a_j + n0 + n1)
        for n in (n0, n1):
            total += math.lgamma(a_jk + n) - math.lgamma(a_jk)
    return total
