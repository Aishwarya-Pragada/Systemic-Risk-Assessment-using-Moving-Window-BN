"""Candidate-parent selection (search-space restriction).

Each node gets C candidate parents: the C other nodes with the largest pairwise BDeu gain
    gain(i | j) = score(i | {j}) - score(i | {})
Parent sets are then searched exhaustively *within* these candidates (see bdeu.py), which makes the
order-MCMC score exact inside the restricted space.
"""
import math

import numpy as np

from .._jit import njit


@njit(cache=True)
def pair_gains(X, ess):
    N = X.shape[0]
    v = X.shape[1]
    gains = np.zeros((v, v))
    for i in range(v):
        n1 = 0
        for r in range(N):
            n1 += X[r, i]
        n0 = N - n1
        s0 = math.lgamma(ess) - math.lgamma(ess + N)
        s0 += math.lgamma(ess / 2.0 + n0) - math.lgamma(ess / 2.0)
        s0 += math.lgamma(ess / 2.0 + n1) - math.lgamma(ess / 2.0)
        for j in range(v):
            if j == i:
                continue
            c = np.zeros((2, 2))
            for r in range(N):
                c[X[r, j], X[r, i]] += 1.0
            s1 = 0.0
            for xj in range(2):
                nj = c[xj, 0] + c[xj, 1]
                s1 += math.lgamma(ess / 2.0) - math.lgamma(ess / 2.0 + nj)
                for xi in range(2):
                    s1 += math.lgamma(ess / 4.0 + c[xj, xi]) - math.lgamma(ess / 4.0)
            gains[i, j] = s1 - s0
    return gains


def select_candidates(X, C, ess):
    """(v, C) int64 array: candidate parents of every node, best pairwise gain first."""
    v = X.shape[1]
    C = min(C, v - 1)
    gains = pair_gains(X, ess)
    cand = np.empty((v, C), dtype=np.int64)
    for i in range(v):
        g = gains[i].copy()
        g[i] = -np.inf
        cand[i] = np.argsort(-g, kind="stable")[:C]
    return cand
