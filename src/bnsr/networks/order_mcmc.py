"""Order MCMC (Friedman & Koller, 2003; paper ref. 41) over node orderings.

State: a permutation of the v nodes.  Score of an order = sum_i F_i[allowed_i], where allowed_i is the set of
candidate parents of i that precede i in the order and F_i marginalises exactly over all admissible parent
sets (|pa| <= M) inside that set.  Proposals swap two positions (adjacent with prob. 1/2, otherwise random);
they are symmetric, so acceptance is min(1, exp(delta)).
For sampled orders we also read off the best DAG consistent with the order (max over parent sets, table G);
the highest-scoring DAG seen over the whole chain is returned (the paper keeps the highest-BDeu structure).
"""
import math

import numpy as np

from .._jit import njit


@njit(cache=True)
def _allowed_mask(i, pos, cand):
    m = 0
    pi = pos[i]
    for c in range(cand.shape[1]):
        if pos[cand[i, c]] < pi:
            m |= (1 << c)
    return m


@njit(cache=True)
def order_mcmc(F, G, A, cand, n_iter, burn, thin, seed):
    np.random.seed(seed)
    v = cand.shape[0]
    order = np.random.permutation(v)
    pos = np.empty(v, dtype=np.int64)
    for p in range(v):
        pos[order[p]] = p
    masks = np.zeros(v, dtype=np.int64)
    cur = 0.0
    for i in range(v):
        masks[i] = _allowed_mask(i, pos, cand)
        cur += F[i, masks[i]]

    best_score = -np.inf
    best_sub = np.zeros(v, dtype=np.int64)
    newm = np.empty(v, dtype=np.int64)
    n_acc = 0

    for it in range(-1, n_iter):
        if it >= 0:
            if v < 2:
                break
            if np.random.random() < 0.5:
                p = np.random.randint(0, v - 1)
                q = p + 1
            else:
                p = np.random.randint(0, v)
                q = np.random.randint(0, v - 1)
                if q >= p:
                    q += 1
                if q < p:
                    tmp = p
                    p = q
                    q = tmp
            u = order[p]
            w = order[q]
            order[p] = w
            order[q] = u
            pos[w] = p
            pos[u] = q
            delta = 0.0
            for s in range(p, q + 1):
                node = order[s]
                nm = _allowed_mask(node, pos, cand)
                newm[s] = nm
                delta += F[node, nm] - F[node, masks[node]]
            if math.log(np.random.random() + 1e-300) < delta:
                for s in range(p, q + 1):
                    masks[order[s]] = newm[s]
                cur += delta
                n_acc += 1
            else:
                order[p] = u
                order[q] = w
                pos[u] = p
                pos[w] = q
            if (it % 20000) == 19999:              # remove accumulated floating-point drift
                cur = 0.0
                for i in range(v):
                    cur += F[i, masks[i]]
        # collect the MAP DAG of the current order (it = -1 is the initial order)
        if it < 0 or (it >= burn and ((it - burn) % thin) == 0):
            tot = 0.0
            for i in range(v):
                tot += G[i, masks[i]]
            if tot > best_score:
                best_score = tot
                for i in range(v):
                    best_sub[i] = A[i, masks[i]]
    return best_score, best_sub, n_acc / max(n_iter, 1)
