"""Structure MCMC (Madigan & York, 1995; paper ref. 40) restricted to the candidate-parent space.

Proposal: pick one of the eligible node pairs uniformly, then move it uniformly to one of its *other* allowed
states {no edge, a->b, b->a}.  This adds, deletes or reverses an edge.  The number of allowed states of a pair is
fixed, so the proposal is symmetric and the Metropolis-Hastings ratio reduces to the BDeu score ratio
(the paper's acceptance ratio with q(G|G')/q(G'|G) = 1).  Moves that break acyclicity or the maximum
parent size M (score = -inf) are rejected.  The highest-BDeu DAG visited is returned.
"""
import math

import numpy as np

from .._jit import njit


@njit(cache=True)
def _reaches(pm, y, x):
    """True if x is reachable from y following parent->child edges (pm[n] = parent bitmask of n)."""
    v = pm.shape[0]
    reach = 1 << y
    changed = True
    while changed:
        changed = False
        for n in range(v):
            if ((reach >> n) & 1) == 0 and (pm[n] & reach) != 0:
                reach |= (1 << n)
                changed = True
    return ((reach >> x) & 1) == 1


@njit(cache=True)
def structure_mcmc(scores, cand, init_sub, n_iter, seed):
    np.random.seed(seed)
    v = cand.shape[0]
    C = cand.shape[1]
    candpos = np.full((v, v), -1, dtype=np.int64)       # candpos[i, j] = bit of j in i's candidate list
    for i in range(v):
        for c in range(C):
            candpos[i, cand[i, c]] = c

    npairs = 0
    for a in range(v):
        for b in range(a + 1, v):
            if candpos[b, a] >= 0 or candpos[a, b] >= 0:
                npairs += 1
    pa = np.empty(npairs, dtype=np.int64)
    pb = np.empty(npairs, dtype=np.int64)
    k = 0
    for a in range(v):
        for b in range(a + 1, v):
            if candpos[b, a] >= 0 or candpos[a, b] >= 0:
                pa[k] = a
                pb[k] = b
                k += 1

    sub = init_sub.copy()
    pm = np.zeros(v, dtype=np.int64)
    cur = 0.0
    for i in range(v):
        cur += scores[i, sub[i]]
        for c in range(C):
            if ((sub[i] >> c) & 1) == 1:
                pm[i] |= (1 << cand[i, c])
    best = cur
    best_sub = sub.copy()
    opts = np.empty(3, dtype=np.int64)
    n_acc = 0
    if npairs == 0:
        return best, best_sub, 0.0

    for it in range(n_iter):
        k = np.random.randint(0, npairs)
        a = pa[k]
        b = pb[k]
        e_ab = candpos[b, a]          # >= 0 iff a -> b is allowed
        e_ba = candpos[a, b]          # >= 0 iff b -> a is allowed
        if e_ab >= 0 and ((sub[b] >> e_ab) & 1) == 1:
            st = 1
        elif e_ba >= 0 and ((sub[a] >> e_ba) & 1) == 1:
            st = 2
        else:
            st = 0
        no = 0
        if st != 0:
            opts[no] = 0
            no += 1
        if st != 1 and e_ab >= 0:
            opts[no] = 1
            no += 1
        if st != 2 and e_ba >= 0:
            opts[no] = 2
            no += 1
        ns = opts[np.random.randint(0, no)]

        nb = sub[b]
        na = sub[a]
        pmb = pm[b]
        pma = pm[a]
        if st == 1:
            nb &= ~(1 << e_ab)
            pmb &= ~(1 << a)
        elif st == 2:
            na &= ~(1 << e_ba)
            pma &= ~(1 << b)
        if ns == 1:
            nb |= (1 << e_ab)
            pmb |= (1 << a)
        elif ns == 2:
            na |= (1 << e_ba)
            pma |= (1 << b)

        sb = scores[b, nb]
        sa = scores[a, na]
        if sb == -np.inf or sa == -np.inf:        # exceeds the maximum number of parents M
            continue
        delta = sb + sa - scores[b, sub[b]] - scores[a, sub[a]]
        if not (math.log(np.random.random() + 1e-300) < delta):
            continue
        if ns != 0:                               # acyclicity check for the new edge x -> y
            old_pa = pm[a]
            old_pb = pm[b]
            pm[a] = pma
            pm[b] = pmb
            if ns == 1:
                cyc = _reaches(pm, b, a)
            else:
                cyc = _reaches(pm, a, b)
            if cyc:
                pm[a] = old_pa
                pm[b] = old_pb
                continue
        else:
            pm[a] = pma
            pm[b] = pmb
        sub[a] = na
        sub[b] = nb
        cur += delta
        n_acc += 1
        if cur > best + 1e-12:
            best = cur
            for i in range(v):
                best_sub[i] = sub[i]
    return best, best_sub, n_acc / max(n_iter, 1)
