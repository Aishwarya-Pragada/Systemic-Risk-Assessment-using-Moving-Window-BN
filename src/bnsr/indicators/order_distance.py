"""Relative order (Eq. 5), normalised order (Eq. 6) and order distance (Eq. 7)."""
import numpy as np


def relative_order(M, v):
    """Eq. (5): RM_t(X_i) = (v_t - M_t(X_i)) / (v_t - 1); 1 = head of the order, 0 = tail."""
    return (v - np.asarray(M, dtype=float)) / (v - 1)


def normalized_order(M, v, idx):
    """Eq. (6): NT_t(V_s) = (v_t - M_t(X_i)) / sum_{j in V_s}(v_t - M_t(X_j)) for i in V_s.
    `idx` selects the nodes of V_s inside the length-v vector M."""
    w = v - np.asarray(M, dtype=float)[idx]
    s = w.sum()
    if s <= 0:
        raise ValueError("degenerate normalisation (all selected nodes at the tail)")
    return w / s


def order_distance(M_prev, nodes_prev, M_cur, nodes_cur):
    """Eq. (7): OD_t = || NT_t(V_s) - NT_{t-1}(V_s) ||_1 with V_s = V_{t-1} intersect V_t.
    M_*: median orders; nodes_*: matching lists of symbols; v_* = len(nodes_*)."""
    common = [s for s in nodes_prev if s in set(nodes_cur)]
    ip = {s: i for i, s in enumerate(nodes_prev)}
    ic = {s: i for i, s in enumerate(nodes_cur)}
    a = normalized_order(M_prev, len(nodes_prev), [ip[s] for s in common])
    b = normalized_order(M_cur, len(nodes_cur), [ic[s] for s in common])
    return float(np.abs(b - a).sum())
