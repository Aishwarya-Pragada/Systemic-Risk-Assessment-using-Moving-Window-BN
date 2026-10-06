"""Network statistics shared by later stages."""


def modified_network_density(n_edges, v, M):
    """Eq. (4): MND = e_t / (v_t * M - M (M + 1) / 2)."""
    return n_edges / (v * M - M * (M + 1) / 2.0)
