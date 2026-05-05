import numpy as np

from utils.constants import trim_margin

# Window descriptor for appendix_B and gradient descriptors D1/D2/D3
# for appendix_C. The gradients are taken on the FULL grid before
# slicing so np.gradient at the inner edges still uses exterior points.


def build_window(x_full, n_full, ell, common_indices=None):
    # common_indices=None: use the per-ell-valid index range (the figure
    # behaviour). Pass an array to evaluate every ell on the same grid.
    dx = x_full[1] - x_full[0]
    k = int(round(ell / dx))
    margin = trim_margin(len(x_full))

    if common_indices is None:
        lo = max(margin, k)
        hi = len(x_full) - max(margin, k)
        if hi - lo < 20:
            return None, None
        indices = np.arange(lo, hi)
    else:
        indices = common_indices
        if indices.min() < k or indices.max() + k >= len(x_full):
            return None, None

    X = np.column_stack([n_full[indices + off] for off in range(-k, k + 1)])
    return indices, X


def build_gradients(x, n, sl):
    # Returns (D1, D2, D3) sliced to the inner grid. n' and n'' are
    # computed on the full grid first, then sliced -- ensures inner-edge
    # values still see exterior neighbours.
    dx = x[1] - x[0]
    n_prime = np.gradient(n, dx)
    n_pp = np.gradient(n_prime, dx)
    d1 = n[sl].reshape(-1, 1)
    d2 = np.column_stack([n[sl], n_prime[sl]])
    d3 = np.column_stack([n[sl], n_prime[sl], n_pp[sl]])
    return d1, d2, d3
