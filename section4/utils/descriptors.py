import numpy as np

# Descriptor builders. build_pointwise covers D_1 = [n] and the oracle
# D = [n, x] used by every figure that fits KRR. build_window slides a
# centred density window of half-width ell over the grid for fig_5_6.


def build_pointwise(x, n, sl):
    n_inner = n[sl]
    x_inner = x[sl]
    return {
        r"$D_1 = [n]$": n_inner.reshape(-1, 1),
        r"$D_{\mathrm{oracle}} = [n, x]$": np.column_stack([n_inner, x_inner]),
    }


def build_window(x_full, n_full, ell):
    # Density at offsets [-k..k] around each retained inner point, where
    # k = round(ell / dx). Returns (None, None) if the window leaves
    # fewer than 20 valid points.
    dx = x_full[1] - x_full[0]
    k = int(round(ell / dx))
    margin = max(3, len(x_full) // 20)
    lo = max(margin, k)
    hi = len(x_full) - max(margin, k)
    indices = np.arange(lo, hi)
    if len(indices) < 20:
        return None, None
    X = np.column_stack([n_full[indices + off] for off in range(-k, k + 1)])
    return indices, X
