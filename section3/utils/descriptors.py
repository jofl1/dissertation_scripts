import numpy as np

# Build the four pointwise descriptor matrices on the trimmed grid:
# D_1 = [n], D_2 = [n, n'], D_3 = [n, n', n''], D_oracle = [n, x].
# Returns a dict keyed by the LaTeX display name so callers can iterate
# while keeping the labels for plotting.


def build_pointwise(x, n, sl):
    dn_dx = np.gradient(n, x)
    d2n_dx2 = np.gradient(dn_dx, x)

    n_inner = n[sl]
    dn_inner = dn_dx[sl]
    d2n_inner = d2n_dx2[sl]
    x_inner = x[sl]

    return {
        r"$D_1 = [n]$": n_inner.reshape(-1, 1),
        r"$D_2 = [n, n']$": np.column_stack([n_inner, dn_inner]),
        r"$D_3 = [n, n', n'']$": np.column_stack([n_inner, dn_inner, d2n_inner]),
        r"$D_{\mathrm{oracle}} = [n, x]$": np.column_stack([n_inner, x_inner]),
    }
