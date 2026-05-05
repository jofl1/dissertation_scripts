import numpy as np

# Shared parameters for the 19-system asymmetric double-well grid. The
# right-well depth a_r is fixed; left-well depth a_left and separation d
# are swept by the four solve_*.py scripts.
length = 30.0
soft = 1.0
dx_target = 0.25
a_r = 2.0
d_default = 10.0

k_neighbours = 5
alpha_grid = np.logspace(-8, -1, 8)
# Two gamma grids -- the originals used a 12-point grid in fig_5_3b, fig_5_4b,
# fig_5_5 and a 10-point grid in fig_5_6. Preserved as separate constants so
# the refactored figure scripts produce the same MAE numbers as before.
gamma_grid_12 = np.logspace(-3, 2, 12)
gamma_grid_10 = np.logspace(-3, 2, 10)


def trim_margin(n_points):
    # max(3, N // 20) -- 5% trim with a hard floor at 3 grid points.
    return max(3, n_points // 20)


rc_params = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}
