import numpy as np

# Hyperparameter grids and shared constants. gamma_grid_12 matches the
# section4 master plot and is used by appendix_C and appendix_D;
# gamma_grid_10 matches fig_5_6 and is only used by appendix_B. a_r is
# the right-well depth, fixed at 2.0 across the 19-system grid.
a_r = 2.0
alpha_grid = np.logspace(-8, -1, 8)
gamma_grid_12 = np.logspace(-3, 2, 12)
gamma_grid_10 = np.logspace(-3, 2, 10)


def trim_margin(n_points):
    return max(3, n_points // 20)
