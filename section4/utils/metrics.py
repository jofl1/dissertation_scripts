import numpy as np
from sklearn.neighbors import NearestNeighbors

# KNN splitting metric in 1D density space. The mean is the dissertation's
# scalar diagnostic <sigma_k>; the per-point form is plotted in
# fig_5_sigma_spatial.


def per_point_sigma(n_inner, v_inner, k=5):
    # For each point, look at its k nearest neighbours by density value
    # (the point itself is included via n_neighbors=k+1) and return the
    # std of v among those k+1 points.
    X = n_inner.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    _, indices = nn.kneighbors(X)
    return np.array([np.std(v_inner[indices[i]]) for i in range(len(X))])


def splitting_metric_knn(n_inner, v_inner, k=5):
    return float(np.mean(per_point_sigma(n_inner, v_inner, k=k)))
