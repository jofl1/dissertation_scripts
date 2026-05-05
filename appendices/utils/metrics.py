import numpy as np
from sklearn.neighbors import NearestNeighbors

# KNN splitting metric in 1D density space. Mirrors section4's
# splitting_metric_knn but takes k as an explicit argument because
# appendix_D sweeps k.


def splitting_metric_knn(n_inner, v_inner, k):
    X = n_inner.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    _, indices = nn.kneighbors(X)
    stds = np.array([np.std(v_inner[indices[i]]) for i in range(len(X))])
    return float(np.mean(stds))
