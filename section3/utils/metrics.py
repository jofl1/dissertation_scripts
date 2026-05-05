import numpy as np
from sklearn.neighbors import NearestNeighbors

# KNN splitting metric and the mirror-pair MAE floor diagnostic.


def splitting_metric_knn(n_inner, v_inner, k=5):
    # Average KNN splitting metric in 1D density space. For each point,
    # find its k nearest neighbours by density value (the point itself is
    # included automatically because the search uses n_neighbors=k+1).
    # Compute the std of v among those k+1 points, then average.
    X = n_inner.reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    stds = np.array([np.std(v_inner[indices[i]]) for i in range(len(X))])
    return float(np.mean(stds))


def mirror_pair_mae_floor(x, v_pred, v_target, atol=1e-9):
    # Mean of pair-averaged absolute errors over off-centre mirror pairs.
    # For any deterministic f(n) reading only the symmetric baseline
    # density, this floor is at least S/2 for the antisymmetric step
    # construction.
    pair_means = []
    used_left = set()
    for i, xi in enumerate(x):
        if xi <= 0:
            continue
        matches = np.where(np.abs(x + xi) < atol)[0]
        if len(matches) == 0:
            continue
        j = int(matches[0])
        if j in used_left:
            continue
        used_left.add(j)
        err_r = abs(v_pred[i] - v_target[i])
        err_l = abs(v_pred[j] - v_target[j])
        pair_means.append(0.5 * (err_r + err_l))
    if not pair_means:
        return float("nan")
    return float(np.mean(pair_means))
