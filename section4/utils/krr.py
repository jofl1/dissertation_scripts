from dataclasses import dataclass

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# KRR fitting with grid-searched LOO-CV. RBF kernel, features standardised
# before fitting. Returns the LOO predictions (for residuals/floor checks),
# the best MAE, and the chosen hyperparameters.


@dataclass
class KRRResult:
    predictions: np.ndarray
    mae: float
    alpha: float
    gamma: float


def fit_krr_loocv(X, y, alphas=None, gammas=None):
    if alphas is None:
        alphas = np.logspace(-8, -1, 8)
    if gammas is None:
        gammas = np.logspace(-3, 2, 12)

    X_scaled = StandardScaler().fit_transform(X)

    best = None
    for alpha in alphas:
        for gamma in gammas:
            krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            y_pred = cross_val_predict(krr, X_scaled, y, cv=len(y))
            mae = float(np.mean(np.abs(y - y_pred)))
            if best is None or mae < best.mae:
                best = KRRResult(predictions=y_pred, mae=mae, alpha=alpha, gamma=gamma)
    return best
