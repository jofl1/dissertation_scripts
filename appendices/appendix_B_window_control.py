"""Control analysis for §4.6 / Figure 5.6: window scan on a common grid.

The original `ch5/fig_5_6_window_scan_exact.py` computes the LOO-CV MAE
of a density-window descriptor of half-width ell, but the valid evaluation
set shrinks as ell grows because the window cannot extend beyond the box.
A reviewer asked whether the observed MAE collapse is genuinely caused by
adding nonlocal density context or partly an artefact of removing
boundary-adjacent points.

This script re-runs the same KRR pipeline twice for each (system, ell):

  (a) Per-ell-valid grid (matches the original script).
  (b) Common grid valid for ell_max = max(WINDOW_HALF_WIDTHS) = 8.0 a.u.
      All ell values are evaluated on the SAME index range, so the
      collapse trend can be read like-for-like.

Inputs (read-only): cached exact solves in ../ch5/data/
Outputs:            results/window_scan_control.csv
                    + a printed summary of collapse ratios.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "ch5" / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AR = 2.0
ALPHA_GRID = np.logspace(-8, -1, 8)
GAMMA_GRID = np.logspace(-3, 2, 10)
WINDOW_HALF_WIDTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

SYSTEMS = [
    (1.0, 10.0, "DA1.0_d10"),
    (1.0, 6.0,  "DA1.0_d6"),
    (1.5, 10.0, "DA0.5_d10"),
]


def load_system(AL, d):
    path = DATA_DIR / f"exact_asymmetric_uu_AL{AL}_AR{AR}_d{d}.npz"
    data = dict(np.load(path))
    v_xc_raw = data["v_xc"]
    data["v_xc"] = v_xc_raw - v_xc_raw[-1]
    return data


def krr_loo_mae(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_mae = np.inf
    for alpha in ALPHA_GRID:
        for gamma in GAMMA_GRID:
            krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            y_pred = cross_val_predict(krr, X_scaled, y, cv=len(y))
            mae = np.mean(np.abs(y - y_pred))
            if mae < best_mae:
                best_mae = mae
    return best_mae


def window_mae(x_full, n_full, v_xc_full, ell, common_indices=None):
    """Compute KRR LOO MAE for a window-of-half-width-ell density descriptor.

    common_indices=None : use per-ell-valid index range (original behaviour).
    common_indices given: use those indices regardless of ell (control).
    """
    dx = x_full[1] - x_full[0]
    k = int(round(ell / dx))
    margin = max(3, len(x_full) // 20)

    if common_indices is None:
        lo = max(margin, k)
        hi = len(x_full) - max(margin, k)
        if hi - lo < 20:
            return np.nan
        indices = np.arange(lo, hi)
    else:
        indices = common_indices
        if indices.min() < k or indices.max() + k >= len(x_full):
            return np.nan

    y = v_xc_full[indices]
    X_window = np.column_stack([
        n_full[indices + offset]
        for offset in range(-k, k + 1)
    ])
    return krr_loo_mae(X_window, y)


def main() -> int:
    rows = []
    ell_max = max(WINDOW_HALF_WIDTHS)

    print(f"\n{'='*72}")
    print("CONTROL ANALYSIS for window descriptor scan (Figure 5.6)")
    print(f"All ell evaluated on the index range valid for ell_max={ell_max} a.u.")
    print(f"{'='*72}")

    for AL, d, sys_label in SYSTEMS:
        print(f"\nSystem: AL={AL}, d={d}  ({sys_label})")
        print("-" * 72)

        data = load_system(AL, d)
        x = data["x"]
        n = data["n"]
        v_xc = data["v_xc"]
        dx = x[1] - x[0]

        k_max = int(round(ell_max / dx))
        margin = max(3, len(x) // 20)
        lo_common = max(margin, k_max)
        hi_common = len(x) - max(margin, k_max)
        common_indices = np.arange(lo_common, hi_common)

        print(f"  Common-grid index range: [{lo_common}, {hi_common})  "
              f"size={len(common_indices)}")
        print(f"  Per-ell range at ell=0:  [{margin}, {len(x)-margin})  "
              f"size={len(x) - 2*margin}")
        print()

        for ell in WINDOW_HALF_WIDTHS:
            print(f"  ell={ell:.1f} a.u. ...", flush=True)

            mae_orig = window_mae(x, n, v_xc, ell)
            mae_common = window_mae(x, n, v_xc, ell,
                                    common_indices=common_indices)

            print(f"    per-ell-valid MAE: {mae_orig:.4f}")
            print(f"    common-grid  MAE: {mae_common:.4f}")

            rows.append({
                "system": sys_label,
                "AL": AL,
                "d": d,
                "ell": ell,
                "MAE_per_ell_valid": mae_orig,
                "MAE_common_grid": mae_common,
            })

    csv_path = RESULTS_DIR / "window_scan_control.csv"
    cols = ["system", "AL", "d", "ell",
            "MAE_per_ell_valid", "MAE_common_grid"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    print(f"\n{'='*72}")
    print("SUMMARY: collapse from ell=0 to ell=8 on each grid")
    print(f"{'='*72}")
    for AL, d, sys_label in SYSTEMS:
        sys_rows = [r for r in rows if r["system"] == sys_label]
        sys_rows.sort(key=lambda r: r["ell"])
        first = sys_rows[0]
        last = sys_rows[-1]
        print(f"\n{sys_label} (AL={AL}, d={d}):")
        print(f"  per-ell:  {first['MAE_per_ell_valid']:.4f} -> "
              f"{last['MAE_per_ell_valid']:.4f}  "
              f"(ratio={last['MAE_per_ell_valid']/first['MAE_per_ell_valid']:.3f})")
        print(f"  common:   {first['MAE_common_grid']:.4f} -> "
              f"{last['MAE_common_grid']:.4f}  "
              f"(ratio={last['MAE_common_grid']/first['MAE_common_grid']:.3f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
