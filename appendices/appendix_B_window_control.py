import csv
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# Control for the fig_5_6 window scan. The MAE collapse with growing
# half-width ell could in principle be partly an artefact of the valid
# evaluation set shrinking near the box boundary. We re-run the same KRR
# pipeline twice for each (system, ell): once on the per-ell-valid grid
# (matches the figure) and once on the common grid valid for ell_max=8,
# so every ell uses the SAME index range. If the collapse persists on the
# common grid it is genuinely caused by the descriptor, not by trimming.

here = Path(__file__).parent
data_dir = here.parent / "section4" / "data"
results_dir = here / "results"
results_dir.mkdir(parents=True, exist_ok=True)

a_r = 2.0
alpha_grid = np.logspace(-8, -1, 8)
gamma_grid = np.logspace(-3, 2, 10)
window_half_widths = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

systems = [
    (1.0, 10.0, "DA1.0_d10"),
    (1.0, 6.0,  "DA1.0_d6"),
    (1.5, 10.0, "DA0.5_d10"),
]


def krr_loo_mae(X, y):
    X_scaled = StandardScaler().fit_transform(X)
    best = np.inf
    for alpha in alpha_grid:
        for gamma in gamma_grid:
            krr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            y_pred = cross_val_predict(krr, X_scaled, y, cv=len(y))
            mae = float(np.mean(np.abs(y - y_pred)))
            if mae < best:
                best = mae
    return best


def window_mae(x_full, n_full, v_xc_full, ell, common_indices=None):
    # Pass common_indices=None for the original per-ell-valid behaviour.
    # Pass an explicit index array to evaluate every ell on the same grid.
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
        n_full[indices + offset] for offset in range(-k, k + 1)
    ])
    return krr_loo_mae(X_window, y)


def main():
    rows = []
    ell_max = max(window_half_widths)

    print(f"\n{'=' * 72}")
    print("CONTROL ANALYSIS for window descriptor scan (Figure 5.6)")
    print(f"All ell evaluated on the index range valid for ell_max={ell_max} a.u.")
    print(f"{'=' * 72}")

    for a_left, d, sys_label in systems:
        print(f"\nSystem: AL={a_left}, d={d}  ({sys_label})")
        print("-" * 72)

        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        with np.load(path) as data:
            x = data["x"].copy()
            n = data["n"].copy()
            v_xc_raw = data["v_xc"].copy()
        v_xc = v_xc_raw - v_xc_raw[-1]
        dx = x[1] - x[0]

        k_max = int(round(ell_max / dx))
        margin = max(3, len(x) // 20)
        lo_common = max(margin, k_max)
        hi_common = len(x) - max(margin, k_max)
        common_indices = np.arange(lo_common, hi_common)

        print(f"  Common-grid index range: [{lo_common}, {hi_common})  "
              f"size={len(common_indices)}")
        print(f"  Per-ell range at ell=0:  [{margin}, {len(x) - margin})  "
              f"size={len(x) - 2 * margin}")
        print()

        for ell in window_half_widths:
            print(f"  ell={ell:.1f} a.u. ...", flush=True)

            mae_orig = window_mae(x, n, v_xc, ell)
            mae_common = window_mae(x, n, v_xc, ell, common_indices=common_indices)

            print(f"    per-ell-valid MAE: {mae_orig:.4f}")
            print(f"    common-grid  MAE: {mae_common:.4f}")

            rows.append({
                "system": sys_label,
                "AL": a_left,
                "d": d,
                "ell": ell,
                "MAE_per_ell_valid": mae_orig,
                "MAE_common_grid": mae_common,
            })

    csv_path = results_dir / "window_scan_control.csv"
    cols = ["system", "AL", "d", "ell",
            "MAE_per_ell_valid", "MAE_common_grid"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row[k] for k in cols])
    print(f"\nSaved: {csv_path}")

    print(f"\n{'=' * 72}")
    print("SUMMARY: collapse from ell=0 to ell=8 on each grid")
    print(f"{'=' * 72}")
    for a_left, d, sys_label in systems:
        sys_rows = [r for r in rows if r["system"] == sys_label]
        sys_rows.sort(key=lambda r: r["ell"])
        first = sys_rows[0]
        last = sys_rows[-1]
        print(f"\n{sys_label} (AL={a_left}, d={d}):")
        print(f"  per-ell:  {first['MAE_per_ell_valid']:.4f} -> "
              f"{last['MAE_per_ell_valid']:.4f}  "
              f"(ratio={last['MAE_per_ell_valid'] / first['MAE_per_ell_valid']:.3f})")
        print(f"  common:   {first['MAE_common_grid']:.4f} -> "
              f"{last['MAE_common_grid']:.4f}  "
              f"(ratio={last['MAE_common_grid'] / first['MAE_common_grid']:.3f})")


if __name__ == "__main__":
    main()
    print("Done.")
