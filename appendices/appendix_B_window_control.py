import csv
from pathlib import Path

import numpy as np

from utils.constants import gamma_grid_10, trim_margin
from utils.descriptors import build_window
from utils.io import load_full
from utils.krr import fit_krr_loocv

# Control for the fig_5_6 window scan. The MAE collapse with growing
# half-width ell could in principle be partly an artefact of the valid
# evaluation set shrinking near the box boundary. We re-run the same KRR
# pipeline twice for each (system, ell): once on the per-ell-valid grid
# (matches the figure) and once on the common grid valid for ell_max=8,
# so every ell uses the SAME index range. If the collapse persists on
# the common grid it is genuinely caused by the descriptor.

here = Path(__file__).parent
data_dir = here.parent / "section4" / "data"
results_dir = here / "results"
results_dir.mkdir(parents=True, exist_ok=True)

window_half_widths = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

systems = [
    (1.0, 10.0, "DA1.0_d10"),
    (1.0, 6.0,  "DA1.0_d6"),
    (1.5, 10.0, "DA0.5_d10"),
]


def window_mae(x, n, v_xc, ell, common_indices=None):
    indices, X = build_window(x, n, ell, common_indices=common_indices)
    if indices is None:
        return np.nan
    return fit_krr_loocv(X, v_xc[indices], gammas=gamma_grid_10).mae


def main():
    rows = []
    ell_max = max(window_half_widths)

    for a_left, d, sys_label in systems:
        x, n, v_xc = load_full(data_dir, a_left, d)
        dx = x[1] - x[0]
        k_max = int(round(ell_max / dx))
        margin = trim_margin(len(x))
        common_indices = np.arange(max(margin, k_max), len(x) - max(margin, k_max))

        for ell in window_half_widths:
            mae_orig = window_mae(x, n, v_xc, ell)
            mae_common = window_mae(x, n, v_xc, ell, common_indices=common_indices)
            print(f"  {sys_label}  ell={ell:.1f}  per-ell={mae_orig:.4f}  "
                  f"common={mae_common:.4f}")
            rows.append({
                "system": sys_label,
                "AL": a_left, "d": d, "ell": ell,
                "MAE_per_ell_valid": mae_orig,
                "MAE_common_grid": mae_common,
            })

    csv_path = results_dir / "window_scan_control.csv"
    cols = ["system", "AL", "d", "ell", "MAE_per_ell_valid", "MAE_common_grid"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row[c] for c in cols])
    print(f"\nSaved: {csv_path}")

    print(f"\n{'system':>12} {'per-ell ratio':>14} {'common ratio':>14}")
    for a_left, d, sys_label in systems:
        sys_rows = sorted([r for r in rows if r["system"] == sys_label],
                          key=lambda r: r["ell"])
        first, last = sys_rows[0], sys_rows[-1]
        ratio_per = last["MAE_per_ell_valid"] / first["MAE_per_ell_valid"]
        ratio_common = last["MAE_common_grid"] / first["MAE_common_grid"]
        print(f"{sys_label:>12} {ratio_per:14.3f} {ratio_common:14.3f}")


if __name__ == "__main__":
    main()
    print("Done.")
