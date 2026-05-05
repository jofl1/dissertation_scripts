import csv
from pathlib import Path

import numpy as np

from utils.constants import a_r, trim_margin
from utils.descriptors import build_gradients
from utils.io import discover_systems, load_full
from utils.krr import fit_krr_loocv_pipeline

# Gradient descriptors D_2 = [n, n'] and D_3 = [n, n', n''] swept across
# the 19 exact systems. The dissertation reports D_1 / D_oracle / windowed
# descriptors but never D_2 / D_3 on the physical systems -- this fills
# the gap. Uses the pipeline KRR (scaler refit inside each LOO fold) so
# the published numbers in tab:gradient_descriptors_exact reproduce.

here = Path(__file__).parent
data_dir = here.parent / "section4" / "data"
results_dir = here / "results"
results_dir.mkdir(parents=True, exist_ok=True)


def main():
    systems = discover_systems(data_dir)
    print(f"Discovered {len(systems)} systems")

    rows = []
    for a_left, d in systems:
        x, n, v_xc = load_full(data_dir, a_left, d)
        sl = slice(trim_margin(len(x)), -trim_margin(len(x)))
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        d1, d2, d3 = build_gradients(x, n, sl)
        d_oracle = np.column_stack([n_in, x_in])

        mae_d1 = fit_krr_loocv_pipeline(d1, v_in).mae
        mae_d2 = fit_krr_loocv_pipeline(d2, v_in).mae
        mae_d3 = fit_krr_loocv_pipeline(d3, v_in).mae
        mae_oracle = fit_krr_loocv_pipeline(d_oracle, v_in).mae

        print(f"  AL={a_left:.1f} d={d:.1f} DA={a_r - a_left:.1f}  "
              f"D1={mae_d1:.4f}  D2={mae_d2:.4f}  D3={mae_d3:.4f}  "
              f"oracle={mae_oracle:.4f}")

        rows.append({
            "AL": a_left, "d": d, "DeltaA": a_r - a_left,
            "mae_d1": mae_d1, "mae_d2": mae_d2, "mae_d3": mae_d3,
            "mae_oracle": mae_oracle,
        })

    csv_path = results_dir / "gradient_descriptors_mae.csv"
    cols = ["AL", "d", "DeltaA", "mae_d1", "mae_d2", "mae_d3", "mae_oracle"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row[c] for c in cols])
    print(f"\nSaved: {csv_path}")

    d1_arr = np.array([r["mae_d1"] for r in rows])
    d2_arr = np.array([r["mae_d2"] for r in rows])
    d3_arr = np.array([r["mae_d3"] for r in rows])
    print(f"\nMedian D2/D1 = {np.median(d2_arr / d1_arr):.3f}, "
          f"Median D3/D1 = {np.median(d3_arr / d1_arr):.3f}")
    print(f"Mean   D2/D1 = {np.mean(d2_arr / d1_arr):.3f}, "
          f"Mean   D3/D1 = {np.mean(d3_arr / d1_arr):.3f}")


if __name__ == "__main__":
    main()
    print("Done.")
