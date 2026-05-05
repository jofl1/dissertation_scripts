import csv
import re
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Gradient descriptors D_2 = [n, n'] and D_3 = [n, n', n''] swept across
# the 19 exact systems, using the same KRR setup as fig_5_5. The
# dissertation reports D_1 / D_oracle / windowed descriptors in chapter 4
# but never D_2 / D_3 on the physical systems -- this fills that gap so
# the table in Appendix C has numbers to point at. Derivatives are taken
# on the full grid then sliced, so the inner edges still see exterior
# neighbours.

here = Path(__file__).parent
data_dir = here.parent / "section4" / "data"
results_dir = here / "results"
results_dir.mkdir(parents=True, exist_ok=True)

a_r = 2.0
alpha_grid = np.logspace(-8, -1, 8)
gamma_grid = np.logspace(-3, 2, 12)

filename_pattern = re.compile(
    r"exact_asymmetric_uu_AL([\d.]+)_AR2\.0_d([\d.]+)\.npz$"
)


def discover_systems():
    systems = []
    for f in sorted(data_dir.glob("exact_asymmetric_uu_AL*_AR2.0_d*.npz")):
        if "_L" in f.name:
            continue
        m = filename_pattern.match(f.name)
        if m:
            systems.append((float(m.group(1)), float(m.group(2))))
    return systems


def krr_loo_mae(X, y):
    # Pipeline form so the StandardScaler is refit inside each LOO fold.
    best = np.inf
    for alpha in alpha_grid:
        for gamma in gamma_grid:
            model = make_pipeline(
                StandardScaler(),
                KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma),
            )
            y_pred = cross_val_predict(model, X, y, cv=len(y))
            mae = float(np.mean(np.abs(y - y_pred)))
            if mae < best:
                best = mae
    return best


def build_descriptors(x, n):
    # Compute n' and n'' on the full grid, then return D1, D2, D3 stacked.
    # The caller slices with the standard boundary-trim afterwards.
    dx = x[1] - x[0]
    n_prime = np.gradient(n, dx)
    n_pp = np.gradient(n_prime, dx)
    d1 = n.reshape(-1, 1)
    d2 = np.column_stack([n, n_prime])
    d3 = np.column_stack([n, n_prime, n_pp])
    return d1, d2, d3


def main():
    systems = discover_systems()
    print(f"Discovered {len(systems)} systems:")
    for a_left, d in systems:
        print(f"  AL={a_left}, d={d}, DeltaA={a_r - a_left:.1f}")

    rows = []
    for a_left, d in systems:
        print(f"\nSystem: AL={a_left}, d={d}, DeltaA={a_r - a_left:.1f}")

        path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        with np.load(path) as data:
            x = data["x"].copy()
            n = data["n"].copy()
            v_xc_raw = data["v_xc"].copy()
        v_xc = v_xc_raw - v_xc_raw[-1]

        margin = max(3, len(x) // 20)
        sl = slice(margin, -margin)
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        d1_full, d2_full, d3_full = build_descriptors(x, n)
        d1 = d1_full[sl]
        d2 = d2_full[sl]
        d3 = d3_full[sl]
        d_oracle = np.column_stack([n_in, x_in])

        mae_d1 = krr_loo_mae(d1, v_in)
        mae_d2 = krr_loo_mae(d2, v_in)
        mae_d3 = krr_loo_mae(d3, v_in)
        mae_oracle = krr_loo_mae(d_oracle, v_in)

        print(f"  D1 MAE:     {mae_d1:.4f}")
        print(f"  D2 MAE:     {mae_d2:.4f}")
        print(f"  D3 MAE:     {mae_d3:.4f}")
        print(f"  Oracle MAE: {mae_oracle:.4f}")

        rows.append({
            "AL": a_left, "d": d, "DeltaA": a_r - a_left,
            "mae_d1": mae_d1,
            "mae_d2": mae_d2,
            "mae_d3": mae_d3,
            "mae_oracle": mae_oracle,
        })

    csv_path = results_dir / "gradient_descriptors_mae.csv"
    cols = ["AL", "d", "DeltaA", "mae_d1", "mae_d2", "mae_d3", "mae_oracle"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row[k] for k in cols])
    print(f"\nSaved: {csv_path}")

    print("\n" + "=" * 78)
    print("SUMMARY: do D_2 and D_3 materially change the local-model trend?")
    print("=" * 78)
    print(f"\n{'AL':>5} {'d':>5} {'DA':>5} {'D1':>10} {'D2':>10} {'D3':>10} "
          f"{'Oracle':>10} {'D2/D1':>7} {'D3/D1':>7}")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: r["mae_d1"]):
        ratio_d2 = r["mae_d2"] / r["mae_d1"] if r["mae_d1"] > 0 else float("nan")
        ratio_d3 = r["mae_d3"] / r["mae_d1"] if r["mae_d1"] > 0 else float("nan")
        print(f"{r['AL']:5.1f} {r['d']:5.1f} {r['DeltaA']:5.1f} "
              f"{r['mae_d1']:10.4f} {r['mae_d2']:10.4f} {r['mae_d3']:10.4f} "
              f"{r['mae_oracle']:10.4f} {ratio_d2:7.3f} {ratio_d3:7.3f}")

    d1 = np.array([r["mae_d1"] for r in rows])
    d2 = np.array([r["mae_d2"] for r in rows])
    d3 = np.array([r["mae_d3"] for r in rows])
    print(f"\nAggregate: mean D2/D1 = {np.mean(d2 / d1):.3f}, "
          f"mean D3/D1 = {np.mean(d3 / d1):.3f}")
    print(f"           median D2/D1 = {np.median(d2 / d1):.3f}, "
          f"median D3/D1 = {np.median(d3 / d1):.3f}")


if __name__ == "__main__":
    main()
    print("Done.")
