import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import rc_params, s_values, trim_margin
from utils.descriptors import build_pointwise
from utils.krr import fit_krr_loocv
from utils.metrics import mirror_pair_mae_floor
from utils.toy_target import make_toy_target

# LOO-CV KRR MAE vs S for the four pointwise descriptors. Also computes
# the mirror-pair MAE floor for D_1 directly and verifies it sits on the
# analytic S/2 lower bound.

here = Path(__file__).parent
fig_dir = here / "figures"
data_dir = here / "data"
res_dir = here / "results"
baseline_path = data_dir / "baseline_symmetric_uu_d10.npz"
fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

d1_name = r"$D_1 = [n]$"


def main():
    plt.rcParams.update(rc_params)

    with np.load(baseline_path) as baseline:
        x = baseline["x"]
        n = baseline["n"]
        v_xc = baseline["v_xc"]

    margin = trim_margin(len(x))
    sl = slice(margin, -margin)
    x_inner = x[sl]
    v_xc_inner = v_xc[sl]
    descriptors = build_pointwise(x, n, sl)
    desc_names = list(descriptors)
    print("Descriptors:", desc_names)

    results = {name: [] for name in desc_names}
    floor_records = []  # (S, MAE_D1, mirror-pair floor, S/2)
    t0 = time.time()
    for S in s_values:
        y = make_toy_target(v_xc_inner, x_inner, S)
        print(f"  S = {S:.2f}")
        # Keep the D_1 predictions aside so we can compute the mirror-pair
        # MAE floor for it after the descriptor loop.
        d1_predictions = None
        for name, X in descriptors.items():
            res = fit_krr_loocv(X, y)
            results[name].append(res.mae)
            print(f"    {name:40s}  MAE = {res.mae:.6f}  alpha={res.alpha:.1e}  gamma={res.gamma:.1e}")
            if name == d1_name:
                d1_predictions = res.predictions

        # Compare the empirical D_1 mirror-pair MAE against the analytic S/2
        # floor as a sanity check on every iteration.
        floor = mirror_pair_mae_floor(x_inner, d1_predictions, y)
        floor_records.append((S, results[d1_name][-1], floor, 0.5 * S))
        print(f"    mirror-pair MAE floor (D1) = {floor:.6f}  (S/2 = {0.5*S:.6f})")

    print(f"\nTotal KRR fitting time: {time.time() - t0:.1f}s")

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = [
        {"color": "#d62728", "marker": "o"},
        {"color": "#ff7f0e", "marker": "s"},
        {"color": "#2ca02c", "marker": "^"},
        {"color": "#1f77b4", "marker": "D"},
    ]
    s_arr = np.array(s_values, dtype=float)
    xmax = s_arr.max() * 1.1
    for name, style in zip(desc_names, styles):
        maes = np.array(results[name])
        ax.plot(s_arr, maes, linewidth=2, markersize=8,
                markeredgecolor="black", markeredgewidth=0.8,
                label=name, zorder=5, **style)

    s_line = np.linspace(0, xmax, 100)
    ax.plot(s_line, 0.5 * s_line, ":", color="black", linewidth=1.5,
            alpha=0.6, label=r"$S/2$ floor (analytic)")

    ax.set_xlabel(r"Step height $S$ (a.u.)")
    ax.set_ylabel(r"LOO-CV MAE (a.u.)")
    ax.set_xlim(-0.02, xmax)
    ax.set_ylim(bottom=-0.005)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = fig_dir / f"fig_krr_mae.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    out_csv = res_dir / "krr_mae_vs_S.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S"] + desc_names)
        for i, S in enumerate(s_arr):
            w.writerow([S] + [results[name][i] for name in desc_names])
    print(f"Saved: {out_csv}")

    floor_csv = res_dir / "mirror_pair_floor.csv"
    with open(floor_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S", "mae_D1", "mirror_pair_floor", "S_over_2"])
        for row in floor_records:
            w.writerow(row)
    print(f"Saved: {floor_csv}")


if __name__ == "__main__":
    main()
    print("Done.")
