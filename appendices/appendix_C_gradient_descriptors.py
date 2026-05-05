"""Control analysis: gradient descriptors D_2 and D_3 on the 19 exact asymmetric systems.

Section 2.3 of the dissertation defines four pointwise descriptor sets:
  D_1 = [n], D_2 = [n, n'], D_3 = [n, n', n''], D_oracle = [n, x].
The toy diagnostic (Section 3.2) uses D_2 and D_3 explicitly to test whether
semi-local information removes the mirror ambiguity. The exact-system results
(Chapter 4) report only D_1, the oracle, and density windows. A reviewer
flagged this asymmetry: what happens to D_2 and D_3 in the physical systems?

This script answers the question. For each of the 19 exact asymmetric systems
auto-discovered from ../ch5/data/, it computes the LOO-CV MAE of KRR with
descriptor sets D_1, D_2, D_3 (and reports D_oracle for context too).

Method matches the existing ch5 pipeline:
  - same alpha/gamma grid as the master plot
  - same boundary trim (margin = max(3, N//20))
  - same gauge convention (v_xc shifted to zero at right boundary)
  - n' and n'' computed by central finite differences on the uniform grid,
    then sliced to the interior so derivatives at the inner edges use
    exterior points (better accuracy than trimming first).

Output: results/gradient_descriptors_mae.csv
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "ch5" / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AR = 2.0
ALPHA_GRID = np.logspace(-8, -1, 8)
GAMMA_GRID = np.logspace(-3, 2, 12)


def discover_systems():
    pattern = re.compile(
        r"exact_asymmetric_uu_AL([\d.]+)_AR2\.0_d([\d.]+)\.npz$"
    )
    systems = []
    for f in sorted(DATA_DIR.glob("exact_asymmetric_uu_AL*_AR2.0_d*.npz")):
        if "_L" in f.name:
            continue
        m = pattern.match(f.name)
        if m:
            al = float(m.group(1))
            d = float(m.group(2))
            systems.append((al, d))
    return systems


def load_system(AL, d):
    path = DATA_DIR / f"exact_asymmetric_uu_AL{AL}_AR{AR}_d{d}.npz"
    data = dict(np.load(path))
    v_xc_raw = data["v_xc"]
    data["v_xc"] = v_xc_raw - v_xc_raw[-1]
    return data


def krr_loo_mae(X, y):
    """Grid-search KRR with LOO-CV; scaler is refit inside each fold."""
    best_mae = np.inf
    for alpha in ALPHA_GRID:
        for gamma in GAMMA_GRID:
            model = make_pipeline(
                StandardScaler(),
                KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma),
            )
            y_pred = cross_val_predict(model, X, y, cv=len(y))
            mae = np.mean(np.abs(y - y_pred))
            if mae < best_mae:
                best_mae = mae
    return best_mae


def build_descriptors(x, n):
    """Build D_1, D_2, D_3 on the FULL grid (slice later)."""
    dx = x[1] - x[0]
    n_prime = np.gradient(n, dx)
    n_pp = np.gradient(n_prime, dx)
    D1 = n.reshape(-1, 1)
    D2 = np.column_stack([n, n_prime])
    D3 = np.column_stack([n, n_prime, n_pp])
    return D1, D2, D3


def main() -> int:
    systems = discover_systems()
    print(f"Discovered {len(systems)} systems:")
    for al, d in systems:
        print(f"  AL={al}, d={d}, DeltaA={AR - al:.1f}")

    rows = []
    for AL, d in systems:
        print(f"\nSystem: AL={AL}, d={d}, DeltaA={AR-AL:.1f}")
        data = load_system(AL, d)
        x = data["x"]
        n = data["n"]
        v_xc = data["v_xc"]
        margin = max(3, len(x) // 20)
        sl = slice(margin, -margin)
        x_in, n_in, v_in = x[sl], n[sl], v_xc[sl]

        D1_full, D2_full, D3_full = build_descriptors(x, n)
        D1 = D1_full[sl]
        D2 = D2_full[sl]
        D3 = D3_full[sl]
        D_oracle = np.column_stack([n_in, x_in])

        mae_d1 = krr_loo_mae(D1, v_in)
        mae_d2 = krr_loo_mae(D2, v_in)
        mae_d3 = krr_loo_mae(D3, v_in)
        mae_oracle = krr_loo_mae(D_oracle, v_in)

        print(f"  D1 MAE:     {mae_d1:.4f}")
        print(f"  D2 MAE:     {mae_d2:.4f}")
        print(f"  D3 MAE:     {mae_d3:.4f}")
        print(f"  Oracle MAE: {mae_oracle:.4f}")

        rows.append({
            "AL": AL, "d": d, "DeltaA": AR - AL,
            "mae_d1": mae_d1,
            "mae_d2": mae_d2,
            "mae_d3": mae_d3,
            "mae_oracle": mae_oracle,
        })

    csv_path = RESULTS_DIR / "gradient_descriptors_mae.csv"
    cols = ["AL", "d", "DeltaA", "mae_d1", "mae_d2", "mae_d3", "mae_oracle"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
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

    # Aggregate stats
    d1 = np.array([r["mae_d1"] for r in rows])
    d2 = np.array([r["mae_d2"] for r in rows])
    d3 = np.array([r["mae_d3"] for r in rows])
    print(f"\nAggregate: mean D2/D1 = {np.mean(d2/d1):.3f}, "
          f"mean D3/D1 = {np.mean(d3/d1):.3f}")
    print(f"           median D2/D1 = {np.median(d2/d1):.3f}, "
          f"median D3/D1 = {np.median(d3/d1):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
