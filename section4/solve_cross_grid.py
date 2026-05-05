from pathlib import Path

from utils.constants import a_r
from utils.solve import solve_exact

# Cross-grid: AL in {1.2, 1.8} crossed with d in {4, 6, 8}. Bridges the
# asymmetry and distance scans so fig_5_5 has 19 systems with both
# parameters varying.

here = Path(__file__).parent
data_dir = here / "data"
data_dir.mkdir(parents=True, exist_ok=True)

cross_grid = [(a_l, d) for a_l in (1.2, 1.8) for d in (4.0, 6.0, 8.0)]


def main():
    for a_left, d in cross_grid:
        out_path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        solve_exact(a_left, a_r, d, out_path)


if __name__ == "__main__":
    main()
    print("Done.")
