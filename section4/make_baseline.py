from pathlib import Path

from utils.constants import a_r, d_default
from utils.solve import solve_exact

# Solve the representative AL=1.0, AR=2.0, d=10 system that fig_5_1,
# fig_5_2, fig_5_3a/b, fig_5_5, fig_5_6, and fig_5_sigma_spatial all
# consume. Cached in data/ -- only solved once. The four solve_*.py
# scripts populate the rest of the 19-system asymmetric grid.

here = Path(__file__).parent
data_dir = here / "data"
data_dir.mkdir(parents=True, exist_ok=True)

a_left = 1.0


def main():
    out_path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d_default}.npz"
    solve_exact(a_left, a_r, d_default, out_path)


if __name__ == "__main__":
    main()
    print("Done.")
