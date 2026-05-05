from pathlib import Path

from utils.constants import a_r, d_default
from utils.solve import solve_exact

# Asymmetry scan at d=10: AL = 1.2, 1.5, 1.8, 2.0. AL=1.0 is solved by
# make_baseline.py. Together these supply the d=10 column of the 19-system
# grid that fig_5_3a/b and fig_5_5 consume.

here = Path(__file__).parent
data_dir = here / "data"
data_dir.mkdir(parents=True, exist_ok=True)

als_to_solve = [1.2, 1.5, 1.8, 2.0]


def main():
    for a_left in als_to_solve:
        out_path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d_default}.npz"
        solve_exact(a_left, a_r, d_default, out_path)


if __name__ == "__main__":
    main()
    print("Done.")
