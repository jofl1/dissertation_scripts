from pathlib import Path

from utils.constants import a_r
from utils.solve import solve_exact

# Distance scan at DeltaA=0.5: AL=1.5, AR=2.0, d in {4, 6, 8, 12}. d=10
# already comes from solve_asymmetry_scan_d10.py.

here = Path(__file__).parent
data_dir = here / "data"
data_dir.mkdir(parents=True, exist_ok=True)

a_left = 1.5
ds_to_solve = [4.0, 6.0, 8.0, 12.0]


def main():
    for d in ds_to_solve:
        out_path = data_dir / f"exact_asymmetric_uu_AL{a_left}_AR{a_r}_d{d}.npz"
        solve_exact(a_left, a_r, d, out_path)


if __name__ == "__main__":
    main()
    print("Done.")
