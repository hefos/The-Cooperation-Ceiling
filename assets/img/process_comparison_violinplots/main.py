import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import argparse

here = Path(__file__).resolve()
assets_path = here.parents[2]

parser = argparse.ArgumentParser(description="Processes to compare")

parser.add_argument("--p1", type=str, required=True, help="Process 1")
parser.add_argument("--p2", type=str, required=True, help="Process 2")

args = parser.parse_args()

p1 = args.p1
p2 = args.p2

try:
    df = pd.read_csv(
        assets_path / f"data/processed/comparison_tables/{p1}_vs_{p2}/main.csv"
    )
except FileNotFoundError:
    df = pd.read_csv(
        assets_path / f"data/processed/comparison_tables/{p2}_vs_{p1}/main.csv"
    )
    p3 = p1
    p1 = p2
    p2 = p3

N = 8
N_frame = df[df["N"] == N]
fig, ax = plt.subplots()
N_frame = N_frame[N_frame["winner"] != "draw"]

N_frame["p_C_difference"] = N_frame[f"p_C_in_{p1}"] - N_frame[f"p_C_in_{p2}"]
N_frame["r"] = N_frame["r"].round(3)

sns.violinplot(data=N_frame, x="r", y="p_C_difference", ax=ax, native_scale=True)

ax.set_ylabel(r"Difference in $p_C$")
folder = Path(here.parent / f"{p1}_vs_{p2}_n_eq_{N}")
folder.mkdir(exist_ok=True)

xmin, xmax = ax.get_xlim()
ax.set_xticks([0.5, N, xmax])
ax.set_xticklabels(["0.5", f"{N}", f"{xmax:.2f}"])

plt.tight_layout()
plt.savefig(here.parent / f"{p1}_vs_{p2}_n_eq_{N}/main.pdf")
plt.close()
