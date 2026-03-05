import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np

here = Path(__file__).resolve()
assets_path = here.parents[3]
df = pd.read_csv(
    assets_path
    / "data/processed/comparison_tables/linear/introspection_against_moran/main.csv"
)


for N, N_frame in df.groupby("N"):
    fig, ax = plt.subplots()
    N_frame = N_frame[N_frame["winner"] != "draw"]

    N_frame["p_C_difference"] = (
        N_frame["p_C_in_introspection"] - N_frame["p_C_in_moran"]
    )
    N_frame["r"] = N_frame["r"].round(3)

    sns.violinplot(data=N_frame, x="r", y="p_C_difference", ax=ax, native_scale=True)

    ax.set_ylabel(r"$p_C(\text{Introspection}) - p_C(\text{Moran Process})$")
    folder = Path(here.parent / f"N_eq_{N}")
    folder.mkdir(exist_ok=True)

    xmin, xmax = ax.get_xlim()

    ax.set_xticks([0.5, N, xmax])
    ax.set_xticklabels(["0.5", f"{N}", f"{xmax:.2f}"])

    plt.tight_layout()
    plt.savefig(here.parent / f"N_eq_{N}/main.pdf")
    plt.close()
