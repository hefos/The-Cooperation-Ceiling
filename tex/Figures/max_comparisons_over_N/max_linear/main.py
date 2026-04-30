import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).resolve()
data_path = here.parents[3]

dataframe_paths = [
    data_path / "raw/linear_population_fermi/main.csv",
    data_path / "raw/linear_population_moran/main.csv",
    data_path / "raw/linear_population_imispection/main.csv",
    data_path / "raw/linear_population_introspection/main.csv",
]

small_frames = []

for df_path in dataframe_paths:
    df = pd.read_csv(df_path, usecols=["N", "p_C", "process"])

    df = df[df["N"].isin([3, 4, 5])]

    small_frames.append(df)

plot_df = pd.concat(small_frames, ignore_index=True)

fig, ax = plt.subplots()

sns.boxplot(data=plot_df, x="N", y="p_C", hue="process", width=0.5, ax=ax)

plt.xlabel("N")
plt.ylabel(f"$p_C$")

plt.tight_layout()
plt.savefig(here.parent / "main.pdf")
