import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).resolve()
data_path = here.parents[3]

dataframes_asp = [
    data_path / "raw/aspiration_mu_eq_0.1/main.csv",
    data_path / "raw/aspiration_mu_eq_0.001/main.csv",
    data_path / "raw/aspiration_mu_eq_0.005/main.csv",
    data_path / "raw/aspiration_mu_eq_0.05/main.csv",
]

dataframes_fermi = [
    data_path / "raw/fermi_mu_eq_0.1/main.csv",
    data_path / "raw/fermi_mu_eq_0.001/main.csv",
    data_path / "raw/fermi_mu_eq_0.005/main.csv",
    data_path / "raw/fermi_mu_eq_0.05/main.csv",
]

dataframes_imispection = [
    data_path / "raw/imispection_mu_eq_0.1/main.csv",
    data_path / "raw/imispection_mu_eq_0.001/main.csv",
    data_path / "raw/imispection_mu_eq_0.005/main.csv",
    data_path / "raw/imispection_mu_eq_0.05/main.csv",
]

dataframes_introspection = [
    data_path / "raw/introspection_mu_eq_0.1/main.csv",
    data_path / "raw/introspection_mu_eq_0.001/main.csv",
    data_path / "raw/introspection_mu_eq_0.005/main.csv",
    data_path / "raw/introspection_mu_eq_0.05/main.csv",
]

dataframes_moran = [
    data_path / "raw/moran_mu_eq_0.1/main.csv",
    data_path / "raw/moran_mu_eq_0.001/main.csv",
    data_path / "raw/moran_mu_eq_0.005/main.csv",
    data_path / "raw/moran_mu_eq_0.05/main.csv",
]

dataframes = [
    dataframes_imispection,
    dataframes_fermi,
    dataframes_asp,
    dataframes_introspection,
    dataframes_moran
]

small_frames = []

for df_path in dataframes:
    df = pd.concat([pd.read_csv(path, usecols=["N", "p_C", "process"]) for path in df_path])

    df = df[df["N"].isin([2, 3, 4, 5, 6, 7, 8])]

    small_frames.append(df)

plot_df = pd.concat(small_frames, ignore_index=True)

fig, ax = plt.subplots()

sns.boxplot(data=plot_df, x="N", y="p_C", hue="process", width=0.5, ax=ax)

plt.xlabel("N")
plt.ylabel("$p_C$")

plt.tight_layout()
plt.savefig(here.parent / "main.pdf")
