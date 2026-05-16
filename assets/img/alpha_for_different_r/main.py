import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
from pathlib import Path

here = Path(__file__).resolve()
data_path = here.parents[2] / "data"


dataframes_asp = [
    data_path / "raw/aspiration_mu_eq_0.0/main.csv",
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
    data_path / "raw/introspection_mu_eq_0.0/main.csv",
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


def aggregate_dataframe(df):

    aggregated_df = df.groupby("UID").agg(
        p_C=("p_C", "first"),
        alpha_min=("alpha_i", "min"),
        alpha_max=("alpha_i", "max"),
        N=("N", "first"),
        r=("r", "first"),
        process=("process", "first"),
    )

    aggregated_df["r_band"] = (aggregated_df["r"] > aggregated_df["N"]).astype(int).map({1:"gt", 0:"lt"}, meta=("r_band", "object"))

    aggregated_df["alpha_range"] = (
        aggregated_df["alpha_max"] - aggregated_df["alpha_min"]
    )

    return aggregated_df.reset_index()


for df_set in dataframes:
    df_total = dd.read_csv(df_set)
    df_total = aggregate_dataframe(df=df_total).compute()
    process = df_total["process"].iloc[0]
    for N, df in df_total.groupby("N"):
        if N == 1:
            continue
        for r_band, r_band_frame in df.groupby("r_band"):
            r_band_frame["alpha_range_bin"] = pd.qcut(
                r_band_frame["alpha_range"], q=10, precision=3, duplicates="drop"
            )

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_ylim(0, 1)

            groups = []
            

            for _, g in r_band_frame.groupby("alpha_range_bin", sort=True):
                vals = g["p_C"].dropna().to_numpy()

                if vals.size > 0:
                    groups.append(vals)

            # HARD SAFETY CHECK
            ax.violinplot(groups, showmeans=True)
            if r_band == "gt":
                ax.set_title(fr"{process}, range($\alpha$) against $p_C$, r > N")
            else:
                ax.set_title(fr"{process}, range($\alpha$) against $p_C$, r < N")
            folder = Path(here.parent / f"{process}_N_eq_{N}_r_{r_band}")
            folder.mkdir(exist_ok=True)

            ax.set_xlabel(r"decile of range($\alpha$)")
            ax.set_ylabel(r"$p_C$")

            plt.savefig(here.parent / f"{process}_N_eq_{N}_r_{r_band}/main.pdf")
            plt.close()
