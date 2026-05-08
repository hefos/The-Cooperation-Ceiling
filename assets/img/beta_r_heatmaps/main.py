import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd

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

dataframes = [
    dataframes_imispection,
    dataframes_fermi,
    dataframes_asp,
    dataframes_introspection,
]


def aggregate_dataframe(df):

    aggregated_df = df.groupby("UID").agg(
        p_C=("p_C", "first"),
        beta=("beta", "first"),
        r=("r", "first"),
        N=("N", "first"),
        process=("process", "first"),
    )

    return aggregated_df.reset_index()


for df_set in dataframes:
    df_total = dd.read_csv(df_set)
    df_total = aggregate_dataframe(df=df_total).compute()
    process = df_total["process"].iloc[0]
    for N, df in df_total.groupby("N"):
        if N == 1:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 1)

        heatmap_df = (
            df.groupby(["beta", "r"])["p_C"]
            .mean()
            .reset_index()
        )

        heatmap_matrix = heatmap_df.pivot(
            index="beta",
            columns="r",
            values="p_C"
        )

        im = ax.imshow(
            heatmap_matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(heatmap_matrix.columns)))
        ax.set_yticks(range(len(heatmap_matrix.index)))

        ax.set_xticklabels([round(v, 2) for v in heatmap_matrix.columns])
        ax.set_yticklabels([round(v, 2) for v in heatmap_matrix.index])

        cbar = plt.colorbar(im)
        cbar.set_label(r"mean $p_C$")

        ax.set_title(f"{process}_N_eq_{N}")
        folder = Path(here.parent / f"{process}_N_eq_{N}")
        folder.mkdir(exist_ok=True)

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\beta$")

        plt.savefig(here.parent / f"{process}_N_eq_{N}/main.pdf")
        plt.close()
