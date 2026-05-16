import matplotlib.pyplot as plt
import dask.dataframe as dd

from pathlib import Path

here = Path(__file__).resolve()
data_path = here.parents[2] / "data"

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


def aggregate_dataframe(df):

    aggregated_df = df.groupby("UID").agg(
        p_C=("p_C", "first"),
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

        groups = []
        r_values = []

        for r, g in df.groupby("r", sort=True):
            vals = g["p_C"].dropna().to_numpy()

            if vals.size > 0:
                groups.append(vals)
                r_values.append(round(r, 2))

        ax.violinplot(groups, positions=r_values, showmeans=True)

        ax.set_xticks(r_values)
        ax.set_xticklabels(r_values)

        ax.set_title(fr"{process} $r$ against $p_C$")
        folder = Path(here.parent / f"{process}_N_eq_{N}")
        folder.mkdir(exist_ok=True)

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$p_C$")

        plt.savefig(here.parent / f"{process}_N_eq_{N}/main.pdf")
        plt.close()
