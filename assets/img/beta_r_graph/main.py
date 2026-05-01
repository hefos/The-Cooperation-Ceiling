import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

here = pathlib.Path(__file__).resolve()
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

dataframes = [dataframes_fermi, dataframes_imispection, dataframes_introspection, dataframes_asp]

process_order = ["Fermi", "Introspective Imitation", "Introspection", "Aspiration"]

proc_to_col = {p: i for i, p in enumerate(process_order)}


N_BINS = 5
param_name = "beta"

fig, axis = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(15, 6),
)

axarray = axis.flatten()


for df_path in dataframes:
    df_plot = pd.concat([pd.read_csv(csv_path) for csv_path in df_path], ignore_index=True).drop_duplicates("UID", keep="first")
    df_plot = df_plot[df_plot["N"] == 8]

    param_name_formatted = param_name[0].upper() + param_name[1:].replace("_", " ")

    r_quantiles = pd.qcut(df_plot["r"], q=N_BINS)
    r_labels = ["r: %.1f-%.1f" % (b.left, b.right) for b in r_quantiles.cat.categories]

    df_plot["r range"] = pd.qcut(
        df_plot["r"], q=N_BINS, labels=r_labels, duplicates="drop"
    )
    process = df_plot["process"].iloc[0]
    if process == "fermi":
        process = "Fermi"
    if process == "imispection":
        process = "Introspective Imitation"
    if process == "aspiration":
        process = "Aspiration"
    if process == "introspection":
        process = "Introspection"
    col = proc_to_col[process]

    ax = axarray[col]
    ax.set_xlabel(r"$\beta$")

    ax.set_title(process)
    ax.set_ylabel(r"mean($p_C$)")

    sns.lineplot(
        data=df_plot,
        x=param_name,
        y="p_C",
        hue="r range",
        estimator="mean",
        errorbar=None,
        ax=ax,
        legend=True,
    )

    plt.subplots_adjust(right=0.82, top=0.88, hspace=0.3, wspace=0.2)

plt.legend(bbox_to_anchor=(1.05,0.5))
plt.savefig(here.parent / "main.pdf")
