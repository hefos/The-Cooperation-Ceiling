import matplotlib.pyplot as plt
import dask.dataframe as dd
import seaborn as sns
from pathlib import Path

here = Path(__file__).resolve()
data_path = here.parents[2] / "data"

dataframes_imispection = [
    data_path / "raw/imispection_mu_eq_0.1/main.csv",
    data_path / "raw/imispection_mu_eq_0.001/main.csv",
    data_path / "raw/imispection_mu_eq_0.005/main.csv",
    data_path / "raw/imispection_mu_eq_0.05/main.csv",
]

df = dd.read_csv(dataframes_imispection)
df = df[df["N"] == 8]

df_lt = df[df["r"] < 8][["epsilon", "beta", "p_C"]].compute()
df_gt = df[df["r"] > 8][["epsilon", "beta", "p_C"]].compute()

df_lt["beta"] = df_lt["beta"].round(2)
df_lt["epsilon"] = df_lt["epsilon"].round(2)

df_gt["beta"] = df_gt["beta"].round(2)
df_gt["epsilon"] = df_gt["epsilon"].round(2)

df_lt = df_lt.groupby(["beta", "epsilon"])["p_C"].mean().reset_index()
df_gt = df_gt.groupby(["beta", "epsilon"])["p_C"].mean().reset_index()

lt_plot = df_lt.pivot(index="beta", columns="epsilon", values="p_C")
gt_plot = df_gt.pivot(index="beta", columns="epsilon", values="p_C")

lt_plot = lt_plot.sort_index(ascending=False).sort_index(axis=1)
gt_plot = gt_plot.sort_index(ascending=False).sort_index(axis=1)

vmin = min(lt_plot.min().min(), gt_plot.min().min())
vmax = max(lt_plot.max().max(), gt_plot.max().max())

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.heatmap(
    lt_plot,
    cmap="coolwarm",
    ax=ax,
    vmin=vmin,
    vmax=vmax,
    cbar=True,
)
ax.set_title(r"$r < N$")
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$\beta$")

folder = Path(here.parent / "imispection_r_lt")
folder.mkdir(exist_ok=True)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(here.parent / "imispection_r_lt/main.pdf")
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

sns.heatmap(
    gt_plot,
    cmap="coolwarm",
    ax=ax,
    vmin=vmin,
    vmax=vmax,
    cbar=True
)
ax.set_title(r"$r > N$")
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$\beta$")

folder = Path(here.parent / "imispection_r_gt")
folder.mkdir(exist_ok=True)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(here.parent / "imispection_r_gt/main.pdf")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(here.parent / "main.pdf")
