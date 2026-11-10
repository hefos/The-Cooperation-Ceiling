import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

here = pathlib.Path(__file__).resolve()

data_path = pathlib.Path(here.parents[2]/"data")
halo = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]

aspiration_df = pd.read_csv(data_path/"subgroups/aspiration/main.csv")
moran_df = pd.read_csv(data_path/"subgroups/moran/main.csv")
moran_df["process"] = moran_df["process"].replace("moran", "Moran")
fermi_df = pd.read_csv(data_path/"subgroups/fermi/main.csv")
fermi_df["process"] = fermi_df["process"].replace("fermi", "Fermi")
introspection_df = pd.read_csv(data_path/"subgroups/introspection/main.csv")
fitness_df = pd.read_csv(data_path/"subgroup_fitness/main.csv")

def add_stats(df):
    df["statistics"] = df.groupby(
        ["N", "M", "alpha_i", "i", "beta", "epsilon", "aspiration"]
    ).ngroup()
    df["r lever"] = np.select(
    [df["r"] > df["d"], df["r"] < df["d"]],
    ["r > d", "r < d"],
    default="r = d"
    )
    df["r/d"] = df["r"]/df["d"]
    df["aspiration ratio"] = df["aspiration"] / df["M"]
    return df

def add_stats_asp_plot(df):
    df["statistics"] = df.groupby(
        ["N", "M", "alpha_i", "i", "beta", "epsilon"]
    ).ngroup()
    df["r lever"] = np.select(
    [df["r"] > df["d"], df["r"] < df["d"]],
    ["r > d", "r < d"],
    default="r = d"
    )
    df["r/d"] = df["r"]/df["d"]
    df["aspiration ratio"] = df["aspiration"] / df["M"]
    return df

moran_colour = "#0072B2"
fermi_colour = "#D55E00"
baseline_colour = "#555555"
aspiration_colour = "#009E73"
introspection_colour = "#E69F00"
r_below_colour = "#0072B2"
r_above_colour = "#D55E00"
fixed_d = 3

fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
(ax_ceiling, ax_intrinsic), (ax_group_size, ax_payoff) = axes

for df, colour, marker, line_style, label in (
    (fermi_df, fermi_colour, "s", "--", "Fermi"),
    (moran_df, moran_colour, "o", "-", "Moran"),
):
    df = add_stats(df[df["d"] == fixed_d])

    sns.lineplot(
        data=df,
        x="r",
        y="p_C",
        units="statistics",
        estimator=None,
        alpha=0.12,
        linewidth=0.4,
        linestyle=line_style,
        legend=False,
        ax=ax_ceiling,
        color=colour,
        zorder=1,
    )

    sns.lineplot(
        data=df,
        x="r",
        y="p_C",
        estimator="median",
        errorbar=None,
        linewidth=1.5,
        ax=ax_ceiling,
        color=colour,
        label=label,
        marker=marker,
        markersize=4,
        markerfacecolor=colour,
        markeredgecolor=colour,
        markeredgewidth=0.8,
        linestyle=line_style,
        zorder=3,
        path_effects=halo,
    )

ax_ceiling.set_xlabel("r")
ax_ceiling.set_ylabel(r"$p_C$")
ax_ceiling.set_title("A) Small groups break the ceiling")
ax_ceiling.legend()

for df, colour, marker, line_style, label in (
    (introspection_df, introspection_colour, "o", "-", "introspection"),
    (aspiration_df, aspiration_colour, "^", "--", "aspiration"),
):
    df = add_stats(df[df["d"] == fixed_d])

    sns.lineplot(
        data=df,
        x="r",
        y="p_C",
        units="statistics",
        estimator=None,
        alpha=0.12,
        linewidth=0.4,
        linestyle=line_style,
        legend=False,
        ax=ax_intrinsic,
        color=colour,
        zorder=1,
    )

    sns.lineplot(
        data=df,
        x="r",
        y="p_C",
        estimator="median",
        errorbar=None,
        linewidth=1.5,
        ax=ax_intrinsic,
        color=colour,
        label=label,
        marker=marker,
        markersize=4,
        markerfacecolor=colour,
        markeredgecolor=colour,
        markeredgewidth=0.8,
        linestyle=line_style,
        zorder=3,
        path_effects=halo,
    )

ax_intrinsic.set_xlabel("r")
ax_intrinsic.set_ylabel(r"$p_C$")
ax_intrinsic.set_title(r"B) The intrinsic crossing is at $r=d$")
ax_intrinsic.axvline(x=fixed_d, color=baseline_colour, linestyle='dashed')
ax_intrinsic.legend()


df = pd.concat([
    introspection_df, aspiration_df, moran_df, fermi_df
])

ax = sns.boxplot(
    data=df,
    x="process",
    y="p_C",
    hue="d",
    palette="tab10",
    width=0.8,
    dodge=True,
    ax=ax_group_size,
    showfliers=False,
    legend=False
)

ax_group_size.set_xlabel("Dynamic")
ax_group_size.set_ylabel(r"$p_C$")
ax_group_size.tick_params(axis="x", labelrotation=-30)
ax_group_size.set_title("C) Increasing group size reduces cooperation")

legend = ax_group_size.legend(
    title="Left to Right:\nAscending d values",
    loc="upper right",
    frameon=True,
    fontsize=8,
    title_fontsize=9
)
legend.get_title().set_ha("center")


pivot_df = fitness_df.pivot_table(
    index=["r", "type"],
    columns="number of cooperators",
    values="payoff",
    aggfunc="mean"
).fillna(0)

r_values = sorted(fitness_df["r"].unique())
types = fitness_df["type"].unique()
cooperators = pivot_df.columns

x = np.arange(len(r_values))
width = 0.35

colours = sns.color_palette("viridis", n_colors=len(cooperators))

for i, t in enumerate(types):
    bottom = np.zeros(len(r_values))

    for j, coop in enumerate(cooperators):
        values = [
            pivot_df.loc[(r, t), coop]
            if (r, t) in pivot_df.index else 0
            for r in r_values
        ]

        ax_payoff.bar(
            x + (i - 0.5) * width,
            values,
            width,
            bottom=bottom,
            color=colours[j],
            label=f"{coop} cooperators" if i == 0 else None
        )

        bottom += np.array(values)
legend = ax_payoff.legend(
    title="Number of cooperators",
    loc="upper right",
    fontsize=8,
    title_fontsize=9
)

legend.get_title().set_ha("center")

ax_payoff.text(
    0.35, 0.95,
    "Left: D | Right: C",
    transform=ax_payoff.transAxes,
    ha="right",
    va="top",
    fontsize=8
)

ax_payoff.set_xticks(x)
ax_payoff.set_xticklabels(r_values)
ax_payoff.set_xlabel("r")

plt.tight_layout()
plt.savefig(here.parent/"main.pdf")