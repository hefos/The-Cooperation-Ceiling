"""The cooperation ceiling for large populations.

Reads the resumable simulation sweep produced by ``data/large_n/main.py`` and
draws a four-panel figure: the ceiling at the largest N, its persistence in N,
the introspection threshold across N (one curve per population size, each with
its own choice intensity so the curves do not collapse), and a validation of the
simulator against the exact steady state at small N.
"""

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

here = pathlib.Path(__file__).resolve()
data_path = here.parents[2] / "data" / "large_n"
figure_outputs = [
    here.parents[2] / "tex" / "ceiling_large_n.pdf",
]

extrinsic = ["moran", "fermi"]
intrinsic = ["introspection", "aspiration"]
colours = {
    "moran": "#0072B2",
    "fermi": "#D55E00",
    "introspection": "#E69F00",
    "aspiration": "#009E73",
}
styles = {
    "moran": ("-", "o"),
    "fermi": ("--", "s"),
    "introspection": ("-.", "^"),
    "aspiration": (":", "D"),
}
labels = {
    "moran": "Moran",
    "fermi": "Fermi",
    "introspection": "introspection",
    "aspiration": "aspiration",
}
baseline_colour = "#555555"
halo = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
collapse_colours = {10: "#56B4E9", 50: "#0072B2", 100: "#CC79A7", 200: "#E69F00"}
collapse_markers = {10: "o", 50: "s", 100: "^", 200: "D"}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

simulations = pd.read_csv(data_path / "simulations.csv")
exact = (
    pd.read_csv(data_path / "exact.csv")
    if (data_path / "exact.csv").exists()
    else pd.DataFrame(columns=["dynamic", "N", "r_over_N", "p_C"])
)

aggregated = (
    simulations.groupby(["dynamic", "N", "r_index", "r_over_N"])["p_C"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
aggregated["sem"] = aggregated["std"].fillna(0.0) / np.sqrt(aggregated["count"])

collapse = (
    pd.read_csv(data_path / "collapse.csv")
    if (data_path / "collapse.csv").exists()
    else pd.DataFrame(columns=["N", "beta", "r_over_N", "p_C"])
)
collapse_aggregated = (
    collapse.groupby(["N", "beta", "r_over_N"])["p_C"].mean().reset_index()
)


def add_baseline(ax, threshold=True):
    ax.axhline(0.5, color=baseline_colour, linestyle="--", linewidth=0.9, zorder=0)
    if threshold:
        ax.axvline(1.0, color=baseline_colour, linestyle=":", linewidth=0.9, zorder=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"cooperation $p_C$")


fig, ((ax_ceiling, ax_scaling), (ax_collapse, ax_validation)) = plt.subplots(
    2, 2, figsize=(8.6, 7.0)
)

largest = aggregated["N"].max()
for dynamic in extrinsic + intrinsic:
    rows = aggregated[
        (aggregated["dynamic"] == dynamic) & (aggregated["N"] == largest)
    ].sort_values("r_over_N")
    if rows.empty:
        continue
    line_style, marker = styles[dynamic]
    ax_ceiling.fill_between(
        rows["r_over_N"], rows["mean"] - 1.96 * rows["sem"],
        rows["mean"] + 1.96 * rows["sem"], color=colours[dynamic], alpha=0.15, zorder=1,
    )
    ax_ceiling.plot(
        rows["r_over_N"], rows["mean"], line_style, marker=marker, ms=4,
        color=colours[dynamic], label=labels[dynamic], zorder=3, path_effects=halo,
    )
add_baseline(ax_ceiling)
ax_ceiling.set_xlabel(r"return per player $r / N$")
ax_ceiling.set_title(rf"(a) the ceiling holds at $N = {int(largest)}$")
ax_ceiling.legend(loc="upper left")

high_return = aggregated["r_over_N"].max()
for dynamic in extrinsic + intrinsic:
    rows = aggregated[
        (aggregated["dynamic"] == dynamic)
        & np.isclose(aggregated["r_over_N"], high_return)
    ].sort_values("N")
    if rows.empty:
        continue
    line_style, marker = styles[dynamic]
    ax_scaling.plot(
        rows["N"], rows["mean"], line_style, marker=marker, ms=4,
        color=colours[dynamic], label=labels[dynamic], zorder=3, path_effects=halo,
    )
ax_scaling.axhline(0.5, color=baseline_colour, linestyle="--", linewidth=0.9, zorder=0)
ax_scaling.set_ylim(0.0, 1.0)
ax_scaling.set_ylabel(rf"$p_C$ at $r / N = {high_return:g}$")
ax_scaling.set_xlabel(r"number of players $N$")
ax_scaling.set_title(r"(b) the ceiling persists as $N$ grows")
ax_scaling.legend(loc="upper left")

for population_size in sorted(collapse_colours):
    rows = collapse_aggregated[collapse_aggregated["N"] == population_size].sort_values(
        "r_over_N"
    )
    if rows.empty:
        continue
    beta = rows["beta"].iloc[0]
    ax_collapse.plot(
        rows["r_over_N"], rows["p_C"], "-", marker=collapse_markers[population_size],
        ms=4, color=collapse_colours[population_size],
        label=rf"$N = {population_size}$, $\beta = {beta:g}$",
        zorder=3, path_effects=halo,
    )
add_baseline(ax_collapse)
ax_collapse.set_xlabel(r"return per player $r / N$")
ax_collapse.set_title(r"(c) introspection: threshold stays at $r = N$")
ax_collapse.legend(loc="upper left")

validation_dynamics = ["moran", "fermi", "introspection", "aspiration"]
offsets = {"moran": -0.27, "fermi": -0.09, "introspection": 0.09, "aspiration": 0.27}
small_sizes = sorted(
    simulations[simulations["N"] <= 8]["N"].unique()
)
for dynamic in validation_dynamics:
    positions, samples = [], []
    for population_size in small_sizes:
        values = simulations[
            (simulations["dynamic"] == dynamic) & (simulations["N"] == population_size)
        ]["p_C"].to_numpy()
        if values.size == 0:
            continue
        positions.append(population_size + offsets[dynamic])
        samples.append(values)
    if not samples:
        continue
    box = ax_validation.boxplot(
        samples, positions=positions, widths=0.15, patch_artist=True, manage_ticks=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(colours[dynamic])
        patch.set_alpha(0.45)
    for element in ("medians", "whiskers", "caps"):
        for line in box[element]:
            line.set_color(colours[dynamic])
    exact_rows = exact[exact["dynamic"] == dynamic].sort_values("N")
    ax_validation.scatter(
        exact_rows["N"] + offsets[dynamic], exact_rows["p_C"],
        marker=styles[dynamic][1], s=22,
        color=colours[dynamic], edgecolor="black", linewidth=0.6, zorder=5,
        label=f"{labels[dynamic]} (exact)",
    )
ax_validation.axhline(0.5, color=baseline_colour, linestyle="--", linewidth=0.9, zorder=0)
ax_validation.set_ylim(0.0, 1.0)
if small_sizes:
    ax_validation.set_xticks(list(small_sizes))
ax_validation.set_xlabel(r"number of players $N$")
ax_validation.set_ylabel(r"cooperation $p_C$")
ax_validation.set_title(r"(d) simulation matches the exact result, $r / N = 1.2$")
ax_validation.legend(loc="upper left")

fig.tight_layout()
for output_path in figure_outputs:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
print("saved figure:", *[str(p) for p in figure_outputs], sep="\n  ")
