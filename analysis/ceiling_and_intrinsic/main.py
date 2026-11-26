"""Hand-picked panel figure: the cooperation ceiling under extrinsic dynamics.

This replaces the parameter-sweep figure for the Moran process and Fermi
imitation dynamics with curves drawn at a single, fully specified parameter set
(N = 8, a linear contribution profile, mutation 0.05). Each marker is the exact
steady-state cooperation p_C for one parameter combination, not an average over
the sweep.
"""

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

here = Path(__file__).resolve()
data_path = here.parents[2] / "data" / "sweep"
output_paths = [
    here.parents[2] / "tex" / "play_field.pdf",
    here.parent/"main.pdf"
]

number_of_players = 8
mutation = "0.05"

moran_colour = "#0072B2"
fermi_colour = "#AE4C02"
aspiration_colour = "#009E73"
introspection_colour = "#E69F00"
wf_colour = "#C06898"
baseline_colour = "#555555"
r_below_colour = "#0072B2"
r_above_colour = "#D55E00"
halo = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def load(dynamic):
    frame = pd.read_csv(data_path / f"{dynamic}_mu_eq_{mutation}/main.csv")
    return frame[frame["N"] == number_of_players].drop_duplicates("UID")


def nearest(values, target):
    unique_values = np.array(sorted(set(values)))
    return unique_values[np.argmin(np.abs(unique_values - target))]


band_mutations = ["0.001", "0.005", "0.05", "0.1"]


def sweep_curves_and_band(dynamic):
    """Return every distinct p_C-against-r curve and the min/max envelope over the
    whole N=8 sweep (every contribution, intensity, aspiration and mutation)."""
    columns = pd.read_csv(data_path / f"{dynamic}_mu_eq_0.05/main.csv", nrows=0).columns
    parameters = [c for c in ["M", "beta", "epsilon", "aspiration"] if c in columns]
    frames = []
    for index, mutation_value in enumerate(band_mutations):
        frame = pd.read_csv(
            data_path / f"{dynamic}_mu_eq_{mutation_value}/main.csv",
            usecols=["UID", "N", "r", "p_C"] + parameters,
        )
        frame = frame[frame["N"] == number_of_players].drop_duplicates("UID").copy()
        frame["__mutation__"] = index
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    envelope = combined.groupby("r")["p_C"].agg(["min", "max"]).sort_index()
    median = combined.groupby("r")["p_C"].median().sort_index()
    for intensity in ("epsilon", "beta"):
        if intensity in combined.columns:
            combined[f"__{intensity}_rank__"] = combined.groupby(
                ["M", "r", "__mutation__"]
            )[intensity].rank(method="dense")
    group_columns = [
        c
        for c in ["M", "aspiration", "__epsilon_rank__", "__beta_rank__"]
        if c in combined.columns
    ] + ["__mutation__"]
    curves = []
    seen = set()
    for _, group in combined.groupby(group_columns):
        group = group.sort_values("r")
        if len(group) < 2:
            continue
        returns = group["r"].to_numpy()
        cooperation = group["p_C"].to_numpy()
        signature = (tuple(np.round(returns, 3)), tuple(np.round(cooperation, 4)))
        if signature in seen:
            continue
        seen.add(signature)
        curves.append((returns, cooperation))
    return (
        curves,
        envelope.index.to_numpy(),
        envelope["min"].to_numpy(),
        envelope["max"].to_numpy(),
        median.index.to_numpy(),
        median.to_numpy(),
    )


def load_full(dynamic, mutation_values=band_mutations):
    """Every N=8 parameter set across the listed mutation rates."""
    columns = pd.read_csv(data_path / f"{dynamic}_mu_eq_0.05/main.csv", nrows=0).columns
    parameters = [
        c for c in ["M", "r", "beta", "epsilon", "aspiration"] if c in columns
    ]
    frames = []
    for index, mutation_value in enumerate(mutation_values):
        frame = pd.read_csv(
            data_path / f"{dynamic}_mu_eq_{mutation_value}/main.csv",
            usecols=["UID", "N", "p_C"] + parameters,
        )
        frame = frame[frame["N"] == number_of_players].drop_duplicates("UID").copy()
        frame["__mutation__"] = index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def load_all_N(dynamic, mutation_values=band_mutations):
    columns = pd.read_csv(data_path / f"{dynamic}_mu_eq_0.05/main.csv", nrows=0).columns
    parameters = [
        c for c in ["M", "r", "beta", "epsilon", "aspiration", "process"] if c in columns
    ]
    frames = []
    for index, mutation_value in enumerate(mutation_values):
        frame = pd.read_csv(
            data_path / f"{dynamic}_mu_eq_{mutation_value}/main.csv",
            usecols=["UID", "N", "p_C"] + parameters,
        )
        if dynamic in ("fermi", "introspection", "aspiration"):
            frame = frame[frame["beta"] != 0].drop_duplicates("UID").copy()
        else:
            frame = frame[frame["epsilon"] != 0].drop_duplicates("UID").copy()
        frame["__mutation__"] = index
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def sampled_curves(frame, x_column, normalise=False):
    """Every distinct p_C-against-x_column curve, one per setting of the rest."""
    group_columns = [
        c
        for c in ["M", "r", "beta", "epsilon", "aspiration", "__mutation__"]
        if c in frame.columns and c != x_column
    ]
    curves = []
    seen = set()
    for _, group in frame.groupby(group_columns, sort=False):
        group = group.sort_values(x_column)
        if len(group) < 2:
            continue
        x = group[x_column].to_numpy(dtype=float)
        y = group["p_C"].to_numpy()
        if normalise and x.max() > 0:
            x = x / x.max()
        signature = (tuple(np.round(x, 3)), tuple(np.round(y, 4)))
        if signature in seen:
            continue
        seen.add(signature)
        curves.append((x, y))
    return curves


def moran_against_contribution_curves(full_frame):
    """Every distinct p_C-against-M curve (strongest selection at each M)."""
    combinations = full_frame[["r", "__mutation__"]].drop_duplicates().to_numpy()
    curves = []
    seen = set()
    for return_value, mutation_index in combinations:
        subset = full_frame[
            (full_frame["r"] == return_value)
            & (full_frame["__mutation__"] == mutation_index)
        ]
        points = []
        for contribution in sorted(subset["M"].unique()):
            at_contribution = subset[subset["M"] == contribution]
            strongest = at_contribution["epsilon"].max()
            points.append(
                (
                    contribution,
                    at_contribution[at_contribution["epsilon"] == strongest][
                        "p_C"
                    ].iloc[0],
                )
            )
        if len(points) < 2:
            continue
        points.sort()
        contributions = np.array([p[0] for p in points])
        cooperation = np.array([p[1] for p in points])
        signature = tuple(np.round(cooperation, 4))
        if signature in seen:
            continue
        seen.add(signature)
        curves.append((contributions, cooperation))
    return curves


def median_curve(frame, x_column):
    """Median p_C over all other parameters, at each value of x_column."""
    grouped = frame.groupby(x_column)["p_C"].median().sort_index()
    return grouped.index.to_numpy(), grouped.to_numpy()


def rank_median(frame, sort_column, n_points=10, aspiration=False):
    """Median p_C at each normalised rank of sort_column, over all settings."""
    rows = []
    if aspiration is True:
        cols = ["M", "r", "__mutation__", "aspiration"]
    else:
        cols = ["M", "r", "__mutation__"]
    for _, group in frame.groupby(cols):
        group = group.sort_values(sort_column)
        if len(group) == n_points:
            rows.append(group["p_C"].to_numpy())
    return np.linspace(0.0, 1.0, n_points), np.median(np.array(rows), axis=0)


moran_full = load_full("moran")
fermi_full = load_full("fermi")
wf_full = load_full("wf")
introspection_full = load_full("introspection", mutation_values=["0.05"])
aspiration_full = load_full("aspiration")


fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
(ax_ceiling, ax_breaking), (ax_intensity, ax_N_resist) = axes


def add_baseline(ax, with_threshold=True):
    ax.axhline(0.5, color=baseline_colour, linestyle="-", linewidth=0.9)
    if with_threshold:
        ax.axvline(
            number_of_players, color=baseline_colour, linestyle=":", linewidth=0.9
        )
    ax.set_ylim(0.0, 1.0)


for dynamic, colour, marker, line_style, label in (
    ("moran", moran_colour, "o", "-", "Moran"),
    ("fermi", fermi_colour, "s", "-", "Fermi"),
    ("wf", wf_colour, "^", "-", "Wright-Fisher")
):
    curves, _, _, _, median_return, median_cooperation = sweep_curves_and_band(dynamic)
    for curve_return, curve_cooperation in curves:
        ax_ceiling.plot(
            curve_return,
            curve_cooperation,
            line_style,
            color=colour,
            alpha=0.12,
            linewidth=0.4,
            zorder=1,
        )
    ax_ceiling.plot(
        median_return,
        median_cooperation,
        line_style,
        marker=marker,
        color=colour,
        ms=4,
        label=label,
        zorder=3,
        path_effects=halo,
    )
add_baseline(ax_ceiling)
ax_ceiling.set_xlabel(r"return on investment $r$")
ax_ceiling.set_ylabel(r"cooperation $p_C$")
ax_ceiling.set_title(r"(a) The ceiling holds across the sweep")

for curve_x, curve_y in sampled_curves(
    moran_full[moran_full["r"] > number_of_players], "epsilon", normalise=True
):
    ax_intensity.plot(
        curve_x, curve_y, "-", color=moran_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for curve_x, curve_y in sampled_curves(
    fermi_full[fermi_full["r"] > number_of_players], "beta", normalise=True
):
    ax_intensity.plot(
        curve_x, curve_y, "-", color=fermi_colour, alpha=0.12, linewidth=0.5, zorder=1
    )

for curve_x, curve_y in sampled_curves(
    wf_full[wf_full["r"] > number_of_players], "epsilon", normalise=True
):
    ax_intensity.plot(
        curve_x, curve_y, "-", color=wf_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for curve_x, curve_y in sampled_curves(
    introspection_full[introspection_full["r"] > number_of_players], "beta", normalise=True
):
    ax_intensity.plot(
        curve_x, curve_y, "-", color=introspection_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for curve_x, curve_y in sampled_curves(
    aspiration_full[aspiration_full["r"] > number_of_players], "beta", normalise=True
):
    ax_intensity.plot(
        curve_x, curve_y, "-", color=aspiration_colour, alpha=0.12, linewidth=0.5, zorder=1
    )

moran_intensity_x, moran_intensity_y = rank_median(
    moran_full[moran_full["r"] > number_of_players], "epsilon"
)
ax_intensity.plot(
    moran_intensity_x,
    moran_intensity_y,
    "-o",
    color=moran_colour,
    label=r"Moran (selection $\varepsilon$)",
    ms=4,
    zorder=3,
    path_effects=halo,
)
fermi_intensity_x, fermi_intensity_y = rank_median(
    fermi_full[fermi_full["r"] > number_of_players], "beta"
)
ax_intensity.plot(
    fermi_intensity_x,
    fermi_intensity_y,
    "-s",
    color=fermi_colour,
    label=r"Fermi (choice $\beta$)",
    ms=4,
    zorder=3,
    path_effects=halo,
)
wf_intensity_x, wf_intensity_y = rank_median(
    wf_full[wf_full["r"] > number_of_players], "epsilon"
)
ax_intensity.plot(
    wf_intensity_x,
    wf_intensity_y,
    "-^",
    color=wf_colour,
    label=r"Wright-Fisher (selection $\varepsilon$)",
    ms=4,
    zorder=3,
    path_effects=halo,
)
aspiration_intensity_x, aspiration_intensity_y = rank_median(
    aspiration_full[aspiration_full["r"] > number_of_players], "beta", aspiration=True
)
ax_intensity.plot(
    aspiration_intensity_x,
    aspiration_intensity_y,
    "-D",
    color=aspiration_colour,
    label=r"aspiration (choice $\beta$)",
    ms=4,
    zorder=3,
    path_effects=halo,
)
introspection_intensity_x, introspection_intensity_y = rank_median(
    introspection_full[introspection_full["r"] > number_of_players], "beta"
)
ax_intensity.plot(
    introspection_intensity_x,
    introspection_intensity_y,
    "-p",
    color=introspection_colour,
    label=r"introspection (choice $\beta$)",
    ms=4,
    zorder=3,
    path_effects=halo,
)

add_baseline(ax_intensity, with_threshold=False)
ax_intensity.set_xlabel("intensity (normalised)")
ax_intensity.set_ylabel(r"cooperation $p_C$")
ax_intensity.set_title(r"(c) Intensities have opposite effects, $r > N$")

for dynamic, colour, marker, line_style, label in (
    ("introspection", introspection_colour, "p", "-", "introspection"),
    ("aspiration", aspiration_colour, "D", "-", "aspiration"),
):
    curves, _, _, _, median_return, median_cooperation = sweep_curves_and_band(dynamic)
    for curve_return, curve_cooperation in curves:
        ax_breaking.plot(
            curve_return, curve_cooperation, line_style, color=colour,
            alpha=0.12, linewidth=0.4, zorder=1,
        )
    ax_breaking.plot(
        median_return, median_cooperation, line_style, marker=marker,
        color=colour, ms=4, label=label, zorder=3, path_effects=halo,
    )
add_baseline(ax_breaking)
ax_breaking.set_xlabel(r"return on investment $r$")
ax_breaking.set_ylabel(r"cooperation $p_C$")
ax_breaking.set_title(r"(b) The ceiling is broken across the sweep")

handles1, labels1 = ax_ceiling.get_legend_handles_labels()
handles2, labels2 = ax_breaking.get_legend_handles_labels()
fig.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=5,
    title=r"Bold: median over sweep",
    title_fontsize=8,
)

for dynamic, colour, marker in (
    ("moran", moran_colour, "o"),
    ("fermi", fermi_colour, "s"),
    ("wf", wf_colour, "^"),
    ("introspection", introspection_colour, "p"),
    ("aspiration", aspiration_colour, "D")
):
    df = load_all_N(dynamic=dynamic)
    df_plot = (
    df.assign(r_category=np.where(df["r"] > df["N"], "r > N", "r < N"))
    .groupby(["N", "r_category"], as_index=False)["p_C"]
    .max()
    )
    sns.scatterplot(
        data=df_plot[df_plot["r_category"] == "r > N"],
        x="N",
        y="p_C",
        marker=marker,
        color=colour,
        ax=ax_N_resist,
    )
add_baseline(ax_N_resist, with_threshold=False)
ax_N_resist.set_ylabel(r"max $p_C$")
ax_N_resist.set_xlabel(r"$N$")
ax_N_resist.set_title(r"(d) The Ceiling Remains Across N, $r>N$")

fig.tight_layout(rect=[0, 0.08, 1, 1])
for output_path in output_paths:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        orientation="portrait",
        bbox_inches="tight",
    )
print("saved:", *[str(path) for path in output_paths], sep="\n  ")