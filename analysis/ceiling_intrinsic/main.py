"""Hand-picked panel figure: intrinsic dynamics exceed the cooperation ceiling.

This replaces the parameter-sweep figure for aspiration and introspection
dynamics with curves drawn at a single, fully specified parameter set
(N = 8, a linear contribution profile, mutation 0.05, choice intensity 2.0).
Each marker is the exact steady-state cooperation p_C for one parameter
combination, not an average over the sweep.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

here = Path(__file__).resolve()
data_path = here.parents[2] / "data" / "raw"
output_paths = [
    here.parent / "main.pdf",
    here.parents[3] / "tex" / "figures" / "ceiling_intrinsic" / "main.pdf",
]

number_of_players = 8
mutation = "0.05"
reference_contribution = 16.0  # M = 2N, the reference contribution scale
choice_intensity = 2.0  # beta
return_below_threshold = 3.0555555555555554  # an r < N slice
return_above_threshold = 9.444444444444445  # an r > N slice

aspiration_colour = "#009E73"
introspection_colour = "#E69F00"
baseline_colour = "#555555"
r_below_colour = "#0072B2"  # r < N
r_above_colour = "#D55E00"  # r > N
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
    columns = pd.read_csv(
        data_path / f"{dynamic}_mu_eq_0.05/main.csv", nrows=0
    ).columns
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
    curves = []
    seen = set()
    for _, group in combined.groupby(parameters + ["__mutation__"]):
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
    columns = pd.read_csv(
        data_path / f"{dynamic}_mu_eq_0.05/main.csv", nrows=0
    ).columns
    parameters = [c for c in ["M", "r", "beta", "epsilon", "aspiration"] if c in columns]
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


def sampled_curves(frame, x_column):
    """Every distinct p_C-against-x_column curve, one per setting of the rest."""
    group_columns = [
        c for c in ["M", "r", "beta", "epsilon", "aspiration", "__mutation__"]
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
        signature = (tuple(np.round(x, 3)), tuple(np.round(y, 4)))
        if signature in seen:
            continue
        seen.add(signature)
        curves.append((x, y))
    return curves


def median_curve(frame, x_column):
    """Median p_C over all other parameters, at each value of x_column."""
    grouped = frame.groupby(x_column)["p_C"].median().sort_index()
    return grouped.index.to_numpy(), grouped.to_numpy()


aspiration = load("aspiration")
introspection = load("introspection")
# Introspection p_C is independent of the mutation rate, so one rate suffices.
introspection_full = load_full("introspection", mutation_values=["0.05"])
aspiration_full = load_full("aspiration")
aspiration_level = nearest(
    aspiration[np.isclose(aspiration["M"], reference_contribution)]["aspiration"],
    reference_contribution / 2,
)


def introspection_against_return(contribution, intensity):
    rows = introspection[
        np.isclose(introspection["M"], contribution)
        & np.isclose(introspection["beta"], intensity)
    ].sort_values("r")
    return rows["r"].to_numpy(), rows["p_C"].to_numpy()


def aspiration_against_return(contribution, intensity, level):
    rows = aspiration[
        np.isclose(aspiration["M"], contribution)
        & np.isclose(aspiration["beta"], intensity)
        & np.isclose(aspiration["aspiration"], level)
    ].sort_values("r")
    return rows["r"].to_numpy(), rows["p_C"].to_numpy()


fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
(ax_breaking, ax_mechanism), (ax_heterogeneity, ax_aspiration_lever) = axes


def add_baseline(ax, with_threshold=True):
    ax.axhline(0.5, color=baseline_colour, linestyle="--", linewidth=0.9)
    if with_threshold:
        ax.axvline(
            number_of_players, color=baseline_colour, linestyle=":", linewidth=0.9
        )
    ax.set_ylim(0.0, 1.0)


# Panel A: the ceiling is broken across the whole sweep, p_C against r.
for dynamic, colour, marker, line_style, label in (
    ("introspection", introspection_colour, "o", "-", "introspection"),
    ("aspiration", aspiration_colour, "^", "--", "aspiration"),
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
ax_breaking.set_title(r"(a) the ceiling is broken across the sweep")
ax_breaking.legend(
    loc="upper left", title="bold: median over sweep", title_fontsize=8
)

# Panel B: the Delta-pi mechanism, introspection at several contribution scales.
for contribution, colour, marker, line_style in (
    (8.0, "#0072B2", "o", "-"),
    (16.0, "#D55E00", "s", "--"),
    (32.0, "#009E73", "^", ":"),
):
    contribution_subset = introspection_full[
        np.isclose(introspection_full["M"], contribution)
    ]
    for curve_x, curve_y in sampled_curves(contribution_subset, "r"):
        ax_mechanism.plot(
            curve_x, curve_y, line_style, color=colour, alpha=0.12, linewidth=0.5, zorder=1
        )
    mechanism_return, mechanism_cooperation = median_curve(contribution_subset, "r")
    ax_mechanism.plot(
        mechanism_return, mechanism_cooperation, line_style, marker=marker, color=colour,
        ms=4, label=rf"$M = {contribution:.0f}$", zorder=3, path_effects=halo,
    )
add_baseline(ax_mechanism)
ax_mechanism.set_xlabel(r"return on investment $r$")
ax_mechanism.set_ylabel(r"cooperation $p_C$")
ax_mechanism.set_title(r"(b) introspection: threshold at $r = N$")
ax_mechanism.legend(loc="upper left", title=r"median over $\beta$", title_fontsize=8)

# Panel C: heterogeneity signature, introspection p_C against contribution scale.
for curve_x, curve_y in sampled_curves(
    introspection_full[introspection_full["r"] < number_of_players], "M"
):
    ax_heterogeneity.plot(
        curve_x, curve_y, "-", color=r_below_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for curve_x, curve_y in sampled_curves(
    introspection_full[introspection_full["r"] > number_of_players], "M"
):
    ax_heterogeneity.plot(
        curve_x, curve_y, "--", color=r_above_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for mask, line_style, marker, colour, label in (
    (introspection_full["r"] < number_of_players, "-", "o", r_below_colour, r"$r < N$"),
    (introspection_full["r"] > number_of_players, "--", "s", r_above_colour, r"$r > N$"),
):
    contribution_grid, cooperation = median_curve(introspection_full[mask], "M")
    ax_heterogeneity.plot(
        contribution_grid, cooperation, line_style, marker=marker,
        color=colour, ms=4, label=label, zorder=3, path_effects=halo,
    )
add_baseline(ax_heterogeneity, with_threshold=False)
ax_heterogeneity.set_xlabel(r"contribution scale $M$")
ax_heterogeneity.set_ylabel(r"cooperation $p_C$")
ax_heterogeneity.set_title(r"(c) introspection responds to $\alpha_i$")
ax_heterogeneity.legend(loc="upper left", title="median per regime", title_fontsize=8)

# Panel D: aspiration's own lever, p_C against the aspiration level A (M fixed).
aspiration_at_reference = aspiration_full[
    np.isclose(aspiration_full["M"], reference_contribution)
]
for curve_x, curve_y in sampled_curves(
    aspiration_at_reference[aspiration_at_reference["r"] < number_of_players], "aspiration"
):
    ax_aspiration_lever.plot(
        curve_x, curve_y, "-", color=r_below_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for curve_x, curve_y in sampled_curves(
    aspiration_at_reference[aspiration_at_reference["r"] > number_of_players], "aspiration"
):
    ax_aspiration_lever.plot(
        curve_x, curve_y, "--", color=r_above_colour, alpha=0.12, linewidth=0.5, zorder=1
    )
for mask, line_style, marker, colour, label in (
    (aspiration_at_reference["r"] < number_of_players, "-", "o", r_below_colour, r"$r < N$"),
    (aspiration_at_reference["r"] > number_of_players, "--", "^", r_above_colour, r"$r > N$"),
):
    aspiration_grid, cooperation = median_curve(
        aspiration_at_reference[mask], "aspiration"
    )
    ax_aspiration_lever.plot(
        aspiration_grid, cooperation, line_style, marker=marker,
        color=colour, ms=4, label=label, zorder=3, path_effects=halo,
    )
add_baseline(ax_aspiration_lever, with_threshold=False)
ax_aspiration_lever.set_xlabel(r"aspiration level $A$")
ax_aspiration_lever.set_ylabel(r"cooperation $p_C$")
ax_aspiration_lever.set_title(r"(d) aspiration: the aspiration lever")
ax_aspiration_lever.legend(
    loc="upper left", title=r"median per regime ($M=16$)", title_fontsize=8
)

fig.tight_layout()
for output_path in output_paths:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
print("saved:", *[str(path) for path in output_paths], sep="\n  ")
