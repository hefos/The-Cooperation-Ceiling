"""Sweep-wide evidence for the cooperation ceiling.

Across the entire parameter sweep (all N, contributions, returns, intensities,
aspirations, and the ergodic mutation rates mu > 0), no purely extrinsic
parameter set exceeds the neutral-drift baseline p_C = 1/2, whereas intrinsic
dynamics routinely do. This script produces the overview figure and a LaTeX
table of the fraction of parameter sets that cross the ceiling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

here = Path(__file__).resolve()
data_path = here.parents[2] / "data" / "sweep"
figure_outputs = [
    here.parents[2] / "tex" / "ceiling_overview.pdf",
]
table_outputs = [
    here.parents[2] / "tex" / "ceiling_crossing_table.tex",
]

mutations = ["0.001", "0.005", "0.05", "0.1"]
extrinsic = ["moran", "fermi"]
intrinsic = ["introspection", "aspiration"]
labels = {
    "moran": "Moran",
    "fermi": "Fermi",
    "introspection": "introspection",
    "aspiration": "aspiration",
}
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
    frames = []
    for mutation in mutations:
        frame = pd.read_csv(
            data_path / f"{dynamic}_mu_eq_{mutation}/main.csv",
            usecols=["UID", "process", "N", "r", "p_C"],
        )
        frames.append(frame.drop_duplicates("UID"))
    combined = pd.concat(frames, ignore_index=True)
    combined["above_ceiling"] = combined["p_C"] > 0.5 + 1e-9
    return combined


sweep = {dynamic: load(dynamic) for dynamic in extrinsic + intrinsic}

fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
(ax_extrinsic, ax_intrinsic), (ax_maximum, ax_fraction) = axes
# The extrinsic panel ends its bins exactly at 0.5 so the pile-up of neutral
# (zero-intensity) runs at p_C = 0.5 lands in the last, both-ends-closed bin,
# flush against the ceiling rather than spilling into the p_C > 0.5 region. The
# intrinsic panel genuinely crosses 0.5, so it runs to 1.0. Both use the same
# bin width.
extrinsic_bins = np.linspace(0.0, 0.5, 21)
intrinsic_bins = np.linspace(0.0, 1.0, 41)


def histogram_panel(ax, dynamics, title, bins):
    for dynamic in dynamics:
        ax.hist(
            sweep[dynamic]["p_C"], bins=bins, density=True, histtype="stepfilled",
            alpha=0.5, color=colours[dynamic], label=labels[dynamic],
        )
    ax.axvspan(0.5, 1.0, color="0.9", zorder=0)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"cooperation $p_C$")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper right")


histogram_panel(
    ax_extrinsic, extrinsic, r"(a) purely extrinsic: all $p_C \leq \frac{1}{2}$",
    bins=extrinsic_bins,
)
histogram_panel(
    ax_intrinsic, intrinsic, r"(b) intrinsic dynamics cross $\frac{1}{2}$",
    bins=intrinsic_bins,
)

for dynamic in extrinsic + intrinsic:
    line_style, marker = styles[dynamic]
    grouped = sweep[dynamic].groupby("N")["p_C"].max()
    ax_maximum.plot(
        grouped.index, grouped.to_numpy(), line_style, marker=marker, ms=4,
        color=colours[dynamic], label=labels[dynamic],
    )
ax_maximum.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
ax_maximum.set_xlabel(r"number of players $N$")
ax_maximum.set_ylabel(r"$\max p_C$ over the sweep")
ax_maximum.set_ylim(0.0, 1.0)
ax_maximum.set_title(r"(c) the sweep maximum is pinned at $\frac{1}{2}$")

ratio_bins = np.linspace(0.0, 1.6, 17)
ratio_centres = 0.5 * (ratio_bins[:-1] + ratio_bins[1:])
for dynamic in extrinsic + intrinsic:
    line_style, marker = styles[dynamic]
    frame = sweep[dynamic]
    ratio = (frame["r"] / frame["N"]).to_numpy()
    above = frame["above_ceiling"].to_numpy()
    fractions = np.array([
        above[(ratio >= low) & (ratio < high)].mean()
        if ((ratio >= low) & (ratio < high)).any()
        else np.nan
        for low, high in zip(ratio_bins[:-1], ratio_bins[1:])
    ])
    populated = ~np.isnan(fractions)  # drop empty r/N bins so the line is unbroken
    ax_fraction.plot(
        ratio_centres[populated], fractions[populated], line_style, marker=marker, ms=4,
        color=colours[dynamic], label=labels[dynamic],
    )
ax_fraction.axvline(1.0, color="#555555", linestyle=":", linewidth=0.9)
ax_fraction.set_xlabel(r"$r / N$")
ax_fraction.set_ylabel(r"fraction with $p_C > \frac{1}{2}$")
ax_fraction.set_ylim(-0.02, 1.0)
ax_fraction.set_title(r"(d) crossing switches on at $r = N$")

# Panels (c) and (d) share the same four dynamics, so a single legend centred
# under the bottom row serves both and avoids repeating it in each panel.
shared_handles, shared_labels = ax_maximum.get_legend_handles_labels()
fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
fig.legend(
    shared_handles, shared_labels, loc="lower center", ncol=4,
    bbox_to_anchor=(0.5, 0.0),
)
for output_path in figure_outputs:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


# Crossing table: percentage of parameter sets with p_C > 1/2.
def crossing_percentage(frame, mask=None):
    rows = frame if mask is None else frame[mask]
    if len(rows) == 0:
        return float("nan")
    return 100.0 * rows["above_ceiling"].mean()


table_rows = []
for dynamic in extrinsic + intrinsic:
    frame = sweep[dynamic]
    overall = crossing_percentage(frame)
    below = crossing_percentage(frame, frame["r"] < frame["N"])
    above = crossing_percentage(frame, frame["r"] > frame["N"])
    table_rows.append((labels[dynamic], len(frame), overall, below, above))

lines = [
    r"% Generated by analysis/ceiling_overview/main.py. Do not edit by hand.",
    r"\begin{tabular}{lrrrr}",
    r"\hline",
    r"dynamic & parameter sets & all & \(r < N\) & \(r > N\) \\",
    r"\hline",
]
for name, count, overall, below, above in table_rows:
    lines.append(
        rf"{name} & {count:,} & {overall:.1f}\% & {below:.1f}\% & {above:.1f}\% \\"
    )
lines += [r"\hline", r"\end{tabular}"]
table_text = "\n".join(lines) + "\n"
for output_path in table_outputs:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text)

print("saved figure:", *[str(p) for p in figure_outputs], sep="\n  ")
print("saved table:", *[str(p) for p in table_outputs], sep="\n  ")
print("\ncrossing percentages (p_C > 1/2):")
for name, count, overall, below, above in table_rows:
    print(f"  {name:24s} n={count:7d}  all={overall:5.1f}%  r<N={below:5.1f}%  r>N={above:5.1f}%")
