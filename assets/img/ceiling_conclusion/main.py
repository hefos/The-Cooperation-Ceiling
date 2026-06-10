"""Conclusion figure: heterogeneous update rules and heterogeneous returns.

Two illustrations of natural extensions of the framework, both computed exactly
from the steady state of the corresponding transition matrix via ``ludics``.

Panel (a) treats the update rule as a heterogeneous attribute: a fraction of the
population learns by the (extrinsic) Moran process and the remainder by
(intrinsic) introspection dynamics. As more players adopt the intrinsic rule the
cooperation probability is able to cross the neutral-drift baseline once the
return exceeds the threshold r = N.

Panel (b) treats the return as a heterogeneous attribute under introspection
dynamics: a fraction of the population receives a high return and the remainder a
low return. Cooperation crosses the ceiling once enough players sit above the
r = N threshold.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import ludics
import ludics.fitness_functions
import public_goods_games.contribution_rules

here = pathlib.Path(__file__).resolve()
figure_outputs = [
    here.parent / "main.pdf",
    here.parents[3] / "tex" / "figures" / "ceiling_conclusion" / "main.pdf",
]

number_of_players = 8
reference_contribution = 16.0
mutation = 0.05
baseline_colour = "#555555"
palette = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
halo = [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]

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

state_space = ludics.get_state_space(N=number_of_players, k=2)
contribution_vector = np.array(
    [
        public_goods_games.contribution_rules.linear_contribution_rule(
            player_index, number_of_players, reference_contribution
        )
        for player_index in range(number_of_players)
    ]
)
mutation_matrix = np.full(shape=(number_of_players, 2), fill_value=mutation)


def cooperation_probability(transition_matrix):
    steady_state = ludics.compute_steady_state(transition_matrix)
    return (steady_state @ state_space).sum() / number_of_players


def hybrid_cooperation(return_value, choice_intensity, selection_intensity, intrinsic):
    """p_C when ``intrinsic`` players use introspection and the rest use Moran."""
    per_player_dynamic = np.array(
        [ludics.compute_moran_transition_probability] * (number_of_players - intrinsic)
        + [ludics.compute_introspection_transition_probability] * intrinsic,
        dtype=object,
    )
    hybrid_dynamic = ludics.build_hybrid_population_dynamic(per_player_dynamic)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=hybrid_dynamic,
        alpha=contribution_vector,
        r=return_value,
        selection_intensity=selection_intensity,
        choice_intensity=choice_intensity,
        number_of_strategies=2,
        individual_to_action_mutation_probability=mutation_matrix,
    )
    return cooperation_probability(transition_matrix)


def heterogeneous_return_cooperation(return_low, return_high, choice_intensity, high):
    """p_C under introspection when ``high`` players receive return_high."""
    return_vector = np.array(
        [return_high] * high + [return_low] * (number_of_players - high)
    )
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        alpha=contribution_vector,
        r=return_vector,
        selection_intensity=0.0,
        choice_intensity=choice_intensity,
        number_of_strategies=2,
        individual_to_action_mutation_probability=mutation_matrix,
    )
    return cooperation_probability(transition_matrix)


fig, (ax_hybrid, ax_return) = plt.subplots(1, 2, figsize=(8.4, 3.6))
counts = np.arange(0, number_of_players + 1)

# Panel A: a mix of Moran (extrinsic) and introspection (intrinsic) players.
hybrid_sets = [
    (12.0, 2.0, 0.1, r"$r=12>N$, $\beta=2$, $\varepsilon=0.1$"),
    (9.0, 0.5, 2.0, r"$r=9>N$, $\beta=0.5$, $\varepsilon=2$"),
    (4.0, 2.0, 0.1, r"$r=4<N$, $\beta=2$, $\varepsilon=0.1$"),
]
markers = ["o", "s", "^", "D"]
line_styles = ["-", "--", "-.", ":"]
for index, (return_value, choice_intensity, selection_intensity, label) in enumerate(
    hybrid_sets
):
    cooperation = [
        hybrid_cooperation(return_value, choice_intensity, selection_intensity, intrinsic)
        for intrinsic in counts
    ]
    ax_hybrid.plot(
        counts, cooperation, line_styles[index], marker=markers[index], ms=4,
        color=palette[index], label=label, path_effects=halo, zorder=3,
    )
ax_hybrid.axhline(0.5, color=baseline_colour, linestyle="--", linewidth=0.9)
ax_hybrid.set_xlabel("number of intrinsic (introspection) players")
ax_hybrid.set_ylabel(r"cooperation $p_C$")
ax_hybrid.set_ylim(0.0, 1.0)
ax_hybrid.set_title(r"(a) mixing extrinsic and intrinsic update rules")
ax_hybrid.legend(loc="upper left")

# Panel B: introspection with a heterogeneous return r.
return_sets = [
    (4.0, 12.0, 2.0, r"$r_{\mathrm{low}}=4$, $r_{\mathrm{high}}=12$"),
    (6.0, 10.0, 2.0, r"$r_{\mathrm{low}}=6$, $r_{\mathrm{high}}=10$"),
    (2.0, 14.0, 2.0, r"$r_{\mathrm{low}}=2$, $r_{\mathrm{high}}=14$"),
]
for index, (return_low, return_high, choice_intensity, label) in enumerate(return_sets):
    cooperation = [
        heterogeneous_return_cooperation(return_low, return_high, choice_intensity, high)
        for high in counts
    ]
    ax_return.plot(
        counts, cooperation, line_styles[index], marker=markers[index], ms=4,
        color=palette[index], label=label, path_effects=halo, zorder=3,
    )
ax_return.axhline(0.5, color=baseline_colour, linestyle="--", linewidth=0.9)
ax_return.set_xlabel(r"number of players with high return $r_{\mathrm{high}}$")
ax_return.set_ylabel(r"cooperation $p_C$")
ax_return.set_ylim(0.0, 1.0)
ax_return.set_title(r"(b) introspection with heterogeneous return $r$")
ax_return.legend(loc="upper left")

fig.tight_layout()
for output_path in figure_outputs:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
print("saved figure:", *[str(p) for p in figure_outputs], sep="\n  ")
