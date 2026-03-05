import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

np.set_printoptions(suppress=False)
import matplotlib.patches as mpatches
import matplotlib.lines as mlinesge
import scipy
import sys
import pathlib

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions


r = sym.Symbol("r")
epsilon = sym.Symbol("epsilon")
N = 2
alpha = sym.Symbol("alpha")
difference = sym.Symbol("d")
contribution_vector = np.array([alpha, alpha * difference])
state_space = main.get_state_space(N=N, k=2)
beta = sym.Symbol("beta")

imitation_introspection_transition_matrix = main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
    compute_transition_probability=main.compute_imitation_introspection_transition_probability,
    selection_intensity=epsilon,
    choice_intensity=beta,
    number_of_strategies=2,
    r=r,
    contribution_vector=contribution_vector,
    epsilon=epsilon,
)

fermi_transition_matrix = main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
    compute_transition_probability=main.compute_fermi_transition_probability,
    choice_intensity=beta,
    number_of_strategies=2,
    r=r,
    contribution_vector=contribution_vector,
    epsilon=epsilon,
)

moran_transition_matrix = main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
    compute_transition_probability=main.compute_moran_transition_probability,
    selection_intensity=epsilon,
    number_of_strategies=2,
    r=r,
    contribution_vector=contribution_vector,
    epsilon=epsilon,
)


absorption_matrix_imispection = main.calculate_absorption_matrix(
    imitation_introspection_transition_matrix
)

expression_imispection_p1 = sym.lambdify(
    (r, alpha, beta, epsilon, difference),
    sym.Matrix(absorption_matrix_imispection)[0, 1],
    "numpy",
)


fermi_absorption_matrix = main.calculate_absorption_matrix(fermi_transition_matrix)

expression_fermi_p1 = sym.lambdify(
    (r, alpha, beta, epsilon, difference),
    sym.Matrix(fermi_absorption_matrix)[0, 1],
    "numpy",
)


moran_absorption_matrix = main.calculate_absorption_matrix(moran_transition_matrix)

expression_moran_p1 = sym.lambdify(
    (r, alpha, beta, epsilon, difference),
    sym.Matrix(moran_absorption_matrix)[0, 1],
    "numpy",
)


d_values = np.linspace(0, 2, 4000)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

fig.suptitle(
    r"Absorption probabilities for Full Cooperation in the Heterogeneous Public Goods Game with 2 Players, Starting with player 1 contributing."
)

axis = 0
for alpha_i in np.array([2, 3, 4, 5]):

    ax = axes[axis]
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 0.6)
    d_for_fermi = expression_fermi_p1(1.8, alpha_i, 0.5, 0.3, d_values)
    d_for_fermi = d_for_fermi * np.ones_like(d_values)
    d_for_imispection = expression_imispection_p1(1.8, alpha_i, 0.5, 0.3, d_values)
    d_for_moran = expression_moran_p1(1.8, alpha_i, 0.5, 0.3, d_values)

    ax.axvspan(
        0,
        1 / 9,
        alpha=0.3,
        color="red",
        label="No possible benefit for player 1",
        zorder=0,
    )
    ax.plot(d_values, d_for_fermi, label="Fermi Imitation Dynamics", zorder=2)
    ax.plot(d_values, d_for_moran, label="Moran Process", zorder=2)
    ax.plot(
        d_values, d_for_imispection, label="Introspective Imitation Dynamics", zorder=2
    )

    ax.set_title(r"$v_0=(C, D)$, $\alpha =$" + str(alpha_i))
    ax.set_xlabel(r"$\alpha_2 / \alpha_1$", fontsize=14)
    ax.set_ylabel(r"$\rho_{C,C}$", fontsize=14)
    axis += 1
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout(rect=[0, 0, 1, 0.93])

fig.text(
    1.25,
    0.02,
    r"$\beta = 0.5$, $\epsilon = 0.2$, $r = 1.8$",
    ha="right",
    va="bottom",
    fontsize=12,
)

plt.savefig(file_path.parent / "main.pdf", bbox_inches="tight")
