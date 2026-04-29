import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import ludics
import ludics.fitness_functions
import numpy as np
import public_goods_games.contribution_rules

here = pathlib.Path(__file__).resolve()
assets_path = here.parents[2]

def get_pc(transition_matrix, state_space, N):
    steady_state = ludics.compute_steady_state(transition_matrix)
    pc = (steady_state @ state_space).sum() / N
    return pc


def get_data(beta, epsilon, r, contribution_vector, split, N=8):
    state_space = ludics.get_state_space(N=N,k=2)

    population_dynamic_array = np.concatenate([
    np.array(
        [ludics.compute_moran_transition_probability for _ in range(N-split)],
        dtype=object
    ),
    np.array(
        [ludics.compute_imitation_introspection_transition_probability for _ in range(split)],
        dtype=object
    )
])

    hybrid_population_dynamic = ludics.build_hybrid_population_dynamic(population_dynamic_array)

    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
        compute_transition_probability=hybrid_population_dynamic,
        contribution_vector=contribution_vector,
        r=r,
        selection_intensity=epsilon,
        choice_intensity=beta,
        individual_to_action_mutation_probability=np.full(shape=(8,2), fill_value=0.05)
    )
    pc = get_pc(transition_matrix=transition_matrix,state_space=state_space, N=N)
    return pc


parameter_sets = [
    [7, 8, 0.5, 0.1, 'o'],
    [10, 8, 0.8, 0.01, 'x'],
    [15, 8, 0.1, 0.1, '^'],
    [13, 8, 0.5, 0.1, 'v']
]
fig, ax = plt.subplots()
for r, M, beta, epsilon, marker in parameter_sets:
    data = []
    for split in range(0,8):
        print([public_goods_games.contribution_rules.linear_contribution_rule(i, 8, M) for i in range(8)])
        contribution_vector = np.array([public_goods_games.contribution_rules.linear_contribution_rule(i, 8, M) for i in range(8)])

        data.append(get_data(beta=beta, epsilon=epsilon, r=r, contribution_vector=contribution_vector, split=split))
    ax.scatter(x=range(0,8), y=data, marker=marker, color='black', label=fr"$r$={r}, $\beta$={beta}, $\epsilon$={epsilon}")
ax.axhline(y=0.5, color='black', linestyle='dashed')
ax.set_ylabel(r"$p_C$")
ax.set_xlabel("Number of intrinsic players")
plt.legend()
plt.savefig(here.parent/"main.pdf")
