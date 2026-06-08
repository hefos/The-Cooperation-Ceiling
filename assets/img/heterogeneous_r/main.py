import matplotlib.pyplot as plt
import pathlib
import ludics
import ludics.fitness_functions
import numpy as np

here = pathlib.Path(__file__).resolve()
assets_path = here.parents[2]


def get_pc(transition_matrix, state_space, N):
    steady_state = ludics.compute_steady_state(transition_matrix)
    pc = (steady_state @ state_space).sum() / N
    return pc


def get_data(beta, epsilon, alpha, r_high, r_low, split, N=8):
    state_space = ludics.get_state_space(N=N, k=2)

    r_vector = np.concatenate(
        [
            np.array([r_high for _ in range(split)]),
            np.array([r_low for _ in range(N - split)]),
        ]
    )

    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=ludics.compute_imitation_introspection_transition_probability,
        alpha=alpha,
        r=r_vector,
        selection_intensity=epsilon,
        choice_intensity=beta,
        individual_to_action_mutation_probability=np.full(
            shape=(8, 2), fill_value=0.05
        ),
    )
    pc = get_pc(transition_matrix=transition_matrix, state_space=state_space, N=N)
    return pc


alpha = 2
parameter_sets = [
    [4, 10, 0.5, 0.1, "o"],
    [2, 20, 0.8, 0.01, "x"],
    [7, 9, 0.5, 0.3, "^"],
]

fig, ax = plt.subplots()
for r_low, r_high, beta, epsilon, marker in parameter_sets:
    data = []
    for split in range(0, 9):
        data.append(
            get_data(
                beta=beta,
                epsilon=epsilon,
                r_high=r_high,
                r_low=r_low,
                alpha=alpha,
                split=split,
            )
        )
    ax.scatter(
        x=range(0, 9),
        y=data,
        marker=marker,
        color="black",
        label=rf"$r_{{low}}$={r_low}, $r_{{high}}$={r_high}, $\beta$={beta}, $\epsilon$={epsilon}",
    )
ax.axhline(y=0.5, color="black", linestyle="dashed")
ax.set_ylabel(r"$p_C$")
ax.set_xlabel("Number of high return players")
plt.legend()
plt.savefig(here.parent / "main.pdf")
