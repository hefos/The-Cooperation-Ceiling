import pandas as pd
import numpy as np
import sympy as sym
import pathlib
import sys
import uuid

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions
import src.contribution_rules as contribution_rules


r_min = 0.5
r_step_size = 0.02
choice_intensity_range = np.linspace(0, 2, num=30)

df = pd.DataFrame(
    columns=[
        "UID",
        "alpha_i",
        "i",
        "N",
        "r",
        "beta",
        "i_C",
        "p_C",
        "process",
        "population",
        "stochastic",
    ]
)
df.to_csv(file_path.parent / "main.csv", index=False)
N = 3
while True:
    for M in np.linspace(N, 4 * N, 30):
        for r in np.linspace(0.5, 1.5 * N, 30):
            for choice_intensity in choice_intensity_range:
                id = uuid.uuid4()
                alphas = main.get_deterministic_contribution_vector(
                    N=N,
                    contribution_rule=contribution_rules.linear_contribution_rule,
                    M=M,
                )
                state_space = main.get_state_space(N=N, k=2)

                transition_matrix = main.generate_transition_matrix(
                    state_space=state_space,
                    fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                    compute_transition_probability=main.compute_introspection_transition_probability,
                    r=r,
                    contribution_vector=alphas,
                    choice_intensity=choice_intensity,
                    number_of_strategies=2,
                )

                steady_state = main.approximate_steady_state(transition_matrix)
                cooperation_per_player = steady_state @ state_space
                p_C = sum(cooperation_per_player) / N
                data = []
                for i, alpha in enumerate(alphas):
                    i_C = cooperation_per_player[i]
                    row = [
                        id,
                        alpha,
                        i,
                        N,
                        r,
                        choice_intensity,
                        i_C,
                        p_C,
                        "introspection",
                        "linear",
                        False,
                    ]
                    data.append(row)
                df = pd.DataFrame(data)
                df.to_csv(
                    file_path.parent / "main.csv", mode="a", header=False, index=False
                )
    N += 1
