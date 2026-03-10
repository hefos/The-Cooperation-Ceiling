import pandas as pd
import numpy as np
import sympy as sym
import pathlib
import sys
import uuid
import math

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../../").resolve()

sys.path.append(str(root_path))
import ludics.main
import ludics.fitness_functions
import ludics.contribution_rules


r_min = 0.5

df = pd.DataFrame(
    columns=[
        "UID",
        "alpha_i",
        "i",
        "N",
        "n",
        "r",
        "epsilon",
        "beta",
        "i_C",
        "p_C",
        "mu",
        "process",
        "population",
        "stochastic",
    ]
)
df.to_csv(file_path.parent / "main.csv", index=False)
N = 3
while True:
    for mu in (0.001, 0.01, 0.05, 0.1):
        for M in np.linspace(N, 4 * N, 30):
            for n in range(1, N - 1):
                for alpha_h in np.linspace(M / N, M / (N - n) * 0.95, 30):
                    alphas = ludics.main.get_deterministic_contribution_vector(
                        N=N,
                        contribution_rule=ludics.contribution_rules.binomial_contribution_rule,
                        M=M,
                        alpha_h=alpha_h,
                        n=n,
                    )
                    for r in np.linspace(0.5, 1.5 * N, 30):
                        for selection_intensity in np.linspace(
                            0, (1 / alphas[-1]) * 0.99, 30
                        ):
                            for choice_intensity in np.linspace(0, 2, 30):
                                id = uuid.uuid4()
                                state_space = ludics.main.get_state_space(N=N, k=2)

                                individual_to_action_mutation_probability = np.full(
                                    (N, 2), mu
                                )

                                transition_matrix = ludics.main.generate_transition_matrix(
                                    state_space=state_space,
                                    fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                                    compute_transition_probability=ludics.main.compute_imitation_introspection_transition_probability,
                                    r=r,
                                    contribution_vector=alphas,
                                    selection_intensity=selection_intensity,
                                    choice_intensity=choice_intensity,
                                    individual_to_action_mutation_probability=individual_to_action_mutation_probability,
                                    number_of_strategies=2,
                                )

                                steady_state = ludics.main.approximate_steady_state(
                                    transition_matrix
                                )
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
                                        n,
                                        r,
                                        selection_intensity,
                                        choice_intensity,
                                        i_C,
                                        p_C,
                                        mu,
                                        "introspective imitation",
                                        "binomial",
                                        False,
                                    ]
                                    data.append(row)
                                df = pd.DataFrame(data)
                                df.to_csv(
                                    file_path.parent / "main.csv",
                                    mode="a",
                                    header=False,
                                    index=False,
                                )
    N += 1
