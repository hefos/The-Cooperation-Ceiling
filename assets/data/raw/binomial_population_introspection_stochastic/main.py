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
import src.main as main
import src.fitness_functions as fitness_functions
import src.contribution_rules as contribution_rules


r_min = 0.5
r_step_size = 0.02
choice_intensity_range = np.linspace(0, 2, num=30)
M = 20

df = pd.DataFrame(
    columns=[
        "UID",
        "alpha_i",
        "i",
        "N",
        "n",
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
    for alpha_h in np.linspace(0.5, 0.99, 20):
        for n in range(1, N - 1):
            for r in np.linspace(0.5, 1.5 * N, 30):
                for choice_intensity in choice_intensity_range:
                    for scale in np.linspace(0.1, 10, 30):
                        for repetitions in range(200):
                            id = uuid.uuid4()
                            alphas = main.get_dirichlet_contribution_vector(
                                N=N,
                                alpha_rule=contribution_rules.dirichlet_binomial_alpha_rule,
                                M=M,
                                scale=scale,
                                n=n,
                                low_alpha=1 - alpha_h,
                                high_alpha=alpha_h,
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

                            steady_state = main.approximate_steady_state(
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
                                    choice_intensity,
                                    i_C,
                                    p_C,
                                    "introspection",
                                    "binomial",
                                    True,
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
