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
r_step_size = 0.02
choice_intensity_range = np.linspace(0, 2, num=30)

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
    for M in np.linspace(N, 4 * N, 30):
        for n in range(1, N - 1):
            for alpha_h in np.linspace(M / N, M / (N - n) * 0.95, 30):
                for r in np.linspace(0.5, 1.5 * N, 30):
                    for choice_intensity in choice_intensity_range:
                        id = uuid.uuid4()
                        alphas = ludics.main.get_deterministic_contribution_vector(
                            N=N,
                            contribution_rule=ludics.contribution_rules.binomial_contribution_rule,
                            M=M,
                            alpha_h=alpha_h,
                            n=n,
                        )
                        state_space = ludics.main.get_state_space(N=N, k=2)

                        transition_matrix = ludics.main.generate_transition_matrix(
                            state_space=state_space,
                            fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                            compute_transition_probability=ludics.main.compute_introspection_transition_probability,
                            r=r,
                            contribution_vector=alphas,
                            choice_intensity=choice_intensity,
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
                                choice_intensity,
                                i_C,
                                p_C,
                                "introspection",
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
