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

df = pd.DataFrame(
    columns=[
        "UID",
        "alpha_i",
        "i",
        "mutant_alpha",
        "N",
        "r",
        "beta",
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
                alphas = main.get_deterministic_contribution_vector(
                    N=N,
                    contribution_rule=contribution_rules.linear_contribution_rule,
                    M=M,
                )

                state_space = main.get_state_space(N=N, k=2)

                transition_matrix = main.generate_transition_matrix(
                    state_space=state_space,
                    fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                    compute_transition_probability=main.compute_fermi_transition_probability,
                    r=r,
                    contribution_vector=alphas,
                    choice_intensity=choice_intensity,
                    number_of_strategies=2,
                )

                absorption_matrix = main.approximate_absorption_matrix(
                    transition_matrix
                )

                for first_contribution in np.unique(alphas):
                    data = []
                    id = uuid.uuid4()
                    approximate_state = np.zeros(N)
                    approximate_state[np.where(alphas == first_contribution)[0]] = 1
                    p_C = absorption_matrix[
                        np.where(np.all(state_space == approximate_state, axis=1))[0]
                        - 1,
                        -1,
                    ][0]
                    for i, alpha in enumerate(alphas):
                        row = [
                            id,
                            alpha,
                            i,
                            first_contribution,
                            N,
                            r,
                            choice_intensity,
                            p_C,
                            "fermi",
                            "linear",
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
