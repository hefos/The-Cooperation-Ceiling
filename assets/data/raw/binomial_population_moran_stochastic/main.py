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
M = 20

df = pd.DataFrame(
    columns=[
        "UID",
        "alpha_i",
        "i",
        "mutant_alpha",
        "N",
        "n",
        "r",
        "epsilon",
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
                for scale in np.linspace(0.1, 10, 30):
                    for repetitions in range(200):

                        alphas = main.get_dirichlet_contribution_vector(
                            N=N,
                            alpha_rule=contribution_rules.dirichlet_binomial_alpha_rule,
                            M=M,
                            scale=scale,
                            n=n,
                            low_alpha=1 - alpha_h,
                            high_alpha=alpha_h,
                        )
                        for selection_intensity in np.linspace(
                            0, (1 / max(alphas)) * 0.99, 30
                        ):

                            state_space = main.get_state_space(N=N, k=2)

                            transition_matrix = main.generate_transition_matrix(
                                state_space=state_space,
                                fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                                compute_transition_probability=main.compute_moran_transition_probability,
                                r=r,
                                contribution_vector=alphas,
                                selection_intensity=selection_intensity,
                                number_of_strategies=2,
                            )

                            absorption_matrix = main.approximate_absorption_matrix(
                                transition_matrix
                            )

                            for first_contribution in np.unique(alphas):
                                id = uuid.uuid4()
                                data = []
                                approximate_state = np.zeros(N)
                                approximate_state[
                                    np.where(alphas == first_contribution)[0]
                                ] = 1
                                p_C = absorption_matrix[
                                    np.where(
                                        np.all(state_space == approximate_state, axis=1)
                                    )[0]
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
                                        n,
                                        r,
                                        selection_intensity,
                                        p_C,
                                        "moran",
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
