import pandas as pd
import numpy as np
import pathlib
import sys
import uuid

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../../").resolve()

sys.path.append(str(root_path))
import ludics.main as main
import ludics.fitness_functions as fitness_functions
import public_goods_games.contribution_rules
import stet

try:
    df = pd.read_csv(file_path.parent / "main.csv")
except (FileNotFoundError, pd.errors.EmptyDataError):
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
            "mu",
            "process",
            "population",
            "stochastic",
        ]
    )

    df.to_csv(file_path.parent / "main.csv", index=False)


@stet.once(
    store=file_path.parent / "_stet_store.sqlite",
    key=["N", "M", "r", "choice_intensity", "i", "mu"],
)
def run_experiment(
    N, M, r, choice_intensity, i, alphas, id, mu, steady_state, state_space
):
    cooperation_per_player = steady_state @ state_space
    p_C = sum(cooperation_per_player) / N
    data = [
        [
            id,
            alphas[i],
            i,
            N,
            r,
            choice_intensity,
            cooperation_per_player[i],
            p_C,
            mu,
            "fermi",
            "linear",
            False,
        ]
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "UID",
            "alpha_i",
            "i",
            "N",
            "r",
            "beta",
            "i_C",
            "p_C",
            "mu",
            "process",
            "population",
            "stochastic",
        ],
    )
    df.to_csv(
        file_path.parent / "main.csv",
        mode="a",
        header=False,
        index=False,
    )


N = 3
while True:
    state_space = main.get_state_space(N=N, k=2)
    for mu in (0.001, 0.01, 0.05, 0.1):
        individual_to_action_mutation_probability = np.full((N, 2), mu)
        for M in np.linspace(N, 4 * N, 30):
            alphas = public_goods_games.contribution_rules.get_deterministic_contribution_vector(
                N=N, contribution_rule=public_goods_games.contribution_rules.linear_contribution_rule, M=M
            )
            for r in np.linspace(0.5, 1.5 * N, 30):
                for choice_intensity in np.linspace(0, 2, num=30):
                    id = uuid.uuid4()
                    transition_matrix = main.generate_transition_matrix(
                        state_space=state_space,
                        fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                        compute_transition_probability=main.compute_fermi_transition_probability,
                        r=r,
                        contribution_vector=alphas,
                        choice_intensity=choice_intensity,
                        number_of_strategies=2,
                        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
                    )
                    steady_state = main.approximate_steady_state(transition_matrix)
                    for i in range(N):
                        run_experiment(
                            N=N,
                            M=M,
                            r=r,
                            choice_intensity=choice_intensity,
                            i=i,
                            alphas=alphas,
                            id=id,
                            mu=mu,
                            steady_state=steady_state,
                            state_space=state_space,
                        )
    N += 1
