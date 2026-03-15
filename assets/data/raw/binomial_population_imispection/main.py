import pandas as pd
import numpy as np
import pathlib
import sys
import uuid

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../../").resolve()

sys.path.append(str(root_path))
import ludics.main
import ludics.fitness_functions
import src.contribution_rules as contribution_rules
import stet

try:
    df = pd.read_csv(file_path.parent / "main.csv")
except (FileNotFoundError, pd.errors.EmptyDataError):
    df = pd.DataFrame(
        columns=[
            "UID",
            "alpha_i",
            "i",
            "mutant_alpha",
            "N",
            "n",
            "r",
            "beta",
            "epsilon",
            "p_C",
            "process",
            "population",
            "stochastic",
        ]
    )

    df.to_csv(file_path.parent / "main.csv", index=False)


@stet.once(
    store=file_path.parent / "_stet_store.sqlite",
    key=[
        "N",
        "M",
        "low_players",
        "alpha_h",
        "r",
        "choice_intensity",
        "first_contribution",
        "i",
        "selection_intensity",
    ],
)
def run_experiment(
    N,
    M,
    low_players,
    alpha_h,
    r,
    choice_intensity,
    first_contribution,
    i,
    alphas,
    id,
    selection_intensity,
    absorption_matrix,
    state_space,
):
    approximate_state = np.zeros(N)
    approximate_state[np.where(alphas == first_contribution)[0][0]] = 1
    p_C = absorption_matrix[
        np.where(np.all(state_space == approximate_state, axis=1))[0] - 1,
        -1,
    ][0]
    data = [
        [
            id,
            alphas[i],
            i,
            first_contribution,
            N,
            low_players,
            r,
            choice_intensity,
            selection_intensity,
            p_C,
            "introspective imitation",
            "binomial",
            False,
        ]
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "UID",
            "alpha_i",
            "i",
            "mutant_alpha",
            "N",
            "n",
            "r",
            "beta",
            "epsilon",
            "p_C",
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
    state_space = ludics.main.get_state_space(N=N, k=2)
    for M in np.linspace(N, 4 * N, 30):
        for low_players in range(1, N - 1):
            for alpha_h in np.linspace(M / N, M / (N - low_players) * 0.95, 30):
                alphas = ludics.main.get_deterministic_contribution_vector(
                    N=N,
                    contribution_rule=contribution_rules.binomial_contribution_rule,
                    M=M,
                    alpha_h=alpha_h,
                    n=low_players,
                )
                for r in np.linspace(0.5, 1.5 * N, 30):
                    for choice_intensity in np.linspace(0, 2, num=30):
                        for selection_intensity in np.linspace(
                            0, (1 / alphas[-1]) * 0.99, 30
                        ):
                            transition_matrix = ludics.main.generate_transition_matrix(
                                state_space=state_space,
                                fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                                compute_transition_probability=ludics.main.compute_introspection_imitation_transition_probability,
                                r=r,
                                contribution_vector=alphas,
                                choice_intensity=choice_intensity,
                                number_of_strategies=2,
                                selection_intensity=selection_intensity,
                            )
                            absorption_matrix = (
                                ludics.main.approximate_absorption_matrix(
                                    transition_matrix
                                )
                            )
                            for first_contribution in np.unique(alphas):
                                id = uuid.uuid4()
                                for i, alpha in enumerate(alphas):
                                    run_experiment(
                                        N=N,
                                        M=M,
                                        low_players=low_players,
                                        alpha_h=alpha_h,
                                        r=r,
                                        choice_intensity=choice_intensity,
                                        first_contribution=first_contribution,
                                        selection_intensity=selection_intensity,
                                        i=i,
                                        alphas=alphas,
                                        id=id,
                                        absorption_matrix=absorption_matrix,
                                        state_space=state_space,
                                    )

    N += 1
