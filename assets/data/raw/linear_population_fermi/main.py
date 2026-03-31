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
import public_goods_games.contribution_rules
import stet

csv_path = pathlib.Path(file_path.parent / "main.csv")
if csv_path.exists():
    pass
else:
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


@stet.once(
    store=file_path.parent / "_stet_store.sqlite",
    key=[
        "N",
        "M",
        "r",
        "choice_intensity",
        "first_contribution",
        "i",
    ],
)
def run_experiment(
    N,
    M,
    r,
    choice_intensity,
    first_contribution,
    i,
    alphas,
    id,
    state_space,
    absorption_matrix,
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
            r,
            choice_intensity,
            p_C,
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
            "mutant_alpha",
            "N",
            "r",
            "beta",
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


N = 8
while True:
    state_space = ludics.main.get_state_space(N=N, k=2)
    for M in np.linspace(N, 4 * N, 30):
        alphas = public_goods_games.contribution_rules.get_deterministic_contribution_vector(
            N=N,
            contribution_rule=public_goods_games.contribution_rules.linear_contribution_rule,
            M=M,
        )
        for r in np.linspace(0.5, 1.5 * N, 30):
            for choice_intensity in np.linspace(0, 2, num=30):
                transition_matrix = ludics.main.generate_transition_matrix(
                    state_space=state_space,
                    fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                    compute_transition_probability=ludics.main.compute_fermi_transition_probability,
                    r=r,
                    contribution_vector=alphas,
                    choice_intensity=choice_intensity,
                    number_of_strategies=2,
                )

                absorption_matrix = ludics.main.approximate_absorption_matrix(
                    transition_matrix
                )
                for first_contribution in np.unique(alphas):
                    id = uuid.uuid4()
                    for i, alpha in enumerate(alphas):
                        run_experiment(
                            N=N,
                            M=M,
                            r=r,
                            choice_intensity=choice_intensity,
                            first_contribution=first_contribution,
                            i=i,
                            alphas=alphas,
                            id=id,
                            absorption_matrix=absorption_matrix,
                            state_space=state_space,
                        )

    N += 1
