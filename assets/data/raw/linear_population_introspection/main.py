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

csv_path = pathlib.Path(file_path.parent / "main.csv")
if csv_path.exists():
    pass
else:
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


@stet.once(
    store=file_path.parent / "_stet_store.sqlite",
    key=["N", "M", "r", "choice_intensity", "i"],
)
def run_experiment(N, M, r, choice_intensity, i, alphas, id, steady_state, state_space):
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
            "introspection",
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
    state_space = main.get_state_space(N=N, k=2)
    for M in np.linspace(N, 4 * N, 30):
        alphas = public_goods_games.contribution_rules.get_deterministic_contribution_vector(
            N=N,
            contribution_rule=public_goods_games.contribution_rules.linear_contribution_rule,
            M=M,
        )
        for r in np.linspace(0.5, 1.5 * N, 30):
            for choice_intensity in np.linspace(0, 2, num=30):
                id = uuid.uuid4()
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
                for i in range(N):
                    run_experiment(
                        N=N,
                        M=M,
                        r=r,
                        choice_intensity=choice_intensity,
                        i=i,
                        alphas=alphas,
                        id=id,
                        steady_state=steady_state,
                        state_space=state_space,
                    )
    N += 1
