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
            "n",
            "r",
            "beta",
            "epsilon",
            "i_C",
            "p_C",
            "mu",
            "process",
            "population",
            "stochastic",
            "scale",
            "seed",
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
        "i",
        "mu",
        "scale",
        "seed",
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
    selection_intensity,
    i,
    alphas,
    id,
    mu,
    scale,
    seed,
    steady_state,
    state_space,
):
    cooperation_per_player = steady_state @ state_space
    p_C = sum(cooperation_per_player) / N
    data = [
        [
            id,
            alphas[i],
            i,
            N,
            low_players,
            r,
            choice_intensity,
            selection_intensity,
            cooperation_per_player[i],
            p_C,
            mu,
            "introspective imitation",
            "binomial",
            True,
            scale,
            seed,
        ]
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "UID",
            "alpha_i",
            "i",
            "N",
            "n",
            "r",
            "beta",
            "epsilon",
            "i_C",
            "p_C",
            "mu",
            "process",
            "population",
            "stochastic",
            "scale",
            "seed",
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
    for mu in (0.001, 0.01, 0.05, 0.1):
        individual_to_action_mutation_probability = np.full((N, 2), mu)
        for M in np.linspace(N, 4 * N, 30):
            for low_players in range(1, N - 1):
                for alpha_h in np.linspace(0.5, 0.99, 20):
                    for scale in np.linspace(0.1, 10, 30):
                        for seed in range(100):
                            np.random.seed(seed)
                            alphas = public_goods_games.contribution_rules.get_dirichlet_contribution_vector(
                                N=N,
                                alpha_rule=public_goods_games.contribution_rules.dirichlet_binomial_alpha_rule,
                                M=M,
                                scale=scale,
                                n=low_players,
                                low_alpha=1 - alpha_h,
                                high_alpha=alpha_h,
                            )
                            for r in np.linspace(0.5, 1.5 * N, 30):
                                for choice_intensity in np.linspace(0, 2, num=30):
                                    for selection_intensity in np.linspace(
                                        0, (1 / alphas[-1]) * 0.99, 30
                                    ):
                                        transition_matrix = main.generate_transition_matrix(
                                            state_space=state_space,
                                            fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                                            compute_transition_probability=main.compute_imitation_introspection_transition_probability,
                                            r=r,
                                            contribution_vector=alphas,
                                            choice_intensity=choice_intensity,
                                            number_of_strategies=2,
                                            individual_to_action_mutation_probability=individual_to_action_mutation_probability,
                                            selection_intensity=selection_intensity,
                                        )
                                        steady_state = main.approximate_steady_state(
                                            transition_matrix
                                        )
                                        id = uuid.uuid4()
                                        for i in range(N):
                                            run_experiment(
                                                N=N,
                                                M=M,
                                                low_players=low_players,
                                                alpha_h=alpha_h,
                                                r=r,
                                                choice_intensity=choice_intensity,
                                                i=i,
                                                alphas=alphas,
                                                id=id,
                                                mu=mu,
                                                scale=scale,
                                                seed=seed,
                                                steady_state=steady_state,
                                                state_space=state_space,
                                                selection_intensity=selection_intensity,
                                            )
    N += 1
