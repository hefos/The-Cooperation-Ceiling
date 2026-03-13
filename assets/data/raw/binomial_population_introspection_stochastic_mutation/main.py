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
import src.contribution_rules as contribution_rules
import stet
from stet.backends import get_backend

try:
    df = pd.read_csv(file_path.parent / "main.csv")
    backend = get_backend(pathlib.Path(file_path.parent / "_stet_store.sqlite"))
    for uid, experiment_frame in df.groupby("UID"):
        M_in_experiment = experiment_frame["alpha_i"].sum()
        alpha_h_in_experiment = experiment_frame["alpha_i"].max()
        for _, row in experiment_frame.iterrows():
            backend.record(
                {
                    "M": M_in_experiment,
                    "alpha_h": alpha_h_in_experiment,
                    "low_players": row["n"],
                    "r": row["r"],
                    "choice_intensity": row["beta"],
                    "N": row["N"],
                    "i": row["i"],
                    "mu": row["mu"],
                    "scale": row["scale"],
                    "seed": row["seed"],
                }
            )
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
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
    ],
)
def run_experiment(
    N,
    M,
    low_players,
    alpha_h,
    r,
    choice_intensity,
    i,
    alphas,
    id,
    individual_to_action_mutation_probability,
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
            cooperation_per_player[i],
            p_C,
            mu,
            "introspection",
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


N = 3
while True:
    state_space = main.get_state_space(N=N, k=2)
    for mu in (0.001, 0.01, 0.05, 0.1):
        individual_to_action_mutation_probability = np.full((N, 2), mu)
        for M in np.linspace(N, 4 * N, 30):
            for low_players in range(1, N - 1):
                for alpha_h in np.linspace(M / N, M / (N - low_players) * 0.95, 30):
                    for scale in np.linspace(0.1, 10, 30):
                        for seed in range(100):
                            np.random.seed(seed)
                            alphas = main.get_dirichlet_contribution_vector(
                                N=N,
                                alpha_rule=contribution_rules.dirichlet_binomial_alpha_rule,
                                M=M,
                                scale=scale,
                                n=low_players,
                                low_alpha=1 - alpha_h,
                                high_alpha=alpha_h,
                            )
                            for r in np.linspace(0.5, 1.5 * N, 30):
                                for choice_intensity in np.linspace(0, 2, num=30):
                                    transition_matrix = main.generate_transition_matrix(
                                        state_space=state_space,
                                        fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                                        compute_transition_probability=main.compute_introspection_transition_probability,
                                        r=r,
                                        contribution_vector=alphas,
                                        choice_intensity=choice_intensity,
                                        number_of_strategies=2,
                                        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
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
                                            individual_to_action_mutation_probability=individual_to_action_mutation_probability,
                                            mu=mu,
                                            scale=scale,
                                            seed=seed,
                                            steady_state=steady_state,
                                            state_space=state_space,
                                        )
    N += 1
