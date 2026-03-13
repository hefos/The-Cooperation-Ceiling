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
from stet.backends import get_backend

try:
    df = pd.read_csv(file_path.parent / "main.csv")
    backend = get_backend(pathlib.Path(file_path.parent / "_stet_store.sqlite"))
    for uid, experiment_frame in df.groupby("UID"):
        M_in_experiment = experiment_frame["alpha_i"].sum()
        for _, row in experiment_frame.iterrows():
            backend.record(
                {
                    "M": M_in_experiment,
                    "r": row["r"],
                    "choice_intensity": row["beta"],
                    "first_contribution": row["mutant_alpha"],
                    "N": row["N"],
                    "i": row["i"],
                    "aspiration": row["aspiration"],
                }
            )
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
    df = pd.DataFrame(
        columns=[
            "UID",
            "alpha_i",
            "i",
            "mutant_alpha",
            "N",
            "r",
            "beta",
            "aspiration",
            "i_C" "p_C",
            "process",
            "population",
            "stochastic",
        ]
    )

    df.to_csv(file_path.parent / "main.csv", index=False)


@stet.once(
    store=file_path.parent / "_stet_store.sqlite",
    key=["N", "M", "r", "choice_intensity", "first_contribution", "i", "aspiration"],
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
    steady_state,
    aspiration,
):
    cooperation_per_player = steady_state @ state_space
    p_C = sum(cooperation_per_player) / N
    data = [
        [
            id,
            alphas[i],
            i,
            first_contribution,
            N,
            r,
            choice_intensity,
            aspiration,
            cooperation_per_player[i],
            p_C,
            "aspiration",
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
            "aspiration",
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


N = 3
while True:
    state_space = ludics.main.get_state_space(N=N, k=2)
    for M in np.linspace(N, 4 * N, 30):
        alphas = ludics.main.get_deterministic_contribution_vector(
            N=N,
            contribution_rule=contribution_rules.linear_contribution_rule,
            M=M,
        )
        for r in np.linspace(0.5, 1.5 * N, 30):
            for aspiration in np.linspace(1, np.sum(alphas) * 1.01 / N):
                aspiration_vector = np.array([aspiration for _ in range(N)])
                for choice_intensity in np.linspace(0, 2, num=30):
                    transition_matrix = ludics.main.generate_transition_matrix(
                        state_space=state_space,
                        fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
                        compute_transition_probability=ludics.main.compute_aspiration_transition_probability,
                        r=r,
                        contribution_vector=alphas,
                        choice_intensity=choice_intensity,
                        number_of_strategies=2,
                        aspiration_vector=aspiration_vector,
                    )

                    steady_state = ludics.main.approximate_steady_state(
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
                                steady_state=steady_state,
                                state_space=state_space,
                                aspiration=aspiration,
                            )

    N += 1
