import pandas as pd
import numpy as np
import sympy as sym
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
                    "seed": row["seed"],
                    "scale": row["scale"],
                }
            )
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
    df = pd.DataFrame(
        columns=[
            "UID",
            "alpha_i",
            "i",
            "mutant_alpha",
            "scale",
            "N",
            "r",
            "beta",
            "p_C",
            "process",
            "population",
            "stochastic",
            "seed",
        ]
    )

    df.to_csv(file_path.parent / "main.csv", index=False)


@stet.once(
    store=file_path.parent / "_stet_store.sqlite",
    key=["N", "M", "r", "choice_intensity", "first_contribution", "i", "scale", "seed"],
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
    scale,
    seed,
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
            scale,
            N,
            r,
            choice_intensity,
            p_C,
            "fermi",
            "linear",
            True,
            seed,
        ]
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "UID",
            "alpha_i",
            "i",
            "mutant_alpha",
            "scale",
            "N",
            "r",
            "beta",
            "p_C",
            "process",
            "population",
            "stochastic",
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
    state_space = ludics.main.get_state_space(N=N, k=2)
    for M in np.linspace(N, 4 * N, 30):
        for r in np.linspace(0.5, 1.5 * N, 30):
            for choice_intensity in np.linspace(0, 2, num=30):
                for scale in np.linspace(0.1, 10, 30):
                    for seed in range(100):
                        np.random.seed(seed)
                        alphas = ludics.main.get_dirichlet_contribution_vector(
                            N=N,
                            alpha_rule=contribution_rules.dirichlet_linear_alpha_rule,
                            M=M,
                            scale=scale,
                        )
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
                                    scale=scale,
                                    seed=seed,
                                    absorption_matrix=absorption_matrix,
                                    state_space=state_space,
                                )

    N += 1
