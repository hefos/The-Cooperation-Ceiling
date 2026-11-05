import pandas as pd
import numpy as np
import pathlib
import uuid
import ludics
import ludics.fitness_functions
import public_goods_games.contribution_rules
import stet
import itertools
import argparse

file_path = pathlib.Path(__file__)

parser = argparse.ArgumentParser(description="Process mu and dynamic inputs")

parser.add_argument(
    "--dynamic", type=str, required=True, help="A string value for dynamic"
)

args = parser.parse_args()

mu = 0.05
dynamic = args.dynamic

folder = pathlib.Path(file_path.parent / f"{dynamic}")
folder.mkdir(exist_ok=True)

csv_path = pathlib.Path(file_path.parent / f"{dynamic}/main.csv")
if csv_path.exists():
    pass
else:
    df = pd.DataFrame(
        columns=[
            "UID",
            "process",
            "N",
            "d",
            "M",
            "r",
            "alpha_i",
            "i",
            "beta",
            "epsilon",
            "aspiration",
            "i_C",
            "p_C",
        ]
    )

    df.to_csv(file_path.parent / f"{dynamic}/main.csv", index=False)


@stet.once(
    store=file_path.parent / f"{dynamic}/_stet_store.sqlite",
    key=[
        "N",
        "M",
        "r",
        "d",
        "aspiration",
        "selection_intensity",
        "choice_intensity",
        "mu",
    ],
)
def run_experiment(
    N,
    M,
    r,
    d,
    alphas,
    aspiration,
    selection_intensity,
    choice_intensity,
    dynamic,
    mu,
):

        aspiration_vector = np.array([aspiration for _ in range(N)])
        dynamic_to_process = {
            "aspiration": ludics.compute_aspiration_transition_probability,
            "fermi": ludics.compute_fermi_transition_probability,
            "moran": ludics.compute_moran_transition_probability,
            "introspection": ludics.compute_introspection_transition_probability
        }
        state_space = ludics.get_state_space(N=N, k=2)
        transition_matrix = ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=public_goods_games.contribution_rules.subgroup_pgg_fitness_function,
            compute_transition_probability=dynamic_to_process[dynamic],
            aspiration_vector=aspiration_vector,
            selection_intensity=selection_intensity,
            choice_intensity=choice_intensity,
            individual_to_action_mutation_probability=np.full(
                shape=(N, 2), fill_value=mu
            ),
            r=r,
            contribution_vector=alphas,
            number_of_strategies=2,
            d=d
        )
        steady_state = ludics.compute_steady_state(transition_matrix)
        id = uuid.uuid4()
        cooperation_per_player = steady_state @ state_space
        p_C = sum(cooperation_per_player) / N
        for i, alpha_i in enumerate(alphas):
            i_C = cooperation_per_player[i]
            data = [
                [
                    id,
                    dynamic,
                    N,
                    d,
                    M,
                    r,
                    alpha_i,
                    i,
                    choice_intensity,
                    selection_intensity,
                    aspiration,
                    i_C,
                    p_C,
                ]
            ]
            df = pd.DataFrame(
                data=data,
                columns=[
                    "UID",
                    "process",
                    "N",
                    "d",
                    "M",
                    "r",
                    "alpha_i",
                    "i",
                    "beta",
                    "epsilon",
                    "aspiration",
                    "i_C",
                    "p_C",
                ],
            )
        df.to_csv(
            file_path.parent / f"{dynamic}/main.csv",
            mode="a",
            header=False,
            index=False,
        )


N = 8
mu = 0.05
for d in (2, 3, 4, 5, 6, 7):
    M_range = np.linspace(N, 4 * N, 5)
    r_range = np.linspace(0.5, 1.5 * d, 5)
    for M, r in itertools.product(M_range, r_range):
        alphas = public_goods_games.contribution_rules.get_deterministic_contribution_vector(
            N=N,
            contribution_rule=public_goods_games.contribution_rules.linear_contribution_rule,
            M=M,
        )
        aspiration_range = np.linspace(1, M, 5)
        if r < d:
            selection_intensity_range = np.linspace(
                0, 0.99 / (1 - alphas[-1] * ((r / d) - 1)), 5
            )
        else:
            selection_intensity_range = np.linspace(0, (alphas[-1] * ((r / d) - 1) + 1) / (alphas[-1] * ((r / d) - 1) + 2), 10)
            
        choice_intensity_range = np.linspace(0, 2, 5)
        null_array = np.array([0])
        dynamic_to_iteration = {
            "aspiration": itertools.product(
                choice_intensity_range, null_array, aspiration_range
            ),
            "fermi": itertools.product(choice_intensity_range, null_array, null_array),
            "introspection": itertools.product(
                choice_intensity_range, null_array, null_array
            ),
            "moran": itertools.product(
                null_array, selection_intensity_range, null_array
            ),
        }

        for choice_intensity, selection_intensity, aspiration in dynamic_to_iteration[
            dynamic
        ]:
            run_experiment(
                N=N,
                M=M,
                r=r,
                d=d,
                alphas=alphas,
                aspiration=aspiration,
                selection_intensity=selection_intensity,
                choice_intensity=choice_intensity,
                dynamic=dynamic,
                mu=mu,
            )

print(f"finished sweep job: dynamic={dynamic}, mu={mu}", flush=True)
