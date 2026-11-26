import argparse
import itertools
import pathlib
import uuid

import ludics
import numpy as np
import pandas as pd
import stet

from public_goods_games import simulate

file_path = pathlib.Path(__file__)

parser = argparse.ArgumentParser(description="Process mu and dynamic inputs")

parser.add_argument(
    "--dynamic", type=str, required=True, help="A string value for dynamic"
)

args = parser.parse_args()
dynamic = args.dynamic
columns=[
            "UID",
            "process",
            "seed",
            "iterations",
            "N",
            "d",
            "r",
            "beta",
            "epsilon",
            "aspiration",
            "p_C",
        ]

folder = pathlib.Path(file_path.parent / f"{dynamic}")
folder.mkdir(exist_ok=True)

csv_path = pathlib.Path(file_path.parent / f"{dynamic}/main.csv")
if csv_path.exists():
    pass
else:
    df = pd.DataFrame(
        columns=columns
    )

    df.to_csv(file_path.parent / f"{dynamic}/main.csv", index=False)

@stet.once(
    store=file_path.parent / f"{dynamic}/_stet_store.sqlite",
    key=[
        "N",
        "r",
        "d",
        "aspiration",
        "selection_intensity",
        "choice_intensity",
        "mu",
        "seed"
    ],
)
def run_experiment(N,r,d,aspiration,selection_intensity,choice_intensity,dynamic,mu,seed,iterations):
    id = uuid.uuid4()
    generator = simulate.generator_for(seed=seed)
    dynamic_to_process = {
                "aspiration": simulate.aspiration_transition_probability,
                "fermi": simulate.fermi_transition_probability,
                "moran": simulate.moran_transition_probability,
                "introspection": simulate.introspection_transition_probability,
                "wf":simulate.wright_fisher_transition_probability
            }
    if dynamic == "wf":
        individual_to_action_mutation_probability = None
    else:
        individual_to_action_mutation_probability = np.full(shape=(N,2), fill_value=mu)

    _, visit_counts = ludics.simulate_markov_chain(
        initial_state=np.zeros(N, dtype=int),
        number_of_strategies=2,
        fitness_function=simulate.subgroup_pgg_fitness_function,
        compute_transition_probability=dynamic_to_process[dynamic],
        seed=seed,
        individual_to_action_mutation_probability=individual_to_action_mutation_probability,
        mu=mu,
        warmup=iterations // 5,
        iterations=iterations,
        contribution_vector=contribution_vector,
        r=r,
        d=d,
        aspiration=aspiration,
        selection_intensity=selection_intensity,
        choice_intensity=choice_intensity,
        generator=generator
    )
    p_C = sum(count * sum(state) for state, count in visit_counts.items()) / (sum(visit_counts.values()) * N)

    data = [[
        id,
        dynamic,
        seed,
        iterations,
        N,
        d,
        r,
        choice_intensity,
        selection_intensity,
        aspiration,
        p_C
    ]]

    pd.DataFrame(data=data, columns=columns).to_csv(
                file_path.parent / f"{dynamic}/main.csv",
                mode="a",
                header=False,
                index=False,
            )

N = 8
contribution_vector = np.array([1,2,3,4,5,6,7,8])
mu = 0.05
iterations = 1000
for d in (2, 3, 4, 5, 6, 7):
    for seed in range(5):
        for r in np.linspace(d, 1.5 * d, 5):
            aspiration_range = np.linspace(1, 10, 5)
            selection_intensity_range = np.linspace(0,1,5)
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
                "wf": itertools.product(
                    null_array, selection_intensity_range, null_array
                )
            }

            for choice_intensity, selection_intensity, aspiration in dynamic_to_iteration[
                dynamic
            ]:
                run_experiment(
                    N=N,
                    r=r,
                    d=d,
                    aspiration=aspiration,
                    selection_intensity=selection_intensity,
                    choice_intensity=choice_intensity,
                    dynamic=dynamic,
                    mu=mu,
                    seed=seed,
                    iterations=iterations
                )

print(f"finished subgroup sweep job: dynamic={dynamic}", flush=True)