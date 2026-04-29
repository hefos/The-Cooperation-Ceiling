import ludics
import ludics.fitness_functions
import public_goods_games.contribution_rules
import numpy as np
import pathlib
file_path = pathlib.Path(__file__).resolve()
import pandas as pd
import argparse
import uuid
import stet

parser = argparse.ArgumentParser(description="Process mu and dynamic inputs")

parser.add_argument("--p1", type=str, required=True, help="The process with lower contributors")
parser.add_argument("--p2", type=str, required=True, help="The process with higher contributors")

args = parser.parse_args()

process_to_compute_transition_probability = {
    "Moran":ludics.compute_moran_transition_probability,
    "introspection":ludics.compute_introspection_transition_probability,
    "imispection":ludics.compute_imitation_introspection_transition_probability
}

process_1 = process_to_compute_transition_probability[args.p1]
process_2 = process_to_compute_transition_probability[args.p2]

folder = pathlib.Path(file_path.parent / f"{args.p1}_{args.p2}")    
folder.mkdir(exist_ok=True)

csv_path = pathlib.Path(file_path.parent / f"{args.p1}_{args.p2}/main.csv")
if csv_path.exists():
    pass
else:
    df = pd.DataFrame(
        columns=[
            "UID",
            "seed",
            "low_process",
            "high_process",
            "split",
            "r",
            "p_C"
        ]
    )
    df.to_csv(file_path.parent / f"{args.p1}_{args.p2}/main.csv", index=False)



@stet.once(
        store=file_path.parent / f"{args.p1}_{args.p2}/_stet_store.sqlite",
        key=[
            "seed",
            "split",
            "r",
        ]
)
def do_simulation(process_1, process_2,seed, split, r, p1, p2, N=15):
    dynamic_array = np.concat((np.array([process_1 for _ in range(split)]), np.array([process_2 for _ in range(N - split)])))
    hybrid_dynamic = ludics.build_hybrid_population_dynamic(dynamic_array)
    contribution_vector = np.array([public_goods_games.contribution_rules.linear_contribution_rule(i, N, 15) for i in range(N)])
    id = uuid.uuid4()

    initial_state = np.array([0 for _ in range(N)])
    initial_state[-1] = 1
        
    _,simulation_distribution = ludics.simulate_markov_chain(
            initial_state=initial_state,
            number_of_strategies=2,
            fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
            compute_transition_probability=hybrid_dynamic,
            seed=seed,
            individual_to_action_mutation_probability=np.full((15,2), 0.05),
            iterations=100000,
            r=r,
            contribution_vector=contribution_vector,
            choice_intensity=0.5,
            selection_intensity=0.05
        )

    simulation_pc = sum(([sum(key) * (value/100000) for (key, value) in simulation_distribution.items()])) / N

    data = [[id, seed, p1, p2, split, r, simulation_pc]]

    df = pd.DataFrame(
                    data=data,
                    columns=[
            "UID",
            "seed",
            "low_process",
            "high_process",
            "split",
            "r",
            "p_C"
        ]
                )
    df.to_csv(file_path.parent / f"{p1}_{p2}/main.csv",
                  mode="a",
                  header=False,
                  index=False)

for split in range(1,14):
    for seed in range(1,10):
        for r in [7,20]:
            do_simulation(process_1=process_1, process_2=process_2,split=split,seed=seed, r=r, p1=args.p1, p2=args.p2)
