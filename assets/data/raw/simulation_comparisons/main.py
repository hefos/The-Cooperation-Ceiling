import stet
import itertools
import uuid
import argparse
import pandas as pd
import ludics
import ludics.fitness_functions
import numpy as np
import public_goods_games.contribution_rules
import pathlib
file_path = pathlib.Path(__file__).resolve()


parser = argparse.ArgumentParser(description="Process mu and dynamic inputs")

parser.add_argument("--process", type=str, required=True, help="The process with lower contributors")

args = parser.parse_args()

process = args.process
process_to_compute_transition_probability = {
    "Moran":ludics.compute_moran_transition_probability,
    "introspection":ludics.compute_introspection_transition_probability,
    "imispection":ludics.compute_imitation_introspection_transition_probability,
    "Fermi":ludics.compute_fermi_transition_probability,
    "aspiration":ludics.compute_aspiration_transition_probability
}


N=8
r_values = [N/2, 2*N]
beta_values = [0.5, 2]
M_values = [N]
mu_values = [0.01]
process_function = process_to_compute_transition_probability[process]
iteration_values = np.arange(1000, 10000, 100)
seed_range=range(1,10)
aspiration_range = [1]
selection_intensity_range = [0.01],
null_array = np.array([0])

dynamic_to_iteration = {
        "aspiration": itertools.product(r_values, M_values, mu_values, beta_values,null_array, aspiration_range, iteration_values, seed_range),
        "Fermi": itertools.product(r_values, M_values, mu_values, beta_values,null_array, null_array, iteration_values, seed_range),
        "introspection": itertools.product(r_values, M_values, mu_values, beta_values,null_array, null_array, iteration_values, seed_range),
        "imispection":itertools.product(r_values, M_values, mu_values, beta_values,selection_intensity_range, null_array, iteration_values, seed_range),
        "Moran":itertools.product(r_values, M_values, mu_values, null_array,selection_intensity_range, null_array, iteration_values, seed_range),
    }

iteration_values = dynamic_to_iteration[process]

folder = pathlib.Path(file_path.parent / f"{args.process}")    
folder.mkdir(exist_ok=True)

csv_path = pathlib.Path(file_path.parent / f"{args.process}/main.csv")
if csv_path.exists():
    pass
else:
    df = pd.DataFrame(
        columns=[
            "UID",
            "seed",
            "process",
            "iterations",
            "r",
            "mu",
            "beta",
            "M",
            "epsilon",
            "aspiration",
            "simulation_pc",
            "theoretical_pc",
            "difference"
            ]
    )
    df.to_csv(file_path.parent / f"{args.process}/main.csv", index=False)




@stet.once(
        store=file_path.parent / f"{args.process}/_stet_store.sqlite",
        key=[
            "seed",
            "iterations",
            "r",
            "mu",
            "beta",
            "M",
            "epsilon",
            "aspiration"
        ]
)
def run_sim(process_function, id, r, beta, mu, M, epsilon, iterations, aspiration, seed, process, N=8):
    state_space=ludics.get_state_space(N=N, k=2)
    initial_state = np.array([0 for _ in range(N)])
    initial_state[-1] = 1
    alphas = public_goods_games.contribution_rules.get_deterministic_contribution_vector(contribution_rule=public_goods_games.contribution_rules.linear_contribution_rule, N=N, M=M)
    individual_to_action_mutation_probability=np.full(shape=(N,2), fill_value=mu)
    aspiration_vector = np.array([aspiration for _ in range(N)])
    transition_matrix = ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
            compute_transition_probability=process_function,
            individual_to_action_mutation_probability=individual_to_action_mutation_probability,
            r=r,
            alpha=alphas,
            choice_intensity=beta,
            aspiration_vector=aspiration_vector,
            selection_intensity=epsilon,
            number_of_strategies=2
        )

    theoretical_pc = (ludics.compute_steady_state(transition_matrix) @ state_space).sum() / N

    _, simulation_distribution = ludics.simulate_markov_chain(initial_state=initial_state, number_of_strategies=2, fitness_function=ludics.fitness_functions.public_goods_game_fitness_function, compute_transition_probability=process_function, seed=seed, individual_to_action_mutation_probability=individual_to_action_mutation_probability, iterations=iterations, warmup=50, choice_intensity=beta, r=r, alpha=alphas, aspiration_vector=aspiration_vector, selection_intensity=epsilon)

    simulation_pc = sum(([sum(key) * (value/iterations) for (key, value) in simulation_distribution.items()])) / N

    diff = (simulation_pc - theoretical_pc)

    data = [[id, seed, process, iterations, r, mu, beta, M, epsilon, aspiration, simulation_pc, theoretical_pc, diff]]

    df = pd.DataFrame(
                    data=data,
                    columns=[
            "UID",
            "seed",
            "process",
            "iterations",
            "r",
            "mu",
            "beta",
            "M",
            "epsilon",
            "aspiration",
            "simulation_pc",
            "theoretical_pc",
            "difference"
        ]
                )
    df.to_csv(file_path.parent / f"{process}/main.csv",
                  mode="a",
                  header=False,
                  index=False)


for r,M,mu,beta,epsilon,aspiration,iteration,seed in iteration_values:
    id = uuid.uuid4()
    run_sim(process_function=process_function, r=r, id=id, beta=beta, mu=mu, M=M, epsilon=epsilon, iterations=iteration, aspiration=aspiration, seed=seed, process=process)
