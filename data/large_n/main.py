import csv
import pathlib

import numpy as np
import stet

import ludics
import ludics.fitness_functions
import public_goods_games.contribution_rules as contribution_rules

here = pathlib.Path(__file__).resolve().parent
simulations_csv = here / "simulations.csv"
exact_csv = here / "exact.csv"
collapse_csv = here / "collapse.csv"
store = here / "_stet_store.sqlite"

dynamics = ["moran", "fermi", "introspection", "aspiration"]
returns_over_n = [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 0.8]
validation_return_index = returns_over_n.index(1.2)
large_population_sizes = [10, 30, 50, 75, 100, 200]
small_population_sizes = [2, 3, 4, 5, 6, 7, 8]
seeds_main = range(0, 15)
seeds_validation = range(0, 40)
iterations = 10000

mutation = 0.05
selection_intensity = 0.5
choice_intensity = 2.0
aspiration_fraction = 0.7

gap_fill_sizes = [125, 150, 175]
high_return_index = returns_over_n.index(1.3)

collapse_betas = {10: 1.0, 50: 2.0, 100: 3.0, 200: 4.0}
collapse_seeds = range(0, 10)


def contribution_scale(number_of_players):
    return 2.0 * number_of_players


def contributions(number_of_players):
    scale = contribution_scale(number_of_players)
    return np.array(
        [
            contribution_rules.linear_contribution_rule(index, number_of_players, scale)
            for index in range(number_of_players)
        ]
    )


def dynamic_function_and_kwargs(dynamic, number_of_players):
    """Return the ludics transition function and its dynamic-specific kwargs."""
    if dynamic == "moran":
        return ludics.compute_moran_transition_probability, {
            "selection_intensity": selection_intensity
        }
    if dynamic == "fermi":
        return ludics.compute_fermi_transition_probability, {
            "choice_intensity": choice_intensity
        }
    if dynamic == "introspection":
        return ludics.compute_introspection_transition_probability, {
            "choice_intensity": choice_intensity
        }
    if dynamic == "aspiration":
        aspiration_level = aspiration_fraction * contribution_scale(number_of_players)
        return ludics.compute_aspiration_transition_probability, {
            "choice_intensity": choice_intensity,
            "aspiration_vector": np.full(number_of_players, aspiration_level),
        }
    raise ValueError(f"unknown dynamic {dynamic!r}")


def simulated_cooperation(dynamic, number_of_players, return_value, seed):
    transition_function, extra = dynamic_function_and_kwargs(dynamic, number_of_players)
    mutation_matrix = np.full((number_of_players, 2), mutation)
    _, visit_counts = ludics.simulate_markov_chain(
        initial_state=np.zeros(number_of_players, dtype=int),
        number_of_strategies=2,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=transition_function,
        seed=seed,
        individual_to_action_mutation_probability=mutation_matrix,
        warmup=iterations // 5,
        iterations=iterations,
        alpha=contributions(number_of_players),
        r=return_value,
        **extra,
    )
    total_visits = sum(visit_counts.values())
    cooperators = sum(count * sum(state) for state, count in visit_counts.items())
    return cooperators / (total_visits * number_of_players)


def exact_cooperation(dynamic, number_of_players, return_value):
    transition_function, extra = dynamic_function_and_kwargs(dynamic, number_of_players)
    state_space = ludics.get_state_space(N=number_of_players, k=2)
    transition_matrix = ludics.generate_transition_matrix(
        state_space=state_space,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=transition_function,
        alpha=contributions(number_of_players),
        r=return_value,
        number_of_strategies=2,
        individual_to_action_mutation_probability=np.full(
            (number_of_players, 2), mutation
        ),
        **extra,
    )
    steady_state = ludics.compute_steady_state(transition_matrix)
    return (steady_state @ state_space).sum() / number_of_players


def simulated_introspection(number_of_players, return_value, beta, seed):
    """p_C under introspection dynamics at an explicit choice intensity beta."""
    _, visit_counts = ludics.simulate_markov_chain(
        initial_state=np.zeros(number_of_players, dtype=int),
        number_of_strategies=2,
        fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
        compute_transition_probability=ludics.compute_introspection_transition_probability,
        seed=seed,
        individual_to_action_mutation_probability=np.full(
            (number_of_players, 2), mutation
        ),
        warmup=iterations // 5,
        iterations=iterations,
        alpha=contributions(number_of_players),
        r=return_value,
        choice_intensity=beta,
    )
    total_visits = sum(visit_counts.values())
    cooperators = sum(count * sum(state) for state, count in visit_counts.items())
    return cooperators / (total_visits * number_of_players)


def append_row(path, row):
    is_new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


@stet.once(store=str(store), key=["dynamic", "N", "r_index", "seed"])
def run_simulation(dynamic, N, r_index, seed):
    return_value = returns_over_n[r_index] * N
    cooperation = simulated_cooperation(dynamic, N, return_value, seed)
    append_row(
        simulations_csv,
        {
            "dynamic": dynamic,
            "N": N,
            "r_index": r_index,
            "r_over_N": returns_over_n[r_index],
            "r": return_value,
            "seed": seed,
            "iterations": iterations,
            "p_C": cooperation,
        },
    )


@stet.once(store=str(store), key=["dynamic", "N", "r_index", "exact"])
def run_exact(dynamic, N, r_index):
    return_value = returns_over_n[r_index] * N
    cooperation = exact_cooperation(dynamic, N, return_value)
    append_row(
        exact_csv,
        {
            "dynamic": dynamic,
            "N": N,
            "r_index": r_index,
            "r_over_N": returns_over_n[r_index],
            "r": return_value,
            "p_C": cooperation,
        },
    )


@stet.once(store=str(store), key=["N", "r_index", "seed", "collapse"])
def run_collapse(N, r_index, seed):
    beta = collapse_betas[N]
    return_value = returns_over_n[r_index] * N
    cooperation = simulated_introspection(N, return_value, beta, seed)
    append_row(
        collapse_csv,
        {
            "N": N,
            "beta": beta,
            "r_index": r_index,
            "r_over_N": returns_over_n[r_index],
            "r": return_value,
            "seed": seed,
            "p_C": cooperation,
        },
    )


def main():
    for dynamic in dynamics:
        for number_of_players in large_population_sizes:
            for r_index in range(len(returns_over_n)):
                for seed in seeds_main:
                    run_simulation(
                        dynamic=dynamic, N=number_of_players, r_index=r_index, seed=seed
                    )
        for number_of_players in gap_fill_sizes:
            for seed in seeds_main:
                run_simulation(
                    dynamic=dynamic,
                    N=number_of_players,
                    r_index=high_return_index,
                    seed=seed,
                )
        for number_of_players in small_population_sizes:
            for seed in seeds_validation:
                run_simulation(
                    dynamic=dynamic,
                    N=number_of_players,
                    r_index=validation_return_index,
                    seed=seed,
                )
            run_exact(
                dynamic=dynamic, N=number_of_players, r_index=validation_return_index
            )
    for number_of_players in collapse_betas:
        for r_index in range(len(returns_over_n)):
            for seed in collapse_seeds:
                run_collapse(N=number_of_players, r_index=r_index, seed=seed)
    print("done; rows:")
    for path in (simulations_csv, exact_csv, collapse_csv):
        if path.exists():
            print(" ", path, sum(1 for _ in open(path)) - 1)


if __name__ == "__main__":
    main()
