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
root_path = (file_path / "../../../../../").resolve()

parser = argparse.ArgumentParser(description="Process mu and dynamic inputs")

parser.add_argument("--mu", type=float, required=True, help="A float value for mu")
parser.add_argument("--dynamic", type=str, required=True, help="A string value for dynamic")
parser.add_argument("--absorbing", action="store_true", help="Set if the process is absorbing")

args = parser.parse_args()

mu = args.mu
dynamic = args.dynamic
absorbing = args.absorbing

folder = pathlib.Path(file_path.parent / f"{dynamic}_mu_eq_{mu}")    
folder.mkdir(exist_ok=True)

csv_path = pathlib.Path(file_path.parent / f"{dynamic}_mu_eq_{mu}/main.csv")
if csv_path.exists():
    pass
else:
    if absorbing is True:
        df = pd.DataFrame(
            columns=[
                    "UID",
                    "process",
                    "N",
                    "M",
                    "r",
                    "alpha_i",
                    "mutant_alpha",
                    "i",
                    "beta",
                    "epsilon",
                    "p_C",
                ]
        )
    else:
        df = pd.DataFrame(
            columns=[
                    "UID",
                    "process",
                    "N",
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

    df.to_csv(file_path.parent / f"{dynamic}_mu_eq_{mu}/main.csv", index=False)

@stet.once(
        store=file_path.parent / f"{dynamic}_mu_eq_{mu}/_stet_store.sqlite",
        key=[
            "N",
            "M",
            "r",
            "aspiration",
            "selection_intensity",
            "choice_intensity",
            "mu",
        ]
)
def run_experiment(N,M,r,alphas,aspiration,selection_intensity,choice_intensity,dynamic,mu,absorbing):
    if dynamic == "introspection":
        id = uuid.uuid4()
        running_phis = []

        for i in range(N):
            numerator = 0.8
            denominator = 1 + np.exp(0.5 * (alphas[i] * (1 - (r/N))))

            running_phis.append((numerator / denominator) + 0.1)
        
        p_C = sum(running_phis) / N
        for j, alpha_j in enumerate(alphas):
            data = [[id, dynamic, N, M, r, alpha_j, j, choice_intensity, selection_intensity, p_C]]

            df = pd.DataFrame(
                    data=data,
                    columns=[
                        "UID",
                        "process",
                        "N",
                        "M",
                        "r",
                        "alpha_i",
                        "i",
                        "beta",
                        "epsilon",
                        "p_C"
                    ]
                )
    else:
        aspiration_vector = np.array([aspiration for _ in range(N)])
        dynamic_to_process = {
            "aspiration":ludics.compute_aspiration_transition_probability,
            "fermi":ludics.compute_fermi_transition_probability,
            "imispection":ludics.compute_imitation_introspection_transition_probability,
            "moran":ludics.compute_moran_transition_probability,
        }
        state_space = ludics.get_state_space(N=N,k=2)
        transition_matrix = ludics.generate_transition_matrix(
            state_space=state_space,
            fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
            compute_transition_probability=dynamic_to_process[dynamic],
            aspiration_vector=aspiration_vector,
            selection_intensity=selection_intensity,
            choice_intensity=choice_intensity,
            individual_to_action_mutation_probability=np.full(shape=(N,2),fill_value=mu),
            r=r,
            contribution_vector=alphas,
            number_of_strategies=2
        )
        if absorbing:
            absorption_matrix = ludics.compute_absorption_matrix(transition_matrix)
            for first_contribution in np.unique(alphas):
                id = uuid.uuid4()
                approximate_state = np.zeros(N)
                approximate_state[np.where(alphas == first_contribution)[0][0]] = 1
                p_C = absorption_matrix[
                    np.where(np.all(state_space == approximate_state, axis=1))[0] - 1,
                    -1,
                ][0]
                for i,alpha_i in enumerate(alphas):
                    data = [id, dynamic, N, M, r, alpha_i, i, first_contribution, choice_intensity, selection_intensity, p_C]

                    df = pd.DataFrame(
                        data=data,
                        columns=[
                            "UID",
                            "process",
                            "N",
                            "M",
                            "r",
                            "alpha_i",
                            "i",
                            "mutant_alpha",
                            "beta",
                            "epsilon",
                            "p_C"
                        ]
                    )
        else:
            steady_state = ludics.compute_steady_state(
                                        transition_matrix
                                    )
            id = uuid.uuid4()
            cooperation_per_player = steady_state @ state_space
            p_C = sum(cooperation_per_player) / N
            for i,alpha_i in enumerate(alphas):
                i_C = cooperation_per_player[i]
                data = [
                    [id,dynamic,N,M,r,alpha_i,i,choice_intensity,selection_intensity,aspiration,i_C,p_C]
                ]
                print(data)
                df = pd.DataFrame(
                    data=data,
                    columns=[
                        "UID",
                        "process",
                        "N",
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
    df.to_csv(file_path.parent / f"{dynamic}_mu_eq_{mu}/main.csv",
                  mode="a",
                  header=False,
                  index=False)



for N in np.array([8,7,6,5,4,3,2,1]):
    M_range = np.linspace(N, 4 * N, 10)
    r_range = np.linspace(0.5, 1.5 * N, 10)
    for M,r in itertools.product(M_range, r_range):
        alphas = public_goods_games.contribution_rules.get_deterministic_contribution_vector(
                N=N,
                contribution_rule=public_goods_games.contribution_rules.linear_contribution_rule,
                M=M,
            )
        aspiration_range = np.linspace(1, M, 10)
        selection_intensity_range = np.linspace(0, 0.99/alphas[-1], 10)
        choice_intensity_range = np.linspace(0,2,10)
        null_array = np.array([0])
        dynamic_to_iteration = {
            "aspiration": itertools.product(choice_intensity_range,null_array,aspiration_range),
            "fermi": itertools.product(choice_intensity_range,null_array,null_array),
            "introspection": itertools.product(choice_intensity_range,null_array,null_array),
            "imispection":itertools.product(choice_intensity_range, selection_intensity_range, null_array),
            "moran":itertools.product(null_array, selection_intensity_range, null_array)
        }
        
        for choice_intensity, selection_intensity, aspiration in dynamic_to_iteration[dynamic]:
            
            run_experiment(N=N, M=M, r=r, alphas=alphas, aspiration=aspiration, selection_intensity=selection_intensity, choice_intensity=choice_intensity, dynamic=dynamic, mu=mu, absorbing=absorbing)