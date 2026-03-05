import numpy as np
import sympy as sym
import pathlib
import sys

file_path = pathlib.Path(__file__)

root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions


r = sym.Symbol("r")
epsilon = sym.Symbol("epsilon")
N = 2
M = sym.Symbol("alpha_1") + sym.Symbol("alpha_2")
generic_alphas_N_eq_2 = [sym.Symbol("alpha_1"), sym.Symbol("alpha_2")]
state_space = main.get_state_space(N=N, k=2)

general_heterogeneous_contribution_transition_matrix_n_2 = main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=fitness_functions.heterogeneous_contribution_pgg_fitness_function,
    compute_transition_probability=main.compute_moran_transition_probability,
    selection_intensity=epsilon,
    r=r,
    N=N,
    contribution_vector=generic_alphas_N_eq_2,
)

general_heterogeneous_absorption_matrix_n_2 = main.calculate_absorption_matrix(
    general_heterogeneous_contribution_transition_matrix_n_2
)

with open(
    file_path.parent / "main.tex",
    "w",
) as f:
    f.write(sym.latex(sym.Matrix(general_heterogeneous_absorption_matrix_n_2)))
