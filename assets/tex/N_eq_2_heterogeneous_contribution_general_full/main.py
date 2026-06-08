import sympy as sym
import pathlib
import sys
import ludics
import ludics.fitness_functions

file_path = pathlib.Path(__file__)

root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))

r = sym.Symbol("r")
epsilon = sym.Symbol("epsilon")
N = 2
M = sym.Symbol("alpha_1") + sym.Symbol("alpha_2")
generic_alphas_N_eq_2 = [sym.Symbol("alpha_1"), sym.Symbol("alpha_2")]
state_space = ludics.get_state_space(N=N, k=2)

general_heterogeneous_contribution_transition_matrix_n_2 = ludics.generate_transition_matrix(
    state_space=state_space,
    fitness_function=ludics.fitness_functions.public_goods_game_fitness_function,
    compute_transition_probability=ludics.compute_moran_transition_probability,
    selection_intensity=epsilon,
    r=r,
    N=N,
    alpha=generic_alphas_N_eq_2,
)

general_heterogeneous_absorption_matrix_n_2 = ludics.calculate_absorption_matrix(
    general_heterogeneous_contribution_transition_matrix_n_2
)

with open(
    file_path.parent / "main.tex",
    "w",
) as f:
    f.write(sym.latex(sym.Matrix(general_heterogeneous_absorption_matrix_n_2)))
