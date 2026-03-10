import numpy as np
import sympy as sym
import pathlib
import sys

file_path = pathlib.Path(__file__)
root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import ludics.main


def fitness_function_2_by_2(state, **kwargs):

    f = sym.Function("f")
    if (state == np.array([0, 0])).all():
        state_symbol = sym.Symbol("a")
    elif (state == np.array([0, 1])).all():
        state_symbol = sym.Symbol("b")
    elif (state == np.array([1, 0])).all():
        state_symbol = sym.Symbol("c")
    elif (state == np.array([1, 1]=ludics.main.():
        state_symbol = sym.Symbol("d")

    return np.array(
        [sym.Function(f"f_{i+1}")(state_symbol) for i, j in enumerate(state)]
    )


r = sym.Symbol("r")
epsilon = sym.Symbol("epsilon")
N = 2
state_space = ludics.main.get_state_space(N=N, k=2)
beta = sym.Symbol("beta")

transition_matrix = ludics.main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=fitness_functions.general_four_state_fitness_function,
    compute_transition_probability=main.compute_introspection_transition_probability,
    selection_intensity=beta,
    number_of_strategies=2,
)

steady_state = ludics.main.calculate_steady_state(transition_matrix)


with open(
    file_path.parent / "main.tex",
    "w",
) as f:
    f.write(sym.latex(sym.Matrix(steady_state)))
