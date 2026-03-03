import numpy as np
import sympy as sym
import sys

sys.path.append("../../../src/")
import src.main


def heterogeneous_contribution_fitness_function(
    state, omega, r, contribution_vector, **kwargs
):
    """Public goods fitness function where each player contributes H times

    their position in the state."""

    total_goods = (
        r
        * sum(
            action * contribution
            for action, contribution in zip(state, contribution_vector)
        )
        / len(state)
    )

    payoff_vector = np.array(
        [
            total_goods - (action * contribution)
            for action, contribution in zip(state, contribution_vector)
        ]
    )


root_path = (file_path / "../../../../").resolve()

sys.path.append(str(root_path))
import src.main as main
import src.fitness_functions as fitness_functions

r = sym.Symbol("r")
omega = sym.Symbol(r"\omega")
N = 2
M = sym.Symbol(r"\alpha_1") + sym.Symbol(r"\alpha_2")
generic_alphas_N_eq_2 = [sym.Symbol(r"\alpha_1"), sym.Symbol(r"\alpha_2")]
state_space = src.main.get_state_space(N=N, k=2)


general_heterogeneous_contribution_transition_matrix_N_2 = (
    src.main.generate_transition_matrix(
        state_space=state_space,
        fitness_function=heterogeneous_contribution_fitness_function,
        r=r,
        omega=omega,
        N=N,
        contribution_vector=generic_alphas_N_eq_2,
    )
)

general_heterogeneous_absorption_matrix_N_2 = src.main.generate_absorption_matrix(
    general_heterogeneous_contribution_transition_matrix_N_2, symbolic=True
)

with open(
    "../Assets/tex/N_eq_2_heterogeneous_contribution_general_full/main.tex", "w"
) as f:
    f.write(
        sym.latex(sym.simplify(sym.Matrix(general_heterogeneous_absorption_matrix_N_2)))
    )
