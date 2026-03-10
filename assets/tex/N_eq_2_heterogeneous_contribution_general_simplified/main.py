import numpy as np
import sympy as sym
import ludics.main
import ludics.fitness_functions

r = sym.Symbol("r")
omega = sym.Symbol(r"\omega")
N = 2
M = sym.Symbol(r"\alpha_1") + sym.Symbol(r"\alpha_2")
generic_alphas_N_eq_2 = [sym.Symbol(r"\alpha_1"), sym.Symbol(r"\alpha_2")]
state_space = ludics.main.get_state_space(N=N, k=2)


general_heterogeneous_contribution_transition_matrix_N_2 = ludics.main.generate_transition_matrix(
    state_space=state_space,
    fitness_function=ludics.fitness_functions.heterogeneous_contribution_pgg_fitness_function,
    r=r,
    omega=omega,
    N=N,
    contribution_vector=generic_alphas_N_eq_2,
)

general_heterogeneous_absorption_matrix_N_2 = ludics.main.generate_absorption_matrix(
    general_heterogeneous_contribution_transition_matrix_N_2, symbolic=True
)

with open(
    "../assets/tex/N_eq_2_heterogeneous_contribution_general_full/main.tex", "w"
) as f:
    f.write(
        sym.latex(sym.simplify(sym.Matrix(general_heterogeneous_absorption_matrix_N_2)))
    )
