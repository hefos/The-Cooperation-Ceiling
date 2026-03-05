"""
Code for the general heterogeneous Moran process

This corresponds to the model described in `main.tex`

Assume we have N ordered individuals of k types: $A_1$, ... $A_K$

We have a state v \in S = [$v_1$, .... $v_n$] as the set of types of individuals in the population, so that $v_i$ is the type of individual i. |S| = $k^N$

There is also a fitness function f: S -> $R^N$, giving us an array of the fitness of individuals
"""

import itertools
import numpy as np
import sympy as sym
import scipy


def get_state_space(N, k):
    """
    Return state space for a given N and K

    Parameters:
    -----------
    N: integer, number of individuals

    k: integer, number of possible types

    Returns:
    --------
    Array of possible states within the system, sorted based on the
    total values of the rows, in order to ensure a consistent result
    """
    state_space = np.array(list(itertools.product(range(k), repeat=N)))

    return state_space


def compute_moran_transition_probability(
    source, target, fitness_function, selection_intensity, **kwargs
):
    """
    Given two states and a fitness function, returns the transition probability

    when moving from the source state to the target state. Must move between

    states with a Hamming distance of 1. Returns 0 if Hamming distance > 1.

    Returns None if Hamming distance = 0. For an absorbing state, this will

    naturally return 0 for all off-diagonal entries, and None on the diagonal.

    This is adressed in the get_transition_matrix function.

    $\frac{\sum_{v_i = u_{i*}}{f(v_i)}}{\sum_{v_i}f(v_i)}$

    Parameters
    ----------
    source: numpy.array, the starting state

    target: numpy.array, what the source transitions to

    fitness_function: func, The fitness function which maps a state to a
    numpy.array where each entry represents the fitness of the given individual

    selection_intensity: float, the selection intensity $\epsilon$ of the
    system

    Returns
    ---------
    Float: the transition pobability from source to target
    """
    different_indices = np.where(source != target)
    if len(different_indices[0]) > 1:
        return 0
    if len(different_indices[0]) == 0:
        return None
    fitness = 1 + (selection_intensity * fitness_function(source, **kwargs))
    denominator = fitness.sum() * len(source)
    numerator = fitness[source == target[different_indices]].sum()
    return numerator / denominator


def fermi_imitation_function(delta, choice_intensity=0.5, **kwargs):
    """
    Given the fitness of the focal individual who changes action type, and the

    target individual who is being copied, as well as the choice intensity,

    returns $\phi(a_i, a_j) = \frac{1}{1 + \exp({\frac{f(a_{i}) - f(a_{j})
    }{\beta}})}$

    choice intensity is set to 0.5 by default, as is common according to:

    Xiaojian Maa, Ji Quana, Xianjia Wang (2021): Effect of reputation-based
    heterogeneous investment on cooperation in spatial public goods games


    Parameters
    -----------

    delta: float or sym.Symbol, the difference between the current fitness of
    an individual and the considered fitness.

    choice_intensity: float or sym.Symbol, a parameter which determines the
    effect the difference in fitness has on the transition probability. As
    choice_intensity goes to infinity, the probability of transitioning goes
    to $\frac{1}{2}$
    """
    return 1 / (1 + sym.E ** (choice_intensity * (delta)))


def compute_fermi_transition_probability(
    source, target, fitness_function, choice_intensity, **kwargs
):
    """
    Given two states, a fitness function, and a choice intensity, returns

    the transition probability when moving from the source state to the target

    state. Must move between states with a Hamming distance of 1. Returns 0 if

    Hamming distance > 1.

    Returns None if Hamming distance = 0. For an absorbing state, this will

    naturally return 0 for all off-diagonal entries, and None on the diagonal.

    This is adressed in the get_transition_matrix function.

    The following equation is the subject of this function:

    $\sum_{a_j=b_{i^*}}^N\frac{1}{N(N-1)}\phi(f_i(a) - f(a_j))$

    where $\phi(f_i(a) - f(a_j)) = \frac{1}{1 + \exp({\frac{f(a_{i}) - f(a_{j})
    }{\beta}})}$

    Parameters
    ----------
    source: numpy.array, the starting state

    target: numpy.array, what the source transitions to

    fitness_function: func, The fitness function which maps a state to a
    numpy.array

    choice_intensity: float or sympy.Symbol: the choice intensity of the
    function. The lower the value, the higher the probability that a player
    will choose the higher fitness strategy in $\phi$

    Returns
    ---------
    Float: the transition pobability from source to target"""

    different_indices = np.where(source != target)
    if len(different_indices[0]) > 1:
        return 0
    if len(different_indices[0]) == 0:
        return None
    fitness = fitness_function(source, **kwargs)

    changes = [
        fermi_imitation_function(
            delta=fitness[different_indices] - fitness[i],
            choice_intensity=choice_intensity,
            **kwargs,
        )
        for i in np.where(source == target[different_indices])
    ]

    scalar = 1 / (len(source) * (len(source) - 1))
    return (scalar * np.array(changes)).sum()


def compute_imitation_introspection_transition_probability(
    source, target, fitness_function, choice_intensity, selection_intensity, **kwargs
):
    """
    Given two states, a fitness function, and a choice intensity, returns

    the transition probability when moving from the source state to the target

    state in introspective imitation dynamics. Must move between states with a]

    Hamming distance of 1. Returns 0 if Hamming distance > 1.

    Returns None if Hamming distance = 0. For an absorbing state, this will

    naturally return 0 for all off-diagonal entries, and None on the diagonal.

    This is adressed in the get_transition_matrix function.

    The following equation is the subject of this function:

    $\frac{1}{N}\frac{\sum_{a_{j} = b_{I(\textbf{a}, \textbf{b})}}f_j(\textbf{a})}{\sum_{k}f_k(\textbf{a})}\phi(\Delta(f_{I(\textbf{a,b})}))$

    Parameters
    ----------
    source: numpy.array, the starting state

    target: numpy.array, what the source transitions to

    fitness_function: func, The fitness function which maps a state to a
    numpy.array

    choice_intensity: float or sympy.Symbol: the choice intensity of the
    function. The lower the value, the higher the probability that a player
    will choose the higher fitness strategy in $\phi$

    selection_intensity: float or sympy.Symbol: the selection intensity
    $\epsilon$ of the system

    Returns
    ---------
    Float: the transition pobability from source to target"""

    different_indices = np.where(source != target)
    if len(different_indices[0]) > 1:
        return 0
    if len(different_indices[0]) == 0:
        return None

    fitness = fitness_function(source, **kwargs)
    fitness_before = fitness[different_indices][0]
    fitness_after = fitness_function(target, **kwargs)[different_indices][0]

    selection_fitness = 1 + (selection_intensity * fitness)
    selection_denominator = selection_fitness.sum() * len(source)
    selection_numerator = selection_fitness[source == target[different_indices]].sum()
    selection_probability = selection_numerator / selection_denominator

    delta = fitness_before - fitness_after

    return selection_probability * fermi_imitation_function(
        delta=delta, choice_intensity=choice_intensity
    )


def compute_introspection_transition_probability(
    source, target, fitness_function, choice_intensity, number_of_strategies, **kwargs
):
    """
    Given two states, a fitness function, and a choice intensity, returns

    the transition probability when moving from the source state to the target

    state in introspective imitation dynamics. Must move between states with a]

    Hamming distance of 1. Returns 0 if Hamming distance > 1.

    Returns None if Hamming distance = 0.

    This is adressed in the get_transition_matrix function.

    The following equation is the subject of this function:

    $\frac{1}{N(m_j - 1)}\phi(f_i(a) - f_i(b))$

    Parameters
    ----------
    source: numpy.array, the starting state

    target: numpy.array, what the source transitions to

    fitness_function: func, The fitness function which maps a state to a
    numpy.array

    choice_intensity: float or sympy.Symbol: the choice intensity of the
    function. The lower the value, the higher the probability that a player
    will choose the higher fitness strategy in $\phi$

    number_of_strategies: the number of strategies available to each player in
    the population. What we call "k" in the get_state_space function

    Returns
    ---------
    Float: the transition pobability from source to target"""

    different_indices = np.where(source != target)
    if len(different_indices[0]) > 1:
        return 0
    if len(different_indices[0]) == 0:
        return None

    fitness = fitness_function(source, **kwargs)
    fitness_before = fitness[different_indices][0]
    fitness_after = fitness_function(target, **kwargs)[different_indices][0]

    selection_probability = 1 / (len(source) * ((number_of_strategies) - 1))

    delta = fitness_before - fitness_after

    return selection_probability * fermi_imitation_function(
        delta=delta, choice_intensity=choice_intensity
    )


def generate_transition_matrix(
    state_space, fitness_function, compute_transition_probability, **kwargs
):
    """
    Given a state space and a fitness function, returns the transition matrix

    for the heterogeneous Moran process.

    Parameters
    ----------
    state_space: numpy.array, the state space for the transition matrix.

    fitness_function: function, should return a size N numpy.array when passed
    a state

    compute_transition_probability: function, takes a source state, a target
    state, and a fitness function, and returns the probability of transitioning
    from the source state to the target state

    Returns
    ----------
    numpy.array: the transition matrix
    """
    N = len(state_space)
    transition_matrix = np.zeros(shape=(N, N))
    for row_index, source in enumerate(state_space):
        for col_index, target in enumerate(state_space):
            if row_index != col_index:
                try:
                    transition_matrix[row_index, col_index] = (
                        compute_transition_probability(
                            source=source,
                            target=target,
                            fitness_function=fitness_function,
                            **kwargs,
                        )
                    )
                except TypeError:
                    transition_matrix = transition_matrix.astype(object)
                    transition_matrix[row_index, col_index] = (
                        compute_transition_probability(
                            source=source,
                            target=target,
                            fitness_function=fitness_function,
                            **kwargs,
                        )
                    )
    np.fill_diagonal(transition_matrix, 1 - transition_matrix.sum(axis=1))
    return transition_matrix


def get_absorbing_state_index(state_space):
    """Given a state space, returns the indexes of the absorbing states
    (i.e, states with only one value repeated).

    Parameters
    -------------
    state_space: numpy.array, an array of states

    Returns
    --------------
    numpy.array of index values for the absorbing states"""

    absorbing_index = np.where(np.all(state_space == state_space[:, [0]], axis=1))[0]

    return absorbing_index if len(absorbing_index) >= 1 else None


def get_absorbing_states(state_space):
    """Given a state space, returns the absorbing states

    Parameters
    -----------
    state_space: numpy.array, a state space

    Returns
    ---------
    numpy.array, a list of absorbing states, in order"""

    index_array = get_absorbing_state_index(state_space=state_space)

    return (
        None
        if index_array is None
        else np.array([state_space[index] for index in index_array])
    )


def get_absorption_probabilities(
    transition_matrix, state_space, exponent_coefficient=50
):
    """Given a transition matrix and a corresponding state space

    generate the absorption probabilities. This does not yet support a

    symbolic transition matrix input

    Parameters
    -------------
    state_space: numpy.array, a state space

    transition matrix: numpy.array, a matrix of transition probabilities corresponding to the state space

    Returns
    -------------
    Dictionary of values: tuple([starting state]): [[absorbing state 1, absorption probability 1], [absorbing state 2, absorption probability 2]]
    """

    absorption_index = get_absorbing_state_index(state_space=state_space)

    absorbing_transition_matrix = np.linalg.matrix_power(
        transition_matrix, exponent_coefficient
    )

    # TODO this method of getting absorption probabilities will change, but we need to set up benchmarks first

    absorbing_collums = np.array(
        [absorbing_transition_matrix[:, index] for index in absorption_index]
    )

    combined_values = np.array(
        [
            np.ravel(np.column_stack((absorption_index, absorbing_collums[:, k])))
            for k, y in enumerate(absorbing_collums.transpose())
        ]
    )

    return {
        state_index: combined_values[state_index]
        for state_index, state in enumerate(state_space)
    }


def extract_Q(transition_matrix):
    """
    For a transition matrix, compute the corresponding matrix Q

    Parameters
    ----------
    transition_matrix: numpy.array, the transition matrix

    Returns
    -------
    np.array, the matrix Q
    """
    indices_without_1_in_diagonal = np.where(transition_matrix.diagonal() != 1)[0]
    Q = transition_matrix[
        indices_without_1_in_diagonal.reshape(-1, 1), indices_without_1_in_diagonal
    ]
    return Q


def extract_R_numerical(transition_matrix):
    """
    For a transition matrix, compute the corresponding matrix R

    Parameters
    ----------
    transition_matrix: numpy.array, the transition matrix

    Returns
    ----------
    np.array, the matrix R
    """

    # TODO merge with symbolic version and Q as function: obtain canonical form

    absorbing_states = np.isclose(np.diag(transition_matrix), 1.0)
    non_absorbing_states = ~absorbing_states
    R = transition_matrix[np.ix_(non_absorbing_states, absorbing_states)]

    return R


def extract_R_symbolic(transition_matrix):

    n = transition_matrix.shape[0]

    absorbing_states = np.array(
        [sym.simplify(transition_matrix[i, i] - 1) in (0, float(0)) for i in range(n)],
        dtype=bool,
    )

    non_absorbing_states = ~absorbing_states

    R = transition_matrix[np.ix_(non_absorbing_states, absorbing_states)]

    return R


def approximate_absorption_matrix(transition_matrix):
    """
    Given a transition matrix, NOT allowing for symbolic values,

    returns the absorption matrix


    Parameters:
    ------------

    transition_matrix: numpy.array: a transition matrix with no symbolic values


    Returns:
    -----------

    numpy.array: the probability of transitioning from

    each transitive state (row) to each absorbing state(column).
    """

    Q = extract_Q(transition_matrix=transition_matrix)

    R = extract_R_numerical(transition_matrix=transition_matrix)

    B = scipy.linalg.solve(np.eye(Q.shape[0]) - Q, R)

    return B


def calculate_absorption_matrix(transition_matrix):
    """
    Given a transition matrix, allowing for symbolic values,

    returns the absorption matrix


    Parameters:
    ------------

    transition_matrix: numpy.array: a transition matrix allowing for symbolic

    values, that has at least 1 symbolic value.

    symbolic: boolean, states whether symbolic values appear in the matrix


    Returns:
    -----------

    sympy.Matrix: the probability of transitioning from

    each transitive state (row) to each absorbing state(column).
    """

    Q = extract_Q(transition_matrix=transition_matrix)

    R = extract_R_symbolic(transition_matrix=transition_matrix)

    Q_symbolic = sym.Matrix(Q)
    R_symbolic = sym.Matrix(R)

    I = sym.eye(Q_symbolic.shape[0])
    B = (I - Q_symbolic) ** -1 * R_symbolic

    return sym.Matrix(B)


def get_deterministic_contribution_vector(contribution_rule, N, **kwargs):
    """
    Given the number of players and a function defining the contribution

    given by each player, generates the contribution vector

    for the state. The contribution vector may be stochastic, however in such

    case this function cannot guarentee the sum of entries within the

    contribution vector, and get_dirichlet_contribution_vector is better

    placed to run.

    Parameters
    ------------

    contribution_rule: a function that takes a player's index and Ns, and returns the contribution of that player.

    N: int, the number of players

    Returns
    ---------

    numpy.array: a vector of contributions by player"""

    return np.array([contribution_rule(index=x, N=N, **kwargs) for x in range(N)])


def get_dirichlet_contribution_vector(N, alpha_rule, M, scale, **kwargs):
    """
    Given the number of players and a function to generate a set of alpha
    values, returns the contribution vector for a population according to a
    dirichlet distribution. Creates a set of realisations from the dirichlet
    distribution, then applies the transformation:

    realisation * M

    in order to guarentee that players contribute according to their action,
    and that the population maximum contribution is M

    The dirichlet distribution's components all sum to 1, and therefore we can
    see that multiplying this realisation by M component-wise, we will have
    that each vector sums to M - thus we make our maximum population
    contribution equal to M. Taking the mean across these 100 realisations, we
    therefore obtain a vector who's sum is also M (proof in main.tex).

    Parameters
    ------------

    N: int, the number of players

    alpha_rule: function, takes **kwargs and returns an array of alpha values
    for the dirichlet distribution's parameters. Must return alphas with length

    M: the population maximum contribution - the contribution when all players
    give to the public good.

    scale: float - alphas are multiplied by this value. Controls variance.


    Returns
    ---------

    numpy.array: a vector of contributions by player"""

    alphas = np.array(alpha_rule(N=N, **kwargs)) * scale

    if len(alphas) != N:
        raise ValueError("Expected alphas of length", N, "but received ", len(alphas))
    else:
        realisation = np.random.dirichlet(alpha=alphas, size=100).mean(axis=0)

    return realisation * M


def approximate_steady_state(transition_matrix, tolerance=10**-6, initial_dist=None):
    """
    Returns the steady state vector of a given transition matrix that is
    entirely numeric. The steady state is approximated as the left eigenvector
    of the transition matrix of a Markov chain which corresponds to the
    eigenvalue 1.

    Parameters
    ----------
    transition_matrix - numpy.array, a transition matrix.

    tolerance - float. The maximum change when taking next_pi = pi @
    transition_matrix

    initial_dist - numpy.array: the starting state distribution.

    Returns
    ----------
    numpy.array - steady state of transition_matrix."""
    N, _ = transition_matrix.shape
    if initial_dist is None:
        pi = np.ones(N) / N
    else:
        pi = initial_dist
    while np.max(np.abs((next_pi := pi @ transition_matrix) - pi)) > tolerance:
        pi = next_pi
    return pi


def calculate_steady_state(transition_matrix):
    """
    Returns the steady state vectors of a given transition matrix. The steady
    state is calculated as the left eigenvector of the transition matrix of a
    Markov chain. This is achieved by noticing that this is equivalent to
    solving $xA = x$ is equivalent to $(A^T - I)x^T = 0$. Thus, we find the
    right-nullspace of $(A^T - I)$.

    Parameters
    ----------
    transition_matrix - numpy.array or sympy.Matrix, a transition matrix.

    Returns
    ----------
    numpy.array - steady state of transition_matrix. For the symbolic case,
    this will always be simplified.
    """
    transition_matrix = sym.Matrix(transition_matrix)

    nullspace = (transition_matrix.T - sym.eye(transition_matrix.rows)).nullspace()

    try:
        one_eigenvector = nullspace[0]
    except:
        raise ValueError("No eigenvector found")

    return np.array(sym.simplify(one_eigenvector / sum(one_eigenvector)).T)[0]
