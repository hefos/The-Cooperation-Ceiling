import main
import numpy as np
import sympy as sym
import pytest


def test_compute_moran_transition_probability_for_trivial_fitness_function():
    """
    Tests whether the compute_moran_transition_probability

    works properly for a standard fitness function. Given two states

    (source and target, both numpy.arrays) and a trivial

    fitness function (returning 1 for all entries within the state),

    test that compute_moran_transition_probability returns the

    correct value. Here we see (0,1,0) -> (1,1,0) with a correct

    value of 1/9, and then we see a transition with Hamming distance

    2, correct value 0, and then a transition with Hamming distance

    0, correct value None."""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    selection_intensity = 0.5
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            selection_intensity=0.5,
        )
        == 1 / 9
    )
    source = np.array((0, 1, 0))
    target = np.array((1, 1, 1))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            selection_intensity=0.5,
        )
        == 0
    )
    source = np.array((0, 0, 0))
    target = np.array((0, 0, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=trivial_fitness_function,
            selection_intensity=0.5,
        )
        is None
    )


def test_compute_moran_transition_probability_for_specific_fitness_function():
    """
    Tests to see that the compute_moran_transition_probability

    function works correctly when the fitness function takes into account

    all entries within the state. Given two states (source and target, both numpy.arrays)

    and a specific fitness function (which returns the number of entries

    in the state sharing a type with a given entry (including itself)),

    test that compute_moran_transition_probability returns the

    correct value. Here we see (0,1,0) -> (1,1,0) with a correct

    value of 1/11, and then we see a transition with Hamming distance

    2, correct value 0, and then a transition with Hamming distance

    0, correct value None.

    An example for the fitness function can be seen as in the state

    f((0,0,1)) = (2, 2, 1)"""

    def fitness_function(state):
        return np.array([np.count_nonzero(state == _) for _ in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    assert main.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=fitness_function,
        selection_intensity=0.5,
    ) == 1.5 / (3 * 5.5)
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=fitness_function,
            selection_intensity=0.5,
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=fitness_function,
            selection_intensity=0.5,
        )
        is None
    )


def test_compute_moran_transition_probability_for_ordered_fitness_function():
    """
    Tests to see that the compute_moran_transition_probability

    function works correctly when the fitness function takes into account

    the position of entries within the state, both in relation to an entry and

    the position of an entry itself. Given two states (source and target, both numpy.arrays)

    and a specific fitness function (which for a given entry in position i

    (indexed from 0) will return the number of prior (self-included) entries

    with the same value as the entry + (i % 2)), tests that

    compute_moran_transition_probability returns the correct value. Here we see (0,1,0) -> (1,1,0)

    with an expected value of 4/33, and then we see a transition with Hamming

    distance 2, correct value 0, and then a transition with Hamming distance

    0, correct value None.

    An example for the fitness function can be seen as in the state

    f((0,0,1)) = (1, 3, 1)"""

    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = zero_encountered + (position % 2)
            else:
                one_encountered += 1
                fitness[position] = one_encountered + (position % 2)
        return fitness

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    assert main.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=ordered_fitness_function,
        selection_intensity=0.5,
    ) == 2 / (3 * 5.5)
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=ordered_fitness_function,
            selection_intensity=0.5,
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=ordered_fitness_function,
            selection_intensity=0.5,
        )
        is None
    )


def test_compute_moran_transition_probability_for_symbolic_fitness_function():
    """
    Tests for whether compute_transition_prbability returns the correct

    value for a fitness function which works symbolically.

    Given two states (source and target, both numpy.arrays) and a

    symbolic fitness function (i.e, replacing 1 with x and 0 with y, via

    sympy), tests that compute_moran_transition_probability returns the correct

    value. tests (0,1,0) -> (1,1,0), with correct value

    x / ((3 * x) + (6 * y)), then transitions with Hamming distances

    2 and 0, with correct values 0 and None respectively."""

    def symbolic_fitness_function(state):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    x = sym.symbols("x")
    y = sym.symbols("y")
    epsilon = sym.Symbol("\epsilon")
    assert main.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        selection_intensity=epsilon,
    ) == (1 + epsilon * x) / ((3 * (1 + epsilon * x)) + (6 * (1 + epsilon * y)))
    source = np.array((0, 1, 1))
    target = np.array((0, 0, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=symbolic_fitness_function,
            selection_intensity=epsilon,
        )
        == 0
    )
    source = np.array((1, 1, 0))
    target = np.array((1, 1, 0))
    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=symbolic_fitness_function,
            selection_intensity=epsilon,
        )
        is None
    )
    source = np.array((0, 1))
    target = np.array((0, 0))
    assert main.compute_moran_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        selection_intensity=epsilon,
    ) == (1 + epsilon * y) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y))

    source = np.array((0, 1))
    target1 = np.array((0, 0))
    target2 = np.array((1, 1))
    assert 1 - main.compute_moran_transition_probability(
        source=source,
        target=target1,
        fitness_function=symbolic_fitness_function,
        selection_intensity=epsilon,
    ) - main.compute_moran_transition_probability(
        source=source,
        target=target2,
        fitness_function=symbolic_fitness_function,
        selection_intensity=epsilon,
    ) == (
        1
        - ((1 + epsilon * y) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y)))
        - (1 + epsilon * x) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y))
    )


def test_compute_moran_transition_probability_for_kwargs_fitness_function():
    """
    tests the compute_moran_transition_probability function for

    a fitness function which takes kwargs
    """

    def kwargs_fitness_function(state, c, r):
        return np.array([c if individual == 1 else r for individual in state])

    source = np.array((0, 1, 0))
    target = np.array((1, 1, 0))
    c = 2
    r = 3

    expected_transition_probability = 2 / 21

    assert (
        main.compute_moran_transition_probability(
            source=source,
            target=target,
            fitness_function=kwargs_fitness_function,
            selection_intensity=0.5,
            c=c,
            r=r,
        )
        == expected_transition_probability
    )


def test_generate_state_space_for_N_eq_3_and_k_eq_2():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 3, k = 2.
    """
    k = 2
    N = 3
    expected_state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    obtained_state_space = main.get_state_space(N=N, k=k)
    np.testing.assert_array_equal(
        sorted(tuple(x) for x in obtained_state_space),
        sorted(tuple(x) for x in expected_state_space),
    )


def test_generate_state_space_for_N_eq_3_and_k_eq_1():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 3, k = 1.
    """
    k = 1
    N = 3
    expected_state_space = [
        (0, 0, 0),
    ]
    obtained_state_space = main.get_state_space(N=N, k=k)
    np.testing.assert_allclose(
        sorted(expected_state_space), sorted(obtained_state_space)
    )


def test_generate_state_space_for_N_eq_1_and_k_eq_3():
    """
    Given a value of $N$: the number of individuals and a value of $k$: the
    number of types generate $S = [1, ..., k] ^ N$.

    This tests this for N = 1, k = 3.
    """
    k = 3
    N = 1
    expected_state_space = [
        (0,),
        (1,),
        (2,),
    ]
    obtained_state_space = main.get_state_space(N=N, k=k)
    np.testing.assert_allclose(
        sorted(expected_state_space), sorted(obtained_state_space)
    )


def test_generate_transition_matrix_for_trivial_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a trivial fitness function an a state space N = 3, K = 2.
    """

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    expected_transition_matrix = np.array(
        [
            [0.55555556, 0.0, 0.0, 0.11111111, 0.11111111, 0.0, 0.22222222, 0.0],
            [0.0, 0.55555556, 0.0, 0.11111111, 0.0, 0.11111111, 0.22222222, 0.0],
            [0.0, 0.0, 0.55555556, 0.0, 0.11111111, 0.11111111, 0.22222222, 0.0],
            [0.11111111, 0.11111111, 0.0, 0.55555556, 0.0, 0.0, 0.0, 0.22222222],
            [0.11111111, 0.0, 0.11111111, 0.0, 0.55555556, 0.0, 0.0, 0.22222222],
            [0.0, 0.11111111, 0.11111111, 0.0, 0.0, 0.55555556, 0.0, 0.22222222],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(
        main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=trivial_fitness_function,
            compute_transition_probability=main.compute_moran_transition_probability,
            selection_intensity=0.5,
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_ordered_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a fitness function based on order (see test_compute_transition_matrix_for_ordered_fitness_function

    for a description of the fitness function) and a state space N = 3, K = 2.
    """

    def ordered_fitness_function(state):
        fitness = np.array([0 for _ in state])
        zero_encountered = 0
        one_encountered = 0
        for position, value in enumerate(state):
            if value == 0:
                zero_encountered += 1
                fitness[position] = zero_encountered + (position % 2)
            else:
                one_encountered += 1
                fitness[position] = one_encountered + (position % 2)
        return fitness

    state_space = np.array(
        [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 1),
        ]
    )
    expected_transition_matrix = np.array(
        [
            [0.57037037, 0.0, 0.0, 0.0962963, 0.0962963, 0.0, 0.23703704, 0.0],
            [0.0, 0.54814815, 0.0, 0.11851852, 0.0, 0.11851852, 0.21481481, 0.0],
            [0.0, 0.0, 0.57037037, 0.0, 0.0962963, 0.0962963, 0.23703704, 0.0],
            [0.0962963, 0.0962963, 0.0, 0.57037037, 0.0, 0.0, 0.0, 0.23703704],
            [0.11851852, 0.0, 0.11851852, 0.0, 0.54814815, 0.0, 0.0, 0.21481481],
            [0.0, 0.0962963, 0.0962963, 0.0, 0.0, 0.57037037, 0.0, 0.23703704],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=ordered_fitness_function,
            compute_transition_probability=main.compute_moran_transition_probability,
            selection_intensity=0.3,
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_different_state_space():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a state space N = 2, K = 3.
    """

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    state_space = np.array(
        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
    )
    expected_transition_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.5, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=trivial_fitness_function,
            compute_transition_probability=main.compute_moran_transition_probability,
            selection_intensity=0.5,
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_symbolic_fitness_function():
    """
    Tests whether generate_transition_matrix returns the correct matrix

    for a symbolic fitness function function based on (see test_compute_transition_matrix_for_symbolic_fitness_function

    for a description of the fitness function) and a smaller state space N = 2, K = 2.
    """

    def symbolic_fitness_function(state):
        return np.array(
            [
                sym.Symbol("x") if individual == 1 else sym.Symbol("y")
                for individual in state
            ]
        )

    state_space = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])

    x = sym.Symbol("x")
    y = sym.Symbol("y")
    epsilon = sym.Symbol("\epsilon")

    expected_transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [
                (1 + epsilon * y) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y)),
                (
                    1
                    - (
                        (1 + epsilon * y)
                        / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y))
                    )
                    - (1 + epsilon * x)
                    / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y))
                ),
                0,
                (1 + epsilon * x) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y)),
            ],
            [
                (1 + epsilon * y) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y)),
                0,
                (
                    1
                    - (
                        (1 + epsilon * y)
                        / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y))
                    )
                    - (1 + epsilon * x)
                    / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y))
                ),
                (1 + epsilon * x) / (2 * (1 + epsilon * x) + 2 * (1 + epsilon * y)),
            ],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(
        main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=symbolic_fitness_function,
            compute_transition_probability=main.compute_moran_transition_probability,
            selection_intensity=epsilon,
        ),
        expected_transition_matrix,
    )


def test_generate_transition_matrix_for_kwargs_fitness_function():
    """
    tests the generate_transition_matrix function for

    a fitness function which takes kwargs
    """

    def kwargs_fitness_function(state, c, r):
        return np.array([c if individual == 1 else r for individual in state])

    state_space = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    c = 1
    r = 4
    expected_transition_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.33333333, 0.5, 0.0, 0.16666667],
            [0.33333333, 0.0, 0.5, 0.16666667],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(
        expected_transition_matrix,
        main.generate_transition_matrix(
            state_space=state_space,
            fitness_function=kwargs_fitness_function,
            compute_transition_probability=main.compute_moran_transition_probability,
            selection_intensity=0.5,
            c=c,
            r=r,
        ),
    )


def test_get_absorbing_state_index_for_N_eq_2_k_eq_4():
    """
    Tests that get_absorbing_state_index correctly identifies

    the absorbing states in a standard state space"""

    state_space = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ]
    )

    expected_absorbing_states = np.array([0, 5, 10, 15])

    np.testing.assert_array_equal(
        expected_absorbing_states,
        main.get_absorbing_state_index(state_space=state_space),
    )


def test_get_absorbing_state_index_for_no_absorbing_states():
    """
    Tests that get_absorbing_state_index correctly identifies

    that there are no absorbing states in a given state

    space"""

    non_absorbing_state_space = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
        ]
    )

    expected_absorbing_states = None

    assert expected_absorbing_states == main.get_absorbing_state_index(
        state_space=non_absorbing_state_space
    )


def test_get_absorbing_state_index_for_symbolic_state_space():
    """Tests the get_absorbing_state_index function for
    a symbolic state space."""

    A = sym.Symbol("A")
    B = sym.Symbol("B")

    symbolic_state_space = np.array(
        [
            [A, B],
            [A, A],
            [B, B],
            [B, A],
        ]
    )

    expected_absorbing_states = np.array([1, 2])
    np.testing.assert_array_equal(
        expected_absorbing_states,
        main.get_absorbing_state_index(state_space=symbolic_state_space),
    )


def test_get_absorbing_states_for_standard_state_space():
    """Tests the get_absorbing_states function

    for a standard state space"""

    state_space = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ]
    )

    expected_absorbing_states = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    )

    np.testing.assert_array_equal(
        expected_absorbing_states,
        main.get_absorbing_states(state_space=state_space),
    )


def test_get_absorbing_states_for_no_absorbing_states():
    """
    Tests that get_absorbing_states correctly identifies

    that there are no absorbing states in a given state

    space"""

    non_absorbing_state_space = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
        ]
    )

    assert main.get_absorbing_states(state_space=non_absorbing_state_space) is None


def test_get_absorbing_states_for_symbolic_state_space():
    """Tests the get_absorbing_states function for
    a symbolic state space."""

    A = sym.Symbol("A")
    B = sym.Symbol("B")

    symbolic_state_space = np.array(
        [
            [A, B],
            [A, A],
            [B, B],
            [B, A],
        ]
    )
    expected_absorbing_states = np.array(
        [
            [A, A],
            [B, B],
        ]
    )

    np.testing.assert_array_equal(
        expected_absorbing_states,
        main.get_absorbing_states(state_space=symbolic_state_space),
    )


def test_get_absorption_probabilities_for_trivial_transition_matrix_and_standard_state_space():
    """Tests the get_absorption_probabilities function for a transition matrix that guarentees absorption into a certain absorbing state."""

    state_space = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 0],
        ]
    )

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [1 / 2, 1 / 2, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1 / 2, 1 / 2],
        ]
    )

    expected = {
        0: np.array([0, 1, 2, 0], dtype=float),
        1: np.array([0, 1, 2, 0], dtype=float),
        2: np.array([0, 0, 2, 1], dtype=float),
        3: np.array([0, 0, 2, 1], dtype=float),
    }

    actual = main.get_absorption_probabilities(
        transition_matrix=transition_matrix,
        state_space=state_space,
        exponent_coefficient=50,
    )

    for key in expected:
        np.testing.assert_allclose(expected[key], actual[key])


def test_extract_Q_for_numeric_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with numeric values

    and no symbolic values. We take N=2 and K=2"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.25, 0.3, 0.45],
            [0, 0, 1, 0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    expected_Q = np.array(
        [
            [0.25, 0.45],
            [0.25, 0.25],
        ]
    )

    np.testing.assert_array_equal(
        expected_Q, main.extract_Q(transition_matrix=transition_matrix)
    )


def test_extract_Q_for_symbolic_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with just symbolic values. We take N=2 and K=2
    """

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, A, B, B],
            [0, 0, 1, 0],
            [C + A, C, B, C + A],
        ]
    )

    expected_Q = np.array(
        [
            [A, B],
            [C, C + A],
        ]
    )

    np.testing.assert_array_equal(
        expected_Q, main.extract_Q(transition_matrix=transition_matrix)
    )


def test_extract_Q_for_mixed_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with symbolic values

    and numeric values. We take N=2 and K=2"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, A, B, B / 3],
            [0, 0, 1, 0],
            [C + A, 0.5, B, C + 0.2],
        ]
    )

    expected_Q = np.array(
        [
            [A, B / 3],
            [0.5, C + 0.2],
        ]
    )

    np.testing.assert_array_equal(
        expected_Q, main.extract_Q(transition_matrix=transition_matrix)
    )


def test_extract_R_numerical_for_numeric_transition_matrix():
    """
    Tests the extract_R_numerical function for a transition matrix with numeric

    values and no symbolic values. We take N=2 and K=2"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.25, 0.3, 0.45],
            [0, 0, 1, 0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    expected_R = np.array(
        [
            [0, 0.3],
            [0.25, 0.25],
        ]
    )

    np.testing.assert_array_equal(
        expected_R, main.extract_R_numerical(transition_matrix=transition_matrix)
    )


def test_extract_R_symbolic_for_mixed_transition_matrix():
    """
    Tests the extract_R_symbolic function for a transition matrix with symbolic values

    and numeric values. We take N=2 and K=2"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0.5, A, B, B / 3],
            [0, 0, 1, 0],
            [C + A, 0.2, 0.3, C],
        ]
    )

    expected_R = np.array(
        [
            [0.5, B],
            [C + A, 0.3],
        ]
    )

    np.testing.assert_array_equal(
        expected_R, main.extract_R_symbolic(transition_matrix=transition_matrix)
    )


def test_extract_R_symbolic_for_purely_symbolic_transition_matrix():
    """
    Tests the extract_Q function for a transition matrix with symbolic values

    and no numeric values. We take N=2 and K=2"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, A, B, B],
            [0, 0, 1, 0],
            [C + A, C, B, C + A],
        ]
    )

    expected_R = np.array(
        [
            [0, B],
            [C + A, B],
        ]
    )

    np.testing.assert_array_equal(
        expected_R, main.extract_R_symbolic(transition_matrix=transition_matrix)
    )


def test_approximate_absorption_matrix_for_numeric_transition_matrix():
    """
    Tests the approximate_absorption_matrix function for an entirely

    numeric transition matrix"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0.25, 0.75, 0],
            [0.3, 0, 0, 0.7],
        ]
    )

    expected_absorption_matrix = np.array([[0, 1], [1, 0]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        main.approximate_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_calculate_absorption_matrix_for_symbolic_transition_matrix():
    """
    Tests the calculate_absorption_matrix function for an symbolic

    transition matrix"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, A, B, 0],
            [C, C, 0, 0],
        ]
    )

    expected_absorption_matrix = np.array([[0, A / (1 - B)], [C, C]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        main.calculate_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_approximate_absorption_matrix_for_standard_transition_matrix():
    """
    Tests the approximate_absorption_matrix function for an entirely

    numeric transition matrix"""

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0.25, 0.75, 0],
            [0.3, 0, 0, 0.7],
        ]
    )

    expected_absorption_matrix = np.array([[0, 1], [1, 0]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        main.approximate_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_calculate_absorption_matrix_for_standard_transition_matrix():
    """
    Tests the calculate_absorption_matrix function for a standard
    symbolic transition matrix"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, A, B, 0],
            [C, C, 0, 0],
        ]
    )

    expected_absorption_matrix = np.array([[0, A / (1 - B)], [C, C]])

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix,
        main.calculate_absorption_matrix(transition_matrix=transition_matrix),
    )


def test_generate_absorption_matrix_functions_accuracy_for_r_values():
    """Tests that the equations generated by the symbolic

    generate_absorption_matrix function will give the correct value for various

    r values"""

    def public_goods_fitness_function(state, alpha, r, omega):
        number_of_contributors = state.sum()
        big_bit = r * alpha * (number_of_contributors) / (len(state))
        payoff = np.array([big_bit - alpha * x for x in state])
        return (1) + (omega * payoff)

    r = sym.Symbol("r")
    alpha = sym.Symbol("a")
    omega = sym.Symbol("w")

    r_test_values = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    expected_results = [
        0.2883263,
        0.28963365,
        0.29085873,
        0.29200983,
        0.29309407,
        0.29411765,
        0.29508594,
        0.29600366,
    ]

    state_space = main.get_state_space(N=3, k=2)

    transition_matrix = main.generate_transition_matrix(
        state_space=state_space,
        fitness_function=public_goods_fitness_function,
        compute_transition_probability=main.compute_moran_transition_probability,
        selection_intensity=0.5,
        r=r,
        alpha=alpha,
        omega=omega,
    )

    absorption_matrix = main.calculate_absorption_matrix(transition_matrix)

    symbolic_expression = sym.lambdify(
        (r, alpha, omega), sym.Matrix(absorption_matrix)[0, 1], "numpy"
    )

    obtained_results = symbolic_expression(r_test_values, 2, 0.2)

    np.testing.assert_array_almost_equal(expected_results, obtained_results)


def test_calculate_absorption_matrix_for_5_by_5_symbolic_transition_matrix():
    """
    Tests the calculate_absorption_matrix function for a 5x5 symbolic

    transition matrix"""

    A = sym.Symbol("A")
    B = sym.Symbol("B")
    C = sym.Symbol("C")
    D = sym.Symbol("D")

    transition_matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [A, 1 / 3, B, 0, 0],
            [0, A, 0, C, 0],
            [0, 0, C, D, 1 / 3],
            [0, 0, 0, 0, 1],
        ]
    )

    Q = sym.Matrix(np.array([[1 / 3, B, 0], [A, 0, C], [0, C, D]]))

    I = sym.Matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    R = sym.Matrix(np.array([[A, 0], [0, 0], [0, 1 / 3]]))

    expected_absorption_matrix = ((I - Q) ** -1) * R

    obtained_absorption_matrix = main.calculate_absorption_matrix(
        transition_matrix=transition_matrix
    )

    zero_matrix = sym.Matrix(np.zeros((3, 2)))

    np.testing.assert_array_almost_equal(
        expected_absorption_matrix - obtained_absorption_matrix, zero_matrix
    )


def test_get_deterministic_contribution_vector_for_homogeneous_case():
    """Tests the get_deterministic_contribution_vector function for a homogeneous
    case"""

    def homogeneous_contribution_rule(index, N):
        """The contribution of player i (indexed from 1) is always equal to 2

        This is a test that shows the ability of get_deterministic_contribution_vector to
        handle standard contribution rules, not relying on both action and index."""

        return 2

    N = 3

    expected_contribution_vector = np.array([2, 2, 2])

    np.testing.assert_array_equal(
        main.get_deterministic_contribution_vector(
            contribution_rule=homogeneous_contribution_rule, N=N
        ),
        expected_contribution_vector,
    )


def test_get_deterministic_contribution_vector_for_heterogeneous_case():
    """Tests the get_deterministic_contribution_vector function for a homogeneous
    case"""

    def heterogeneous_contribution_rule(index, N):
        """The contribution of player i (indexed from 1) is given by:

        2 * i.

        For example, player 2 performing action 3 would contribute 12

        This is a test that shows the use of the (index)
        parameter for the required contribution_rule function in get_deterministic_contribution_vector
        """

        return 2 * (index + 1)

    N = 3

    expected_contribution_vector = np.array([2, 4, 6])

    np.testing.assert_array_equal(
        main.get_deterministic_contribution_vector(
            contribution_rule=heterogeneous_contribution_rule, N=N
        ),
        expected_contribution_vector,
    )


def test_get_deterministic_contribution_vector_for_kwargs_case():
    """Tests the get_deterministic_contribution_vector function for a homogeneous
    case"""

    def homogeneous_contribution_rule(index, N, discount):
        """The contribution of player i (indexed from 1), with a discount
        value <2, is given by:

        (2-discount) * i.

        For example, player 2 with 0.5 discount would contribute 3

        This is a test that shows the use of **kwargs arguments in a
        contribution rule passde to get_deterministic_contribution_vector"""

        return (2 - discount) * (index + 1)

    N = 3

    expected_contribution_vector = np.array([1, 2, 3])

    np.testing.assert_array_equal(
        main.get_deterministic_contribution_vector(
            contribution_rule=homogeneous_contribution_rule, N=N, discount=1
        ),
        expected_contribution_vector,
    )


def test_get_dirichlet_contribution_vector_for_trivial_alpha_rule_and_large_repitions():
    """
    Tests the get_dirichlet_contribution_vector function for a trivial alpha
    rule in which all alphas are equal to 2. In this case, all the means should
    be equal (with a margin of error due to the stochastic nature of the
    function). We also test the stochasticity of the function by testing across
    100 iterations with a different seed.

    With np.random.seed(1), we expect to obtain
    [4.14781218, 4.12911919, 3.72306863]

    With np.random.seed(5), we expect to obtain a mean over 100 iterations of
    [3.98697183, 3.99898138, 4.01404679]

    The empirical mean would be [4,4,4]"""

    def trivial_alpha_rule(N):

        return np.array([2 for _ in range(N)])

    np.random.seed(1)
    M = 12
    N = 3
    scale = 1

    expected_return = np.array([4.14781218, 4.12911919, 3.72306863])

    actual_return = main.get_dirichlet_contribution_vector(
        N=N, alpha_rule=trivial_alpha_rule, M=M, scale=scale
    )

    np.random.seed(5)

    expected_return_iteration = np.array([3.98697183, 3.99898138, 4.01404679])

    actual_return_iteration = np.array(
        [
            main.get_dirichlet_contribution_vector(
                N=N, alpha_rule=trivial_alpha_rule, M=M, scale=scale
            )
            for _ in range(100)
        ]
    ).mean(axis=0)

    np.testing.assert_allclose(actual_return_iteration, expected_return_iteration)

    np.testing.assert_allclose(actual_return, expected_return)


def test_get_dirichlet_contribution_vector_for_linear_alpha_rule_and_large_repitions():
    """
    Tests the get_dirichlet_contribution_vector function for a linear alpha
    rule. In this case, all the means should be equal (with a margin of error
    due to the stochastic nature of the function). We also test the
    stochasticity of the function by testing across 100 iterations with a
    different seed.

    With np.random.seed(1), we expect to obtain
    [1.9269376 , 3.90995069, 6.16311171]

    With np.random.seed(4), we expect to obtain a mean over 100 iterations of
    [1.96018551, 4.0127389 , 6.02707559]

    The empirical mean would be [2,4,6]"""

    def linear_alpha_rule(N):
        """Returns a numpy.array 1, 2, ..., N. This test allows us to see that
        alphas are not all treated as the same, but without adding the extra
        complications of long computations."""
        return np.array([_ for _ in range(1, N + 1)])

    M = 12
    N = 3
    np.random.seed(1)
    scale = 1

    expected_return = np.array([1.9269376, 3.90995069, 6.16311171])

    actual_return = main.get_dirichlet_contribution_vector(
        N=N, alpha_rule=linear_alpha_rule, M=M, scale=1
    )

    np.testing.assert_allclose(actual_return, expected_return)

    np.random.seed(4)

    expected_return_iteration = np.array([1.96018551, 4.0127389, 6.02707559])

    actual_return_iteration = np.array(
        [
            main.get_dirichlet_contribution_vector(
                N=N, alpha_rule=linear_alpha_rule, M=M, scale=scale
            )
            for _ in range(100)
        ]
    ).mean(axis=0)

    np.testing.assert_allclose(actual_return_iteration, expected_return_iteration)


def test_get_dirichlet_contribution_vector_for_kwargs_alpha_rule_and_large_repitions():
    """
    Tests the get_dirichlet_contribution_vector function for an alpha
    rule in which all alphas are equal to index + bonus, in order to check that
    kwargs are properly passed to the alpha_rule function. We also test the
    stochasticity of the function by testing across 100 iterations with a
    different seed.

    With np.random.seed(1), we expect to obtain
    [6.59821129, 11.40493245, 17.99685625]

    With np.random.seed(3), we expect to obtain a mean over 100 iterations of
    [5.99449831, 11.97708597, 18.02841572]

    The empirical mean would be [6,12,18]

    """

    def kwargs_alpha_rule(N, bonus):
        """Returns a numpy.array 1, 2, ..., N. This test allows us to see that
        alphas are not all treated as the same, but without adding the extra
        complications of long computations."""
        return np.array([_ * bonus for _ in range(1, N + 1)])

    M = 36
    bonus = 3
    N = 3
    np.random.seed(1)
    scale = 1

    expected_return = np.array([6.59821129, 11.40493245, 17.99685625])
    actual_return = main.get_dirichlet_contribution_vector(
        N=N, alpha_rule=kwargs_alpha_rule, M=M, scale=scale, bonus=bonus
    )

    np.testing.assert_allclose(actual_return, expected_return)

    np.random.seed(3)

    expected_return_iteration = np.array([5.99449831, 11.97708597, 18.02841572])

    actual_return_iteration = np.array(
        [
            main.get_dirichlet_contribution_vector(
                N=N, alpha_rule=kwargs_alpha_rule, M=M, scale=scale, bonus=bonus
            )
            for _ in range(100)
        ]
    ).mean(axis=0)

    np.testing.assert_allclose(actual_return_iteration, expected_return_iteration)


def test_get_dirichlet_contribution_vector_raises_type_error_for_few_alphas():
    """
    Tests whether the get_dirichlet_contribution_vector function correctly
    raises a type error in the case that the number of alphas returned by the alpha_rule function is less than the length of the state.
    """

    def small_alpha_rule(N):

        return np.array([2 for _ in range(N - 1)])

    N = 3
    scale = 1

    with pytest.raises(ValueError):
        main.get_dirichlet_contribution_vector(
            N=N, alpha_rule=small_alpha_rule, scale=scale, M=15
        )


def test_get_dirichlet_contribution_vector_raises_type_error_for_many_alphas():
    """
    Tests whether the get_dirichlet_contribution_vector function correctly
    raises a type error in the case that the number of alphas returned by the
    alpha_rule function is more than the length of the state."""

    def small_alpha_rule(N):

        return np.array([2 for _ in range(N + 1)])

    N = 5
    scale = 1

    with pytest.raises(ValueError):
        main.get_dirichlet_contribution_vector(
            N=N, alpha_rule=small_alpha_rule, scale=scale, M=15
        )


def test_approximate_steady_state_for_trivial_transition_matrix():
    """
    Tests approximate_steady_state for a trivial transition matrix
    """

    numeric_matrix = np.array([[0.4, 0.6], [0.4, 0.6]])

    expected_numeric_output = np.array([0.4, 0.6])

    np.testing.assert_allclose(
        expected_numeric_output, main.approximate_steady_state(numeric_matrix)
    )


def test_approximate_steady_state_for_absorbing_transition_matrix():
    """
    Tests approximate_steady_state for an absorbing transition matrix
    """

    numeric_matrix = np.array(
        [[1, 0, 0, 0], [0.3, 0.6, 0, 0.1], [0, 0.3, 0.4, 0.3], [0.2, 0.1, 0.1, 0.6]]
    )

    expected_numeric_output = np.array([1, 0, 0, 0])

    np.testing.assert_allclose(
        expected_numeric_output,
        main.approximate_steady_state(numeric_matrix),
        rtol=1**-5,
    )


def test_calculate_steady_state_for_trivial_transition_matrix():
    """
    Tests whether the calculate_steady_state function returns the correct matrix for
    a 2x2 transition matrix with the simple form [[p, 1-p], [p,1-p]]"""

    p = sym.Symbol("p")
    q = sym.Symbol("q")

    symbolic_matrix = sym.Matrix(
        [[0.5 + p + q, 0.5 - p - q], [0.5 + p + q, 0.5 - p - q]]
    )

    expected_symbolic_output = np.array([0.5 + p + q, 0.5 - p - q])

    np.testing.assert_array_almost_equal(
        expected_symbolic_output, main.calculate_steady_state(symbolic_matrix)
    )


def test_calculate_steady_state_for_absorbing_symbolic_transition_matrix():
    """
    Tests whether the calculate_steady_state function still returns the
    correct value if the matrix passed to it is absorbing and symbolic. It
    should return a steady state corresponding to just the absorbing state of
    the transition matrix"""

    p = sym.Symbol("p")

    transition_matrix = np.array([[p, 1 - p - 0.1, 0.1], [0, 1, 0], [0.6, 0.2, 0.2]])

    expected_output = np.array([0, 1, 0])

    np.testing.assert_array_equal(
        expected_output, main.calculate_steady_state(transition_matrix)
    )


def test_calculate_steady_state_errors():
    """
    Tests whether the errors in get_steady_state are correctly raised for:
    Symbolic matrix with no real solutions"""
    p = sym.Symbol("p")

    test_no_solution_matrix_symbolic = np.array([[p, 0], [0, p]])

    with pytest.raises(ValueError):
        main.calculate_steady_state(test_no_solution_matrix_symbolic)


def test_fermi_imitation_function_for_numeric_value():
    """
    Tests whether the fermi_imitation_function returns the desired value for
    numeric values of delta and selection_intesntiy"""

    delta = 3
    choice_intensity = 0.5

    expected_fermi_value = 0.1824255238

    actual_fermi_value = main.fermi_imitation_function(
        delta=delta, choice_intensity=choice_intensity
    )

    np.testing.assert_almost_equal(expected_fermi_value, actual_fermi_value)


def test_fermi_imitation_function_for_symbolic_value():
    """
    Tests whether the fermi_imitation_function returns the desired expression
    for symbolic values of delta and selection_intensity"""

    delta = sym.Symbol("Delta")
    choice_intensity = sym.Symbol("beta")

    expected_fermi_value = 1 / (1 + sym.E ** (delta * choice_intensity))

    actual_fermi_value = main.fermi_imitation_function(
        delta=delta, choice_intensity=choice_intensity
    )

    assert expected_fermi_value == actual_fermi_value


def test_compute_fermi_transition_probability_for_trivial_fitness_function():
    """
    Tests whether the compute_fermi_transition_probability function returns the
    desired value for a trivial fitness function"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source = np.array([0, 1])
    target = np.array([1, 1])
    choice_intensity = 0.5

    actual_probability = main.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    expected_probability = 0.25

    assert expected_probability == actual_probability


def test_compute_fermi_transition_probability_for_symbolic_fitness_function():
    """
    Tests whether the compute_fermi_transition_probability function returns the
    correct expression for a symbolic fitness function"""

    def symbolic_fitness_function(state, **kwargs):
        return np.array([sym.Symbol("x") if i == 0 else sym.Symbol("y") for i in state])

    source = np.array([0, 1, 1])
    target = np.array([1, 1, 1])
    beta = sym.Symbol("beta")

    actual_probability = main.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        choice_intensity=beta,
    )

    x = sym.Symbol("x")
    y = sym.Symbol("y")

    expected_probability = (1 / 6) * (
        1 / (1 + sym.E ** (((x - y) * beta))) + 1 / (1 + sym.E ** (((x - y) * beta)))
    )

    assert actual_probability == expected_probability


def test_compute_fermi_transition_probability_for_infeasible_states_and_no_change():
    """
    Tests whether compute_fermi_transition_probability returns the correct
    values when the state transition is not of hamming distance 1"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source1 = np.array([0, 1])
    target1 = np.array([1, 0])
    choice_intensity = 0.5

    actual_probability1 = main.compute_fermi_transition_probability(
        source=source1,
        target=target1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    expected_probability1 = 0

    assert expected_probability1 == actual_probability1

    source2 = np.array([0, 1])
    target2 = np.array([0, 1])

    actual_probability2 = main.compute_fermi_transition_probability(
        source=source2,
        target=target2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
    )

    assert actual_probability2 is None

    _ = trivial_fitness_function(source1)  # prevents unused function warning


def test_compute_fermi_transition_probability_for_impossible_transition():
    """Tests compute_fermi_transition_probability for a
    transition which introduces a new strategy to the population"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 2, 0])

    choice_intensity = 0.5
    selection_intensity = 0.5

    actual_probability = main.compute_fermi_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )

    expected_probability = 0.0

    np.testing.assert_almost_equal(actual_probability, expected_probability)


def test_compute_imitation_introspection_transition_probability_for_trivial_fitenss_function():
    """
    Tests that the compute_imitation_introspection_transition_probability
    function returns the correct value for a trivial fitness function."""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 1, 0])

    selection_intensity = 0.1
    choice_intensity = 0.8

    actual_probability = main.compute_imitation_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )

    expected_probability = 0.08999667145

    np.testing.assert_almost_equal(
        actual_probability, expected_probability, err_msg=actual_probability
    )


def test_compute_imitation_introspection_transition_probability_for_symbolic_fitness_function():
    """
    Tests whether the compute_imitation_introspection_transition_probability
    function returns the correct expression for a symbolic fitness function"""

    def symbolic_fitness_function(state, **kwargs):
        return np.array([sym.Symbol("x") if i == 0 else sym.Symbol("y") for i in state])

    source = np.array([0, 1, 1, 0, 0])
    target = np.array([1, 1, 1, 0, 0])
    beta = sym.Symbol("\beta")
    epsilon = sym.Symbol("\epsilon")

    actual_probability = main.compute_imitation_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        choice_intensity=beta,
        selection_intensity=epsilon,
    )

    x = sym.Symbol("x")
    y = sym.Symbol("y")
    fy = 1 + epsilon * y
    fx = 1 + epsilon * x

    expected_probability = (
        (1 / 5)
        * (2 * fy)
        * (1 / ((2 * fy) + (3 * fx)))
        * (1 / (1 + sym.E ** ((x - y) * beta)))
    )

    assert sym.simplify(actual_probability - expected_probability) == 0


def test_compute_imitation_introspection_transition_probability_for_infeasible_states_and_no_change():
    """
    Tests whether compute_imitation_introspection_transition_probability returns the correct
    values when the state transition is not of hamming distance 1"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source1 = np.array([0, 1])
    target1 = np.array([1, 0])
    choice_intensity = 0.5
    selection_intensity = 0.8

    actual_probability1 = main.compute_imitation_introspection_transition_probability(
        source=source1,
        target=target1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )

    expected_probability1 = 0

    assert expected_probability1 == actual_probability1

    source2 = np.array([0, 1])
    target2 = np.array([0, 1])

    actual_probability2 = main.compute_imitation_introspection_transition_probability(
        source=source2,
        target=target2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )

    assert actual_probability2 is None

    _ = trivial_fitness_function(source1)  # prevents unused function warning


def test_compute_imitation_introspection_for_impossible_transition():
    """Tests compute_imitation_introspection_transition_probability for a
    transition which introduces a new strategy to the population"""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 2, 0])

    choice_intensity = 0.5
    selection_intensity = 0.8

    actual_probability = main.compute_imitation_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )

    expected_probability = 0.0

    np.testing.assert_almost_equal(actual_probability, expected_probability)


def test_compute_imitation_introspection_for_global_transition():
    """Tests compute_imitation_introspection_transition_probability for a
    transition which gives a different fitness to the changing player in the
    new state."""

    def heterogeneous_fitness_function(state, **kwargs):
        return np.array([i + np.sum(state) for i in state])

    source = np.array([1, 1, 0, 0])
    target = np.array([1, 1, 1, 0])

    choice_intensity = 0.8
    selection_intensity = 0.5

    actual_probability = main.compute_imitation_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=heterogeneous_fitness_function,
        choice_intensity=choice_intensity,
        selection_intensity=selection_intensity,
    )

    expected_probability = 0.115558109
    np.testing.assert_almost_equal(
        actual_probability, expected_probability, err_msg=actual_probability
    )


def test_compute_introspection_transition_probability_for_trivial_fitness_function():
    """
    Tests that the compute_imitation_introspection_transition_probability
    function returns the correct value for a trivial fitness function."""

    def trivial_fitness_function(state, **kwargs):
        return np.array([i + 1 for i in state])

    source = np.array([1, 1, 0])
    target = np.array([1, 1, 2])

    choice_intensity = 0.5
    number_of_strategies = 3

    actual_probability = main.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
    )

    expected_probability = 0.1218430964

    np.testing.assert_almost_equal(actual_probability, expected_probability)


def test_compute_introspection_transition_probability_for_symbolic_fitness_function():
    """
    Tests that the compute_imitation_introspection_transition_probability
    function returns the correct value for a trivial fitness function."""

    def symbolic_fitness_function(state, **kwargs):
        return np.array([sym.Symbol(f"x_{i}") for i in state])

    source = np.array([1, 1, 0])
    target = np.array([1, 1, 2])

    choice_intensity = sym.Symbol("Beta")
    number_of_strategies = sym.Symbol("k")
    x_0 = sym.Symbol("x_0")
    x_2 = sym.Symbol("x_2")

    actual_probability = main.compute_introspection_transition_probability(
        source=source,
        target=target,
        fitness_function=symbolic_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
    )

    expected_probability = (
        (1 / (3 * (number_of_strategies - 1)))
        * 1
        / (1 + sym.E ** ((x_0 - x_2) * choice_intensity))
    )

    assert sym.simplify(actual_probability == expected_probability)


def test_compute_introspection_transition_probability_for_infeasible_states_and_no_change():
    """
    Tests whether compute_introspection_transition_probability returns the correct
    values when the state transition is not of hamming distance 1"""

    def trivial_fitness_function(state):
        return np.array([1 for _ in state])

    source1 = np.array([0, 1])
    target1 = np.array([1, 0])
    choice_intensity = 0.5
    number_of_strategies = 2

    actual_probability1 = main.compute_introspection_transition_probability(
        source=source1,
        target=target1,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
    )

    expected_probability1 = 0

    assert expected_probability1 == actual_probability1

    source2 = np.array([0, 1])
    target2 = np.array([0, 1])

    actual_probability2 = main.compute_introspection_transition_probability(
        source=source2,
        target=target2,
        fitness_function=trivial_fitness_function,
        choice_intensity=choice_intensity,
        number_of_strategies=number_of_strategies,
    )

    assert actual_probability2 is None

    _ = trivial_fitness_function(source1)  # prevents unused function warning


def test_approximate_steady_state_for_different_initial_dist():
    """tests that the approximate_steady_state function correctly
    approximates a system's steady state for a different initial distribution"""

    initial_dist_1 = np.array([1, 0, 0, 0])
    initial_dist_2 = np.array([0, 0, 0, 1])

    transition_matrix = np.array(
        [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1]]
    )

    steady_state_1 = np.array([1, 0, 0, 0])
    steady_state_2 = np.array([0, 0, 0, 1])

    np.testing.assert_array_equal(
        main.approximate_steady_state(
            transition_matrix=transition_matrix, initial_dist=initial_dist_1
        ),
        steady_state_1,
    )

    np.testing.assert_array_equal(
        main.approximate_steady_state(
            transition_matrix=transition_matrix, initial_dist=initial_dist_2
        ),
        steady_state_2,
    )
