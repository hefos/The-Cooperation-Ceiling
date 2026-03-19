import public_goods_games.contribution_rules as contribution_rules
import numpy as np
import pytest


def test_dirichlet_linear_alpha_rule_for_N_eq_3():
    """Tests that the diriclet_linear_alpha_rule function correctly returns the
    numpy.array [1,2,3] for a population with 3 individuals."""

    N = 3

    expected_alphas = np.array([1, 2, 3])

    obtained_alphas = contribution_rules.dirichlet_linear_alpha_rule(N)

    np.testing.assert_array_equal(expected_alphas, obtained_alphas)


def test_dirichlet_binomial_alpha_rule_for_N_eq_5_n_eq_3():
    """Tests that the diriclet_binomial_alpha_rule function correctly returns
    the numpy.array [1,1,1,3,3] for a population with 5 individuals, 2
    contributing high and 3 contributing low, with a difference of 2 between them."""

    N = 5
    n = 3
    low_alpha = 1
    high_alpha = 3

    expected_alphas = np.array([1, 1, 1, 3, 3])

    obtained_alphas = contribution_rules.dirichlet_binomial_alpha_rule(
        N=N, n=n, low_alpha=low_alpha, high_alpha=high_alpha
    )

    np.testing.assert_array_equal(expected_alphas, obtained_alphas)


def test_dirichlet_log_alpha_rule_for_N_eq_3():
    """Tests that the dirichlet_log_alpha_rule correctly returns the
    numpy.array (log(1) + 1, log(2) + 1, log(3) + 1) for a population with 3 qqindividuals.
    """

    N = 3

    expected_alphas = np.array([1, 1.693147, 2.098612])

    obtained_alphas = contribution_rules.dirichlet_log_alpha_rule(N=N)

    np.testing.assert_array_almost_equal(expected_alphas, obtained_alphas)


def test_log_contribution_rule_for_player_2_N_eq_3_M_eq_12_contributing():
    """
    Tests that the log_contribution_rule function correctly calculates the
    contribution of player 2 of 3 when M = 12"""

    N = 3
    M = 12
    index = 1

    expected_contribution = 4.095894024

    obtained_contribution = contribution_rules.log_contribution_rule(
        index=index, M=M, N=N
    )

    np.testing.assert_almost_equal(expected_contribution, obtained_contribution)


def test_linear_contribution_rule_for_N_eq_3_M_eq_12_contributing():
    """
    Tests that the linear_contribution_rule function correctly calculates
    the contribution of player 2 of 3 when M=12"""

    N = 3
    M = 12
    index = 1

    expected_contribution = 4

    obtained_contribution = contribution_rules.linear_contribution_rule(
        index=index, M=M, N=N
    )

    assert expected_contribution == obtained_contribution


def test_binomial_contribution_rule_for_N_eq_5_n_eq_3():
    """
    Tests that the binomial_contribution_rule function correctly calculates the
    contribution of two players, player 2 and player 4, when N=5 and n=3. We take alpha_h = 3 and M=9
    """

    N = 5
    M = 9
    n = 3
    index_1 = 1
    index_2 = 3
    alpha_h = 3

    expected_contribution_1 = 1
    expected_contribution_2 = 3

    obtained_contribution_1 = contribution_rules.binomial_contribution_rule(
        index=index_1, M=M, N=N, alpha_h=alpha_h, n=3
    )

    obtained_contribution_2 = contribution_rules.binomial_contribution_rule(
        index=index_2, M=M, N=N, alpha_h=alpha_h, n=3
    )

    assert expected_contribution_1 == obtained_contribution_1
    assert expected_contribution_2 == obtained_contribution_2


def test_binomial_contribution_rule_fails_for_negative_alpha_l():
    """
    Tests that the binomial_contribution_rule fails correctly when alpha_l is a
    negative value with a given set of parameters
    """

    N = 5
    M = 9
    n = 3
    index = 1
    alpha_h = 8

    with pytest.raises(ValueError):
        contribution_rules.binomial_contribution_rule(
            index=index, M=M, N=N, alpha_h=alpha_h, n=n
        )


def test_binomial_contribution_rule_fails_for_big_alpha_l():
    """
    Tests that the binomial_contribution_rule function fails correctly when
    alpha_l is greater than alpha_h with a given set of parameters
    """

    N = 5
    M = 9
    n = 3
    index = 1
    alpha_h = 0.1

    with pytest.raises(ValueError):
        contribution_rules.binomial_contribution_rule(
            index=index, M=M, N=N, alpha_h=alpha_h, n=n
        )


def test_dirichlet_power_law_alpha_rule_for_N_eq_6_a_eq_2():
    """
    Tests a standard form of the dirichlet power law, with 6 players and
    $a=2$"""

    a = 2
    N = 6

    actual_alphas = contribution_rules.dirichlet_power_law_alpha_rule(N=N, a=a)

    expected_alphas = np.array([2, 4, 8, 16, 32, 64])

    np.testing.assert_array_equal(actual_alphas, expected_alphas)


def test_dirichlet_power_law_alpha_rule_for_N_eq_2_a_eq_e():
    """
    Tests a standard form of the dirichlet power law, with 2 players and
    $a=e$"""

    N = 2

    actual_alphas = contribution_rules.dirichlet_power_law_alpha_rule(N=N)

    expected_alphas = np.array([np.exp(1), np.exp(2)])

    np.testing.assert_array_almost_equal(actual_alphas, expected_alphas)


def test_dirichlet_power_law_alpha_rule_fails_for_negative_a():
    """
    Tests that a negative value of $a$ will raise a ValueError in the dirichlet
    power law alpha rule"""

    a = -2
    N = 6

    with pytest.raises(ValueError):
        contribution_rules.dirichlet_power_law_alpha_rule(N=N, a=a)


def test_power_law_contribution_rule_for_N_eq_3_M_eq_40():
    """
    Tests that the power_law_contribution_rule function returns the correct
    values summing to 40 for a 3 player game with M=40"""

    M = 40
    N = 3
    summation_term = np.exp(1) + np.exp(2) + np.exp(3)

    expected_return_1 = np.exp(1) * (M / summation_term)
    expected_return_2 = np.exp(2) * (M / summation_term)
    expected_return_3 = np.exp(3) * (M / summation_term)

    actual_return_1 = contribution_rules.power_law_contribution_rule(index=0, M=M, N=N)
    actual_return_2 = contribution_rules.power_law_contribution_rule(index=1, M=M, N=N)
    actual_return_3 = contribution_rules.power_law_contribution_rule(index=2, M=M, N=N)

    np.testing.assert_almost_equal(actual_return_1, expected_return_1, err_msg="1")
    np.testing.assert_almost_equal(actual_return_2, expected_return_2, err_msg="2")
    np.testing.assert_almost_equal(actual_return_3, expected_return_3, err_msg="3")

    actual_sum = actual_return_1 + actual_return_2 + actual_return_3

    np.testing.assert_almost_equal(actual_sum, M)


def test_power_law_contribution_rule_fails_for_a_negative():
    """
    Tests that the power_law_contribution_rule fails correctly for a negative
    value of $a$"""

    a = -2
    N = 6
    M = 742
    index = 3

    with pytest.raises(ValueError):
        contribution_rules.power_law_contribution_rule(index=index, N=N, a=a, M=M)


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
        contribution_rules.get_deterministic_contribution_vector(
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
        contribution_rules.get_deterministic_contribution_vector(
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
        contribution_rules.get_deterministic_contribution_vector(
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

    actual_return = contribution_rules.get_dirichlet_contribution_vector(
        N=N, alpha_rule=trivial_alpha_rule, M=M, scale=scale
    )

    np.random.seed(5)

    expected_return_iteration = np.array([3.98697183, 3.99898138, 4.01404679])

    actual_return_iteration = np.array(
        [
            contribution_rules.get_dirichlet_contribution_vector(
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

    actual_return = contribution_rules.get_dirichlet_contribution_vector(
        N=N, alpha_rule=linear_alpha_rule, M=M, scale=1
    )

    np.testing.assert_allclose(actual_return, expected_return)

    np.random.seed(4)

    expected_return_iteration = np.array([1.96018551, 4.0127389, 6.02707559])

    actual_return_iteration = np.array(
        [
            contribution_rules.get_dirichlet_contribution_vector(
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
    actual_return = contribution_rules.get_dirichlet_contribution_vector(
        N=N, alpha_rule=kwargs_alpha_rule, M=M, scale=scale, bonus=bonus
    )

    np.testing.assert_allclose(actual_return, expected_return)

    np.random.seed(3)

    expected_return_iteration = np.array([5.99449831, 11.97708597, 18.02841572])

    actual_return_iteration = np.array(
        [
            contribution_rules.get_dirichlet_contribution_vector(
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
        contribution_rules.get_dirichlet_contribution_vector(
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
        contribution_rules.get_dirichlet_contribution_vector(
            N=N, alpha_rule=small_alpha_rule, scale=scale, M=15
        )