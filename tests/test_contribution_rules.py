import contribution_rules
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
    numpy.array (log(1) + 1, log(2) + 1, log(3) + 1) for a population with 3 individuals.
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
