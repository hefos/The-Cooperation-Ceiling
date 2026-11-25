import numpy as np

from public_goods_games import simulate


def test_simulated_subgroup_pgg_fitness_function():
    """
    Tests the expected result of simulated_subgroup_pgg_fitness_function is
    returned. Also tests that the generator successfully moves on with multiple
    runs of the function"""

    state = np.array([1,0,1,1,0,1])
    d = 2
    generator = simulate.generator_for(seed=3)
    r = 4
    contribution_vector = np.array([1,2,3,4,5,6])
    selection_intensity = 0.5

    actual = simulate.subgroup_pgg_fitness_function(state=state, selection_intensity=selection_intensity, r=r, contribution_vector=contribution_vector, d=d, generator=generator)

    expected = {4:0, 1:0}

    assert actual == expected

    actual = simulate.subgroup_pgg_fitness_function(state=state, selection_intensity=selection_intensity, r=r, contribution_vector=contribution_vector, d=d, generator=generator)

    expected = {4:6, 2:3}

    assert actual == expected

def test_moran_transition_probability_for_feasible_transition():
    """
    Tests that moran_transition_probability returns the expected value for a
    feasible transition"""

    source = np.array([1,0,1,0,1])
    target = np.array([0,0,1,0,1])

    r=4
    contribution_vector = np.array([1,2,3,4,5])
    d=3
    selection_intensity=0.2
    generator=simulate.generator_for(seed=5)

    actual = simulate.moran_transition_probability(source=source, target=target, selection_intensity=selection_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    expected = 0.13333333333333333

    np.testing.assert_almost_equal(actual, expected)

def test_moran_transition_probability_for_infeasible_and_self_transitions():
    """
    Tests that moran_transition_probability returns 0 and None for infeasible
    and self transitions respectively"""

    source = np.array([1,0,1,0,1])
    target = np.array([1,0,0,1,1])

    r=4
    contribution_vector = np.array([1,2,3,4,5])
    d=3
    selection_intensity=0.2
    generator=simulate.generator_for(seed=5)

    actual_self = simulate.moran_transition_probability(source=source, target=source, selection_intensity=selection_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    assert actual_self is None

    actual_infeasible = simulate.moran_transition_probability(source=source, target=target, selection_intensity=selection_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    assert actual_infeasible == 0

def test_moran_transition_probability_for_outsider_transition():
    """
    Tests that moran_transition_probability returns 0 when a player changes
    which did not play in the previous round's game"""

    source = np.array([1,0,1,0,1])
    target = np.array([1,0,0,0,1])

    r=4
    contribution_vector = np.array([1,2,3,4,5])
    d=3
    selection_intensity=0.2
    generator=simulate.generator_for(seed=5)

    actual_self = simulate.moran_transition_probability(source=source, target=target, selection_intensity=selection_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    assert actual_self == 0

def test_fermi_transition_probability_for_feasible_transition():
    """
    Tests fermi_transition_probability for a feasible transition where a player
    in the game transitions to the type of another player in the state."""

    source = np.array([1,1,0,0,0])
    target = np.array([1,0,0,0,0])
    d=4
    r=4
    contribution_vector=np.array([2,4,6,8,10])
    choice_intensity=0.5
    generator=simulate.generator_for(seed=7)

    expected_transition_probability = 0.146799513
    actual_transition_probability = simulate.fermi_transition_probability(source=source, target=target, choice_intensity=choice_intensity, r=r, d=d, contribution_vector=contribution_vector, generator=generator)

    np.testing.assert_almost_equal(actual_transition_probability, expected_transition_probability)

def test_fermi_transition_probability_for_outsider_transition():
    """
    Tests that fermi_transition_probability returns 0 when a player changes
    which did not play in the previous round's game"""

    source = np.array([1,0,1,0,1])
    target = np.array([1,0,0,0,1])

    r=100
    contribution_vector = np.array([1,2,3,4,5])
    d=3
    choice_intensity=0.2
    generator=simulate.generator_for(seed=31)

    actual_self = simulate.fermi_transition_probability(source=source, target=target, choice_intensity=choice_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    assert actual_self == 0

def test_fermi_transition_probability_for_infeasible_and_self_transitions():
    """
    Tests that moran_transition_probability returns 0 and None for infeasible
    and self transitions respectively"""

    source = np.array([1,0,1,0,1])
    target = np.array([1,0,0,1,1])

    r=22
    contribution_vector = np.array([1,2,3,4,5])
    d=2
    choice_intensity=0.3
    generator=simulate.generator_for(seed=4)

    actual_self = simulate.fermi_transition_probability(source=source, target=source, choice_intensity=choice_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    assert actual_self is None

    actual_infeasible = simulate.fermi_transition_probability(source=source, target=target, choice_intensity=choice_intensity, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    assert actual_infeasible == 0

def test_wright_fisher_transition_probability_for_feasible_transition():
    """
    Tests that the wright-fisher transition probability function returns the
    expected value for a standard, feasible transition"""

    source = np.array([1,0,1,0,1])
    target = np.array([1,1,1,1,1])

    contribution_vector = np.array([6,7,8,9,10])
    r=2
    d=3
    selection_intensity=0.1
    mu=0.05
    generator=simulate.generator_for(seed=284)

    actual_transition_probability = simulate.wright_fisher_transition_probability(source=source, target=target, selection_intensity=selection_intensity, mu=mu, r=r, d=d, contribution_vector=contribution_vector, generator=generator)

    expected_transition_probability = 0.0139816867

    np.testing.assert_almost_equal(actual_transition_probability, expected_transition_probability)

def test_wright_fisher_for_infeasible_and_self_transitions():
    """
    Tests that the wright-fisher process returns the correct values for invalid
    transitions, including transitions where players outside the state change
    action type, transitions away from absorbing states without mutation, and self-transitions"""

    source = np.array([1,0,1,0,1])
    target = np.array([0,1,0,1,0])
    source_extra_action = np.array([1,1,1,1,1])
    target_extra_action = np.array([0,0,0,0,0])

    contribution_vector = np.array([6,7,8,9,10])
    r=2
    d=3
    d_extra=5
    selection_intensity=0.1
    mu=0
    generator=simulate.generator_for(seed=83)

    actual_self = simulate.wright_fisher_transition_probability(source=source, target=source, selection_intensity=selection_intensity, mu=mu, r=r, d=d, contribution_vector=contribution_vector, generator=generator)

    assert actual_self is None

    actual_infeasible = simulate.wright_fisher_transition_probability(source=source, target=target, selection_intensity=selection_intensity, mu=mu, r=r, d=d, contribution_vector=contribution_vector, generator=generator)

    assert actual_infeasible == 0

    actual_extra_action = simulate.wright_fisher_transition_probability(source=source_extra_action, target=target_extra_action, selection_intensity=selection_intensity, mu=mu, r=r, d=d_extra, contribution_vector=contribution_vector, generator=generator)

    assert actual_extra_action == 0

def test_introspection_transition_probability_for_feasible_transition_of_cooperator():
    """
    Tests that the introsepction transition probability returns the expected
    value for a transition where a cooperator changes to a defector"""

    source = np.array([1,0,1,1])
    target = np.array([1,0,1,0])

    r=3
    d=2
    choice_intensity=0.15
    contribution_vector = np.array([8,4,2,5])
    generator=simulate.generator_for(seed=73)

    actual = simulate.introspection_transition_probability(source=source, target=target, choice_intensity=choice_intensity, d=d, r=r, contribution_vector=contribution_vector, generator=generator)

    expected = 0.2036667

    np.testing.assert_almost_equal(actual, expected)

def test_introspection_transition_probability_for_feasible_transition_of_defector():
    """
    Tests that the introsepction transition probability returns the expected
    value for a transition where a defector changes to a cooperator"""

    source = np.array([0,0,1,1])
    target = np.array([1,0,1,1])

    r=1
    d=2
    choice_intensity=0.4
    contribution_vector = np.array([3,1,1,2])
    generator=simulate.generator_for(seed=9)

    actual = simulate.introspection_transition_probability(source=source, target=target, choice_intensity=choice_intensity, d=d, r=r, contribution_vector=contribution_vector, generator=generator)

    expected = 0.1771718469

    np.testing.assert_almost_equal(actual, expected)

def test_introsepction_transition_probability_for_infeasible_and_self_transitions():
    """
    Tests that the introspection transition probability function returns the
    expected value for infeasible and self transitions"""

    source = np.array([1,0,1,0,1])
    target = np.array([0,1,0,1,0])

    contribution_vector = np.array([6,7,8,9,10])
    r=2
    d=3
    choice_intensity=0.1
    generator=simulate.generator_for(seed=284)

    actual = simulate.introspection_transition_probability(source=source, target=target, choice_intensity=choice_intensity, d=d, r=r, contribution_vector=contribution_vector, generator=generator)

    assert actual == 0

    actual_self = simulate.introspection_transition_probability(source=source, target=source, choice_intensity=choice_intensity, d=d, r=r, contribution_vector=contribution_vector, generator=generator)

    assert actual_self is None

def test_aspiration_transition_probability_for_feasible_transition():
    """
    Tests that the aspiration transition probability is expected for a feasible
    transition"""

    source = np.array([1,0,0,1,1,0])
    target = np.array([1,1,0,1,1,0])

    r=3
    d=4
    aspiration=5
    choice_intensity=0.3
    contribution_vector = np.array([1,2,3,4,5,6])
    generator=simulate.generator_for(seed=100)

    actual = simulate.aspiration_transition_probability(source=source, target=target, choice_intensity=choice_intensity, aspiration=aspiration, d=d, r=r, generator=generator, contribution_vector=contribution_vector)

    expected = 0.08020532521

    np.testing.assert_almost_equal(actual, expected)

def test_aspiration_transition_probability_for_infeasible_and_self_transitions():
    """
    Tests that the aspiration transition probability function returns the
    expected value for infeasible and self transitions"""

    source = np.array([1,0,1,0,1])
    target = np.array([0,1,0,1,0])

    contribution_vector = np.array([6,7,8,9,10])
    r=2
    d=3
    choice_intensity=0.1
    aspiration = 2
    generator=simulate.generator_for(seed=284)

    actual = simulate.aspiration_transition_probability(source=source, target=target, aspiration=aspiration, choice_intensity=choice_intensity, d=d, r=r, contribution_vector=contribution_vector, generator=generator)

    assert actual == 0

    actual_self = simulate.aspiration_transition_probability(source=source, target=source, aspiration=aspiration, choice_intensity=choice_intensity, d=d, r=r, contribution_vector=contribution_vector, generator=generator)

    assert actual_self is None