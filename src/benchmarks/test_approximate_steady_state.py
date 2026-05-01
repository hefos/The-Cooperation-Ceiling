import pytest
import numpy as np
import pathlib
import sys

file_path = pathlib.Path(__file__)

root_path = (file_path / "../../../").resolve()

sys.path.append(str(root_path))
import src.main as main  # noqa: E402

n_range = 200
step_size_n = 10
prob_range = 20
step_size_prob = 2


def generate_stochastic_2_valency_cycle_matrix(n, prob_prior):
    identity_matrix = np.eye(n)
    return prob_prior * (np.roll(identity_matrix, -1, axis=1)) + (1 - prob_prior) * (
        np.roll(identity_matrix, 1, axis=1)
    )


def generate_full_uniform_matrix(n):
    return np.full((n, n), 1 / n)


def test_generate_stochastic_2_valency_cycle_matrix():
    """Tests that generate_stochastic_2_valency_cycle_matrix is correct for the
    case of both deterministic cycle and stochastic valency"""

    deterministic_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(
        deterministic_matrix,
        generate_stochastic_2_valency_cycle_matrix(n=3, prob_prior=0),
    )

    stochastic_matrix = np.array(
        [[0, 0.3, 0, 0.7], [0.7, 0, 0.3, 0], [0, 0.7, 0, 0.3], [0.3, 0, 0.7, 0]]
    )
    np.testing.assert_array_almost_equal(
        stochastic_matrix,
        generate_stochastic_2_valency_cycle_matrix(n=4, prob_prior=0.7),
    )


def test_generate_full_uniform_matrix():
    """Tests that generate_full_uniform_matrix is correct"""

    uniform_matrix = np.array(
        [
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        ]
    )
    np.testing.assert_array_equal(uniform_matrix, generate_full_uniform_matrix(4))


@pytest.mark.parametrize("n", range(2, n_range, step_size_n))
def test_approximate_steady_state_for_deterministic_cycle_matrix(n, benchmark):
    """Benchmarks approximate_steady_state for the deterministic cycle
    matrix"""

    transition_matrix = generate_stochastic_2_valency_cycle_matrix(n, 0)
    benchmark(main.approximate_steady_state, transition_matrix)


@pytest.mark.parametrize(
    "n, prob_prior",
    [
        (n, p)
        for n in range(2, n_range, step_size_n)
        for p in [1 / (i) for i in range(1, prob_range + 1, step_size_prob)]
    ],
)
def test_approximate_steady_state_for_stochastic_2_valency_cycle_matrix(
    n, prob_prior, benchmark
):
    """Benchmarks approximate_steady_state for the stochastic 2-valency cycle
    matrix"""

    transition_matrix = generate_stochastic_2_valency_cycle_matrix(n, prob_prior)
    benchmark(main.approximate_steady_state, transition_matrix)


@pytest.mark.parametrize("n", range(2, n_range, step_size_n))
def test_approximate_steady_state_for_full_uniform_matrix(n, benchmark):
    """Benchmarks approximate_steady_state for the full uniform
    matrix"""

    transition_matrix = generate_full_uniform_matrix(n)
    benchmark(main.approximate_steady_state, transition_matrix)


def test_approximate_steady_state_for_specific_four_by_four(benchmark):
    """A benchmark with a specific 4 by 4 matrix"""

    transition_matrix = np.array(
        [
            [0.5, 0, 0.2, 0.3],
            [0.1, 0.7, 0.2, 0],
            [0.3, 0.3, 0.3, 0.1],
            [0.3, 0.1, 0.1, 0.5],
        ]
    )

    benchmark(main.approximate_steady_state, transition_matrix)
