
import numpy as np
import pytest

from src.monte_carlo import simulate_normal_returns


def test_simulation_count():
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    simulated = simulate_normal_returns(
        returns, n_simulations=1000
    )

    assert len(simulated) == 1000


def test_reproducibility():
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    simulation_1 = simulate_normal_returns(
        returns, seed=42
    )

    simulation_2 = simulate_normal_returns(
        returns, seed=42
    )

    np.testing.assert_array_equal(
        simulation_1, simulation_2
    )


def test_invalid_simulation_count():
    returns = np.array([-0.02, 0.01, 0.03])

    with pytest.raises(ValueError):
        simulate_normal_returns(
            returns, n_simulations=-100
        )


def test_insufficient_data():
    with pytest.raises(ValueError):
        simulate_normal_returns(
            [0.01], n_simulations=100
        )

from src.monte_carlo import simulate_wealth_paths


def test_wealth_paths_shape():
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    paths = simulate_wealth_paths(
        returns,
        n_days=10,
        n_paths=100,
        seed=42
    )

    assert paths.shape == (11, 100)


def test_initial_wealth():
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    paths = simulate_wealth_paths(
        returns,
        initial_wealth=100_000,
        n_days=10,
        n_paths=100
    )

    assert np.all(paths[0] == 100_000)


def test_wealth_paths_reproducibility():
    returns = np.array([-0.02, 0.01, 0.03, -0.01])

    paths1 = simulate_wealth_paths(returns, seed=42)
    paths2 = simulate_wealth_paths(returns, seed=42)

    np.testing.assert_array_equal(paths1, paths2)
