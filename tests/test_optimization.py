
import numpy as np
import pytest

from src.optimization import (
    minimum_variance,
    maximum_sharpe,
    target_return_min_variance
)


# Two uncorrelated assets
mu = np.array([0.10, 0.20])

cov = np.array([
    [0.04, 0.00],
    [0.00, 0.09]
])


def test_minimum_variance():
    weights = minimum_variance(cov)

    # Analytical solution: inverse-variance weights
    expected = np.array([
        0.09 / 0.13,
        0.04 / 0.13
    ])

    np.testing.assert_allclose(
        weights, expected, atol=1e-4
    )


def test_maximum_sharpe():
    weights = maximum_sharpe(
        mu, cov, risk_free_rate=0.04
    )

    # Analytical solution proportional to
    # inverse covariance times excess returns
    excess = mu - 0.04
    expected = np.linalg.solve(cov, excess)
    expected = expected / expected.sum()

    np.testing.assert_allclose(
        weights, expected, atol=1e-4
    )


def test_target_return():
    weights = target_return_min_variance(
        mu, cov, target_return=0.15
    )

    np.testing.assert_allclose(
        weights, [0.5, 0.5], atol=1e-4
    )

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights @ mu, 0.15)


def test_infeasible_target():
    with pytest.raises(ValueError):
        target_return_min_variance(
            mu, cov, target_return=0.30
        )
