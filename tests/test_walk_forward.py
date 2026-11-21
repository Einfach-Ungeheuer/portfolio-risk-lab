
import numpy as np
import pandas as pd
import pytest

from src.walk_forward import walk_forward_backtest


def sample_returns():
    dates = pd.bdate_range(
        "2021-01-01", "2025-12-31"
    )

    rng = np.random.default_rng(42)

    data = rng.normal(
        loc=[0.0002, 0.0004, 0.0006],
        scale=[0.003, 0.008, 0.012],
        size=(len(dates), 3)
    )

    return pd.DataFrame(
        data,
        index=dates,
        columns=["AGG", "GLD", "SPY"]
    )


def test_walk_forward_dates():
    returns = sample_returns()

    bt, weights = walk_forward_backtest(
        returns,
        training_years=3
    )

    assert bt.index.min().year == 2024
    assert bt.index.max().year == 2025
    assert list(weights.index) == [2024, 2025]


def test_weights_sum_to_one():
    returns = sample_returns()

    _, weights = walk_forward_backtest(
        returns,
        training_years=3
    )

    np.testing.assert_allclose(
        weights.sum(axis=1).to_numpy(),
        np.ones(len(weights))
    )


def test_costs_reduce_wealth():
    returns = sample_returns()

    without_costs, _ = walk_forward_backtest(
        returns, cost_bps=0
    )

    with_costs, _ = walk_forward_backtest(
        returns, cost_bps=10
    )

    assert (
        with_costs["wealth"].iloc[-1]
        <= without_costs["wealth"].iloc[-1]
    )


def test_invalid_strategy():
    returns = sample_returns()

    with pytest.raises(ValueError):
        walk_forward_backtest(
            returns, strategy="unknown"
        )

def test_cross_year_transition_costs():
    returns = sample_returns()

    bt, weights = walk_forward_backtest(
        returns,
        strategy="maximum_sharpe",
        training_years=3,
        cost_bps=10
    )

    first_day_2025 = bt.loc["2025"].iloc[0]

    assert first_day_2025["turnover"] >= 0
    assert first_day_2025["trading_cost"] >= 0

    # Under this cost model, the fee equals pre-trade
    # portfolio wealth multiplied by turnover and the
    # proportional transaction-cost rate.
    #
    # A zero-turnover boundary is possible if the
    # desired weights already match actual holdings.
    if first_day_2025["turnover"] > 0:
        assert first_day_2025["trading_cost"] > 0
