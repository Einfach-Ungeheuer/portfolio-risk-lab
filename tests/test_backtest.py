
import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_portfolio,
    backtest_metrics
)


@pytest.fixture
def sample_data():
    dates = pd.to_datetime([
        "2025-01-30",
        "2025-01-31",
        "2025-02-03",
        "2025-02-04",
        "2025-03-03"
    ])

    returns = pd.DataFrame({
        "A": [0.10, -0.02, 0.03, 0.01, -0.01],
        "B": [0.00, 0.01, -0.02, 0.00, 0.02]
    }, index=dates)

    weights = pd.Series({
        "A": 0.60,
        "B": 0.40
    })

    return returns, weights


def test_initial_day_return(sample_data):
    returns, weights = sample_data

    result = backtest_portfolio(
        returns, weights,
        rebalance="none"
    )

    # 60% x 10% + 40% x 0% = 6%
    assert result["return"].iloc[0] == pytest.approx(0.06)


def test_monthly_rebalance_count(sample_data):
    returns, weights = sample_data

    result = backtest_portfolio(
        returns, weights,
        rebalance="monthly"
    )

    # First day of February and March
    assert (result["turnover"] > 0).sum() == 2


def test_daily_rebalance_count(sample_data):
    returns, weights = sample_data

    result = backtest_portfolio(
        returns, weights,
        rebalance="daily"
    )

    assert (result["turnover"] > 0).sum() == 4


def test_buy_and_hold_has_no_trades(sample_data):
    returns, weights = sample_data

    result = backtest_portfolio(
        returns, weights,
        rebalance="none",
        cost_bps=10
    )

    assert result["turnover"].sum() == 0
    assert result["trading_cost"].sum() == 0


def test_costs_reduce_wealth(sample_data):
    returns, weights = sample_data

    without_costs = backtest_portfolio(
        returns, weights,
        rebalance="daily",
        cost_bps=0
    )

    with_costs = backtest_portfolio(
        returns, weights,
        rebalance="daily",
        cost_bps=10
    )

    assert (
        with_costs["wealth"].iloc[-1]
        < without_costs["wealth"].iloc[-1]
    )


def test_invalid_weights(sample_data):
    returns, _ = sample_data

    invalid = pd.Series({
        "A": 0.80,
        "B": 0.50
    })

    with pytest.raises(ValueError):
        backtest_portfolio(
            returns, invalid
        )


def test_metrics(sample_data):
    returns, weights = sample_data

    result = backtest_portfolio(
        returns, weights,
        rebalance="none"
    )

    metrics = backtest_metrics(result)

    assert np.isfinite(metrics["total_return"])
    assert np.isfinite(metrics["annual_volatility"])
    assert np.isfinite(metrics["max_drawdown"])
