
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

def test_cross_year_rebalancing_charges_exact_cost(monkeypatch):
    import src.walk_forward as wf

    # Two assets with zero daily returns.
    # Consequently, weights cannot drift naturally.
    dates = pd.to_datetime([
        "2021-01-04", "2021-01-05",
        "2022-01-03", "2022-01-04",
        "2023-01-03", "2023-01-04",
        "2024-01-02", "2024-01-03",
        "2025-01-02", "2025-01-03",
    ])

    returns = pd.DataFrame(
        0.0,
        index=dates,
        columns=["Asset_A", "Asset_B"]
    )

    # Force the optimiser to choose:
    # 2024: 100% Asset_A
    # 2025: 100% Asset_B
    allocations = iter([
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0])
    ])

    monkeypatch.setattr(
        wf,
        "minimum_variance",
        lambda cov: next(allocations)
    )

    bt, weights = wf.walk_forward_backtest(
        returns,
        strategy="minimum_variance",
        training_years=3,
        cost_bps=10
    )

    transition = bt.loc["2025-01-02"]

    # Sell 100% of Asset_A and buy 100% of Asset_B.
    assert transition["turnover"] == pytest.approx(2.0)

    # $100,000 × 2.0 × 0.001 = $200.
    assert transition["trading_cost"] == pytest.approx(200.0)

    # No market movement, only transaction costs.
    assert transition["wealth"] == pytest.approx(99800.0)

    assert transition["return"] == pytest.approx(-0.002)

    assert weights.loc[2024, "Asset_A"] == pytest.approx(1.0)
    assert weights.loc[2025, "Asset_B"] == pytest.approx(1.0)
