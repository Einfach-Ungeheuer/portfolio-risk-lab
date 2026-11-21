
import numpy as np
import pandas as pd


def backtest_portfolio(
    returns,
    target_weights,
    initial_wealth=100_000,
    rebalance="monthly",
    cost_bps=0
):
    """
    Backtest a portfolio with rebalancing and costs.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple asset returns.
    target_weights : pd.Series
        Target portfolio weights indexed by ticker.
    initial_wealth : float
        Starting investment value.
    rebalance : str
        "daily", "monthly", or "none".
    cost_bps : float
        Trading cost in basis points per dollar traded.

    Returns
    -------
    pd.DataFrame
        Daily wealth, returns, turnover, and trading costs.
    """

    # 1. Validate inputs
    if rebalance not in ("daily", "monthly", "none"):
        raise ValueError("Invalid rebalancing rule.")

    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a DataFrame.")

    if returns.empty or not isinstance(
        returns.index, pd.DatetimeIndex
    ):
        raise ValueError("Returns need a nonempty date index.")

    if (
        not returns.index.is_monotonic_increasing
        or returns.index.has_duplicates
        or returns.columns.has_duplicates
    ):
        raise ValueError("Invalid dates or asset columns.")

    if not isinstance(target_weights, pd.Series):
        raise ValueError("Weights must be a pandas Series.")

    if (
        target_weights.index.has_duplicates
        or set(target_weights.index) != set(returns.columns)
    ):
        raise ValueError("Weights must match asset columns.")

    if not np.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive.")

    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction costs must be nonnegative.")

    data = returns.astype(float)

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Returns must be finite.")

    if (data.to_numpy() <= -1).any():
        raise ValueError("Asset returns must exceed -100%.")

    target = target_weights.reindex(
        data.columns
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(target).all()
        or (target < 0).any()
        or not np.isclose(target.sum(), 1.0)
    ):
        raise ValueError("Invalid portfolio weights.")

    # 2. Initialise portfolio
    cost_rate = cost_bps / 10_000

    wealth = float(initial_wealth)
    weights = target.copy()

    records = []
    previous_month = None

    # 3. Process each trading day
    for day_number, (date, row) in enumerate(
        data.iterrows()
    ):

        month = date.to_period("M")

        # Determine whether to rebalance
        if day_number == 0:
            should_rebalance = False

        elif rebalance == "daily":
            should_rebalance = True

        elif rebalance == "monthly":
            should_rebalance = month != previous_month

        else:
            should_rebalance = False

        turnover = 0.0
        trading_cost = 0.0

        # 4. Rebalance before applying daily returns
        if should_rebalance:
            turnover = float(
                np.abs(target - weights).sum()
            )

            trading_cost = (
                wealth * cost_rate * turnover
            )

            wealth -= trading_cost
            weights = target.copy()

        previous_month = month

        # Record wealth before this day's return,
        # including any wealth spent on trading costs
        previous_wealth = wealth + trading_cost

        # 5. Apply asset returns
        asset_returns = row.to_numpy(dtype=float)

        gross_return = float(
            weights @ asset_returns
        )

        wealth *= 1 + gross_return

        net_return = (
            wealth / previous_wealth - 1
        )

        # 6. Update weights as market prices change
        weights = (
            weights * (1 + asset_returns)
            / (1 + gross_return)
        )

        # 7. Save daily results
        records.append({
            "date": date,
            "wealth": wealth,
            "return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "trading_cost": trading_cost
        })

    return pd.DataFrame(records).set_index("date")


def backtest_metrics(
    backtest_result,
    initial_wealth=100_000,
    risk_free_rate=0.04
):
    """Calculate basic portfolio backtesting metrics."""

    r = backtest_result["return"]
    wealth = backtest_result["wealth"]

    total_return = (
        wealth.iloc[-1] / initial_wealth - 1
    )

    volatility = r.std() * np.sqrt(252)

    if len(r) < 2 or volatility <= 0:
        sharpe = np.nan
    else:
        sharpe = (
            r.mean() * 252 - risk_free_rate
        ) / volatility

    # Include starting wealth when measuring drawdown
    values = np.r_[
        initial_wealth,
        wealth.to_numpy()
    ]

    peaks = np.maximum.accumulate(values)

    max_drawdown = np.min(
        values / peaks - 1
    )

    return {
        "total_return": total_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "total_trading_cost": (
            backtest_result["trading_cost"].sum()
        )
    }
