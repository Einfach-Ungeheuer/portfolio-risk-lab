
import numpy as np
import pandas as pd

from src.optimization import minimum_variance, maximum_sharpe


def walk_forward_backtest(
    returns,
    strategy="minimum_variance",
    training_years=3,
    cost_bps=10,
    risk_free_rate=0.04
):
    """
    Expanding-window, annual walk-forward backtest.

    - Optimise using earlier calendar years.
    - Update target weights annually.
    - Rebalance monthly.
    - Preserve holdings and wealth across years.
    - Charge proportional costs on rebalancing trades.

    Initial acquisition costs are excluded.
    """

    if strategy not in ("minimum_variance", "maximum_sharpe"):
        raise ValueError("Unknown strategy.")

    if not isinstance(training_years, int) or training_years < 1:
        raise ValueError("training_years must be positive.")

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("Returns must be a nonempty DataFrame.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have a DatetimeIndex.")

    if (
        returns.index.has_duplicates
        or not returns.index.is_monotonic_increasing
        or returns.columns.has_duplicates
    ):
        raise ValueError("Invalid dates or asset columns.")

    data = returns.astype(float)

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Returns must be finite.")

    if (data.to_numpy() <= -1).any():
        raise ValueError("Asset returns must exceed -100%.")

    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Invalid transaction cost.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("Invalid risk-free rate.")

    years = sorted(data.index.year.unique())

    if len(years) <= training_years:
        raise ValueError("Not enough years.")

    # Step 1: Estimate annual target allocations.
    annual_targets = {}
    weights_history = []

    for i in range(training_years, len(years)):
        test_year = years[i]

        train = data[
            data.index.year.isin(years[:i])
        ]

        mu = train.mean().to_numpy() * 252
        cov = train.cov().to_numpy() * 252

        if strategy == "minimum_variance":
            w = minimum_variance(cov)
        else:
            w = maximum_sharpe(
                mu, cov, risk_free_rate
            )

        annual_targets[test_year] = w

        weights_history.append({
            "year": test_year,
            **dict(zip(data.columns, w))
        })

    # Step 2: Simulate the out-of-sample period
    # continuously, without restarting each year.
    test_data = data[
        data.index.year.isin(annual_targets)
    ]

    wealth = 100_000.0
    weights = None
    previous_month = None
    records = []

    cost_rate = cost_bps / 10_000

    for date, row in test_data.iterrows():
        month = date.to_period("M")
        target = annual_targets[date.year]

        turnover = 0.0
        trading_cost = 0.0

        if weights is None:
            # Initial portfolio formation;
            # acquisition costs excluded.
            weights = target.copy()

        elif month != previous_month:
            # This includes the transition
            # between consecutive test years.
            turnover = float(
                np.abs(target - weights).sum()
            )

            trading_cost = wealth * cost_rate * turnover

            wealth -= trading_cost
            weights = target.copy()

        previous_month = month

        # Wealth immediately before today's trading costs
        previous_wealth = wealth + trading_cost

        asset_returns = row.to_numpy(dtype=float)

        gross_return = float(weights @ asset_returns)

        wealth *= 1 + gross_return

        net_return = wealth / previous_wealth - 1

        # Update asset weights after returns.
        weights = (
            weights * (1 + asset_returns)
            / (1 + gross_return)
        )

        records.append({
            "date": date,
            "wealth": wealth,
            "return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "trading_cost": trading_cost
        })

    combined = pd.DataFrame(records).set_index("date")

    weight_table = pd.DataFrame(
        weights_history
    ).set_index("year")

    return combined, weight_table
