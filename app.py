
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from src.walk_forward import walk_forward_backtest

from src.backtest import backtest_portfolio, backtest_metrics
from src.risk import historical_var, historical_cvar


st.set_page_config(
    page_title="Portfolio Analytics & Risk Lab",
    page_icon="📈",
    layout="wide"
)

st.title("Portfolio Analytics & Risk Lab")
st.caption(
    "Quantitative portfolio research | "
    "SPY, AGG and GLD | Historical analysis"
)


# ------------------------------------------------
# 1. Load financial data
# ------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    prices = yf.download(
        ["SPY", "AGG", "GLD"],
        start="2021-01-01",
        end="2026-01-01",
        auto_adjust=True,
        progress=False
    )["Close"]

    if prices.empty:
        raise ValueError("No market data returned.")

    prices = prices[["SPY", "AGG", "GLD"]]

    if prices.isna().any().any():
        raise ValueError(
            "Market data contains missing observations."
        )

    return prices.pct_change(
        fill_method=None
    ).dropna()


try:
    returns = load_data()
except Exception as exc:
    st.error(f"Unable to load market data: {exc}")
    st.stop()


# ------------------------------------------------
# 2. Sidebar controls
# ------------------------------------------------

st.sidebar.header("Portfolio Configuration")

spy = st.sidebar.slider(
    "SPY allocation (%)", 0, 100, 50
)

agg = st.sidebar.slider(
    "AGG allocation (%)", 0, 100, 30
)

gld = st.sidebar.slider(
    "GLD allocation (%)", 0, 100, 20
)

total = spy + agg + gld

if total == 0:
    st.error("Choose at least one nonzero allocation.")
    st.stop()

weights = pd.Series({
    "SPY": spy / total,
    "AGG": agg / total,
    "GLD": gld / total
})

st.sidebar.caption(
    "Allocations are normalised to total 100%."
)

rebalance = st.sidebar.selectbox(
    "Rebalancing frequency",
    ["monthly", "daily", "none"]
)

cost_bps = st.sidebar.slider(
    "Transaction costs (basis points)",
    0, 50, 10
)

initial_wealth = st.sidebar.number_input(
    "Initial investment (USD)",
    min_value=1000,
    value=100000,
    step=1000
)


# ------------------------------------------------
# 3. Run portfolio backtest
# ------------------------------------------------

bt = backtest_portfolio(
    returns,
    weights,
    initial_wealth=initial_wealth,
    rebalance=rebalance,
    cost_bps=cost_bps
)

metrics = backtest_metrics(
    bt,
    initial_wealth=initial_wealth
)


# ------------------------------------------------
# 4. Display performance statistics
# ------------------------------------------------

st.subheader("Portfolio Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Return",
    f"{metrics['total_return']:.2%}"
)

c2.metric(
    "Annualised Volatility",
    f"{metrics['annual_volatility']:.2%}"
)

c3.metric(
    "Sharpe Ratio",
    f"{metrics['sharpe']:.3f}"
)

c4.metric(
    "Maximum Drawdown",
    f"{metrics['max_drawdown']:.2%}"
)


# ------------------------------------------------
# 5. Portfolio wealth chart
# ------------------------------------------------

st.subheader("Portfolio Wealth")

st.line_chart(
    bt["wealth"].rename("Portfolio Value (USD)")
)


# ------------------------------------------------
# 6. Drawdown chart
# ------------------------------------------------

st.subheader("Historical Drawdowns")

wealth_values = np.r_[
    initial_wealth,
    bt["wealth"].to_numpy()
]

running_peak = np.maximum.accumulate(
    wealth_values
)

drawdown_values = wealth_values / running_peak - 1

drawdown = pd.Series(
    drawdown_values[1:],
    index=bt.index,
    name="Drawdown"
)

st.line_chart(drawdown)


# ------------------------------------------------
# 7. Tail-risk statistics
# ------------------------------------------------

st.subheader("Historical Tail Risk")

risk_returns = bt["return"]

risk_table = pd.DataFrame({
    "Confidence": ["95%", "99%"],
    "VaR": [
        historical_var(risk_returns, 0.95),
        historical_var(risk_returns, 0.99)
    ],
    "CVaR": [
        historical_cvar(risk_returns, 0.95),
        historical_cvar(risk_returns, 0.99)
    ]
})

st.dataframe(
    risk_table.style.format({
        "VaR": "{:.2%}",
        "CVaR": "{:.2%}"
    }),
    hide_index=True
)


# ------------------------------------------------
# 8. Portfolio composition
# ------------------------------------------------

st.subheader("Portfolio Allocation")

st.bar_chart(
    weights.rename("Portfolio Weight")
)

st.subheader("Trading Activity")

st.write(
    f"Total turnover: {bt['turnover'].sum():.4f}"
)

st.write(
    f"Total trading costs: "
    f"${metrics['total_trading_cost']:,.2f}"
)

st.caption(
    "Educational research application. "
    "Historical performance does not predict future returns. "
    "The model excludes initial acquisition costs, "
    "taxes, FX effects and market impact."
)

# ------------------------------------------------
# 9. Out-of-sample strategy comparison
# ------------------------------------------------

st.divider()
st.header("Out-of-Sample Strategy Comparison")

st.caption(
    "Expanding-window optimisation: train through 2023, "
    "test in 2024; retrain through 2024, test in 2025. "
    "No future returns are used to select weights."
)

with st.spinner("Running walk-forward backtests..."):

    min_bt, min_weights = walk_forward_backtest(
        returns,
        strategy="minimum_variance",
        training_years=3,
        cost_bps=cost_bps
    )

    max_bt, max_weights = walk_forward_backtest(
        returns,
        strategy="maximum_sharpe",
        training_years=3,
        cost_bps=cost_bps
    )

    # Match the benchmark to the same test dates.
    test_returns = returns.loc[min_bt.index]

    original_weights = pd.Series({
        "SPY": 0.50,
        "AGG": 0.30,
        "GLD": 0.20
    })

    original_bt = backtest_portfolio(
        test_returns,
        original_weights,
        initial_wealth=100000,
        rebalance="monthly",
        cost_bps=cost_bps
    )

# The walk-forward engine starts with $100,000.
# The benchmark uses the same initial wealth.

comparison = pd.DataFrame({
    "Original": original_bt["wealth"],
    "Minimum Variance": min_bt["wealth"],
    "Maximum Sharpe": max_bt["wealth"]
})

st.subheader("Growth of $100,000")

st.line_chart(comparison)


def comparison_stats(wealth):
    start_wealth = 100000.0
    wealth = wealth.astype(float)

    full_path = pd.concat([
        pd.Series([start_wealth]),
        wealth.reset_index(drop=True)
    ], ignore_index=True)

    drawdown = (
        full_path / full_path.cummax() - 1
    )

    return {
        "Final Wealth": wealth.iloc[-1],
        "Total Return": wealth.iloc[-1] / start_wealth - 1,
        "Maximum Drawdown": drawdown.min()
    }


summary = pd.DataFrame({
    name: comparison_stats(comparison[name])
    for name in comparison.columns
}).T

st.subheader("Performance Summary")

st.dataframe(
    summary.style.format({
        "Final Wealth": "${:,.2f}",
        "Total Return": "{:.2%}",
        "Maximum Drawdown": "{:.2%}"
    })
)

st.subheader("Optimised Annual Allocations")

left, right = st.columns(2)

with left:
    st.write("Minimum Variance")
    st.dataframe(
        min_weights.style.format("{:.1%}")
    )

with right:
    st.write("Maximum Sharpe")
    st.dataframe(
        max_weights.style.format("{:.1%}")
    )

st.info(
    "Optimised strategies use only earlier observations "
    "to estimate their portfolio weights. Performance "
    "is evaluated on subsequent out-of-sample data."
)

