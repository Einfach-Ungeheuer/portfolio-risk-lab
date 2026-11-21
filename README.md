
# Portfolio Analytics & Risk Lab

An interactive quantitative finance application for portfolio risk analysis, optimisation, simulation, and backtesting.

Built with Python, NumPy, pandas, SciPy, yfinance, and Streamlit.

## Overview

This project examines a portfolio consisting of:

- SPY — US equities
- AGG — US investment-grade bonds
- GLD — Gold

Historical analysis covers January 2021 through December 2025.

The application combines quantitative portfolio research with an interactive dashboard.

## Features

### Portfolio Risk Analytics
- Annualised returns and volatility
- Sharpe ratio
- Maximum drawdown
- Historical Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Correlation analysis

### Monte Carlo Simulation
- Simulated portfolio wealth paths
- Terminal wealth distributions
- Loss probabilities
- Drawdown analysis

### Portfolio Optimisation
- Minimum-variance allocation
- Maximum-Sharpe allocation
- Mean-variance optimisation

### Backtesting
- Daily and monthly rebalancing
- Buy-and-hold comparison
- Transaction costs
- Portfolio turnover
- Walk-forward optimisation
- Out-of-sample strategy comparison

### Interactive Dashboard
- Adjustable portfolio allocations
- Custom initial investment
- Transaction cost assumptions
- Performance and drawdown charts
- Comparison of original and optimised portfolios

## Methodology

The portfolio uses historical daily adjusted closing prices downloaded through yfinance.

Daily asset returns are computed using percentage changes.

Portfolio optimisation uses estimated annualised expected returns and covariance matrices, subject to long-only portfolio constraints.

Walk-forward testing uses an expanding training window:

| Training period | Test period |
|---|---|
| 2021–2023 | 2024 |
| 2021–2024 | 2025 |

This approach reduces look-ahead bias by estimating portfolio weights using observations preceding each test year.

The backtesting engine models rebalancing, portfolio weight drift, turnover, and proportional trading costs.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-risk-lab.git
cd portfolio-risk-lab
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Automated Tests

Run:

```bash
python -m pytest -v
```

The initial test suite contains 25 passing tests covering portfolio risk calculations, Monte Carlo simulation, optimisation, backtesting, and walk-forward functionality.

## Limitations

- Historical performance does not guarantee future results.
- Optimisation is sensitive to estimated expected returns and covariance matrices.
- The model assumes long-only portfolios.
- Transaction costs are simplified and exclude market impact.
- Initial acquisition costs, taxes, and FX effects are excluded.
- The asset universe is limited to three ETFs.
- The analysis is intended for research and education, not investment recommendations.

## Future Development

- Additional portfolio constraints
- Rolling risk estimation
- Benchmark comparisons
- Broader asset universe
- More comprehensive out-of-sample validation

## Author

Mostafa Raziebrahimsaraei

Mathematics PhD | Quantitative Finance & Computational Research
