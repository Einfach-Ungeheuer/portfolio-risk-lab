
import pandas as pd


def historical_var(returns, confidence=0.95):
    """Calculate historical Value at Risk."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    r = pd.Series(returns).dropna()

    if r.empty:
        raise ValueError("Returns cannot be empty.")

    return -float(r.quantile(1 - confidence))


def historical_cvar(returns, confidence=0.95):
    """Calculate historical Conditional Value at Risk."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    r = pd.Series(returns).dropna()

    if r.empty:
        raise ValueError("Returns cannot be empty.")

    cutoff = r.quantile(1 - confidence)

    return -float(r[r <= cutoff].mean())
