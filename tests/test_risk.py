
import pandas as pd

from src.risk import historical_var, historical_cvar


def test_historical_var():
    returns = pd.Series([
        -0.10, -0.05, 0.00, 0.05, 0.10
    ])

    result = historical_var(returns, confidence=0.80)

    assert abs(result - 0.06) < 1e-10


def test_historical_cvar():
    returns = pd.Series([
        -0.10, -0.05, 0.00, 0.05, 0.10
    ])

    result = historical_cvar(returns, confidence=0.80)

    assert abs(result - 0.10) < 1e-10


def test_invalid_confidence():
    returns = pd.Series([-0.02, 0.01, 0.03])

    try:
        historical_var(returns, confidence=1.5)
        assert False, "Expected ValueError"
    except ValueError:
        pass
