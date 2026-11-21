
import numpy as np


def simulate_normal_returns(
    returns,
    n_simulations=10_000,
    seed=42
):
    """
    Simulate one-day portfolio returns under
    a fitted normal distribution.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if r.size < 2:
        raise ValueError(
            "At least two valid returns are required."
        )

    if not isinstance(n_simulations, int) or n_simulations <= 0:
        raise ValueError(
            "n_simulations must be a positive integer."
        )

    mu = r.mean()
    sigma = r.std(ddof=1)

    rng = np.random.default_rng(seed)

    return rng.normal(
        loc=mu,
        scale=sigma,
        size=n_simulations
    )

def simulate_wealth_paths(
    returns,
    initial_wealth=100_000,
    n_days=252,
    n_paths=10_000,
    seed=42
):
    """
    Simulate multi-day portfolio wealth paths.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if r.size < 2:
        raise ValueError("At least two returns required.")

    if not np.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("Initial wealth must be positive.")

    if not isinstance(n_days, int) or n_days <= 0:
        raise ValueError("n_days must be positive.")

    if not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    mu = r.mean()
    sigma = r.std(ddof=1)

    rng = np.random.default_rng(seed)

    simulated_returns = rng.normal(
        loc=mu,
        scale=sigma,
        size=(n_days, n_paths)
    )

    wealth_paths = initial_wealth * np.vstack([
        np.ones(n_paths),
        np.cumprod(1 + simulated_returns, axis=0)
    ])

    return wealth_paths
