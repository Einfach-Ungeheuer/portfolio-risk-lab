
import numpy as np
from scipy.optimize import minimize


def minimum_variance(cov):
    """
    Find the long-only minimum-variance portfolio.

    Parameters
    ----------
    cov : array-like
        Annualised covariance matrix.

    Returns
    -------
    numpy.ndarray
        Optimal asset weights.
    """
    cov = np.asarray(cov, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix must be finite.")

    if not np.allclose(cov, cov.T):
        raise ValueError("Covariance matrix must be symmetric.")

    n = cov.shape[0]

    if n == 0:
        raise ValueError("Covariance matrix cannot be empty.")

    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError(
            "Covariance matrix must be positive semidefinite."
        )

    def objective(w):
        return w @ cov @ w

    initial_weights = np.ones(n) / n

    constraints = [{
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }]

    bounds = [(0, 1)] * n

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000
        }
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x


def maximum_sharpe(mu, cov, risk_free_rate=0.04):
    """
    Find the long-only maximum-Sharpe portfolio.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    n = len(mu)

    if (
        mu.ndim != 1
        or cov.shape != (n, n)
        or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(cov))
    ):
        raise ValueError("Invalid return or covariance inputs.")

    if not np.allclose(cov, cov.T):
        raise ValueError("Covariance matrix must be symmetric.")

    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("Covariance matrix must be positive semidefinite.")

    def objective(w):
        variance = w @ cov @ w
        if variance <= 0:
            return 1e12

        return -(w @ mu - risk_free_rate) / np.sqrt(variance)

    result = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }],
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x

def target_return_min_variance(mu, cov, target_return):
    """
    Minimise portfolio variance subject to
    achieving a specified expected return.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    n = len(mu)

    if (
        mu.ndim != 1
        or cov.shape != (n, n)
        or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(cov))
        or not np.isfinite(target_return)
    ):
        raise ValueError("Invalid optimisation inputs.")

    if not np.allclose(cov, cov.T):
        raise ValueError("Covariance matrix must be symmetric.")

    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("Covariance matrix must be positive semidefinite.")

    if not mu.min() <= target_return <= mu.max():
        raise ValueError(
            "Target return is infeasible without short selling."
        )

    def objective(w):
        return w @ cov @ w

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        },
        {
            "type": "eq",
            "fun": lambda w: w @ mu - target_return
        }
    ]

    result = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x
