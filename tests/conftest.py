import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """Generates 300 days of synthetic correlated daily prices for SCHG and SCHX."""
    rng = np.random.default_rng(42)
    days = 300
    dates = pd.bdate_range(end="2026-11-01", periods=days)

    # Correlated daily returns
    cov = np.array([[0.00020, 0.00014], [0.00014, 0.00012]])
    mean = np.array([0.0006, 0.00045])
    returns = rng.multivariate_normal(mean, cov, size=days)

    prices_schg = 100.0 * np.exp(np.cumsum(returns[:, 0]))
    prices_schx = 100.0 * np.exp(np.cumsum(returns[:, 1]))

    return pd.DataFrame(
        {"SCHG": prices_schg, "SCHX": prices_schx},
        index=dates,
    )
