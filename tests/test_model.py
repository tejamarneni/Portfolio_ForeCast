import numpy as np
import pandas as pd
from src.api.schemas import AssetAllocation
from src.ml.predict import (
    compute_drift_and_covariance,
    run_portfolio_monte_carlo,
    simulate_multivariate_gbm,
)


def test_drift_and_covariance_shapes(sample_price_df: pd.DataFrame):
    drift, cov, log_rets = compute_drift_and_covariance(sample_price_df)

    assert drift.shape == (2,)
    assert cov.shape == (2, 2)
    assert log_rets.shape == (len(sample_price_df) - 1, 2)
    # Covariance matrix must be symmetric positive semi-definite
    assert np.allclose(cov, cov.T)


def test_simulate_multivariate_gbm_output_integrity():
    weights = np.array([0.8, 0.2])
    drift = np.array([0.12, 0.09])
    cov = np.array([[0.04, 0.025], [0.025, 0.0225]])

    paths = simulate_multivariate_gbm(
        initial_investment=10000.0,
        weights=weights,
        annual_drift=drift,
        annual_cov_matrix=cov,
        horizon_days=252,
        num_simulations=500,
        seed=42,
    )

    assert paths.shape == (500, 253)
    assert np.allclose(paths[:, 0], 10000.0)
    assert np.all(paths > 0.0)


def test_collinear_assets_cholesky_jitter_fallback():
    """Ensures duplicate/perfectly correlated covariance matrices do not raise LinAlgError."""
    weights = np.array([0.5, 0.5])
    drift = np.array([0.10, 0.10])
    # Singular covariance matrix (rank 1)
    singular_cov = np.array([[0.04, 0.04], [0.04, 0.04]])

    paths = simulate_multivariate_gbm(
        initial_investment=10000.0,
        weights=weights,
        annual_drift=drift,
        annual_cov_matrix=singular_cov,
        horizon_days=30,
        num_simulations=100,
    )

    assert paths.shape == (100, 31)
    assert not np.isnan(paths).any()


def test_run_portfolio_monte_carlo_ordering(sample_price_df: pd.DataFrame):
    allocations = [
        AssetAllocation(ticker="SCHG", weight=0.80),
        AssetAllocation(ticker="SCHX", weight=0.20),
    ]

    result = run_portfolio_monte_carlo(
        price_df=sample_price_df,
        allocations=allocations,
        initial_investment=10000.0,
        horizon_days=126,
        confidence_interval=0.90,
        num_simulations=1000,
    )

    assert result.final_lower_bound <= result.final_expected_value <= result.final_upper_bound
    assert result.metrics.var_95 > 0
    assert 0.0 <= result.metrics.max_drawdown_estimate <= 1.0
    assert result.trajectory[0].day == 0
    assert result.trajectory[-1].day == 126
