from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from src.api.schemas import (
    AssetAllocation,
    ForecastTrajectoryPoint,
    PortfolioForecastResponse,
    PortfolioMetrics
)

def compute_drift_and_covariance(price_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    trading_days = 252

    mean_daily_return = log_returns.mean().to_numpy()
    daily_cov_matrix = log_returns.cov().to_numpy()

    annual_cov_matrix = daily_cov_matrix * trading_days
    variances = np.diag(annual_cov_matrix)
    annual_drift = (mean_daily_return * trading_days) - (0.5 * variances)

    return annual_drift, annual_cov_matrix

def simulate_multivariate_gbm(
        initial_investment: float,
        weights: np.ndarray,
        annual_drift: np.ndarray,
        annual_cov_matrix: np.ndarray,
        horizon_days: int = 252,
        num_simulations: int = 5000,
        seed: Optional[int] = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_assets = len(weights)
    dt = 1.0 / 252.0

    try:
        lower_cholesky = np.linalg.cholesky(annual_cov_matrix)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(num_assets)
        lower_cholesky = np.linalg.cholesky(annual_cov_matrix + jitter)

    initial_asset_values = initial_investment * weights
    simulated_assets = np.zeros((num_simulations, horizon_days + 1, num_assets))
    simulated_assets[:, 0, :] = initial_asset_values

    drift_term = annual_drift * dt
    sqrt_dt = np.sqrt(dt)

    for t in range(1, horizon_days + 1):
        uncorrelated_z = rng.standard_normal((num_simulations, num_assets))
        correlated_shocks = uncorrelated_z @ lower_cholesky.T
        step_returns = np.exp(drift_term + sqrt_dt * correlated_shocks)
        simulated_assets[:, t, :] = simulated_assets[:, t - 1, :] * step_returns

    return np.sum(simulated_assets, axis=2)

def calculate_portfolio_metrics(
        portfolio_paths: np.ndarray,
        initial_investment: float,
        weights: np.ndarray,
        annual_drift: np.ndarray,
        annual_cov_matrix: np.ndarray,
        horizon_days: int
) -> PortfolioMetrics:
    portfolio_var = weights.T @ annual_cov_matrix @ weights
    portfolio_vol = float(np.sqrt(portfolio_var))
    portfolio_drift = float(np.sum(weights * annual_drift) + 0.5 * portfolio_var)

    # VaR 95% from the actual simulation, not a parametric formula
    # so it matches the trajectory / final bounds
    final_values = portfolio_paths[:, -1]
    var_95 = float(initial_investment - np.percentile(final_values, 5))

    running_peaks = np.maximum.accumulate(portfolio_paths, axis=1)
    drawdowns = (portfolio_paths - running_peaks) / running_peaks
    median_max_drawdown = float(np.abs(np.median(np.min(drawdowns, axis=1))))

    return PortfolioMetrics(
        expected_annual_return=round(portfolio_drift, 4),
        annualized_volatility=round(portfolio_vol, 4),
        var_95=round(var_95, 2),
        max_drawdown_estimate=round(median_max_drawdown, 4)
    )

def run_portfolio_monte_carlo(
        price_df: pd.DataFrame,
        allocations: List[AssetAllocation],
        initial_investment: float = 10000.0,
        horizon_days: int = 252,
        confidence_interval: float = 0.90,
        num_simulations: int = 5000,
        seed: Optional[int] = None
) -> PortfolioForecastResponse:
    tickers = [item.ticker for item in allocations]
    weights = np.array([item.weight for item in allocations], dtype=np.float64)

    # --- Validation ---
    if initial_investment <= 0:
        raise ValueError("initial_investment must be > 0")

    if not np.isclose(weights.sum(), 1.0, atol=0.01):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.4f}")

    missing = [t for t in tickers if t not in price_df.columns]
    if missing:
        raise ValueError(f"No price data found for tickers: {', '.join(missing)}")

    if price_df.empty:
        raise ValueError("No price data returned")

    aligned_prices = price_df[tickers]
    annual_drift, annual_cov = compute_drift_and_covariance(aligned_prices)

    paths = simulate_multivariate_gbm(
        initial_investment=initial_investment,
        weights=weights,
        annual_drift=annual_drift,
        annual_cov_matrix=annual_cov,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        seed=seed
    )

    alpha = (1.0 - confidence_interval) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    step = max(1, horizon_days // 30)
    day_indices = list(range(0, horizon_days + 1, step))
    if day_indices[-1] != horizon_days:
        day_indices.append(horizon_days)

    trajectory: List[ForecastTrajectoryPoint] = []
    for d in day_indices:
        day_slice = paths[:, d]
        trajectory.append(
            ForecastTrajectoryPoint(
                day=d,
                median_value=round(float(np.median(day_slice)), 2),
                lower_bound=round(float(np.percentile(day_slice, lower_pct)), 2),
                upper_bound=round(float(np.percentile(day_slice, upper_pct)), 2)
            )
        )

    metrics = calculate_portfolio_metrics(
        portfolio_paths=paths,
        initial_investment=initial_investment,
        weights=weights,
        annual_drift=annual_drift,
        annual_cov_matrix=annual_cov,
        horizon_days=horizon_days
    )

    # FIX: was paths[:-1] (all but last simulation), now paths[:, -1] (last day, all simulations)
    final_slice = paths[:, -1]
    return PortfolioForecastResponse(
        initial_investment=initial_investment,
        horizon_days=horizon_days,
        final_expected_value=round(float(np.median(final_slice)), 2),
        final_lower_bound=round(float(np.percentile(final_slice, lower_pct)), 2),
        final_upper_bound=round(float(np.percentile(final_slice, upper_pct)), 2),
        metrics=metrics,
        trajectory=trajectory
    )