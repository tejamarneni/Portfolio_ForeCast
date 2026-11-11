from fastapi import APIRouter, HTTPException, status
from src.api.schemas import PortfolioForecastRequest, PortfolioForecastResponse
from src.ml.data_loader import fetch_historical_prices
from src.ml.predict import run_portfolio_monte_carlo
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["Forecasting"])

@router.post(
    "/portfolio",
    response_model=PortfolioForecastResponse,
    status_code=status.HTTP_200_OK
)
def forecast_portfolio(payload: PortfolioForecastRequest) -> PortfolioForecastResponse:
    """
    Run a Monte Carlo forecast for a portfolio.

    - Fetches 3 years of historical prices for the given tickers
    - Runs Monte Carlo simulation with the given allocations
    """
    try:
        tickers = [item.ticker for item in payload.portfolio]

        price_df = fetch_historical_prices(
            tickers=tickers,
            lookback_years=3
        )

        return run_portfolio_monte_carlo(
            price_df=price_df,
            allocations=payload.portfolio,
            initial_investment=payload.initial_investment,
            horizon_days=payload.horizon_days,
            confidence_interval=payload.confidence_interval
        )

    except ValueError as exc:
        # Bad ticker, empty data, bad allocation, etc. - client error, not server crash
        logger.warning("Forecast validation failed for %s: %s", tickers, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Real unexpected crash
        logger.exception("Forecast failed for tickers: %s", tickers)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast execution failed: {str(exc)}",
        ) from exc