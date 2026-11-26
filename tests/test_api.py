from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("src.api.routes.forecast.fetch_historical_prices")
def test_forecast_portfolio_success(mock_fetch, sample_price_df: pd.DataFrame):
    mock_fetch.return_value = sample_price_df

    payload = {
        "portfolio": [
            {"ticker": "schg", "weight": 0.80},
            {"ticker": "schx", "weight": 0.20},
        ],
        "initial_investment": 10000.0,
        "horizon_days": 252,
        "confidence_interval": 0.90,
        "model_type": "geometric_brownian_motion",
    }

    response = client.post("/forecast/portfolio", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["initial_investment"] == 10000.0
    assert data["horizon_days"] == 252
    assert len(data["trajectory"]) > 2
    assert "expected_annual_return" in data["metrics"]
    mock_fetch.assert_called_once_with(tickers=["SCHG", "SCHX"], lookback_years=3)


def test_forecast_invalid_weight_sum_returns_422():
    payload = {
        "portfolio": [
            {"ticker": "SCHG", "weight": 0.50},
            {"ticker": "SCHX", "weight": 0.20},
        ],
        "initial_investment": 10000.0,
        "horizon_days": 252,
    }

    response = client.post("/forecast/portfolio", json=payload)
    assert response.status_code == 422


def test_forecast_empty_portfolio_returns_422():
    payload = {
        "portfolio": [],
        "initial_investment": 10000.0,
        "horizon_days": 252,
    }

    response = client.post("/forecast/portfolio", json=payload)
    assert response.status_code == 422
