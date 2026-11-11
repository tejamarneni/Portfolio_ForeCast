from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator

class AssetAllocation(BaseModel):
    ticker: str = Field(..., description="Ticker Symbol")
    weight: float = Field(..., gt=0.0, le=1.0, description="Allocation weight")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        normalized = v.strip().upper()
        if not normalized:
            raise ValueError("Ticker cannot be empty")
        return normalized

class PortfolioForecastRequest(BaseModel):
    portfolio: List[AssetAllocation] = Field(..., min_length=1)
    initial_investment: float = Field(default=10000.0, gt=0.0)
    horizon_days: int = Field(default=252, ge=5, le=1260)
    confidence_interval: float = Field(default=0.90, gt=0.50, lt=1.0)

    @model_validator(mode="after")
    def validate_portfolio(self):
        # 1. No duplicate tickers - AAPL 0.5 + AAPL 0.5 would break covariance
        tickers = [item.ticker for item in self.portfolio]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers are not allowed")

        # 2. Weights must sum to ~1.0
        total_weight = sum(item.weight for item in self.portfolio)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0 (current sum: {total_weight:.4f})")
        return self

class ForecastTrajectoryPoint(BaseModel):
    day: int
    median_value: float
    lower_bound: float
    upper_bound: float

class PortfolioMetrics(BaseModel):
    expected_annual_return: float
    annualized_volatility: float
    var_95: float
    max_drawdown_estimate: float

class PortfolioForecastResponse(BaseModel):
    initial_investment: float
    horizon_days: int
    final_expected_value: float
    final_lower_bound: float
    final_upper_bound: float
    metrics: PortfolioMetrics
    trajectory: List[ForecastTrajectoryPoint]