# Portfolio Forecaster

A production-ready FastAPI service and machine learning engine for financial asset data ingestion, portfolio simulation, and risk forecasting.

---

## Architecture Overview

```text
portfolio-forecaster/
├── data/
│   ├── raw/             # Parquet-cached price histories
│   └── processed/       # Calculated returns & covariance matrices
├── models/              # Serialized model artifacts & metadata
├── notebooks/           # Exploratory analysis & prototyping
├── src/
│   ├── api/             # FastAPI app, schemas, and routes
│   ├── core/            # Centralized settings & path configuration
│   └── ml/              # Data ingestion, feature engineering, and inference
└── tests/               # Unit, integration, and API test suites