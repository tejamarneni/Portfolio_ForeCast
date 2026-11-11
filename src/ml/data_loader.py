from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_path(ticker: str, lookback_years: int) -> Path:
    # lookback is in the filename so a 1y cache is never reused for a 3y request
    return CACHE_DIR / f"{ticker.upper()}_{lookback_years}y.parquet"

def is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
    if not cache_path.exists():
        return False
    modified = datetime.fromtimestamp(cache_path.stat().st_mtime)  # mtime, not atime
    return (datetime.now() - modified) < timedelta(hours=max_age_hours)

def _download_close(tickers: List[str], lookback_years: int) -> pd.DataFrame:
    """One batched yfinance call; returns adjusted close, one column per ticker."""
    end = datetime.now() + timedelta(days=1)  # yfinance `end` is exclusive
    start = end - timedelta(days=lookback_years * 365)

    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,  # "Close" is already split/dividend adjusted
    )
    if data.empty:
        raise ValueError(f"No price history found for: {tickers}")

    # yfinance can return MultiIndex columns (Price, Ticker) in newer versions
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data["Close"]

    if isinstance(close, pd.Series):  # defensive: single-ticker edge case
        close = close.to_frame(tickers[0])
    return close

def fetch_historical_prices(
    tickers: List[str], lookback_years: int = 3, min_rows: int = 50
) -> pd.DataFrame:
    # dedupe while preserving order
    clean = list(dict.fromkeys(t.strip().upper() for t in tickers))

    frames, to_download = {}, []
    for t in clean:
        path = get_cache_path(t, lookback_years)
        if is_cache_valid(path):
            frames[t] = pd.read_parquet(path)["Close"]
        else:
            to_download.append(t)

    if to_download:
        close = _download_close(to_download, lookback_years)
        for t in to_download:
            if t not in close.columns or close[t].dropna().empty:
                raise ValueError(f"No price history found for ticker: '{t}'")
            s = close[t].dropna().rename("Close")
            s.to_frame().to_parquet(get_cache_path(t, lookback_years))
            frames[t] = s

    price_df = pd.concat([frames[t] for t in clean], axis=1, keys=clean)

    # report which ticker is limiting the overlap before dropping rows
    first_valid = price_df.apply(pd.Series.first_valid_index)
    limiting = first_valid.idxmax()

    price_df = price_df.dropna()
    if len(price_df) < min_rows:
        raise ValueError(
            f"Only {len(price_df)} overlapping rows (need {min_rows}). "
            f"Shortest history: '{limiting}', starting {first_valid[limiting].date()}."
        )

    return price_df

def fetch_single_ticker(ticker: str, lookback_years: int = 3) -> pd.Series:
    ticker = ticker.strip().upper()

    # FIX 1: was get_cache_path(ticker) - missing lookback_years, crashed with TypeError
    cache_file = get_cache_path(ticker, lookback_years)

    if is_cache_valid(cache_file):
        df_cached = pd.read_parquet(cache_file)
        # FIX 2: was "Adj Close", now "Close" to match fetch_historical_prices
        # Both functions now read/write the same column, so caches are interchangeable
        return df_cached["Close"]

    # yf.Ticker(...).history() returns standard single-level columns
    t = yf.Ticker(ticker)
    history = t.history(period=f"{lookback_years}y")

    if history.empty:
        raise ValueError(f"No price history found for ticker: '{ticker}'")

    # history() auto-adjusts close prices by default
    close_series = history["Close"].copy()
    close_series.name = ticker

    # FIX 2: was {"Adj Close": ...}, now {"Close": ...}
    cache_df = pd.DataFrame({"Close": close_series})
    cache_df.to_parquet(cache_file)

    return close_series

if __name__ == "__main__":
    df = fetch_historical_prices(["AAPL", "MSFT", "SPY"])
    print(df.tail())
    print(df.shape)