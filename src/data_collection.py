"""Download cryptocurrency OHLCV data from Yahoo Finance."""

from pathlib import Path
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = [
    "Date", "Asset", "Open", "High", "Low", "Close", "Volume"
]


def download_crypto_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Download daily OHLCV data for one cryptocurrency."""

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Newer yfinance versions may return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Asset"] = ticker

    return df[REQUIRED_COLUMNS]


def collect_multiple_assets(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Download and combine data for multiple assets."""

    frames = [
        download_crypto_data(ticker, start_date, end_date)
        for ticker in tickers
    ]

    return pd.concat(frames, ignore_index=True)


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save a DataFrame as a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

# Test if it can download the data

#if __name__ == "__main__":

#    print("Starting data download...")

#   df = collect_multiple_assets(
#        tickers=["BTC-USD", "ETH-USD"],
#        start_date="2025-01-01",
#        end_date="2025-01-10"
#    )

#    print("\nDownload successful!")

#    print("\nFirst 5 rows:")
#   print(df.head())

#    print("\nDataset shape:")
#    print(df.shape)

#    print("\nColumns:")
#    print(df.columns.tolist())

#    print("\nAssets:")
#    print(df["Asset"].value_counts())

#    save_dataset(
#        df,
#        "data/raw/test_crypto_data.csv"
#    )

#    print("\nFile saved successfully!")
