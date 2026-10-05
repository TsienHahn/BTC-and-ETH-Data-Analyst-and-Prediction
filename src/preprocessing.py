"""Data cleaning, validation, and temporal splitting."""

import pandas as pd


REQUIRED_COLUMNS = [
    "Date", "Asset", "Open", "High", "Low", "Close", "Volume"
]

NUMERIC_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume"
]


def validate_columns(df: pd.DataFrame) -> None:
    """Check whether required columns are present."""

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize cryptocurrency market data."""

    validate_columns(df)

    cleaned = df.copy()

    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"],
        errors="coerce"
    )

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce"
        )

    cleaned = cleaned.drop_duplicates(
        subset=["Date", "Asset"]
    )

    cleaned = cleaned.dropna(
        subset=REQUIRED_COLUMNS
    )

    cleaned = cleaned[
        cleaned["Volume"] >= 0
    ]

    # Validate OHLC relationships.
    valid_prices = (
        (cleaned["High"] >= cleaned["Open"]) &
        (cleaned["High"] >= cleaned["Close"]) &
        (cleaned["Low"] <= cleaned["Open"]) &
        (cleaned["Low"] <= cleaned["Close"])
    )

    cleaned = cleaned[valid_prices]

    cleaned = cleaned.sort_values(
        ["Asset", "Date"]
    ).reset_index(drop=True)

    return cleaned


def generate_quality_report(df: pd.DataFrame) -> dict:
    """Generate basic data-quality statistics."""

    return {
        "rows": len(df),
        "duplicate_rows": int(
            df.duplicated(subset=["Date", "Asset"]).sum()
        ),
        "missing_values": df.isna().sum().to_dict(),
        "assets": sorted(df["Asset"].unique().tolist()),
        "start_date": str(df["Date"].min().date()),
        "end_date": str(df["Date"].max().date())
    }


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10
):
    """Split time-series data without random shuffling."""

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Train and validation ratios must sum to less than 1."
        )

    df = df.sort_values("Date").reset_index(drop=True)

    train_end = int(len(df) * train_ratio)
    validation_end = int(
        len(df) * (train_ratio + validation_ratio)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test
