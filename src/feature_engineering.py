"""Technical indicators and sequence generation."""

import numpy as np
import pandas as pd


def add_technical_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate technical indicators separately for each asset."""

    result_frames = []

    for asset, asset_df in df.groupby("Asset"):
        data = asset_df.copy()
        data = data.sort_values("Date")

        data["SMA_10"] = (
            data["Close"].rolling(window=10).mean()
        )

        data["SMA_30"] = (
            data["Close"].rolling(window=30).mean()
        )

        data["EMA_5"] = (
            data["Close"].ewm(span=5, adjust=False).mean()
        )

        data["EMA_20"] = (
            data["Close"].ewm(span=20, adjust=False).mean()
        )

        # RSI
        change = data["Close"].diff()
        gain = change.clip(lower=0)
        loss = -change.clip(upper=0)

        average_gain = gain.rolling(window=14).mean()
        average_loss = loss.rolling(window=14).mean()

        relative_strength = (
            average_gain /
            average_loss.replace(0, np.nan)
        )

        data["RSI_14"] = (
            100 - (100 / (1 + relative_strength))
        )

        # MACD
        ema_12 = data["Close"].ewm(
            span=12,
            adjust=False
        ).mean()

        ema_26 = data["Close"].ewm(
            span=26,
            adjust=False
        ).mean()

        data["MACD"] = ema_12 - ema_26

        data["MACD_Signal"] = (
            data["MACD"]
            .ewm(span=9, adjust=False)
            .mean()
        )

        result_frames.append(data)

    result = pd.concat(
        result_frames,
        ignore_index=True
    )

    INDICATOR_COLUMNS = [
    "SMA_10",
    "SMA_30",
    "EMA_5",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal"
    ]

    return (
    result
    .dropna(subset=INDICATOR_COLUMNS)
    .sort_values(["Asset", "Date"])
    .reset_index(drop=True)
)


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    time_steps: int = 10
):
    """Convert chronological observations into LSTM sequences."""

    x_values = []
    y_values = []

    for index in range(time_steps, len(features)):
        x_values.append(
            features[index - time_steps:index]
        )

        y_values.append(
            targets[index]
        )

    return np.asarray(x_values), np.asarray(y_values)
