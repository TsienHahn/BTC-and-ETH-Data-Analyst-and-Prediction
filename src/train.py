"""Train the cryptocurrency forecasting model."""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.preprocessing import clean_market_data, temporal_split
from src.feature_engineering import (
    add_technical_indicators,
    create_sequences
)
from src.model import build_lstm_model
from src.utils import (
    load_config,
    set_random_seed,
    create_project_directories
)


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_10",
    "SMA_30",
    "EMA_5",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal"
]

TARGET_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close"
]


def main():
    config = load_config("config/config.yaml")

    set_random_seed(
        config["project"]["random_seed"]
    )

    create_project_directories()

    df = pd.read_csv(
        "data/raw/crypto_market_data.csv"
    )

    df = clean_market_data(df)
    df = add_technical_indicators(df)

    # Train one model for one asset.
    asset = "ETH-USD"

    asset_df = (
        df[df["Asset"] == asset]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    train_df, validation_df, test_df = temporal_split(
        asset_df,
        train_ratio=0.80,
        validation_ratio=0.10
    )

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Important: fit scalers only on training data.
    feature_scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    target_scaler.fit(
        train_df[TARGET_COLUMNS]
    )

    train_features = feature_scaler.transform(
        train_df[FEATURE_COLUMNS]
    )

    train_targets = target_scaler.transform(
        train_df[TARGET_COLUMNS]
    )

    validation_features = feature_scaler.transform(
        validation_df[FEATURE_COLUMNS]
    )

    validation_targets = target_scaler.transform(
        validation_df[TARGET_COLUMNS]
    )

    time_steps = config["data"]["time_steps"]

    x_train, y_train = create_sequences(
        train_features,
        train_targets,
        time_steps
    )

    x_validation, y_validation = create_sequences(
        validation_features,
        validation_targets,
        time_steps
    )

    model = build_lstm_model(
        input_shape=(
            x_train.shape[1],
            x_train.shape[2]
        ),
        units=config["model"]["units"],
        output_size=len(TARGET_COLUMNS),
        learning_rate=config["model"]["learning_rate"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath="models/best_eth_lstm.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation
        ),
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        callbacks=callbacks,
        shuffle=False
    )

    model.save(
        "models/final_eth_lstm.keras"
    )

    joblib.dump(
        feature_scaler,
        "models/feature_scaler.pkl"
    )

    joblib.dump(
        target_scaler,
        "models/target_scaler.pkl"
    )


if __name__ == "__main__":
    main()
