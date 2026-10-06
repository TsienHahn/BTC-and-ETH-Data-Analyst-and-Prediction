
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.preprocessing import (
    clean_market_data,
    temporal_split
)

from src.feature_engineering import (
    add_technical_indicators,
    create_sequences
)

from src.model import build_lstm_model

from src.evaluation import (
    calculate_metrics,
    plot_predictions
)

from src.utils import (
    PROJECT_ROOT,
    load_config,
    set_random_seed,
    create_project_directories
)


# ============================================================
# Feature and target configuration
# ============================================================

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


# ============================================================
# Main training pipeline
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------

    config = load_config(
        "config/config.yaml"
    )

    set_random_seed(
        config["project"]["random_seed"]
    )

    create_project_directories()

    # --------------------------------------------------------
    # 2. Load raw market data
    # --------------------------------------------------------

    data_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "test_crypto_data.csv"
    )

    print("\nLoading dataset:")
    print(data_path)

    df = pd.read_csv(
        data_path
    )

    print(
        f"Raw dataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # 3. Clean data and calculate technical indicators
    # --------------------------------------------------------

    df = clean_market_data(
        df
    )

    df = add_technical_indicators(
        df
    )

    print(
        f"Rows after preprocessing: {len(df)}"
    )

    # --------------------------------------------------------
    # 4. Select asset
    # --------------------------------------------------------

    asset = "ETH-USD"

    asset_df = (
        df[
            df["Asset"] == asset
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if asset_df.empty:
        raise ValueError(
            f"No data found for asset: {asset}"
        )

    print(
        f"\nSelected asset: {asset}"
    )

    print(
        f"Asset rows: {len(asset_df)}"
    )

    # --------------------------------------------------------
    # 5. Temporal train / validation / test split
    # --------------------------------------------------------

    train_df, validation_df, test_df = temporal_split(
        asset_df,
        train_ratio=0.80,
        validation_ratio=0.10
    )

    print("\nDataset split:")

    print(
        f"Train rows:      {len(train_df)}"
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    print(
        f"Test rows:       {len(test_df)}"
    )

    # --------------------------------------------------------
    # 6. Scaling
    # --------------------------------------------------------

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit ONLY on training data to avoid data leakage.
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

    test_features = feature_scaler.transform(
        test_df[FEATURE_COLUMNS]
    )

    test_targets = target_scaler.transform(
        test_df[TARGET_COLUMNS]
    )

    # --------------------------------------------------------
    # 7. Create LSTM sequences
    # --------------------------------------------------------

    time_steps = config["data"]["time_steps"]

    if len(train_df) <= time_steps:
        raise ValueError(
            "Training dataset is too small "
            f"for time_steps={time_steps}."
        )

    if len(validation_df) <= time_steps:
        raise ValueError(
            "Validation dataset is too small "
            f"for time_steps={time_steps}."
        )

    if len(test_df) <= time_steps:
        raise ValueError(
            "Test dataset is too small "
            f"for time_steps={time_steps}."
        )

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

    x_test, y_test = create_sequences(
        test_features,
        test_targets,
        time_steps
    )

    print("\nSequence shapes:")

    print(
        f"X train:      {x_train.shape}"
    )

    print(
        f"Y train:      {y_train.shape}"
    )

    print(
        f"X validation: {x_validation.shape}"
    )

    print(
        f"Y validation: {y_validation.shape}"
    )

    print(
        f"X test:       {x_test.shape}"
    )

    print(
        f"Y test:       {y_test.shape}"
    )

    # --------------------------------------------------------
    # 8. Build LSTM model
    # --------------------------------------------------------

    model = build_lstm_model(
        input_shape=(
            x_train.shape[1],
            x_train.shape[2]
        ),
        units=config["model"]["units"],
        output_size=len(TARGET_COLUMNS),
        learning_rate=config["model"]["learning_rate"]
    )

    print("\nModel architecture:")

    model.summary()

    # --------------------------------------------------------
    # 9. Training callbacks
    # --------------------------------------------------------

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=(
                PROJECT_ROOT
                / "models"
                / "best_eth_lstm.keras"
            ),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # --------------------------------------------------------
    # 10. Train model
    # --------------------------------------------------------

    print("\nStarting model training...\n")

    history = model.fit(
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

    # --------------------------------------------------------
    # 11. Save final model
    # --------------------------------------------------------

    model.save(
        PROJECT_ROOT
        / "models"
        / "final_eth_lstm.keras"
    )

    # --------------------------------------------------------
    # 12. Test prediction
    # --------------------------------------------------------

    print("\nEvaluating model on test data...")

    y_pred_scaled = model.predict(
        x_test,
        verbose=0
    )

    # Convert normalized values back to USD prices.
    y_test_original = (
        target_scaler.inverse_transform(
            y_test
        )
    )

    y_pred_original = (
        target_scaler.inverse_transform(
            y_pred_scaled
        )
    )

    # Target dates corresponding to the sequences.
    test_dates = (
        test_df["Date"]
        .iloc[time_steps:]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 13. Calculate evaluation metrics
    # --------------------------------------------------------

    metrics_results = []

    print("\n==============================")
    print("ETH-USD TEST RESULTS")
    print("==============================")

    for index, target_name in enumerate(
        TARGET_COLUMNS
    ):

        metrics = calculate_metrics(
            y_test_original[:, index],
            y_pred_original[:, index]
        )

        metrics["Target"] = target_name

        metrics_results.append(
            metrics
        )

        print(
            f"\n{target_name}"
        )

        print(
            f"MAE:      {metrics['MAE']:.4f}"
        )

        print(
            f"MSE:      {metrics['MSE']:.4f}"
        )

        print(
            f"RMSE:     {metrics['RMSE']:.4f}"
        )

        print(
            f"R²:       {metrics['R2']:.4f}"
        )

        print(
            "Accuracy: "
            f"{metrics['Custom_Accuracy'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # 14. Actual vs Predicted figure
        # ----------------------------------------------------

        plot_predictions(
            dates=test_dates,
            actual=y_test_original[:, index],
            predicted=y_pred_original[:, index],
            target_name=target_name,
            output_path=(
                PROJECT_ROOT
                / "results"
                / "figures"
                / f"ETH_{target_name}.png"
            )
        )

    # --------------------------------------------------------
    # 15. Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics_results
    )

    # Put Target as first column.
    metrics_df = metrics_df[
        [
            "Target",
            "MAE",
            "MSE",
            "RMSE",
            "R2",
            "Custom_Accuracy"
        ]
    ]

    metrics_path = (
        PROJECT_ROOT
        / "results"
        / "eth_test_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # 16. Save scalers
    # --------------------------------------------------------

    joblib.dump(
        feature_scaler,
        PROJECT_ROOT
        / "models"
        / "feature_scaler.pkl"
    )

    joblib.dump(
        target_scaler,
        PROJECT_ROOT
        / "models"
        / "target_scaler.pkl"
    )

    # --------------------------------------------------------
    # 17. Final output
    # --------------------------------------------------------

    print("\n==============================")
    print("TRAINING COMPLETED")
    print("==============================")

    print(
        f"Metrics saved to:\n{metrics_path}"
    )

    print(
        "\nPrediction figures saved to:"
    )

    print(
        PROJECT_ROOT
        / "results"
        / "figures"
    )


if __name__ == "__main__":
    main()

# input in Terminal  “python -m src.train“
