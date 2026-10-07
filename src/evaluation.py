"""Model evaluation and result visualization."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict:
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    denominator = np.mean(np.abs(actual))

    custom_accuracy = (
        1 - mae / denominator
        if denominator != 0
        else np.nan
    )

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Custom_Accuracy": float(custom_accuracy)
    }


def plot_predictions(
    dates,
    actual,
    predicted,
    target_name: str,
    output_path: str
) -> None:
    """Plot actual and predicted values."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        dates,
        actual,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        dates,
        predicted,
        label="Predicted",
        linewidth=2
    )

    plt.title(
        f"Actual vs Predicted {target_name}"
    )

    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
