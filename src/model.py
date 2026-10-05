"""RNN-LSTM model architecture."""

from keras import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam


def build_lstm_model(
    input_shape: tuple,
    units: int = 256,
    output_size: int = 4,
    learning_rate: float = 0.0005
):
    """Build and compile a two-layer LSTM model."""

    model = Sequential([
        Input(shape=input_shape),

        LSTM(
            units=units,
            return_sequences=True
        ),

        Dropout(0.20),

        LSTM(
            units=units,
            return_sequences=False
        ),

        Dropout(0.20),

        Dense(64, activation="relu"),

        Dense(output_size)
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mean_squared_error"
    )

    return model

#test, can be deleted

if __name__ == "__main__":

    print("Testing LSTM model...")

    # assume:
    # time_steps = 10
    # 12 features per day
    input_shape = (10, 12)

    model = build_lstm_model(
        input_shape=input_shape,
        units=256,
        output_size=4,
        learning_rate=0.0005
    )

    print("\nModel created successfully!\n")

    model.summary()

    print("\n=== Model test completed ===")
