import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

from data_preparation import load_data, preprocess_data


def train_model_lstm(data):
    """Placeholder for LSTM model training.
    LSTM models are typically used for sequential data and may not be directly applicable here.

    Training data: time,A,B,C,D,E,F,G,H,I,J,K,L,M,N,Y1,Y2
    """
    # Extract features and targets
    feature_cols = data.columns[1:-2]  # A to N
    X = data[feature_cols].values
    y = data[['Y1', 'Y2']].values
    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    # Split into training and validation sets
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(2))  # Two outputs: Y1 and Y2
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), verbose=1)
    # Evaluate model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f'LSTM Model - MSE: {mse}, R^2: {r2}')
    # Note: Further hyperparameter tuning and model adjustments may be needed for optimal performance

    return model, scaler


if __name__ == "__main__":
    # Example usage (not integrated into main workflow)
    data = load_data("train.csv")
    # Normalize features except the first column (time) and last two columns (Y1, Y2)
    data = preprocess_data(data)
    # Train LSTM model
    print("Training LSTM model...")
    model = train_model_lstm(data)