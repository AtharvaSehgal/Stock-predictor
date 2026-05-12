# models/lstm_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved')
os.makedirs(MODELS_DIR, exist_ok=True)


def prepare_data(df: pd.DataFrame, lookback: int = 60):
    """Prepare sequences for LSTM training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Use multiple features
    features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_width', 'ATR']
    available = [f for f in features if f in df.columns]
    
    data = df[available].values
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])  # Predict Close price

    X, y = np.array(X), np.array(y)

    # 80/20 train-test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, available


def build_model(input_shape: tuple) -> Sequential:
    """Build the LSTM neural network."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber')
    return model


def train_model(symbol: str, df: pd.DataFrame, lookback: int = 60, epochs: int = 50):
    """Train and save LSTM model for a stock."""
    print(f"🧠 Training LSTM for {symbol}...")
    print(f"   Data points: {len(df)} | Lookback: {lookback} days")

    X_train, X_test, y_train, y_test, scaler, features = prepare_data(df, lookback)

    print(f"   Features used: {features}")
    print(f"   Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    model = build_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test)
    
    # Inverse scale for real prices
    dummy = np.zeros((len(y_test), len(features)))
    dummy[:, 0] = y_test
    actual = scaler.inverse_transform(dummy)[:, 0]

    dummy[:, 0] = y_pred.flatten()
    predicted = scaler.inverse_transform(dummy)[:, 0]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    accuracy = 100 - (mae / actual.mean() * 100)

    print(f"\n📊 Model Performance:")
    print(f"   MAE:  ₹{mae:.2f}")
    print(f"   RMSE: ₹{rmse:.2f}")
    print(f"   Accuracy: ~{accuracy:.1f}%")

    # Save model + scaler
    model_path = os.path.join(MODELS_DIR, f"{symbol.replace('.', '_')}.keras")
    model.save(model_path)
    print(f"   ✅ Model saved to {model_path}")

    return model, scaler, features, {'mae': mae, 'rmse': rmse, 'accuracy': accuracy}


def predict_next_days(
    symbol: str,
    df: pd.DataFrame,
    scaler,
    features: list,
    days_ahead: int = 5,
    lookback: int = 60
):
    """Predict next N days of prices."""
    model_path = os.path.join(MODELS_DIR, f"{symbol.replace('.', '_')}.keras")
    
    if not os.path.exists(model_path):
        print(f"No saved model for {symbol}. Train first.")
        return None

    model = load_model(model_path)
    data = df[features].values
    scaled = scaler.transform(data)

    predictions = []
    current_seq = scaled[-lookback:].copy()

    for _ in range(days_ahead):
        input_seq = current_seq.reshape(1, lookback, len(features))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]

        # Decode prediction to real price
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0][0]
        predictions.append(round(pred_price, 2))

        # Slide window forward
        new_row = current_seq[-1].copy()
        new_row[0] = pred_scaled
        current_seq = np.vstack([current_seq[1:], new_row])

    current_price = df['Close'].iloc[-1]
    change_pct = ((predictions[-1] - current_price) / current_price) * 100

    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'predictions': predictions,
        'days_ahead': days_ahead,
        'predicted_trend': 'UP 📈' if change_pct > 0 else 'DOWN 📉',
        'expected_change_pct': round(change_pct, 2)
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    
    from data.fetcher import get_stock_data
    from analysis.technical import add_technical_indicators

    symbol = 'RELIANCE.NS'
    print(f"🚀 Training LSTM model for {symbol}\n")

    df = get_stock_data(symbol, period_days=730)  # 2 years of data
    df = add_technical_indicators(df)

    model, scaler, features, metrics = train_model(symbol, df, epochs=50)

    print("\n🔮 Predicting next 5 days...")
    result = predict_next_days(symbol, df, scaler, features, days_ahead=5)

    if result:
        print(f"\nCurrent price: ₹{result['current_price']}")
        print(f"Trend: {result['predicted_trend']}")
        print(f"Expected change: {result['expected_change_pct']}%")
        print("Day-by-day predictions:")
        for i, price in enumerate(result['predictions'], 1):
            print(f"  Day {i}: ₹{price}")