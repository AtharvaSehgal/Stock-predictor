# utils/backtester.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px


def run_backtest(df: pd.DataFrame, scaler, features: list, model, lookback: int = 60) -> dict:
    """
    Run a full backtest — predict every day in test set
    and compare against actual prices.
    """
    from sklearn.preprocessing import MinMaxScaler

    data = df[features].values
    scaled = scaler.transform(data)

    # Use last 20% as test window
    split = int(len(scaled) * 0.8)
    test_scaled = scaled[split - lookback:]

    actuals = []
    predictions = []

    for i in range(lookback, len(test_scaled)):
        seq = test_scaled[i - lookback:i].reshape(1, lookback, len(features))
        pred_scaled = model.predict(seq, verbose=0)[0][0]

        # Decode to real price
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0][0]

        actual_price = df['Close'].iloc[split + (i - lookback)]

        predictions.append(round(pred_price, 2))
        actuals.append(round(actual_price, 2))

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    # ---- Metrics ----
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Directional accuracy — did it predict UP/DOWN correctly?
    actual_dirs = np.sign(np.diff(actuals))
    pred_dirs = np.sign(np.diff(predictions))
    directional_accuracy = np.mean(actual_dirs == pred_dirs) * 100

    # Within 2% accuracy
    within_2pct = np.mean(np.abs((actuals - predictions) / actuals) < 0.02) * 100
    within_5pct = np.mean(np.abs((actuals - predictions) / actuals) < 0.05) * 100

    # Daily returns comparison
    actual_returns = pd.Series(actuals).pct_change().dropna()
    pred_returns = pd.Series(predictions).pct_change().dropna()
    correlation = actual_returns.corr(pred_returns)

    # Build results dataframe
    test_dates = df.index[split:][:len(predictions)]
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': predictions - actuals,
        'Error_Pct': ((predictions - actuals) / actuals) * 100
    })

    return {
        'metrics': {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'Directional Accuracy': round(directional_accuracy, 2),
            'Within 2%': round(within_2pct, 2),
            'Within 5%': round(within_5pct, 2),
            'Correlation': round(correlation, 4),
        },
        'results_df': results_df,
        'actuals': actuals,
        'predictions': predictions,
        'test_dates': test_dates,
    }


def plot_backtest(results: dict, symbol: str):
    """Plot actual vs predicted prices."""
    df = results['results_df']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual'],
        name='Actual Price',
        line=dict(color='#00CC96', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Predicted'],
        name='Predicted Price',
        line=dict(color='#FFA500', width=2, dash='dot')
    ))

    # Shade error region
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([df['Actual'], df['Predicted'][::-1]]),
        fill='toself',
        fillcolor='rgba(255,165,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Error Band'
    ))

    fig.update_layout(
        title=f'{symbol} — Actual vs Predicted (Backtest)',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=450,
        hovermode='x unified'
    )

    return fig


def plot_error_distribution(results: dict):
    """Plot prediction error distribution."""
    errors = results['results_df']['Error_Pct']

    fig = px.histogram(
        errors,
        nbins=40,
        title='Prediction Error Distribution (%)',
        labels={'value': 'Error %', 'count': 'Frequency'},
        color_discrete_sequence=['#636EFA']
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Perfect")
    fig.add_vline(x=errors.mean(), line_dash="dash", line_color="orange",
                  annotation_text=f"Mean: {errors.mean():.2f}%")
    fig.update_layout(height=350)
    return fig


def plot_directional_accuracy(results: dict):
    """Plot day by day directional prediction (UP/DOWN)."""
    df = results['results_df'].copy()
    df['Actual_Dir'] = df['Actual'].diff().apply(lambda x: '📈 UP' if x > 0 else '📉 DOWN')
    df['Pred_Dir'] = df['Predicted'].diff().apply(lambda x: '📈 UP' if x > 0 else '📉 DOWN')
    df['Correct'] = df['Actual_Dir'] == df['Pred_Dir']
    df = df.dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual'],
        mode='markers',
        marker=dict(
            color=df['Correct'].map({True: 'green', False: 'red'}),
            size=8,
            symbol=df['Actual_Dir'].map({'📈 UP': 'triangle-up', '📉 DOWN': 'triangle-down'})
        ),
        text=df.apply(lambda r: f"{'✅ Correct' if r['Correct'] else '❌ Wrong'}<br>Actual: {r['Actual_Dir']}<br>Predicted: {r['Pred_Dir']}", axis=1),
        hoverinfo='text+x',
        name='Direction Accuracy'
    ))
    fig.update_layout(
        title='Directional Accuracy (Green = Correct, Red = Wrong)',
        height=350
    )
    return fig