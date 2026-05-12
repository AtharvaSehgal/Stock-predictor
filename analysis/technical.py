# analysis/technical.py
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = df.copy()

    # ---- Trend Indicators ----
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_diff'] = ta.trend.macd_diff(df['Close'])

    # ---- Momentum Indicators ----
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Stoch_RSI'] = ta.momentum.stochrsi(df['Close'], window=14)

    # ---- Volatility Indicators ----
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_mid'] = bollinger.bollinger_mavg()
    df['BB_width'] = bollinger.bollinger_wband()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # ---- Volume Indicators ----
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['VWAP'] = ta.volume.volume_weighted_average_price(
        df['High'], df['Low'], df['Close'], df['Volume']
    )

    # ---- Signal Generation ----
    df['RSI_signal'] = np.where(df['RSI'] < 30, 'OVERSOLD',
                        np.where(df['RSI'] > 70, 'OVERBOUGHT', 'NEUTRAL'))

    df['MACD_signal_dir'] = np.where(df['MACD_diff'] > 0, 'BULLISH', 'BEARISH')

    df['BB_signal'] = np.where(df['Close'] < df['BB_lower'], 'BUY',
                       np.where(df['Close'] > df['BB_upper'], 'SELL', 'HOLD'))

    df.dropna(inplace=True)
    return df


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Get the latest signal summary from all indicators."""
    latest = df.iloc[-1]

    signals = {
        'RSI': round(latest['RSI'], 2),
        'RSI_signal': latest['RSI_signal'],
        'MACD': round(latest['MACD'], 4),
        'MACD_direction': latest['MACD_signal_dir'],
        'BB_signal': latest['BB_signal'],
        'price_vs_EMA20': 'ABOVE' if latest['Close'] > latest['EMA_20'] else 'BELOW',
        'price_vs_EMA50': 'ABOVE' if latest['Close'] > latest['EMA_50'] else 'BELOW',
        'ATR': round(latest['ATR'], 2),
        'current_price': round(latest['Close'], 2),
    }

    # Overall signal score
    bull_signals = sum([
        latest['RSI'] < 60,
        latest['MACD_diff'] > 0,
        latest['Close'] > latest['EMA_20'],
        latest['Close'] > latest['EMA_50'],
        latest['BB_signal'] == 'BUY',
    ])

    signals['overall_score'] = bull_signals
    signals['overall_signal'] = 'STRONG BUY' if bull_signals >= 4 else \
                                 'BUY' if bull_signals == 3 else \
                                 'HOLD' if bull_signals == 2 else \
                                 'SELL' if bull_signals == 1 else 'STRONG SELL'
    return signals


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data.fetcher import get_stock_data

    df = get_stock_data('RELIANCE.NS', period_days=180)
    df_with_indicators = add_technical_indicators(df)

    print("📊 Technical Indicators (last 3 rows):")
    print(df_with_indicators[['Close','RSI','MACD','BB_upper','BB_lower']].tail(3))

    print("\n🚦 Signal Summary:")
    summary = get_signal_summary(df_with_indicators)
    for k, v in summary.items():
        print(f"  {k}: {v}")