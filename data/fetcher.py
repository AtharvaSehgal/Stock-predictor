# data/fetcher.py
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

try:
    import streamlit as st
    STREAMLIT = True
except:
    STREAMLIT = False


def cache_it(func):
    """Apply streamlit cache if available."""
    if STREAMLIT:
        return st.cache_data(ttl=3600)(func)
    return func


@cache_it
def get_stock_data(symbol: str, period_days: int = 365) -> pd.DataFrame:
    """Fetch stock data with retry logic and fallback methods."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    # Method 1 — yf.download (different endpoint, less rate limited)
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )

            if df is not None and not df.empty:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index = pd.to_datetime(df.index)
                df.dropna(inplace=True)
                print(f"✅ Fetched {len(df)} days for {symbol}")
                return df

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    # Method 2 — use period string instead of dates
    try:
        ticker = yf.Ticker(symbol)
        period_str = "2y" if period_days >= 365 else "1y" if period_days >= 180 else "6mo"
        df = ticker.history(period=period_str)

        if not df.empty:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            print(f"✅ Fetched {len(df)} days for {symbol} (fallback)")
            return df

    except Exception as e:
        print(f"Fallback also failed: {e}")

    return None


@cache_it
def get_live_price(symbol: str) -> dict:
    """Get current live price."""
    empty = {
        'symbol': symbol,
        'current_price': 'N/A', 'previous_close': 'N/A',
        'day_high': 'N/A', 'day_low': 'N/A',
        'volume': 'N/A', 'market_cap': 'N/A',
        'pe_ratio': 'N/A', 'fifty_two_week_high': 'N/A',
        'fifty_two_week_low': 'N/A',
    }

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info  # faster, less rate limited

            return {
                'symbol':              symbol,
                'current_price':       round(info.last_price, 2) if info.last_price else 'N/A',
                'previous_close':      round(info.previous_close, 2) if info.previous_close else 'N/A',
                'day_high':            round(info.day_high, 2) if info.day_high else 'N/A',
                'day_low':             round(info.day_low, 2) if info.day_low else 'N/A',
                'volume':              info.last_volume if info.last_volume else 'N/A',
                'market_cap':          info.market_cap if info.market_cap else 'N/A',
                'pe_ratio':            'N/A',
                'fifty_two_week_high': round(info.year_high, 2) if info.year_high else 'N/A',
                'fifty_two_week_low':  round(info.year_low, 2) if info.year_low else 'N/A',
            }
        except Exception as e:
            print(f"Live price attempt {attempt+1} failed: {e}")
            time.sleep(1)

    return empty


def get_multiple_stocks(symbols: list, period_days: int = 365) -> dict:
    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period_days)
        if df is not None:
            data[symbol] = df
    return data