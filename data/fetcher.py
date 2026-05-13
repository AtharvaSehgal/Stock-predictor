import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour — avoids rate limiting
def get_stock_data(symbol: str, period_days: int = 365) -> pd.DataFrame:
    """Fetch historical stock data for NSE/BSE stocks."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print(f"No data found for {symbol}")
            return None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index)
        df.dropna(inplace=True)

        print(f"✅ Fetched {len(df)} days of data for {symbol}")
        return df

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


@st.cache_data(ttl=300)  # Cache live price for 5 mins
def get_live_price(symbol: str) -> dict:
    """Get current live price and basic info."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'symbol':             symbol,
            'current_price':      info.get('currentPrice', 'N/A'),
            'previous_close':     info.get('previousClose', 'N/A'),
            'day_high':           info.get('dayHigh', 'N/A'),
            'day_low':            info.get('dayLow', 'N/A'),
            'volume':             info.get('volume', 'N/A'),
            'market_cap':         info.get('marketCap', 'N/A'),
            'pe_ratio':           info.get('trailingPE', 'N/A'),
            'fifty_two_week_high':info.get('fiftyTwoWeekHigh', 'N/A'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'current_price': 'N/A', 'previous_close': 'N/A',
            'day_high': 'N/A', 'day_low': 'N/A',
            'volume': 'N/A', 'market_cap': 'N/A',
            'pe_ratio': 'N/A', 'fifty_two_week_high': 'N/A',
            'fifty_two_week_low': 'N/A',
        }


def get_multiple_stocks(symbols: list, period_days: int = 365) -> dict:
    """Fetch data for multiple stocks at once."""
    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period_days)
        if df is not None:
            data[symbol] = df
    return data