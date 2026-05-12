# data/fetcher.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(symbol: str, period_days: int = 365) -> pd.DataFrame:
    """
    Fetch historical stock data for NSE/BSE stocks.
    symbol examples: 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS'
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        print(f"No data found for {symbol}")
        return None

    # Clean up
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    print(f"✅ Fetched {len(df)} days of data for {symbol}")
    return df


def get_multiple_stocks(symbols: list, period_days: int = 365) -> dict:
    """Fetch data for multiple stocks at once."""
    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period_days)
        if df is not None:
            data[symbol] = df
    return data


def get_live_price(symbol: str) -> dict:
    """Get the current live price and basic info."""
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        'symbol': symbol,
        'current_price': info.get('currentPrice', 'N/A'),
        'previous_close': info.get('previousClose', 'N/A'),
        'day_high': info.get('dayHigh', 'N/A'),
        'day_low': info.get('dayLow', 'N/A'),
        'volume': info.get('volume', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'pe_ratio': info.get('trailingPE', 'N/A'),
        'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
        'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
    }


# ---- Test it directly ----
if __name__ == "__main__":
    # Test with some popular NSE stocks
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

    print("📈 Fetching NSE stock data...\n")

    for sym in symbols:
        print(f"--- {sym} ---")
        df = get_stock_data(sym, period_days=90)
        if df is not None:
            print(df.tail(3))
            print()

    print("\n💰 Live prices:")
    for sym in symbols:
        price = get_live_price(sym)
        print(f"{sym}: ₹{price['current_price']} | High: ₹{price['day_high']} | Low: ₹{price['day_low']}")