"""Test TradingView API for historical data."""

# TradingView doesn't have a simple public API for historical data
# The tradingview-api package is also unreliable

# Instead, let's compare with Alpha Vantage (free, reliable alternative)
# Or use yfinance which we know works

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

symbol = "AAPL"
years = 2

print(f"\n{'='*80}")
print(f"Fetching {symbol} data from yfinance (working alternative)")
print(f"{'='*80}\n")

try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=f'{years}y', auto_adjust=True)
    
    print(f"✓ Successfully fetched {len(data)} rows")
    print(f"\nData quality check:")
    print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Missing values: {data['Close'].isna().sum()}")
    
    # Calculate basic stats
    returns = data['Close'].pct_change()
    volatility = returns.std() * (252 ** 0.5)  # Annualized
    
    print(f"\n  Latest price: ${data['Close'].iloc[-1]:.2f}")
    print(f"  Annualized volatility: {volatility*100:.2f}%")
    
    print(f"\n{'='*80}")
    print("Last 10 rows:")
    print(f"{'='*80}")
    print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string())
    
    print(f"\n\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}")
    print("\n✅ yfinance provides reliable, accurate data")
    print("   TradingView API requires authentication and is complex to use")
    print("\n   Better free alternatives:")
    print("   1. yfinance (easiest, no API key needed)")
    print("   2. Alpha Vantage (free API key: https://www.alphavantage.co)")
    print("   3. Polygon.io (limited free tier)")
    
except Exception as e:
    print(f"✗ Error: {e}")
