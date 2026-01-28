import os
import sys
import urllib.request
from urllib.request import urlopen 
import pandas as pd
import numpy as np
from datetime import date
import time
from src.file_utils import upload_excel_file, save_dataframe_to_gcs
from yahoofinance import HistoricalPrices
def build_dataframe(uploaded_file=None):
    from google.cloud import storage
    from google.api_core import exceptions as _gcs_exceptions
    import io as _io
    import yfinance as yf
    import pandas as pd
    import time

    # If no uploaded_file provided, try interactive upload
    if uploaded_file is None:
        try:
            uploaded_file = upload_excel_file()
            if uploaded_file:
                print("File uploaded successfully.")
            else:
                print("No file was uploaded.")
        except Exception as e:
            print(f"File upload step failed: {e}")
            uploaded_file = None

    # #=========1.create the globals dataframe#=================
    ### List of Global Macro Commodities
    symbols = ['^GSPC', '000001.SS', 'DX=F', 'JPY=X', 'INR=X', '^NSEI', 'CL=F', 'NG=F', 'GC=F', 'SI=F', 'HG=F', 'ZN=F', ]

    # Create a dictionary to map original commodity labels to desired ones
    commodity_mapping = {
        'HG=F': 'Copper', 'SI=F': 'Silver', 'GC=F': 'Gold', 'NG=F': 'Natural Gas',
        'BZ=F': 'Brent Crude', 'CL=F': 'Crude Oil', 'INR=X': 'Indian Rupee','JPY=X': 'Japanese Yen',
        'EURUSD=X': 'Euro', 'DX=F': 'USD Index', '^NSEI': 'Nifty 50', '^IXIC': 'NASDAQ',
        '^GSPC': 'S&P 500','000001.SS': 'Shanghai Composite', 'ZN=F': 'US 10-Y BOND PRICE',
    }

    # Helper function to fetch with yahoofinance library
    def fetch_yahoofinance(symbol, years=5):
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            hp = HistoricalPrices(symbol, 
                                  start_date.strftime('%Y-%m-%d'), 
                                  end_date.strftime('%Y-%m-%d'))
            df = hp.to_dfs()['Historical Prices']
            df = df.reset_index()
            df.columns = ['date', 'Close', 'High', 'Low', 'Open', 'Volume']
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            return df[['date', 'Close']]
        except Exception as e:
            print(f"  yahoofinance error: {e}")
            return None

    # Compare both libraries for first symbol
    print("\n" + "="*80)
    print("COMPARING DATA SOURCES: yfinance vs yahoofinance")
    print("="*80)
    test_symbol = 'JPY=X'
    
    # Try yfinance
    print(f"\n1. Testing yfinance for {test_symbol}...")
    try:
        ticker = yf.Ticker(test_symbol)
        yf_data = ticker.history(period='5y', auto_adjust=True)
        yf_rows = len(yf_data)
        yf_gaps = yf_data['Close'].isna().sum()
        yf_latest = yf_data.index[-1].strftime('%Y-%m-%d') if not yf_data.empty else 'N/A'
        print(f"   ✓ Rows: {yf_rows} | Gaps: {yf_gaps} | Latest: {yf_latest}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        yf_data = None
    
    # Try yahoofinance
    print(f"\n2. Testing yahoofinance for {test_symbol}...")
    yf2_data = fetch_yahoofinance(test_symbol)
    if yf2_data is not None:
        yf2_rows = len(yf2_data)
        yf2_gaps = yf2_data['Close'].isna().sum()
        yf2_latest = yf2_data['date'].iloc[-1] if not yf2_data.empty else 'N/A'
        print(f"   ✓ Rows: {yf2_rows} | Gaps: {yf2_gaps} | Latest: {yf2_latest}")
    
    # Comparison
    if yf_data is not None and yf2_data is not None:
        print(f"\n📊 Comparison:")
        print(f"   Row difference: {abs(yf_rows - yf2_rows)}")
        print(f"   yfinance gaps: {yf_gaps} | yahoofinance gaps: {yf2_gaps}")
        
        if yf2_gaps < yf_gaps:
            print(f"   → yahoofinance has FEWER gaps! Using it as primary source.")
            use_yahoofinance = True
        else:
            print(f"   → yfinance has better or equal quality. Using it.")
            use_yahoofinance = False
    else:
        print("\n   → Defaulting to yfinance")
        use_yahoofinance = False
    
    print("="*80 + "\n")

    all_close_data = pd.DataFrame()
    for commodity in symbols:
        commodity_name = commodity_mapping.get(commodity, commodity)
        
        # Try selected library based on comparison
        if use_yahoofinance:
            print(f"Fetching {commodity_name} using yahoofinance...")
            yf_alt_data = fetch_yahoofinance(commodity)
            
            if yf_alt_data is not None and not yf_alt_data.empty:
                close_data = yf_alt_data.rename(columns={'Close': commodity_name})
                all_close_data = pd.concat([all_close_data, close_data], axis=1)
                time.sleep(0.5)
                continue
            else:
                print(f"  ⚠ yahoofinance failed, falling back to yfinance...")
        
        # Use yfinance (default or fallback)
        try:
            ticker = yf.Ticker(commodity)
            # Try with auto_adjust=True to get cleaner data
            data = ticker.history(period='5y', auto_adjust=True)
            
            if data.empty:
                print(f"⚠ No data returned for {commodity}")
                continue
                
            data = data.reset_index()
            data['date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            # Check for data quality issues
            gaps = data['Close'].isna().sum()
            if gaps > 0:
                print(f"⚠ {commodity_name}: Found {gaps} missing values, forward-filling...")
                data['Close'] = data['Close'].ffill().bfill()
            
            # Check for discontinuities (large jumps)
            pct_change = data['Close'].pct_change().abs()
            large_jumps = (pct_change > 0.5).sum()  # >50% change in one day
            if large_jumps > 0:
                print(f"⚠ {commodity_name}: {large_jumps} suspicious price jumps detected")
            
            close_data = data[['date', 'Close']].rename(columns={'Close': commodity_name})
            all_close_data = pd.concat([all_close_data, close_data], axis=1)
            time.sleep(1)  # Reduced delay
            
        except Exception as e:
            print(f"✗ Error fetching data for {commodity}: {e}")

    df = all_close_data.loc[:,~all_close_data.columns.duplicated(keep='first')].copy()
    
    # Better gap filling: interpolate first, then forward fill
    df = df.interpolate(method='linear', limit=3)  # Fill small gaps with interpolation
    df = df.ffill().bfill()  # Then forward/backward fill remaining
    
    print(f"\n✓ Data quality: {df.shape[0]} rows, {df.shape[1]-1} commodities")
    print(f"  Missing values after cleaning: {df.isna().sum().sum()}")
    


