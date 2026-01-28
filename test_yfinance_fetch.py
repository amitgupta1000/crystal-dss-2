"""Test yfinance data fetching for all symbols from dss_builder."""

import pandas as pd
import yfinance as yf
import time
from datetime import datetime

# Symbols from dss_builder.py
symbols = ['^GSPC', '000001.SS', 'DX=F', 'JPY=X', 'INR=X', '^NSEI', 'CL=F', 'NG=F', 'GC=F', 'SI=F', 'HG=F', 'ZN=F']

commodity_mapping = {
    'HG=F': 'Copper', 'SI=F': 'Silver', 'GC=F': 'Gold', 'NG=F': 'Natural Gas',
    'BZ=F': 'Brent Crude', 'CL=F': 'Crude Oil', 'INR=X': 'Indian Rupee','JPY=X': 'Japanese Yen',
    'EURUSD=X': 'Euro', 'DX=F': 'USD Index', '^NSEI': 'Nifty 50', '^IXIC': 'NASDAQ',
    '^GSPC': 'S&P 500','000001.SS': 'Shanghai Composite', 'ZN=F': 'US 10-Y BOND PRICE',
}

print("\n" + "="*80)
print("TESTING YFINANCE DATA FETCH FOR ALL SYMBOLS")
print("="*80 + "\n")

all_data = {}
issues_found = []

for symbol in symbols:
    commodity_name = commodity_mapping.get(symbol, symbol)
    
    try:
        print(f"Fetching {commodity_name} ({symbol})...")
        ticker = yf.Ticker(symbol)
        
        # Fetch with auto_adjust to avoid split issues
        data = ticker.history(period='5y', interval='1d', auto_adjust=True)
        
        if data.empty:
            print(f"  ✗ NO DATA RETURNED")
            issues_found.append(f"{commodity_name}: No data")
            continue
        
        # Reset index and handle timezone
        data = data.reset_index()
        
        # Remove timezone if present
        if pd.api.types.is_datetime64tz_dtype(data['Date']):
            data['Date'] = data['Date'].dt.tz_localize(None)
        
        data['date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        
        # Check for duplicate dates BEFORE cleaning
        duplicates_before = data['date'].duplicated().sum()
        if duplicates_before > 0:
            print(f"  ⚠ WARNING: {duplicates_before} DUPLICATE DATES found!")
            issues_found.append(f"{commodity_name}: {duplicates_before} duplicate dates")
            
            # Show some duplicate examples
            dup_dates = data[data['date'].duplicated(keep=False)]['date'].unique()[:3]
            print(f"    Example duplicate dates: {list(dup_dates)}")
        
        # Remove duplicates (keep last)
        data = data.drop_duplicates(subset=['date'], keep='last')
        
        # Check for invalid prices
        invalid_prices = (data['Close'].isna() | (data['Close'] <= 0)).sum()
        if invalid_prices > 0:
            print(f"  ⚠ WARNING: {invalid_prices} invalid prices (null or ≤0)")
            issues_found.append(f"{commodity_name}: {invalid_prices} invalid prices")
            data = data[data['Close'].notna() & (data['Close'] > 0)]
        
        # Check for suspicious price patterns
        price_changes = data['Close'].pct_change().abs()
        large_jumps = (price_changes > 0.5).sum()  # >50% change
        if large_jumps > 2:
            print(f"  ⚠ WARNING: {large_jumps} large price jumps (>50% change)")
            issues_found.append(f"{commodity_name}: {large_jumps} suspicious jumps")
        
        # Store data
        all_data[commodity_name] = data[['date', 'Close']].copy()
        
        # Summary
        print(f"  ✓ Rows: {len(data)}")
        print(f"    Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
        print(f"    Latest price: ${data['Close'].iloc[-1]:.2f}")
        print(f"    Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        time.sleep(0.5)  # Rate limiting
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        issues_found.append(f"{commodity_name}: {str(e)}")

# Combine all data
print("\n" + "="*80)
print("COMBINING DATA INTO SINGLE DATAFRAME")
print("="*80 + "\n")

if not all_data:
    print("✗ NO DATA FETCHED - All symbols failed!")
else:
    # Merge all dataframes
    combined = None
    for name, df in all_data.items():
        df_renamed = df.rename(columns={'Close': name})
        if combined is None:
            combined = df_renamed
        else:
            combined = pd.merge(combined, df_renamed, on='date', how='outer')
    
    # Sort by date
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date')
    
    # Check for duplicate dates in combined dataframe
    dup_dates_combined = combined['date'].duplicated().sum()
    if dup_dates_combined > 0:
        print(f"⚠ WARNING: {dup_dates_combined} DUPLICATE DATES IN COMBINED DATA!")
        print("  Removing duplicates...")
        combined = combined.drop_duplicates(subset=['date'], keep='last')
    
    combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
    
    # Quality summary
    print(f"✓ Combined dataframe created:")
    print(f"  Total rows: {len(combined)}")
    print(f"  Columns: {len(combined.columns)}")
    print(f"  Date range: {combined['date'].iloc[0]} to {combined['date'].iloc[-1]}")
    print(f"  Missing values: {combined.drop(columns=['date']).isna().sum().sum()}")
    
    # Show last 10 rows
    print(f"\n{'='*80}")
    print("LAST 10 ROWS:")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(combined.tail(10).to_string(index=False))
    
    # Save to CSV for inspection
    output_file = 'test_yfinance_output.csv'
    combined.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")

# Summary of issues
print(f"\n{'='*80}")
print("ISSUES SUMMARY:")
print("="*80)

if not issues_found:
    print("\n✅ NO ISSUES FOUND - All data looks clean!")
else:
    print(f"\n⚠ {len(issues_found)} issues detected:\n")
    for issue in issues_found:
        print(f"  • {issue}")
    
    print("\n💡 Recommendations:")
    print("  1. Duplicate dates: Fixed by drop_duplicates()")
    print("  2. Invalid prices: Removed automatically")
    print("  3. Large jumps: May indicate data errors or real market events")
    print("  4. Use auto_adjust=True to avoid split/dividend issues")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80 + "\n")
