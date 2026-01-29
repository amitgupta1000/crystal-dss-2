import os
import sys
import urllib.request
from urllib.request import urlopen 
import pandas as pd
import numpy as np
from datetime import date
import time
from src.file_utils import upload_excel_file, save_dataframe_to_gcs
from google.cloud import storage
from google.api_core import exceptions as _gcs_exceptions
import io as _io
import yfinance as yf
import pandas as pd
import time

def build_dataframe(uploaded_file=None):

    # #=========1.create the globals dataframe#=================
    ### List of Global Macro Commodities
    symbols = ['^GSPC', '000001.SS', '^NSEI', '^IXIC', 'DX=F', 'JPY=X', 'INR=X', 'CL=F', 'NG=F', 'GC=F', 'SI=F', 'HG=F', 'ZN=F', ]

    # Create a dictionary to map original commodity labels to desired ones
    commodity_mapping = {
        'HG=F': 'Copper', 'SI=F': 'Silver', 'GC=F': 'Gold', 'NG=F': 'Natural Gas',
        'BZ=F': 'Brent Crude', 'CL=F': 'Crude Oil', 'INR=X': 'Indian Rupee','JPY=X': 'Japanese Yen',
        'EURUSD=X': 'Euro', 'DX=F': 'USD Index', '^NSEI': 'Nifty 50', '^IXIC': 'NASDAQ',
        '^GSPC': 'S&P 500','000001.SS': 'Shanghai Composite', 'ZN=F': 'US 10-Y BOND PRICE',
    }

    print("\n" + "="*80)
    print("FETCHING YFINANCE DATA FOR GLOBAL COMMODITIES")
    print("="*80 + "\n")
    
    all_data = {}
    issues_found = []
    
    for symbol in symbols:
        commodity_name = commodity_mapping.get(symbol, symbol)
        
        try:
            print(f"Fetching {commodity_name} ({symbol})...")
            ticker = yf.Ticker(symbol)
            
            # Fetch with auto_adjust to avoid split/dividend issues
            data = ticker.history(period='6y', interval='1d', auto_adjust=True)
            
            if data.empty:
                print(f"  ✗ NO DATA RETURNED for {commodity_name}")
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
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ ERROR fetching {commodity_name}: {e}")
            issues_found.append(f"{commodity_name}: {str(e)}")
    
    # Merge all dataframes
    print("\n" + "="*80)
    print("COMBINING DATA INTO SINGLE DATAFRAME")
    print("="*80 + "\n")
    
    if not all_data:
        print("✗ NO DATA FETCHED - All symbols failed!")
        GLOBALS_RAW = pd.DataFrame()
    else:
        combined = None
        for name, df_temp in all_data.items():
            df_renamed = df_temp.rename(columns={'Close': name})
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
        
        # Forward fill and copy
        combined = combined.ffill()
        GLOBALS_RAW = combined.copy()
        
        # Quality summary
        print(f"✓ Combined dataframe created:")
        print(f"  Total rows: {len(GLOBALS_RAW)}")
        print(f"  Columns: {len(GLOBALS_RAW.columns)}")
        print(f"  Date range: {GLOBALS_RAW['date'].iloc[0]} to {GLOBALS_RAW['date'].iloc[-1]}")
        print(f"  Missing values: {GLOBALS_RAW.drop(columns=['date']).isna().sum().sum()}")
        
        print(f"\nLast 10 rows:")
        print(GLOBALS_RAW.tail(10))
    
    # Summary of issues
    if issues_found:
        print(f"\n{'='*80}")
        print(f"DATA QUALITY ISSUES: {len(issues_found)} issues detected")
        print("="*80)
        for issue in issues_found:
            print(f"  • {issue}")
    else:
        print(f"\n✅ NO ISSUES FOUND - All data looks clean!")
    
    print("\n" + "="*80 + "\n")

    #=========2.upload PLATTS sheet#=================
    import io
    excel_file_name = None

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


    if uploaded_file:
        excel_file_name = next(iter(uploaded_file))
        print(f"Uploaded file: {excel_file_name}")
    else:
        print("No file was uploaded or the upload failed. Please ensure a file is selected and uploaded successfully in the previous step.")

    if excel_file_name: # Check if a file was successfully uploaded
        # Use io.BytesIO to read the content of the uploaded Excel file
        excel_data = io.BytesIO(uploaded_file[excel_file_name])

        # Load the 'PLATTS_RAW' sheet from the BytesIO object into a pandas DataFrame
        PLATTS_RAW_DF = pd.read_excel(excel_data, sheet_name='PLATTS_RAW')

        print("PLATTS_RAW sheet loaded successfully into PLATTS_RAW_DF.")
        print(PLATTS_RAW_DF.head(5))
    else:
        print("Cannot load PLATTS_RAW sheet: No Excel file was uploaded or its name could not be determined.")

    for col in PLATTS_RAW_DF.columns:
        PLATTS_RAW_DF[col] = PLATTS_RAW_DF[col].astype(str)
    print("All columns in PLATTS_RAW_DF converted to string type.")

    # GLOBALS_RAW was already created in a previous step, so ensuring it's processed here
    if 'GLOBALS_RAW' in locals() and isinstance(GLOBALS_RAW, pd.DataFrame):
        for col in GLOBALS_RAW.columns:
            GLOBALS_RAW[col] = GLOBALS_RAW[col].astype(str)
        #print("All columns in GLOBALS_RAW converted to string type.")
    else:
        print("GLOBALS_RAW DataFrame not found or not a DataFrame. Skipping string conversion for GLOBALS_RAW.")

    #=========3. Clean the dataframes#=================
    def clean_and_format_df(df, divisor=None):
        # 1. Rename first column as 'date'
        df = df.rename(columns={df.columns[0]: 'date'})

        # Convert 'date' column to datetime objects first for proper sorting
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows where date conversion failed (if any)
        df = df.dropna(subset=['date'])

        # 2. Sort all data in ascending date order
        df = df.sort_values(by='date', ascending=True)

        # 3. Format date values as mm-yy
        df['date'] = df['date'].dt.strftime('%d-%m-%y')

        # 4. Format all other columns as numeric (rounding deferred until after ffill)
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if divisor is not None:
                    df[col] = df[col] / divisor
                # do not round here — rounding after forward-fill preserves small values for filling

        return df
    
    ### GLOBALS CLEANER================================================================
    #Make a copy of the GLOBALS_RAW DataFrame
    globals = GLOBALS_RAW.copy()
    #print(f"Original GLOBALS_RAW_DF Shape: {globals.shape}")

    # Apply the clean_and_format_df utility function directly
    globals_cleaned = clean_and_format_df(globals)
    print(f"Cleaned GLOBALS_DF Shape: {globals_cleaned.shape}")

    ### PLATTS CLEANER===============================================================
    # 1. Make a copy of the PLATTS_RAW_DF DataFrame
    platts = PLATTS_RAW_DF.copy()

    # Rename the 'Unnamed: 0' column to 'date' as it's the date identifier
    if 'Unnamed: 0' in platts.columns:
        platts = platts.rename(columns={'Unnamed: 0': 'date'})
        print("Renamed 'Unnamed: 0' column to 'date'.")

    # Drop the first 3 rows, which contain unit information and other metadata
    # These rows are: index 0 (MT), index 1 (USD), index 2 (c/AssessDate)
    platts = platts.iloc[3:].copy()
    platts.reset_index(drop=True, inplace=True)
    print("Dropped the first 3 rows containing metadata.")

    # Apply the clean_and_format_df utility function with divisor
    platts_cleaned = clean_and_format_df(platts, divisor=1000)
    # Treat exact zeros as missing values so forward-fill can fill them
    numeric_cols = platts_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        platts_cleaned[numeric_cols] = platts_cleaned[numeric_cols].replace(0, np.nan)
    # Forward-fill missing numeric values, then round to 2 decimals
    platts_cleaned = platts_cleaned.ffill()
    platts_cleaned = platts_cleaned.round(2)


    # --- New code to clean PLATTS headers ---
    # Define a list of keywords to look for
    keywords = ['CFR Taiwan/China Marker (NextGen MOC)', 'Pressurized', 'Refrigerated', 'DAP', 'FCA', 'FOB', 'CIF', 'ARA', 'CFR', 'FCA']

    def clean_platts_header(header):
        if header == 'date':  # Keep 'date' column name as is
            return header
        for keyword in keywords:
            if keyword in header:  # Check for keyword (case-insensitive)
                # Find the index of the keyword and take the part before it
                return header.split(keyword)[0].strip()
        return header  # Return original header if no keyword is found

    # Apply the cleaning function to all column names except 'date'
    platts_cleaned.columns = [clean_platts_header(col) for col in platts_cleaned.columns]
    print("Cleaned PLATTS headers.")
    print(platts_cleaned.head(10))


    concatenated_df = pd.merge(globals_cleaned, platts_cleaned, on='date', how='left')
    concatenated_df = concatenated_df.ffill()
    concatenated_df['date'] = pd.to_datetime(concatenated_df['date'], format='%d-%m-%y', errors='coerce')
    concatenated_df = concatenated_df.sort_values(by='date', ascending=True)
    concatenated_df['date'] = concatenated_df['date'].dt.strftime('%d-%m-%y')

    #=========4. Calculate spreads (differences)#=================
    print("\n" + "="*80)
    print("CALCULATING SPREADS")
    print("="*80 + "\n")
    
    # Spread 1: Crude Oil vs. Naphtha (convert Crude Oil $/barrel to $/kg by dividing by 135)
    if 'Crude Oil' in concatenated_df.columns and 'Naphtha' in concatenated_df.columns:
        concatenated_df['SPREAD: Crude Oil - Naphtha'] = (
            (pd.to_numeric(concatenated_df['Crude Oil'], errors='coerce') / 135) - 
            pd.to_numeric(concatenated_df['Naphtha'], errors='coerce')
        )
        print("✓ Calculated: SPREAD: Crude Oil - Naphtha (Crude Oil converted to $/kg)")
    else:
        print("⚠ WARNING: Could not calculate Crude Oil - Naphtha spread (missing columns)")
    
    # Spread 2: Benzene vs. Styrene
    if 'Benzene' in concatenated_df.columns and 'Styrene' in concatenated_df.columns:
        concatenated_df['SPREAD: Benzene - Styrene'] = (
            pd.to_numeric(concatenated_df['Benzene'], errors='coerce') - 
            pd.to_numeric(concatenated_df['Styrene'], errors='coerce')
        )
        print("✓ Calculated: SPREAD: Benzene - Styrene")
    else:
        print("⚠ WARNING: Could not calculate Benzene - Styrene spread (missing columns)")
    
    # Spread 3: Propylene vs. PP Inj
    if 'Propylene Poly Grade' in concatenated_df.columns and 'PP Inj' in concatenated_df.columns:
        concatenated_df['SPREAD: Propylene - PP Inj'] = (
            pd.to_numeric(concatenated_df['Propylene Poly Grade'], errors='coerce') - 
            pd.to_numeric(concatenated_df['PP Inj'], errors='coerce')
        )
        print(f"✓ Calculated: SPREAD: Propylene - PP Inj (using 'Propylene Poly Grade' column)")
    else:
        print(f"⚠ WARNING: Could not calculate Propylene - PP Inj spread")


    # Spread 4: Paraxylene vs. PET
    if 'Paraxylene' in concatenated_df.columns and 'Recycled-PET Clear Flakes' in concatenated_df.columns:
        concatenated_df['SPREAD: Paraxylene - PET'] = (
            pd.to_numeric(concatenated_df['Paraxylene'], errors='coerce') - 
            pd.to_numeric(concatenated_df['Recycled-PET Clear Flakes'], errors='coerce')
        )
        print("✓ Calculated: SPREAD: Paraxylene - PET")
    else:
        print("⚠ WARNING: Could not calculate Paraxylene - PET spread (missing columns)")
    

    # Spread 5: Ethylene vs. Polyethylene
    if 'Ethylene' in concatenated_df.columns and 'HDPE Film' in concatenated_df.columns:
        concatenated_df['SPREAD: Ethylene - HDPE'] = (
            pd.to_numeric(concatenated_df['Ethylene'], errors='coerce') - 
            pd.to_numeric(concatenated_df['HDPE Film'], errors='coerce')
        )
        print("✓ Calculated: SPREAD: Ethylene - HDPE")
    else:
        print("⚠ WARNING: Could not calculate Ethylene - HDPE spread (missing columns)")
    
    print("\n" + "="*80 + "\n")

    print("\nHead of the concatenated DataFrame:")
    print(concatenated_df.head())
    print("\nShape of the concatenated DataFrame:")
    print(concatenated_df.shape)

    csv_file_path = 'ALL_CLEAN_DATA.csv'
    print(f"Saving concatenated DataFrame to '{csv_file_path}'...")
    concatenated_df.to_csv(csv_file_path, index=False)

    bucket_name = 'crystal-dss'
    gcs_prefix = 'cleaned_data/clean_df.csv'
    save_dataframe_to_gcs(
        df=concatenated_df,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        validate_rows=False
    )
    print(f"Concatenated data saved to '{csv_file_path}' and gs://{bucket_name}/{gcs_prefix}")
    return concatenated_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build concatenated commodity dataframe from globals and PLATTS sheets")
    parser.add_argument('--excel', '-e', help='Local Excel file path to use instead of interactive upload', default=None)
    args = parser.parse_args()

    uploaded = None
    if args.excel:
        try:
            uploaded = upload_excel_file(source=args.excel)
            print(f"Loaded Excel from {args.excel}.")
        except Exception as exc:
            print(f"Failed to load provided Excel file: {exc}")
            return

    # Run the build
    build_dataframe(uploaded_file=uploaded)


if __name__ == '__main__':
    main()
