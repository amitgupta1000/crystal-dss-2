import os
import sys
import urllib.request
from urllib.request import urlopen 
import pandas as pd
import numpy as np
from datetime import date
import time
from src.file_utils import upload_excel_file, save_dataframe_to_gcs

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
    symbols = ['^GSPC', '000001.SS', 'DX=F', 'JPY=X', 'CL=F', 'NG=F', 'GC=F', 'SI=F', 'HG=F', 'ZN=F', ]

    # Create a dictionary to map original commodity labels to desired ones
    commodity_mapping = {
        'HG=F': 'Copper', 'SI=F': 'Silver', 'GC=F': 'Gold', 'NG=F': 'Natural Gas',
        'BZ=F': 'Brent Crude', 'CL=F': 'Crude Oil', 'INR=X': 'Indian Rupee','JPY=X': 'Japanese Yen',
        'EURUSD=X': 'Euro', 'DX=F': 'USD Index', '^NSEI': 'Nifty 50', '^IXIC': 'NASDAQ',
        '^GSPC': 'S&P 500','000001.SS': 'Shanghai Composite', 'ZN=F': 'US 10-Y BOND PRICE',
    }

    all_close_data = pd.DataFrame()
    for commodity in symbols:
        try:
            ticker = yf.Ticker(commodity)
            data = ticker.history(period='6y', interval='1wk')
            data = data.reset_index()
            data['date'] = data['Date'].dt.strftime('%Y-%m-%d')
            close_data = data[['date', 'Close']].rename(columns=lambda x: f"{commodity_mapping.get(commodity, commodity)}" if x == 'Close' else x)
            all_close_data = pd.concat([all_close_data, close_data], axis=1)
            time.sleep(2)
        except Exception as e:
            print(f"Error fetching data for {commodity}: {e}")

    df = all_close_data.loc[:,~all_close_data.columns.duplicated(keep='first')].copy()
    df = df.ffill()
    GLOBALS_RAW = df.copy()

    #=========2.upload PLATTS sheet#=================
    import io
    excel_file_name = None

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
    keywords = ['DAP', 'FCA', 'FOB', 'CIF', 'ARA', 'CFR', 'FCA']

    def clean_platts_header(header):
        if header == 'date':  # Keep 'date' column name as is
            return header
        for keyword in keywords:
            if keyword in header.upper():  # Check for keyword (case-insensitive)
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
    