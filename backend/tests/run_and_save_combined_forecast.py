import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

from src.arima import generate_and_save_combined_arima_forecast
from src.file_utils import download_latest_csv_from_gcs


def prepare_prices(bucket_name: str) -> pd.DataFrame:
    prices_df = download_latest_csv_from_gcs(bucket_name=bucket_name, gcs_prefix='cleaned_data')
    prices_df = prices_df.copy()
    if 'date' in prices_df.columns:
        prices_df['date'] = pd.to_datetime(prices_df['date'], format='%d-%m-%y', errors='coerce')
        prices_df.dropna(subset=['date'], inplace=True)
        prices_df.set_index('date', inplace=True)
        prices_df.index.name = 'Date'
    else:
        raise RuntimeError("Expected 'date' column not found in cleaned price data")
    prices_df.columns = prices_df.columns.str.strip()
    prices_df.sort_index(ascending=True, inplace=True)
    prices_df.dropna(axis=1, how='all', inplace=True)
    prices_df.dropna(axis=0, how='all', inplace=True)
    return prices_df


if __name__ == '__main__':
    bucket = os.getenv('GCS_BUCKET_NAME', 'crystal-dss')
    prices_df = prepare_prices(bucket)
    commodity_columns = [
        col for col in prices_df.columns if pd.api.types.is_numeric_dtype(prices_df[col])
    ]
    combined_df, gcs_prefix = generate_and_save_combined_arima_forecast(
        prices_df=prices_df,
        commodity_columns=commodity_columns,
        forecast_steps=52,
        gcs_prefix='forecast_data/arima_combined.csv',
        bucket_name=bucket,
    )
    print('Combined DataFrame shape:', combined_df.shape)
    print('Saved to GCS prefix:', gcs_prefix)
