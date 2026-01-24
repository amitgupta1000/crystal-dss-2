import os, sys
sys.path.insert(0, os.getcwd())
from dss_main import generate_and_save_combined_arima_forecast

if __name__ == '__main__':
    combined_df, gcs_prefix = generate_and_save_combined_arima_forecast(forecast_steps=52, gcs_prefix='forecast_data/arima_combined.csv')
    print('Combined DataFrame shape:', combined_df.shape)
    print('Saved to GCS prefix:', gcs_prefix)
