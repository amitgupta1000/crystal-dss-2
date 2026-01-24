
import os
import re
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import warnings
from google.cloud import storage
from google.auth import default
from src import arima
from src.fbprophet import generate_and_save_prophet_forecast
from src.fbprophet_covariates import generate_and_save_prophet_covariate_forecast
from src.file_utils import save_dataframe_to_gcs, download_latest_csv_from_gcs
import logging

generate_and_save_combined_arima_forecast = arima.generate_and_save_combined_arima_forecast
run_model_training = arima.run_model_training

# Suppress common SARIMA warnings
warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average")
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Initialize GCS credentials and client at module level (lightweight)
creds, _ = default()
storage_client = storage.Client(credentials=creds)
bucket_name = 'crystal-dss'

load_dotenv()

# Define the target commodities
target_commodities = [
    'Acetic Acid', 'Butyl Acetate', 'Toluene', 'Isomer-MX', 'Solvent-MX', 'Methanol',
    'MTBE', 'Benzene'
]

# Define the other commodities to correlate against
other_commodities = [
    'Gold', 'Silver', 'Copper', 'S&P 500', 'Shanghai Composite', 'USD Index',
    'Japanese Yen', 'US 10-Y BOND PRICE', 'Crude Oil', 'Natural Gas', 'Naphtha',
    'EDC', 'Ethylene', 'Propylene', 'N-Butanol', 'Paraxylene', 'OrthXylene',
    'Cyclohexane', 'Styrene', 'DEG', '2 EH', 'Acetic Acid', 'Butyl Acetate',
    'Toluene', 'Isomer-MX', 'Solvent-MX', 'Methanol', 'MTBE', 'Benzene'
]



def run_dummy_forecast(model_name, bucket_name):
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} FORECAST PLACEHOLDER")
    print("=" * 80)
    print(f"Skipping {model_name} forecast. Placeholder artifact will be written to GCS.")
    slug = re.sub(r"[^0-9A-Za-z]+", "_", model_name.strip()).strip("_").lower() or "model"
    gcs_prefix = f"forecast_data/{slug}_forecast_placeholder.csv"
    placeholder_df = pd.DataFrame([
        {
            'Model': model_name,
            'Status': 'Not Implemented',
        }
    ])
    save_dataframe_to_gcs(
        df=placeholder_df,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )
    print(f"  ✓ Placeholder saved to GCS prefix: {gcs_prefix}")
    return {
        'model': model_name,
        'gcs_prefix': gcs_prefix,
        'status': 'Placeholder saved',
    }


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FORECASTING WORKFLOW")
    print("="*80)

    # Load latest cleaned prices once for all pathways
    print("\n" + "="*80)
    print("LOADING PRICE DATA FROM GCS")
    print("="*80)
    try:
        prices_df = download_latest_csv_from_gcs(bucket_name=bucket_name, gcs_prefix='cleaned_data')
        print("  ✓ Downloaded latest cleaned DataFrame from GCS")
    except Exception as e:
        raise RuntimeError(f"Failed to load cleaned price data from GCS: {e}")

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

    commodity_columns = [col for col in prices_df.columns if pd.api.types.is_numeric_dtype(prices_df[col])]

    if not commodity_columns:
        raise RuntimeError("No numeric commodity columns found in cleaned price data")

    print(f"  ✓ Price data prepared. Rows: {prices_df.shape[0]} | Commodities: {len(commodity_columns)}")

    print("\nSelect forecaster(s) to run:")
    print("  1. ARIMA")
    print("  2. FB Prophet")
    print("  3. FB Prophet with Covariates")
    print("  4. All of the above (sequential)")
    selection_input = input("\nEnter selection (e.g., 1 or 1,2 or 4 for all): ").strip()

    tokens = [token.strip() for token in selection_input.split(',') if token.strip()]
    selected_options = []
    for token in tokens:
        if token == '4':
            selected_options = ['1', '2', '3']
            break
        if token in {'1', '2', '3'} and token not in selected_options:
            selected_options.append(token)
    if not selected_options:
        selected_options = ['1', '2', '3']

    run_all_selected = set(selected_options) == {'1', '2', '3'}

    print("\n" + "=" * 80)
    print("Generating Forecasts with Confidence Intervals")
    print("=" * 80)
    forecast_steps = int(
        input("\nEnter number of forecast periods (default 250): ").strip() or "250"
    )
    print(
        f"\nGenerating {forecast_steps}-step forecast with 5% and 10% confidence intervals..."
    )

    model_results = []
    arima_training_summary = None

    if '1' in selected_options:
        print("\n" + "="*80)
        print("ARIMA CONFIGURATION")
        print("="*80)
        print("\nDo you want to:")
        print("  1. Load saved models from GCS (faster)")
        print("  2. Run new ARIMA grid search and train models (slower, more accurate)")
        arima_choice = input("\nEnter your choice (1 or 2): ").strip()

        if arima_choice == '2':
            arima_training_summary = run_model_training(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                bucket_name=bucket_name,
            )
        elif arima_choice == '1':
            print("\n" + "="*80)
            print("Loading Saved Models from GCS")
            print("="*80)
            print("Models will be loaded during forecasting...")
        else:
            print("\nInvalid choice. Defaulting to loading saved models.")
            print("\n" + "="*80)
            print("Loading Saved Models from GCS")
            print("="*80)
            print("Models will be loaded during forecasting...")

        try:
            arima_combined_df, arima_gcs_prefix = arima.generate_and_save_combined_arima_forecast(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                forecast_steps=forecast_steps,
                gcs_prefix='forecast_data/arima_forecast.csv',
                conf_interval_05=True,
                conf_interval_10=True,
                bucket_name=bucket_name,
            )
            model_results.append(
                {
                    'model': 'ARIMA',
                    'gcs_prefix': arima_gcs_prefix,
                    'rows': arima_combined_df.shape[0],
                    'columns': arima_combined_df.shape[1],
                    'status': 'Completed',
                }
            )
        except Exception as exc:
            model_results.append(
                {
                    'model': 'ARIMA',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

    if '2' in selected_options:
        try:
            prophet_combined_df, prophet_gcs_prefix = generate_and_save_prophet_forecast(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                forecast_steps=forecast_steps,
                gcs_prefix='forecast_data/fb_prophet_forecast.csv',
                conf_interval_05=True,
                conf_interval_10=True,
                bucket_name=bucket_name,
            )
            model_results.append(
                {
                    'model': 'FB Prophet',
                    'gcs_prefix': prophet_gcs_prefix,
                    'rows': prophet_combined_df.shape[0],
                    'columns': prophet_combined_df.shape[1],
                    'status': 'Completed',
                }
            )
        except Exception as exc:
            model_results.append(
                {
                    'model': 'FB Prophet',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

    if '3' in selected_options:
        try:
            prophet_cov_df, prophet_cov_gcs_prefix = generate_and_save_prophet_covariate_forecast(
                prices_df=prices_df,
                target_commodities=target_commodities,
                other_commodities=other_commodities,
                forecast_steps=forecast_steps,
                gcs_prefix='forecast_data/fb_prophet_covariates_forecast.csv',
                conf_interval_05=True,
                conf_interval_10=True,
                bucket_name=bucket_name,
            )
            model_results.append(
                {
                    'model': 'FB Prophet with Covariates',
                    'gcs_prefix': prophet_cov_gcs_prefix,
                    'rows': prophet_cov_df.shape[0],
                    'columns': prophet_cov_df.shape[1],
                    'status': 'Completed',
                }
            )
        except Exception as exc:
            model_results.append(
                {
                    'model': 'FB Prophet with Covariates',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

    if run_all_selected:
        additional_models = [
            'Google TimesFM',
            'KAN',
            'MAMBA',
            'GRU',
        ]
        for model_name in additional_models:
            model_results.append(run_dummy_forecast(model_name, bucket_name))

    print("\n" + "=" * 80)
    print("MODEL EXECUTION SUMMARY")
    print("=" * 80)
    for result in model_results:
        dims = (
            f"{result['rows']}x{result['columns']}"
            if 'rows' in result and 'columns' in result
            else 'n/a'
        )
        print(
            f"  - {result['model']}: {result['status']} (GCS: {result['gcs_prefix']} | Dimensions: {dims})"
        )
    if arima_training_summary and arima_training_summary.get('evaluation_df') is not None:
        eval_df = arima_training_summary['evaluation_df']
        print(
            f"  • ARIMA training evaluations captured {eval_df.shape[0]} rows across {eval_df.shape[1]} columns"
        )
    print("=" * 80)


