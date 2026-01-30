
import os
import re
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import warnings
from google.cloud import storage
from google.auth import default

# Set Hugging Face cache directory before importing TimesFM
os.environ.setdefault("HF_HOME", r"C:\hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from src import arima
from src.fbprophet import generate_and_save_prophet_forecast
from src.fbprophet_covariates import generate_and_save_prophet_covariate_forecast
from src.gru import generate_and_save_gru_forecast
from src.kan import generate_and_save_kan_forecast
from src.mamba import generate_and_save_mamba_forecast
from src.file_utils import save_dataframe_to_gcs, download_latest_csv_from_gcs
import logging

generate_and_save_combined_arima_forecast = arima.generate_and_save_combined_arima_forecast

# Wrap ARIMA training so the training summary is printed immediately after it returns.
arima_training_summary = None
def _run_model_training_and_print(*args, **kwargs):
    """Call ARIMA training, capture its summary, and print a short diagnostics line.

    Returns the original training summary dict from `arima.run_model_training`.
    """
    global arima_training_summary
    arima_training_summary = arima.run_model_training(*args, **kwargs)
    if isinstance(arima_training_summary, dict) and arima_training_summary.get('evaluation_df') is not None:
        eval_df = arima_training_summary['evaluation_df']
        try:
            print(
                f"  • ARIMA training evaluations captured {eval_df.shape[0]} rows across {eval_df.shape[1]} columns"
            )
        except Exception:
            # Be tolerant of unexpected shapes/types in evaluation_df
            print("  • ARIMA training completed (evaluation_df present)")
    return arima_training_summary

# Expose wrapper under the original name so callers can remain unchanged
run_model_training = _run_model_training_and_print

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

# Number of KAN training epochs can be controlled via environment variable KAN_EPOCHS
try:
    kan_num_epochs = int(os.getenv("KAN_EPOCHS", "20"))
except Exception:
    kan_num_epochs = 20

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
    print("  4. TimesFM")
    print("  5. GRU")
    print("  6. KAN")
    print("  7. MAMBA (Unified Multivariate)")
    print("  8. All of the above (sequential)")
    selection_input = input("\nEnter selection (e.g., 1 or 1,3,7 or 8 for all): ").strip()

    tokens = [token.strip() for token in selection_input.split(',') if token.strip()]
    selected_options = []
    for token in tokens:
        if token == '8':
            selected_options = ['1', '2', '3', '4', '5', '6', '7']
            break
        if token in {'1', '2', '3', '4', '5', '6', '7'} and token not in selected_options:
            selected_options.append(token)
    if not selected_options:
        selected_options = ['1', '2', '3', '4', '5', '6', '7']

    run_all_selected = set(selected_options) == {'1', '2', '3', '4', '5', '6', '7'}

    print("\n" + "=" * 80)
    print("Generating Forecasts with Confidence Intervals")
    print("=" * 80)
    forecast_steps = int(
        input("\nEnter number of forecast periods (default 128): ").strip() or "128"
    )
    print(
        f"\nGenerating {forecast_steps}-step forecast with 5% and 10% confidence intervals..."
    )

    model_results = []

    if '1' in selected_options:
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
                other_commodities=commodity_columns,
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

    if '4' in selected_options:
        try:
            from src.timesfm import generate_and_save_timesfm_forecast

            timesfm_combined_df, timesfm_gcs_prefix = generate_and_save_timesfm_forecast(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                forecast_steps=forecast_steps,
                gcs_prefix='forecast_data/timesfm_forecast.csv',
                conf_interval_05=True,
                conf_interval_10=True,
                bucket_name=bucket_name,
            )
            model_results.append(
                {
                    'model': 'TimesFM',
                    'gcs_prefix': timesfm_gcs_prefix,
                    'rows': timesfm_combined_df.shape[0],
                    'columns': timesfm_combined_df.shape[1],
                    'status': 'Completed',
                }
            )
        except Exception as exc:
            model_results.append(
                {
                    'model': 'TimesFM',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

    if '5' in selected_options:
        try:
            gru_combined_df, gru_gcs_prefix = generate_and_save_gru_forecast(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                forecast_steps=forecast_steps,
                gcs_prefix='forecast_data/gru_forecast.csv',
                train_new_models=False,  # Auto: load if exists, train if not
                conf_interval_05=True,
                conf_interval_10=True,
                bucket_name=bucket_name,
            )
            model_results.append(
                {
                    'model': 'GRU',
                    'gcs_prefix': gru_gcs_prefix,
                    'rows': gru_combined_df.shape[0],
                    'columns': gru_combined_df.shape[1],
                    'status': 'Completed',
                }
            )
        except Exception as exc:
            model_results.append(
                {
                    'model': 'GRU',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

    if '6' in selected_options:
        try:
            kan_combined_df, kan_gcs_prefix = generate_and_save_kan_forecast(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                forecast_steps=forecast_steps,
                gcs_prefix='forecast_data/kan_forecast.csv',
                train_new_models=False,  # Auto: load if exists, train if not
                num_epochs=100,
                conf_interval_05=True,
                conf_interval_10=True,
                bucket_name=bucket_name,
            )
            model_results.append(
                {
                    'model': 'KAN',
                    'gcs_prefix': kan_gcs_prefix,
                    'rows': kan_combined_df.shape[0],
                    'columns': kan_combined_df.shape[1],
                    'status': 'Completed',
                }
            )
        except Exception as exc:
            model_results.append(
                {
                    'model': 'KAN',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

    if '7' in selected_options:
        # Option 7: Unified MAMBA (multivariate)
        try:
            print("\n" + "=" * 80)
            print("OPTION 7: UNIFIED MAMBA (MULTIVARIATE)")
            print("=" * 80)
            
            mamba_gcs_prefix = "forecast_data/mamba_forecast.csv"
            print(f"  → Running unified Mamba forecasting for {len(commodity_columns)} commodities...")
            print(f"  → Learning cross-commodity dynamics from causality relationships")
            
            mamba_df, mamba_gcs_path = generate_and_save_mamba_forecast(
                prices_df=prices_df,
                commodity_columns=commodity_columns,
                forecast_steps=forecast_steps,
                gcs_prefix=mamba_gcs_prefix,
                train_new_model=False,  # Auto: load if exists, train if not
                sequence_length=21,
                n_layers=4,
                num_epochs=30,
                batch_size=32,
                learning_rate=0.001,
                bucket_name=bucket_name,
            )
            
            model_results.append(
                {
                    'model': 'MAMBA (Unified)',
                    'gcs_prefix': mamba_gcs_path,
                    'status': 'Completed',
                    'rows': mamba_df.shape[0] if mamba_df is not None else 0,
                    'columns': mamba_df.shape[1] if mamba_df is not None else 0,
                }
            )
        except Exception as exc:
            print(f"\n✗ MAMBA forecasting failed: {exc}")
            import traceback
            traceback.print_exc()
            model_results.append(
                {
                    'model': 'MAMBA',
                    'gcs_prefix': 'n/a',
                    'status': f'Failed: {exc}',
                }
            )

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

    print("=" * 80)


