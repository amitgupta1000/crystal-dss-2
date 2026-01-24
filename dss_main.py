
"""# 5. FORECASTER - ARIMA / PROPHET / PROPHET_COVARIATE / SARIMAX / TIMESFM / GRU_MAMBA_KAN

## Forecasting with increasing orders of complexity

"""
import os, io, re, tempfile
from urllib.request import urlopen
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from google.cloud import storage
from google.auth import default
from src.file_utils import save_dataframe_to_gcs, upload_excel_file, download_latest_csv_from_gcs
import logging

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

target_commodities = [
    'Acetic Acid',
    'Butyl Acetate',
    'Toluene',
    'Isomer-MX',
    'Solvent-MX',
    'Methanol',
    'MTBE',
    'Benzene'
]

#==================================#============================================
# SEASONALITY-GUIDED ARIMA PARAMETER OPTIMIZATION
#==================================#============================================

def load_seasonality_study(gcs_path='stats_studies_data/seasonality/seasonality_results_20260123.csv'):
    """
    Load seasonality study results from GCS and extract the strongest seasonal period
    for each commodity.
    
    Returns:
        dict: Mapping of commodity name to best seasonal period (m value)
    """
    try:
        print(f"\nLoading seasonality study from GCS: {gcs_path}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        data = blob.download_as_bytes()
        seasonality_df = pd.read_csv(io.BytesIO(data))
        
        # Filter for seasonal periods only
        seasonal_only = seasonality_df[seasonality_df['Is Seasonal for Period'] == True].copy()
        
        if seasonal_only.empty:
            print("  Warning: No seasonal periods found in study")
            return {}
        
        # Get the period with highest seasonality strength for each commodity
        best_periods = seasonal_only.loc[
            seasonal_only.groupby('Commodity')['Seasonality Strength (STL)'].idxmax()
        ]
        
        # Create mapping: commodity -> best period
        seasonality_map = dict(zip(
            best_periods['Commodity'], 
            best_periods['Period'].astype(int)
        ))
        
        print(f"  ✓ Loaded seasonality data for {len(seasonality_map)} commodities")
        return seasonality_map
    
    except Exception as e:
        print(f"  ✗ Failed to load seasonality study: {e}")
        print(f"  Falling back to default m_values")
        return {}


def get_robust_m_values(period, standard_periods=[52, 75, 130, 156, 195, 250, 312, 390, 520, 781]):
    """
    Given a detected seasonal period, return a list of m values to test including:
    - 1 (non-seasonal baseline)
    - The detected period itself
    - Nearest standard periods for robustness
    
    Args:
        period: The detected seasonal period from the study
        standard_periods: List of standard seasonal periods to consider
    
    Returns:
        list: Optimized list of m values to test
    """
    m_values = [1, period]  # Always include non-seasonal and detected period
    
    # Find nearest standard periods (within 20% of detected period)
    tolerance = 0.20
    for std_period in standard_periods:
        if std_period == period:
            continue  # Already included
        ratio = abs(std_period - period) / period
        if ratio <= tolerance:
            m_values.append(std_period)
    
    # Remove duplicates and sort
    m_values = sorted(list(set(m_values)))
    return m_values


def build_commodity_m_mapping(commodity_columns, seasonality_map):
    """
    Build a mapping of commodity to optimized m_values based on seasonality study.
    
    Args:
        commodity_columns: List of commodity column names from prices_df
        seasonality_map: Dict mapping commodity names to best seasonal periods
    
    Returns:
        dict: Mapping of commodity to list of m values to test
    """
    commodity_m_map = {}
    default_m_values = [75]  # Fallback for commodities not in study
    
    for commodity in commodity_columns:
        if commodity in seasonality_map:
            period = seasonality_map[commodity]
            m_values = get_robust_m_values(period)
            commodity_m_map[commodity] = m_values
        else:
            # Use default for commodities not in seasonality study
            commodity_m_map[commodity] = default_m_values
    
    return commodity_m_map


#==================================#============================================
# ARIMA/SARIMA grid testing framework
#==================================#============================================

# ARIMA/SARIMA parameters using pmdarima auto_arima (much faster than manual grid search)
# m_values determined per-commodity based on seasonality study (loaded during grid search)

def _commodity_auto_arima_worker(task):
    """
    Use pmdarima's auto_arima for fast, intelligent model selection.
    Tests both seasonal and non-seasonal models using stepwise search.
    
    task: (commodity, values, index_iso, m_values_for_commodity)
    """
    commodity, values, index_iso, m_values_for_commodity = task
    import pandas as _pd
    import pickle as _pickle
    import pmdarima as pm
    from pmdarima import auto_arima
    
    results = []
    best_map = {}
    s = _pd.Series(values, index=_pd.to_datetime(index_iso))
    
    # Test different d values (differencing orders)
    for d in [0, 1]:
        for m in m_values_for_commodity:
            try:
                # Use auto_arima for intelligent model selection
                if m == 1:
                    # Non-seasonal ARIMA
                    model = auto_arima(
                        s,
                        d=d,  # Fix differencing order
                        start_p=0, max_p=3,
                        start_q=0, max_q=3,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_order=None,
                        trace=False,
                        n_fits=50  # Limit to prevent hanging
                    )
                else:
                    # Seasonal ARIMA (SARIMA)
                    model = auto_arima(
                        s,
                        d=d,  # Fix differencing order
                        start_p=0, max_p=3,
                        start_q=0, max_q=3,
                        seasonal=True,
                        m=m,  # Use detected seasonal period
                        start_P=0, max_P=2,
                        start_Q=0, max_Q=2,
                        D=None,  # Let auto_arima choose seasonal differencing
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_order=None,
                        trace=False,
                        n_fits=50  # Limit to prevent hanging
                    )
                
                # Extract model info
                order = model.order
                seasonal_order = model.seasonal_order if hasattr(model, 'seasonal_order') else (0, 0, 0, 0)
                aic = float(model.aic())
                
                results.append({
                    'Commodity': commodity,
                    'd': d,
                    'm': m,
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'AIC': aic,
                    'Status': 'Success',
                    'Error Message': None
                })
                
                # Serialize model for storage
                try:
                    best_bytes = _pickle.dumps(model)
                except Exception:
                    best_bytes = None
                
                # Store by (d, m) key
                best_map[(d, m)] = (order, seasonal_order, best_bytes, aic)
                
            except Exception as e:
                results.append({
                    'Commodity': commodity,
                    'd': d,
                    'm': m,
                    'order': None,
                    'seasonal_order': None,
                    'AIC': float('nan'),
                    'Status': 'Failed',
                    'Error Message': str(e)
                })
    
    return (commodity, results, best_map)


def _gcs_save_model(model_fit, commodity, prefix='models/arima/', d_override=None, m_override=None):
    try:
        # Determine d and m to encode in filename; allow overrides from caller
        d_val = d_override
        m_val = m_override
        # Try to infer m from model if not provided
        if m_val is None:
            try:
                so = getattr(model_fit, 'seasonal_order', None)
                if so and len(so) == 4 and so[3]:
                    m_val = int(so[3])
                else:
                    m_val = 1
            except Exception:
                m_val = 1
        # Try to infer d from model.order if not provided
        if d_val is None:
            try:
                order = getattr(model_fit, 'order', None)
                if callable(order):
                    order = order()
                if isinstance(order, (list, tuple)) and len(order) >= 2:
                    d_val = int(order[1])
            except Exception:
                d_val = None

        suffix = f"_d{d_val}_m{m_val}" if (d_val is not None or m_val is not None) else "_dNone_mNone"
        key = prefix + re.sub(r"\s+", '_', commodity) + suffix + '.pkl'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        data = pickle.dumps(model_fit)
        blob.upload_from_string(data, content_type='application/octet-stream')
        return key
    except Exception as e:
        print(f"  Failed to save model for {commodity} to GCS: {e}")
        return None


def _is_flatline(fvals, hist_vals, abs_std_thresh=1e-6, rel_range_thresh=1e-3):
    try:
        f = np.array(fvals, dtype=float)
        if f.size == 0:
            return True
        hist = np.array(hist_vals, dtype=float)
        f_std = np.nanstd(f)
        if f_std < abs_std_thresh:
            return True
        hist_mean = np.nanmean(hist) if hist.size else 0.0
        rng = np.nanmax(f) - np.nanmin(f)
        if abs(hist_mean) > 1e-9:
            rel = abs(rng / (abs(hist_mean)))
            if rel < rel_range_thresh:
                return True
        return False
    except Exception:
        return False


def download_model(commodity, preferred_d=1, preferred_m=None, prefix='models/arima/'):
    """Download a saved ARIMA/SARIMA model for `commodity` from GCS.
    Prefer `preferred_d` and `preferred_m`, fall back to the other differencing order.
    If preferred_m is None, try to find the best model across all m values.
    Returns tuple (d_used, model_obj) or (None, None) if not found/failed.
    """
    norm = re.sub(r"\s+", '_', commodity)
    bucket = storage_client.bucket(bucket_name)
    # list candidate blobs under prefix and filter by commodity
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # helper to find blob for a given d and m
    def _find_blob_for_d_m(d, m=None):
        candidates = []
        for b in blobs:
            name = b.name
            if norm in name and f"_d{d}_" in name and name.endswith('.pkl'):
                if m is not None:
                    if f"_m{m}" in name:
                        return b
                else:
                    # Collect all matches for this d
                    candidates.append(b)
        # If no specific m requested, return first match (or None)
        return candidates[0] if candidates else None

    # Try preferred d and m first, then fallback
    m_tries = [preferred_m] if preferred_m is not None else [1, 75, 125, 250, 375]  # Try non-seasonal first, then seasonal
    d_tries = [preferred_d, 0 if preferred_d == 1 else 1]
    
    for d_try in d_tries:
        for m_try in m_tries:
            b = _find_blob_for_d_m(d_try, m_try)
            if b is None:
                continue
            try:
                data = b.download_as_bytes()
                obj = pickle.loads(data)
                return (d_try, obj)
            except Exception as e:
                print(f"  Failed to download/unpickle {b.name}: {e}")
                continue

    return (None, None)


def generate_and_save_combined_arima_forecast(forecast_steps=250, gcs_prefix='forecast_data/arima_forecast.csv', 
                                               conf_interval_05=True, conf_interval_10=True):
    """Generate forecasts for all numeric commodities using saved models,
    combine with historical `prices_df`, extend the date index for the forecast
    horizon and save the combined DataFrame to GCS as CSV.
    
    Args:
        forecast_steps: Number of periods to forecast
        gcs_prefix: GCS path to save the combined forecast
        conf_interval_05: If True, calculate 5% confidence intervals (95% confidence)
        conf_interval_10: If True, calculate 10% confidence intervals (90% confidence)
    
    Returns:
        Tuple of (combined_df, gcs_prefix) where combined_df contains forecasts with confidence intervals
    """
    # Suppress statsmodels index warnings since we handle dates ourselves
    warnings.filterwarnings("ignore", category=FutureWarning, message="No supported index is available.*")
    warnings.filterwarnings("ignore", message="No supported index is available.*")
    
    forecast_series_dict = {}
    conf_lower_05_dict = {}  # 5% alpha = 95% confidence interval
    conf_upper_05_dict = {}
    conf_lower_10_dict = {}  # 10% alpha = 90% confidence interval
    conf_upper_10_dict = {}
    arima_forecast_results = []

    for commodity in commodity_columns:
        series = prices_df[commodity].dropna()
        if series.empty:
            arima_forecast_results.append({'Commodity': commodity, 'Status': 'Skipped', 'Error Message': 'Empty series'})
            continue

        # Try preferred d=1 then fallback to d=0
        d_used, model = download_model(commodity, preferred_d=1)
        if model is None:
            d_used, model = download_model(commodity, preferred_d=0)

        if model is None:
            print(f"  {commodity}: No saved model found")
            arima_forecast_results.append({'Commodity': commodity, 'Status': 'NoModel'})
            continue
        
        print(f"\n  {commodity}: Initial model loaded with d={d_used}")

        try:
            # Get forecast with confidence intervals
            conf_int_05 = None
            conf_int_10 = None
            
            if hasattr(model, 'forecast'):
                fvals = model.forecast(forecast_steps)
            elif hasattr(model, 'predict'):
                fvals = model.predict(forecast_steps)
            else:
                arima_forecast_results.append({'Commodity': commodity, 'Status': 'NoForecastMethod'})
                continue
            
            # Try to get prediction intervals if available
            # Check if model supports get_forecast for confidence intervals (statsmodels ARIMA)
            if hasattr(model, 'get_forecast'):
                try:
                    forecast_obj = model.get_forecast(steps=forecast_steps)
                    if conf_interval_05:
                        conf_int_05 = forecast_obj.conf_int(alpha=0.05)
                    if conf_interval_10:
                        conf_int_10 = forecast_obj.conf_int(alpha=0.10)
                except Exception as e:
                    print(f"  Could not get confidence intervals for {commodity}: {e}")
            # Check if pmdarima model with predict and return_conf_int support
            elif hasattr(model, 'predict'):
                try:
                    if conf_interval_05:
                        _, conf_int_05 = model.predict(n_periods=forecast_steps, return_conf_int=True, alpha=0.05)
                    if conf_interval_10 and not conf_interval_05:
                        _, conf_int_10 = model.predict(n_periods=forecast_steps, return_conf_int=True, alpha=0.10)
                    elif conf_interval_10:
                        # Need separate call for different alpha
                        _, conf_int_10 = model.predict(n_periods=forecast_steps, return_conf_int=True, alpha=0.10)
                except Exception as e:
                    print(f"  Could not get confidence intervals for {commodity}: {e}")
                    
        except Exception as e:
            arima_forecast_results.append({'Commodity': commodity, 'Status': 'ForecastFailed', 'Error Message': str(e)})
            continue

        # If flatline, force d=0
        if _is_flatline(fvals, series.values) and d_used != 0:
            print(f"    → Flatline detected! Attempting to switch from d={d_used} to d=0")
            d0, model0 = download_model(commodity, preferred_d=0)
            if model0 is not None:
                try:
                    if hasattr(model0, 'forecast'):
                        fvals = model0.forecast(forecast_steps)
                    elif hasattr(model0, 'predict'):
                        fvals = model0.predict(forecast_steps)
                    d_used = 0
                    model = model0
                    print(f"    → Successfully switched to d=0 model")
                    
                    # Re-calculate confidence intervals with new model
                    conf_int_05 = None
                    conf_int_10 = None
                    if hasattr(model, 'get_forecast'):
                        try:
                            forecast_obj = model.get_forecast(steps=forecast_steps)
                            if conf_interval_05:
                                conf_int_05 = forecast_obj.conf_int(alpha=0.05)
                            if conf_interval_10:
                                conf_int_10 = forecast_obj.conf_int(alpha=0.10)
                        except Exception:
                            pass
                    elif hasattr(model, 'predict'):
                        try:
                            if conf_interval_05:
                                _, conf_int_05 = model.predict(n_periods=forecast_steps, return_conf_int=True, alpha=0.05)
                            if conf_interval_10:
                                _, conf_int_10 = model.predict(n_periods=forecast_steps, return_conf_int=True, alpha=0.10)
                        except Exception:
                            pass
                except Exception:
                    print(f"    → Failed to switch to d=0 model, keeping d={d_used}")
                    pass
            else:
                print(f"    → No d=0 model available, keeping d={d_used}")
        elif _is_flatline(fvals, series.values):
            print(f"    → Flatline detected (already using d={d_used})")
        else:
            print(f"    → Forecast looks good with d={d_used}")

        # Build forecast dates using inferred frequency
        try:
            freq = pd.infer_freq(series.index)
        except Exception:
            freq = None
        if freq is None:
            # Fallback to daily frequency
            freq = 'D'
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]

        try:
            forecast_series = pd.Series(np.array(fvals).astype(float), index=forecast_dates)
            forecast_series_dict[commodity] = forecast_series
            
            # Store confidence intervals if available
            if conf_int_05 is not None:
                try:
                    # Handle different array structures
                    conf_int_05_array = np.array(conf_int_05)
                    if conf_int_05_array.ndim == 2:
                        conf_lower_05_dict[f"{commodity}_lower_05"] = pd.Series(conf_int_05_array[:, 0], index=forecast_dates)
                        conf_upper_05_dict[f"{commodity}_upper_05"] = pd.Series(conf_int_05_array[:, 1], index=forecast_dates)
                    elif conf_int_05_array.ndim == 1:
                        # If 1D, assume it's already lower/upper bounds separately
                        print(f"  Warning: Unexpected 1D confidence interval structure for {commodity}")
                except Exception as ci_err:
                    print(f"  Could not store 5% confidence intervals for {commodity}: {ci_err}")
            
            if conf_int_10 is not None:
                try:
                    # Handle different array structures
                    conf_int_10_array = np.array(conf_int_10)
                    if conf_int_10_array.ndim == 2:
                        conf_lower_10_dict[f"{commodity}_lower_10"] = pd.Series(conf_int_10_array[:, 0], index=forecast_dates)
                        conf_upper_10_dict[f"{commodity}_upper_10"] = pd.Series(conf_int_10_array[:, 1], index=forecast_dates)
                    elif conf_int_10_array.ndim == 1:
                        print(f"  Warning: Unexpected 1D confidence interval structure for {commodity}")
                except Exception as ci_err:
                    print(f"  Could not store 10% confidence intervals for {commodity}: {ci_err}")
            
            arima_forecast_results.append({
                'Commodity': commodity, 
                'Status': 'Success', 
                'd_used': d_used,
                'Has_CI_05': conf_int_05 is not None,
                'Has_CI_10': conf_int_10 is not None,
                'Forecast_Start': forecast_dates[0],
                'Forecast_End': forecast_dates[-1]
            })
            print(f"    ✓ Forecast complete using ARIMA model with d={d_used}")
        except Exception as e:
            print(f"    ✗ Failed to build forecast series: {e}")
            arima_forecast_results.append({'Commodity': commodity, 'Status': 'SeriesBuildFailed', 'Error Message': str(e)})

    # Convert results to DataFrame
    arima_forecast_df = pd.DataFrame(arima_forecast_results)
    print("\n" + "="*80)
    print("ARIMA Forecast Summary:")
    print("="*80)
    print(arima_forecast_df.to_string())
    print("="*80)

    # Prepare historical dataframe of numeric commodities
    historical_df = prices_df[commodity_columns].copy()

    # Prepare forecast wide dataframe including confidence intervals
    all_forecast_dict = {}
    all_forecast_dict.update(forecast_series_dict)
    all_forecast_dict.update(conf_lower_05_dict)
    all_forecast_dict.update(conf_upper_05_dict)
    all_forecast_dict.update(conf_lower_10_dict)
    all_forecast_dict.update(conf_upper_10_dict)
    
    if all_forecast_dict:
        forecast_wide_df = pd.DataFrame(all_forecast_dict)
    else:
        forecast_wide_df = pd.DataFrame(columns=historical_df.columns)

    combined_df = pd.concat([historical_df, forecast_wide_df], axis=0, join='outer')
    combined_df.sort_index(inplace=True)
    combined_df.index.name = 'Date'
    
    # Reset index to make Date a regular column in the output
    combined_df_to_save = combined_df.reset_index()

    print(f"\nCombined DataFrame shape: {combined_df.shape}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Historical data points: {len(historical_df)}")
    print(f"Forecast data points: {len(forecast_wide_df)}")

    # Save combined DataFrame to GCS
    print(f"\nSaving combined ARIMA forecast results to GCS prefix: {gcs_prefix}")
    save_dataframe_to_gcs(df=combined_df_to_save, bucket_name=bucket_name, gcs_prefix=gcs_prefix, validate_rows=False)

    return combined_df, gcs_prefix


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ARIMA FORECASTING WORKFLOW")
    print("="*80)
    
    # Ask user if they want to load saved models or run tests
    print("\nDo you want to:")
    print("  1. Load saved models from GCS (faster)")
    print("  2. Run new ARIMA grid search and train models (slower, more accurate)")
    
    user_choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if user_choice == '2':
        print("\n" + "="*80)
        print("Running ARIMA Grid Search and Model Training")
        print("="*80)
        
        # Load seasonality study and build optimized m_values mapping
        print("\n" + "="*80)
        print("SEASONALITY-GUIDED PARAMETER OPTIMIZATION")
        print("="*80)
        seasonality_map = load_seasonality_study()
        commodity_m_map = build_commodity_m_mapping(commodity_columns, seasonality_map)

        # Display optimization results
        print("\nOptimized m_values per commodity:")
        for commodity, m_vals in sorted(commodity_m_map.items()):
            if commodity in seasonality_map:
                detected_period = seasonality_map[commodity]
                print(f"  {commodity:20s}: m={m_vals} (detected period: {detected_period})")
            else:
                print(f"  {commodity:20s}: m={m_vals} (using defaults)")

        print("\n" + "="*80)
        print("ARIMA/SARIMA GRID SEARCH CONFIGURATION")
        print("="*80)
        print(f"Using pmdarima auto_arima for intelligent model selection (much faster!)")
        print(f"Non-seasonal: d=[0,1], p/q auto-selected (max_p=3, max_q=3)")
        print(f"Seasonal: D auto-selected, P/Q auto-selected (max_P=2, max_Q=2)")
        print(f"m_values: Commodity-specific based on empirical seasonality analysis")
        print(f"\nEstimated search time per commodity: 10-30 seconds (vs 5-10 minutes with grid)")
        for commodity in sorted(commodity_columns)[:5]:  # Show first 5 as examples
            m_vals = commodity_m_map.get(commodity, [1, 75, 250])
            print(f"  {commodity:20s}: m={m_vals}")
        print(f"  ... (see full list above)")
        print(f"\nBENEFITS: Stepwise search + empirically-detected periods = fast & accurate")
        print(f"          Auto-convergence handling prevents hanging")
        
        # Build per-commodity tasks with optimized m_values
        tasks = []
        for commodity in commodity_columns:
            series = prices_df[commodity].dropna()
            if series.empty:
                print(f"  Skipping {commodity}: empty after dropna")
                continue
            m_values_for_commodity = commodity_m_map.get(commodity, [1, 75, 250])
            tasks.append((commodity, series.values, series.index.astype(str).tolist(), m_values_for_commodity))

        # Run per-commodity grid in parallel using processes
        max_workers = max(1, (os.cpu_count() or 1) - 1)
        print(f"\nRunning per-commodity auto_arima using {max_workers} processes...")
        import pickle as _pickle
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_commodity_auto_arima_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                commodity, eval_rows, best_map_ser = fut.result()
                # append evaluations
                arima_evaluation_results.extend(eval_rows)
                # reconstruct best models
                best_models[commodity] = {}
                for (d_val, m_val), (order, seasonal_order, best_bytes, aic) in best_map_ser.items():
                    try:
                        model_fit = _pickle.loads(best_bytes) if best_bytes is not None else None
                    except Exception:
                        model_fit = None
                    if model_fit is not None:
                        best_models[commodity][(d_val, m_val)] = (order, seasonal_order, model_fit, aic)

        # Convert to DataFrame
        arima_evaluation_df = pd.DataFrame(arima_evaluation_results)
        print("\nARIMA Evaluation Results DataFrame Head:")
        print(arima_evaluation_df.head())

        # Save best models per commodity/d/m to GCS
        arima_fitted_models = {}
        print("\nSaving selected best-fit ARIMA/SARIMA models to GCS...")
        for commodity, dmap in best_models.items():
            arima_fitted_models[commodity] = {}
            for (d_val, m_val), (order, seasonal_order, model_fit, aic) in dmap.items():
                try:
                    key = _gcs_save_model(model_fit, commodity, prefix='models/arima/', d_override=d_val, m_override=m_val)
                    if key:
                        model_type = "SARIMA" if m_val > 1 else "ARIMA"
                        print(f"  Saved {commodity} {model_type} d={d_val} m={m_val} order={order} seasonal={seasonal_order} AIC={aic:.2f} -> {key}")
                    arima_fitted_models[commodity][(d_val, m_val)] = model_fit
                except Exception as e:
                    print(f"  Failed to save selected model for {commodity} d={d_val} m={m_val}: {e}")
        
        print("\n" + "="*80)
        print("Model Training Complete")
        print("="*80)
    
    elif user_choice == '1':
        print("\n" + "="*80)
        print("Loading Saved Models from GCS")
        print("="*80)
        print("Models will be loaded during forecasting...")
    
    else:
        print("\nInvalid choice. Defaulting to loading saved models.")
        user_choice = '1'
    
    # Generate forecasts with confidence intervals
    print("\n" + "="*80)
    print("Generating Forecasts with Confidence Intervals")
    print("="*80)
    
    forecast_steps = int(input("\nEnter number of forecast periods (default 250): ").strip() or "250")
    
    print(f"\nGenerating {forecast_steps}-step forecast with 5% and 10% confidence intervals...")
    combined_df, gcs_prefix = generate_and_save_combined_arima_forecast(
        forecast_steps=forecast_steps,
        gcs_prefix='forecast_data/arima_forecast.csv',
        conf_interval_05=True,
        conf_interval_10=True
    )
    
    print("\n" + "="*80)
    print("ARIMA Forecasting Workflow Complete")
    print("="*80)
    print(f"\nResults saved to GCS: {gcs_prefix}")
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"\nColumns in output:")
    for col in combined_df.columns:
        print(f"  - {col}")

# # ==============================================================================
# # Select best models (by AIC) and forecast using selected orders
# # ==============================================================================
# import warnings
# from statsmodels.tsa.arima.model import ARIMA
# from pandas.tseries.frequencies import infer_freq

# warnings.filterwarnings("ignore")

# # Ensure arima_evaluation_df is present
# if 'arima_evaluation_df' not in locals() or arima_evaluation_df.empty:
#     print("Error: 'arima_evaluation_df' not found or empty. Skipping forecasting.")
#     arima_forecast_df = pd.DataFrame()
# else:
#     print("Selecting best models and forecasting 24 periods per commodity...")

#     successful_fits_df = arima_evaluation_df[arima_evaluation_df['Status'] == 'Success'].copy()

#     if successful_fits_df.empty:
#         print("No successful auto-ARIMA selections found. No forecasts will be produced.")
#         arima_forecast_df = pd.DataFrame()
#     else:
#         # For robustness, pick the min-AIC row per commodity (auto_arima usually gives one row)
#         best_models_indices = successful_fits_df.groupby('Commodity')['AIC'].idxmin()
#         best_arima_models_df = successful_fits_df.loc[best_models_indices].reset_index(drop=True)

#         print("Best ARIMA models selected for forecasting:")
#         print(best_arima_models_df.head())

#         # Forecast settings
#         forecast_steps = 52
#         arima_forecast_results = []

#         # Use the fitted pmdarima models to forecast directly
#         def _is_flatline(fvals, hist_vals):
#             try:
#                 f = np.array(fvals, dtype=float)
#                 if f.size == 0:
#                     return True
#                 hist = np.array(hist_vals, dtype=float)
#                 # absolute and relative thresholds
#                 abs_std_thresh = 1e-6
#                 rel_range_thresh = 1e-3
#                 f_std = np.nanstd(f)
#                 if f_std < abs_std_thresh:
#                     return True
#                 hist_mean = np.nanmean(hist) if hist.size else 0.0
#                 rng = np.nanmax(f) - np.nanmin(f)
#                 if abs(hist_mean) > 1e-9:
#                     rel = abs(rng / (abs(hist_mean)))
#                     if rel < rel_range_thresh:
#                         return True
#                 return False
#             except Exception:
#                 return False

#         for commodity, model_dict in arima_fitted_models.items():
#             # model_dict expected like {0: model_d0, 1: model_d1}
#             series = prices_df[commodity].dropna()
#             if series.empty:
#                 arima_forecast_results.append({
#                     'Commodity': commodity,
#                     'Status': 'Skipped',
#                     'Error Message': 'Empty series',
#                     'ARIMA Order': None,
#                     'Forecast Dates': None,
#                     'Forecast Values': None
#                 })
#                 print(f"  Skipping forecasting for {commodity}: Empty series")
#                 continue

#             try:
#                 # Prefer d=1 model first, fallback to d=0 if needed
#                 chosen_model = None
#                 chosen_d = None
#                 if isinstance(model_dict, dict) and 1 in model_dict:
#                     chosen_model = model_dict[1]
#                     chosen_d = 1
#                 elif isinstance(model_dict, dict) and 0 in model_dict:
#                     chosen_model = model_dict[0]
#                     chosen_d = 0
#                 else:
#                     # fallback: if model_dict is actually a single model (backcompat), use it
#                     chosen_model = model_dict
#                     chosen_d = None

#                 if chosen_model is None:
#                     arima_forecast_results.append({
#                         'Commodity': commodity,
#                         'Status': 'Failed',
#                         'Error Message': 'No fitted model available',
#                         'ARIMA Order': None,
#                         'Forecast Dates': None,
#                         'Forecast Values': None
#                     })
#                     print(f"    No fitted models available for {commodity}")
#                     continue

#                 print(f"  Forecasting {commodity} using fitted auto_arima model (d={chosen_d})...")
#                 forecast_vals = chosen_model.predict(n_periods=forecast_steps)

#                 # If forecast is a flatline, try alternative differencing model (d=1) if available
#                 if _is_flatline(forecast_vals, series.values):
#                     alt_model = None
#                     alt_d = None
#                     if chosen_d == 0 and isinstance(model_dict, dict) and 1 in model_dict:
#                         alt_model = model_dict[1]
#                         alt_d = 1
#                     elif chosen_d == 1 and isinstance(model_dict, dict) and 0 in model_dict:
#                         alt_model = model_dict[0]
#                         alt_d = 0

#                     if alt_model is not None:
#                         try:
#                             alt_forecast = alt_model.predict(n_periods=forecast_steps)
#                             if not _is_flatline(alt_forecast, series.values):
#                                 forecast_vals = alt_forecast
#                                 chosen_model = alt_model
#                                 chosen_d = alt_d
#                                 print(f"    Switched to alternative model d={alt_d} for {commodity} due to flatline.")
#                         except Exception:
#                             pass

#                 # Build forecast dates using inferred frequency
#                 freq = infer_freq(series.index)
#                 if freq is None:
#                     freq = 'W'
#                 last_date = series.index[-1]
#                 forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]

#                 arima_forecast_results.append({
#                     'Commodity': commodity,
#                     'Status': 'Success',
#                     'Error Message': None,
#                     'ARIMA Order': getattr(chosen_model, 'order', None),
#                     'Forecast Dates': forecast_dates.tolist(),
#                     'Forecast Values': np.array(forecast_vals).tolist()
#                 })
#                 print(f"    Forecast generated for {commodity} ({forecast_steps} steps) using d={chosen_d}.")
#             except Exception as e:
#                 print(f"    Forecasting failed for {commodity}: {e}")
#                 arima_forecast_results.append({
#                     'Commodity': commodity,
#                     'Status': 'Failed',
#                     'Error Message': str(e),
#                     'ARIMA Order': getattr(chosen_model, 'order', None) if 'chosen_model' in locals() else None,
#                     'Forecast Dates': None,
#                     'Forecast Values': None
#                 })

#         # Convert results to DataFrame
#         arima_forecast_df = pd.DataFrame(arima_forecast_results)

#         print("\nARIMA Forecast Results DataFrame Head:")
#         print(arima_forecast_df.head())

#         # --- Combine historical and forecast data ---
#         # Prepare historical dataframe of commodities
#         commodity_columns = [c for c in prices_df.columns if pd.api.types.is_numeric_dtype(prices_df[c])]
#         historical_df = prices_df[commodity_columns].copy()

#         # Prepare forecast wide dataframe
#         forecast_series_dict = {}
#         for _, frow in arima_forecast_df.iterrows():
#             if frow['Status'] == 'Success' and frow['Forecast Dates'] is not None and frow['Forecast Values'] is not None:
#                 try:
#                     forecast_series = pd.Series(frow['Forecast Values'], index=pd.to_datetime(frow['Forecast Dates']))
#                     forecast_series_dict[frow['Commodity']] = forecast_series
#                 except Exception as e:
#                     print(f"Warning: could not create forecast series for {frow['Commodity']}: {e}")

#         if forecast_series_dict:
#             forecast_wide_df = pd.DataFrame(forecast_series_dict)
#         else:
#             forecast_wide_df = pd.DataFrame(columns=historical_df.columns)

#         combined_df = pd.concat([historical_df, forecast_wide_df], axis=0, join='outer')
#         combined_df.sort_index(inplace=True)
#         combined_df.index.name = 'Date'

#         print("\nCombined DataFrame head/tail:")
#         print(combined_df.head(2))
#         print(combined_df.tail(2))
#         print("Combined DataFrame shape:", combined_df.shape)

#         if not combined_df.empty:
#             print("\nSaving combined ARIMA forecast results to GCS...")
#             gcs_prefix_arima_forecast = 'forecast_data/arima_forecast.csv'
#             save_dataframe_to_gcs(
#                 df=combined_df,
#                 bucket_name='crystal-dss',
#                 gcs_prefix=gcs_prefix_arima_forecast,
#                 validate_rows=False
#             )
#             print(f"Combined ARIMA forecast results saved to GCS prefix: {gcs_prefix_arima_forecast}")
#         else:
#             print("Combined DataFrame is empty. No data to save.")


# """## FB PROPHET"""

# # PROPHET

# # Imports
# import numpy as np
# import pandas as pd
# from prophet import Prophet
# import nest_asyncio
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import gspread
# from google.colab import auth
# from google.auth import default

# # Allow nested async in notebooks
# nest_asyncio.apply()
# import asyncio, aiohttp

# df = prices_df
# print (f"Original Shape : {df.shape}")

# # Forecast function (runs in threads)
# def forecast_blocking(col_name, series, periods=24):
#     try:
#         ts = pd.DataFrame({'ds': series.index, 'y': series.values})
#         ts['y'] = ts['y'].astype(float)  # Ensure float for NaNs
#         model = Prophet()
#         model.fit(ts)
#         future = model.make_future_dataframe(periods=periods, freq='w')
#         forecast = model.predict(future)
#         return forecast[['ds', 'yhat']].rename(columns={'yhat': col_name}).set_index('ds')
#     except Exception as e:
#         print(f"[{col_name}] forecast error → {e}")
#         return pd.DataFrame()


# # Async wrapper to run forecast in thread pool
# async def run_in_executor(loop, executor, func, *args):
#     return await loop.run_in_executor(executor, func, *args)

# # Orchestration logic
# async def forecast_all_series(df, periods=12, max_workers=8):
#     loop = asyncio.get_event_loop()
#     executor = ThreadPoolExecutor(max_workers=max_workers)
#     # Filter out non-numeric columns before creating tasks
#     numeric_cols = df.select_dtypes(include=np.number).columns
#     tasks = [
#         run_in_executor(loop, executor, forecast_blocking, col, df[col], periods)
#         for col in numeric_cols
#     ]
#     results = await asyncio.gather(*tasks)
#     executor.shutdown(wait=True)
#     # Filter out empty DataFrames which are returned for skipped series
#     results = [res for res in results if not res.empty]
#     if results:
#         return pd.concat(results, axis=1).reset_index()
#     else:
#         return pd.DataFrame()


# # Run everything
# forecast_df = await forecast_all_series(df, periods=24)
# print(forecast_df.shape)
# print(forecast_df.head())

# # Get the last date from the original DataFrame
# last_historical_date = df.index.max()

# # Filter forecast_df to keep only future dates
# # Ensure 'ds' is a datetime type before comparison
# if 'ds' in forecast_df.columns:
#     forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
#     future_forecast_df = forecast_df[forecast_df['ds'] > last_historical_date].set_index('ds')
# else:
#     future_forecast_df = pd.DataFrame() # Handle case where forecast_df is empty


# # Combine the original DataFrame with the future forecast
# # Ensure 'date' column from original df is not included in the concat if it's not the index
# df_to_concat = df.copy()
# if 'date' in df_to_concat.columns:
#     df_to_concat = df_to_concat.drop(columns=['date'])

# # Align columns before concatenation, filling missing columns with NaN
# combined_historical_and_forecast_df = pd.concat([df_to_concat, future_forecast_df], axis=0, join='outer')


# # Convert all columns to numeric, coercing errors
# combined_historical_and_forecast_df = combined_historical_and_forecast_df.apply(pd.to_numeric, errors='coerce')

# # Rename the index to 'date'
# combined_historical_and_forecast_df.index.name = 'date'


# print(combined_historical_and_forecast_df.shape)
# print(combined_historical_and_forecast_df.head(2))
# print(combined_historical_and_forecast_df.tail(2))


# #===============================================================================
# # Save the combined historical and forecast data to Google Sheets
# #===============================================================================
# if not combined_historical_and_forecast_df.empty:
#     print("\nSaving combined historical and Prophet forecast data to GCS...")
#     gcs_prefix_fb_prophet_forecast = 'forecast_data/fb_prophet_forecast.csv'
#     save_dataframe_to_gcs(
#         df=combined_historical_and_forecast_df,
#         bucket_name='crystal-dss',
#         gcs_prefix=gcs_prefix_fb_prophet_forecast,
#         validate_rows=False
# else:
#     print("No combined historical and Prophet forecast data found to save.")


# #===============================================================================
# # Visualize the first 5 time series
# #===============================================================================
# import matplotlib.pyplot as plt
# # Assuming df is the original DataFrame with historical data
# start_date = df.index.min()
# end_date = df.index.max()

# print(f"Start Date: {start_date}")
# print(f"End Date: {end_date}")

# # Get the list of columns to plot (excluding the index and potentially the original 'date' column if it wasn't dropped earlier)
# cols_to_plot = [col for col in combined_historical_and_forecast_df.columns if col != 'date']

# # Visualize the first 5 columns (excluding the 'date' column if present)
# for i, col in enumerate(cols_to_plot[:5]):
#     plt.figure(figsize=(12, 3))
#     plt.plot(combined_historical_and_forecast_df.index, combined_historical_and_forecast_df[col])
#     plt.title(f'Time Series Forecast for {col}')
#     plt.xlabel('Date')
#     plt.ylabel(col)
#     # Add a vertical red dashed line at the end_date of historical data
#     plt.axvline(end_date, color='red', linestyle='--', label='forecast_period_start')
#     plt.legend() # Add legend to show the label for the vertical line
#     plt.show()

# """## PROPHET WITH COVARIATES"""

# import pandas as pd
# from prophet import Prophet
# import numpy as np
# import warnings
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# warnings.filterwarnings("ignore") # Suppress warnings during model fitting

# # ==============================================================================
# # Define exog_regressors, create forecast and save df to be called in next step
# # ==============================================================================
# forecast_steps = 24
# prices_df = prices_df

# # Identify all columns in the prices_df DataFrame
# all_columns = prices_df.columns.tolist()

# # Define columns to exclude (non-commodities and cyclical features)
# cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
#                    'day_of_week_sin', 'day_of_week_cos',
#                    'month_sin', 'month_cos',
#                    'year_sin', 'year_cos']

# # Create a list of commodity columns by excluding the specified columns and keeping only numeric ones
# commodity_columns = [col for col in all_columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]

# # Define the list of the first 7 commodities as covariates




# # Define the list of the remaining commodities as target commodities
# target_commodities = commodity_columns[8:]

# target_commodities = [
#     'Acetic Acid',
#     'Butyl Acetate',
#     'Toluene',
#     'Isomer-MX',
#     'Solvent-MX',
#     'Methanol',
#     'MTBE',
#     'Benzene'
# ]

# # Define the other commodities to correlate against
# exog_regressors = [
#     'Gold',
#     'Silver',
#     'Copper',
#     'S&P 500',
#     'Shanghai Composite',
#     'USD Index',
#     'Japanese Yen',
#     'US 10-Y BOND PRICE',
#     'Crude Oil',
#     'Natural Gas',
#     'Naphtha',
#     'EDC'
#     'Ethylene',
#     'Propylene',
#     'N-Butanol',
#     'OrthXylene',
#     'Cyclohexane',
#     'Acetic Acid',
#     'Toluene',
#     'Methanol',
#     'MTBE',
#     'Benzene'
# ]

# # Print the lists to verify
# print("Exogenous Regressors:")
# print(exog_regressors)
# print("\nTarget Commodities:")
# print(target_commodities)

# if 'exog_regressors' not in locals() or not exog_regressors:
#     print("Error: 'exog_regressors' list is not defined or is empty. Cannot forecast covariates.")
#     # Finish the task with failure if exogenous regressors are not defined
#     # Initialize necessary variables to prevent errors in later steps
#     covariate_forecasts = {}
#     future_exog_data = pd.DataFrame()
# else:
#     print(f"Forecasting the following covariates using separate Prophet models: {exog_regressors}")

#     # Check if forecast_steps is defined
#     if 'forecast_steps' not in locals():
#          print("Error: 'forecast_steps' is not defined. Cannot forecast covariates.")
#          # Finish the task with failure if forecast_steps is not defined
#          covariate_forecasts = {}
#          future_exog_data = pd.DataFrame()
#     else:
#         print(f"Forecasting {forecast_steps} steps into the future for each covariate.")

#         # Dictionary to store forecast results for each covariate
#         covariate_forecasts = {}

#         # Configure the thread pool executor for concurrent forecasting
#         max_workers = 8 # Example: use 8 worker threads
#         executor = ThreadPoolExecutor(max_workers=max_workers)
#         tasks = []

#         # Async function to fit Prophet and forecast for a single covariate
#         async def forecast_single_covariate_async(series, covariate_name, steps, executor):
#              """
#              Fits a Prophet model and generates a forecast for a single covariate
#              in an async context, running synchronous operations in a thread pool.
#              """
#              print(f"  Starting fitting and forecasting for covariate: {covariate_name}...")
#              try:
#                 loop = asyncio.get_event_loop()

#                 # Prepare data for Prophet
#                 covariate_df = series.reset_index().rename(columns={'Date': 'ds', covariate_name: 'y'})
#                 covariate_df['ds'] = pd.to_datetime(covariate_df['ds'])
#                 covariate_df['y'] = pd.to_numeric(covariate_df['y'], errors='coerce')
#                 covariate_df.dropna(inplace=True) # Drop NaNs after coercion

#                 if covariate_df.empty:
#                     print(f"  Skipping fitting for {covariate_name}: Empty data after preparation.")
#                     return covariate_name, pd.DataFrame() # Return empty DataFrame if data is insufficient

#                 # Run the synchronous Prophet fitting in a thread pool
#                 model = Prophet()
#                 model_fit = await loop.run_in_executor(executor, lambda: model.fit(covariate_df))
#                 # print(f"  Prophet model fitted for {covariate_name}.") # Keep print for debugging if needed

#                 # Create future DataFrame for the covariate forecast
#                 future_covariate = model_fit.make_future_dataframe(periods=steps)

#                 # Generate forecast for the covariate
#                 forecast_covariate = model_fit.predict(future_covariate)
#                 # print(f"  Forecast generated for {covariate_name}.") # Keep print for debugging if needed

#                 # Store the forecasted values ('yhat') with the 'ds' (date)
#                 # We only need the future dates beyond the historical data
#                 last_historical_date = covariate_df['ds'].max()
#                 future_covariate_forecast = forecast_covariate[forecast_covariate['ds'] > last_historical_date][['ds', 'yhat']].rename(columns={'yhat': covariate_name})

#                 # Set 'ds' as index
#                 future_covariate_forecast.set_index('ds', inplace=True)

#                 print(f"  Finished fitting and forecasting for covariate: {covariate_name}.")
#                 return covariate_name, future_covariate_forecast

#              except Exception as e:
#                 print(f"  Error forecasting covariate {covariate_name}: {e}")
#                 return covariate_name, pd.DataFrame() # Return empty DataFrame on error

#         # Create a list of async tasks for forecasting each covariate
#         tasks = []
#         for covariate in exog_regressors:
#              # Check if the covariate column exists and is numeric before creating a task
#              if covariate in prices_df.columns and pd.api.types.is_numeric_dtype(prices_df[covariate]):
#                  tasks.append(forecast_single_covariate_async(prices_df[covariate].dropna(), covariate, forecast_steps, executor))
#              else:
#                  print(f"  Skipping {covariate} as it is not found or not numeric in the DataFrame.")
#                  # Add an empty DataFrame entry for skipped covariates
#                  covariate_forecasts[covariate] = pd.DataFrame()


#         # Run the async tasks concurrently and collect the results
#         try:
#             all_covariate_results = await asyncio.gather(*tasks)
#             # Populate the dictionary with results
#             for cov_name, forecast_df in all_covariate_results:
#                 covariate_forecasts[cov_name] = forecast_df

#         except Exception as e:
#             print(f"\nAn error occurred during concurrent covariate forecasting: {e}")
#         finally:
#             # Ensure the executor is shut down
#             executor.shutdown(wait=True)


#         # --- Combine the covariate forecasts into a single DataFrame ---
#         # Filter out empty DataFrames before concatenating
#         non_empty_forecasts = {k: v for k, v in covariate_forecasts.items() if not v.empty}

#         if non_empty_forecasts:
#             # Concatenate the individual covariate forecast DataFrames
#             future_exog_data = pd.concat(non_empty_forecasts.values(), axis=1)

#             print("\nCombined future exogenous data (head):")
#             print(future_exog_data.tail(5))
#             print("\nCombined future exogenous data (shape):", future_exog_data.shape)

#         else:
#             print("\nNo successful covariate forecasts were generated.")
#             future_exog_data = pd.DataFrame() # Create an empty DataFrame if no forecasts


# print("\nCovariate forecasting complete.")

# import pandas as pd
# import numpy as np

# # Ensure prices_df (original historical data) and covariate_forecasts (dictionary of covariate forecasts) are available
# # Assuming exog_regressors (list of covariate column names) is available

# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot create covariate DataFrames.")
#     historical_data_available = False
# else:
#     historical_data_available = True
#     print("Using 'prices_df' for historical data.")

# if 'covariate_forecasts' not in locals() or not covariate_forecasts:
#     print("Error: 'covariate_forecasts' dictionary not found or is empty. Cannot create covariate DataFrames.")
#     forecast_data_available = False
# else:
#     forecast_data_available = True
#     print("Using 'covariate_forecasts' for covariate forecast data.")

# if 'exog_regressors' not in locals() or not exog_regressors:
#     print("Error: 'exog_regressors' list not found or is empty. Cannot identify covariates.")
#     covariates_defined = False
# else:
#     covariates_defined = True
#     print(f"Using defined exogenous regressors: {exog_regressors}")


# # --- Create DataFrame with Covariate Forecasts Only ---
# covariates_forecast_only_df = pd.DataFrame() # Initialize as empty

# if forecast_data_available and covariates_defined:
#     print("\nCreating DataFrame with covariate forecast data only...")

#     # The covariate_forecasts dictionary already contains DataFrames indexed by 'ds' (datetime)
#     # We can concatenate these directly.
#     try:
#         covariates_forecast_only_df = pd.concat(covariate_forecasts.values(), axis=1)

#         # Rename index for clarity
#         covariates_forecast_only_df.index.name = 'Date'
#     except Exception as e:
#         print(f"Error concatenating covariate forecasts: {e}")


# # --- Create DataFrame with Combined History and Forecasts for Covariates ---
# combined_covariates_df = pd.DataFrame() # Initialize as empty

# if historical_data_available and forecast_data_available and covariates_defined:
#     print("\nCreating DataFrame with combined history and forecasts for covariates...")

#     # Get the historical data for the covariates
#     historical_covariates_df = prices_df[exog_regressors].dropna().copy()

#     # Ensure historical_covariates_df has a DatetimeIndex and rename it 'Date'
#     historical_covariates_df.index.name = 'Date'

#     # The covariates_forecast_only_df already has the forecast data with 'Date' as index

#     # Combine the historical and forecast data for covariates
#     # Use outer join to include all dates from both historical and forecast periods
#     combined_covariates_df = pd.concat([historical_covariates_df, covariates_forecast_only_df], axis=0, join='outer')

#     # Sort by date
#     combined_covariates_df.sort_index(inplace=True)
#     print(combined_covariates_df.tail(2))
#     print("\nCombined Covariates History and Forecasts DataFrame Shape:", combined_covariates_df.shape)

# else:
#     print("\nCannot create combined covariates DataFrame: Required data is not available.")

# # You now have 'covariates_forecast_only_df' and 'combined_covariates_df' DataFrames.

# import pandas as pd
# import numpy as np

# # ==============================================================================
# # Prepare Data - historical and futures for exog_regressors
# # ==============================================================================

# # --- Start: Data Preparation ---
# # We will now use prices_df for historical target and covariate data,
# # and covariates_forecast_only_df for future covariate data.

# print("Preparing historical target data and identifying future covariate data...")

# # Dictionary to store historical data for each target commodity in Prophet format
# # This will now also include historical covariate data
# historical_target_data = {}

# # Check if prices_df is available
# if 'prices_df' not in locals() or prices_df is None:
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare historical data.")
# else:
#     # Ensure prices_df has a DatetimeIndex named 'Date'
#     if not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#         print("prices_df index is not a DatetimeIndex named 'Date'. Attempting to set index...")
#         try:
#             # Assuming the 'date' column is the correct datetime column
#             if 'date' in prices_df.columns:
#                 prices_df['Date'] = pd.to_datetime(prices_df['date'])
#                 prices_df.set_index('Date', inplace=True)
#                 prices_df.drop(columns=['date'], errors='ignore', inplace=True)
#                 print("Set 'Date' as index in prices_df.")
#             elif prices_df.index.name is None and pd.api.types.is_datetime64_any_dtype(prices_df.index):
#                  # If index is datetime but unnamed, just name it
#                  prices_df.index.name = 'Date'
#                  print("Named index 'Date' in prices_df.")
#             else:
#                  print("Warning: Could not set 'Date' as DatetimeIndex in prices_df.")
#         except Exception as e:
#             print(f"Error setting 'Date' as DatetimeIndex in prices_df: {e}")


#     # Check if target_commodities and exog_regressors are defined
#     if 'target_commodities' not in locals() or not target_commodities:
#         print("Error: target_commodities are not defined or are empty. Please define them first.")
#     elif 'exog_regressors' not in locals() or not exog_regressors:
#          print("Error: exog_regressors are not defined or are empty. Please define them first.")
#     else:
#         # Prepare the historical covariate data from prices_df
#         # Ensure only the defined exogenous regressors are selected
#         historical_covariates_df = prices_df[exog_regressors].dropna().copy()
#         # Ensure historical_covariates_df has a DatetimeIndex and rename it 'Date' for merging
#         historical_covariates_df.index.name = 'Date'


#         # Iterate through each target commodity to get historical data and merge with historical covariates
#         for target_commodity in target_commodities:
#             if target_commodity in prices_df.columns and pd.api.types.is_numeric_dtype(prices_df[target_commodity]):
#                 # Get the historical series for the target commodity and format for Prophet
#                 target_series = prices_df[target_commodity].dropna().copy()
#                 historical_target_df_prophet_format = target_series.reset_index().rename(columns={'Date': 'ds', target_commodity: 'y'})
#                 # Ensure 'ds' is datetime
#                 historical_target_df_prophet_format['ds'] = pd.to_datetime(historical_target_df_prophet_format['ds'])


#                 # Merge the historical target data with the historical covariate data
#                 # Use 'inner' merge to keep only dates present in both target and all covariates
#                 # or 'left' merge on target_df to keep all target dates and add covariates where available
#                 # Let's use 'left' to keep all historical target dates
#                 historical_data_with_covariates = pd.merge(
#                     historical_target_df_prophet_format,
#                     historical_covariates_df.reset_index().rename(columns={'Date': 'ds'}), # Merge on 'ds'
#                     how='left',
#                     on='ds'
#                 )

#                 # Ensure the 'ds' column is the first column
#                 cols = ['ds', 'y'] + [col for col in historical_data_with_covariates.columns if col not in ['ds', 'y']]
#                 historical_data_with_covariates = historical_data_with_covariates[cols]

#                 # Store the prepared and merged DataFrame (historical target + historical covariates)
#                 historical_target_data[target_commodity] = historical_data_with_covariates
#                 print(f"  Historical data prepared and merged with covariates for {target_commodity}. Shape: {historical_data_with_covariates.shape}")


#             else:
#                 print(f"  Skipping historical data preparation for {target_commodity} as it is not found or not numeric in prices_df.")

#         print("\nHistorical data prepared and merged with covariates for target commodities.")
#         print(f"Historical dataframes available for {len(historical_target_data)} target commodities.")


# # Check if covariates_forecast_only_df is available for future covariates
# if 'covariates_forecast_only_df' not in locals() or covariates_forecast_only_df.empty:
#     print("Error: 'covariates_forecast_only_df' DataFrame not found or is empty. Cannot provide future covariate data for forecasting.")
#     future_covariate_data_available = False
# else:
#     print("\nUsing 'covariates_forecast_only_df' for future covariate data.")
#     # Ensure covariates_forecast_only_df has a DatetimeIndex named 'Date'
#     if not isinstance(covariates_forecast_only_df.index, pd.DatetimeIndex) or covariates_forecast_only_df.index.name != 'Date':
#         print("covariates_forecast_only_df index is not a DatetimeIndex named 'Date'. Attempting to set index...")
#         try:
#             # Assuming the index is the date column, just name it
#             if covariates_forecast_only_df.index.name is None and pd.api.types.is_datetime64_any_dtype(covariates_forecast_only_df.index):
#                  covariates_forecast_only_df.index.name = 'Date'
#                  print("Named index 'Date' in covariates_forecast_only_df.")
#             else:
#                  print("Warning: Could not set 'Date' as DatetimeIndex in covariates_forecast_only_df index.")
#                  future_covariate_data_available = False # Mark as not available if date index is problematic

#         except Exception as e:
#             print(f"Error setting 'Date' as DatetimeIndex in covariates_forecast_only_df: {e}")
#             future_covariate_data_available = False # Mark as not available on error
#     else:
#          print("'Date' is already DatetimeIndex in covariates_forecast_only_df.")
#          future_covariate_data_available = True

# # --- End: Data Preparation ---

# # ==============================================================================
# # Run PROPHET with exog_regressors
# # ==============================================================================

# import pandas as pd
# import numpy as np
# from prophet import Prophet
# import warnings
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import gspread
# from google.colab import auth
# from google.auth import default

# warnings.filterwarnings("ignore") # Suppress warnings during model fitting

# # --- Start: Prophet Forecasting with Covariates Module ---
# # This module assumes historical_target_data (dict of historical target dataframes),
# # covariates_forecast_only_df (future covariate data), and exog_regressors (list of covariate names)
# # are available from previous steps.

# # Check if necessary data is available
# if 'historical_target_data' not in locals() or not historical_target_data:
#     print("Error: 'historical_target_data' not found or is empty. Cannot proceed with forecasting module.")
#     # Initialize results list and forecast DataFrame as empty if data is not available
#     prophet_covariates_forecast_results = []
#     prophet_covariates_forecast_summary_df = pd.DataFrame()
#     all_targets_combined_forecast_df = pd.DataFrame()
#     combined_targets_historical_forecast_df = pd.DataFrame()

# elif 'covariates_forecast_only_df' not in locals() or covariates_forecast_only_df.empty:
#     print("Error: 'covariates_forecast_only_df' DataFrame not found or is empty. Cannot proceed with forecasting module.")
#     # Initialize results list and forecast DataFrame as empty if data is not available
#     prophet_covariates_forecast_results = []
#     prophet_covariates_forecast_summary_df = pd.DataFrame()
#     all_targets_combined_forecast_df = pd.DataFrame()
#     combined_targets_historical_forecast_df = pd.DataFrame()

# elif 'exog_regressors' not in locals() or not exog_regressors:
#     print("Error: 'exog_regressors' list not found or is empty. Cannot proceed with forecasting module.")
#     # Initialize results list and forecast DataFrame as empty if data is not available
#     prophet_covariates_forecast_results = []
#     prophet_covariates_forecast_summary_df = pd.DataFrame()
#     all_targets_combined_forecast_df = pd.DataFrame()
#     combined_targets_historical_forecast_df = pd.DataFrame()

# else:
#     print("\nStarting Prophet Forecasting with Covariates Module...")

#     # Define the number of steps to forecast
#     # We will determine this from the number of rows in covariates_forecast_only_df
#     forecast_steps = len(covariates_forecast_only_df)
#     print(f"Forecasting {forecast_steps} steps based on the number of future covariate data points.")

#     # Configure the thread pool executor for concurrent forecasting
#     max_workers = 8 # Example: use 8 worker threads
#     executor = ThreadPoolExecutor(max_workers=max_workers)

#     # Initialize a list to store the forecast results
#     prophet_covariates_forecast_results = []

#     # Async function to fit Prophet with covariates and forecast for a single commodity
#     async def forecast_prophet_with_covariates(historical_df, target_commodity_name, exog_regressor_names, future_exog_df, executor):
#         """
#         Fits a Prophet model with exogenous regressors using historical data
#         and generates a forecast using future exogenous data.
#         """
#         print(f"  Starting fitting and forecasting for target: {target_commodity_name}...")
#         try:
#             loop = asyncio.get_event_loop()

#             # Initialize Prophet model
#             model = Prophet(daily_seasonality=False) # Disable daily seasonality for simplicity

#             # Add exogenous regressors to the model and keep track of which ones were added
#             added_regressors = []
#             for regressor in exog_regressor_names:
#                  # Ensure regressor column exists in historical_df before adding
#                  if regressor in historical_df.columns:
#                       model.add_regressor(regressor)
#                       added_regressors.append(regressor) # Add to our list if successfully added
#                  else:
#                       print(f"    Warning: Covariate '{regressor}' not found in historical data for {target_commodity_name}. Skipping addition.")


#             # Fit the Prophet model on historical data in a thread pool
#             model_fit = await loop.run_in_executor(executor, lambda: model.fit(historical_df))
#             # print(f"  Prophet model with covariates fitted for {target_commodity_name}.") # Optional print


#             # Create the future DataFrame for prediction
#             # This should contain 'ds' and the regressor columns from future_exog_df
#             future_df_for_predict = future_exog_df.reset_index().rename(columns={'Date': 'ds'})
#             # Ensure 'ds' is datetime
#             future_df_for_predict['ds'] = pd.to_datetime(future_df_for_predict['ds'])

#             # Ensure only regressors that were successfully added to the model are in future_df_for_predict
#             # and that the 'ds' column is also present
#             # Use the 'added_regressors' list we created
#             future_regressor_cols_present = [col for col in added_regressors if col in future_df_for_predict.columns]
#             cols_for_predict_df = ['ds'] + future_regressor_cols_present

#             # Select only the necessary columns for prediction
#             future_df_for_predict = future_df_for_predict[cols_for_predict_df].copy()

#             # Check if future_df_for_predict has the expected columns for prediction
#             # Prophet expects 'ds' and all added regressors
#             expected_predict_cols = ['ds'] + added_regressors
#             if not set(expected_predict_cols).issubset(future_df_for_predict.columns):
#                  missing_cols = set(expected_predict_cols) - set(future_df_for_predict.columns)
#                  raise ValueError(f"Future DataFrame for prediction is missing required columns: {missing_cols}")


#             # Generate forecast using the model and the future DataFrame
#             # Run the synchronous model_fit.predict() in a thread pool
#             forecast = await loop.run_in_executor(executor, lambda: model_fit.predict(future_df_for_predict))
#             # print(f"  Forecast generated for {target_commodity_name}.") # Optional print


#             print(f"  Finished fitting and forecasting for target: {target_commodity_name}.")

#             # Return the relevant forecast columns
#             return {
#                 'Commodity': target_commodity_name,
#                 'Status': 'Success',
#                 'Error Message': None,
#                 'Forecast DataFrame': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], # Include uncertainty intervals
#             }

#         except Exception as e:
#             print(f"  Forecasting failed for target {target_commodity_name}: {e}")
#             return {
#                 'Commodity': target_commodity_name,
#                 'Status': 'Failed',
#                 'Error Message': str(e),
#                 'Forecast DataFrame': pd.DataFrame() # Return empty DataFrame on error
#             }

#     # Create a list of async tasks for forecasting each target commodity
#     tasks = []
#     print("\nStarting concurrent Prophet forecasting with covariates for target commodities...")

#     # Iterate through the historical data for each target commodity
#     # Assuming historical_target_data is a dictionary where keys are commodity names and values are dataframes
#     for target_commodity, historical_df in historical_target_data.items():
#          # Ensure future covariate data is available and pass it to the async function
#          if 'future_covariate_data_available' in locals() and future_covariate_data_available and 'covariates_forecast_only_df' in locals() and not covariates_forecast_only_df.empty:
#               tasks.append(forecast_prophet_with_covariates(historical_df, target_commodity, exog_regressors, covariates_forecast_only_df, executor))
#          else:
#               print(f"  Skipping forecasting for {target_commodity}: Future covariate data not available.")
#               prophet_covariates_forecast_results.append({
#                   'Commodity': target_commodity,
#                   'Status': 'Skipped (Future Covariate Data Missing)',
#                   'Error Message': 'Future covariate data not available for forecasting.',
#                   'Forecast DataFrame': pd.DataFrame() # Return empty DataFrame
#               })


#     # Run the async tasks concurrently and collect the results
#     try:
#         all_forecasts = await asyncio.gather(*tasks)
#         prophet_covariates_forecast_results.extend(all_forecasts) # Add results from async tasks

#     except Exception as e:
#         print(f"\nAn error occurred during concurrent Prophet forecasting with covariates: {e}")
#     finally:
#         # Ensure the executor is shut down
#         executor.shutdown(wait=True)


#     print("\nProphet forecasting with covariates complete for all specified target commodities.")

#     # Convert the list of results into a pandas DataFrame
#     # This DataFrame will summarize the forecasting status for each commodity
#     prophet_covariates_forecast_summary_df = pd.DataFrame([
#         {'Commodity': res['Commodity'], 'Status': res['Status'], 'Error Message': res['Error Message']}
#         for res in prophet_covariates_forecast_results
#     ])

#     print("\nProphet with Covariates Forecast Summary DataFrame Head:")
#     print(prophet_covariates_forecast_summary_df.head())

#     # Combine the successful future forecast DataFrames
#     all_targets_combined_forecast_df = pd.DataFrame()

#     if prophet_covariates_forecast_results:
#          forecast_dfs_list = []
#          for res in prophet_covariates_forecast_results:
#               if res['Status'] == 'Success' and not res['Forecast DataFrame'].empty:
#                    forecast_df = res['Forecast DataFrame'].copy()
#                    # Rename the 'yhat' column to the commodity name
#                    forecast_df.rename(columns={'yhat': res['Commodity'],
#                                               'yhat_lower': f"{res['Commodity']}_lower",
#                                               'yhat_upper': f"{res['Commodity']}_upper"}, inplace=True)
#                    # Set 'ds' as the index for merging
#                    forecast_df.set_index('ds', inplace=True)
#                    # Append the forecast DataFrame to the list
#                    forecast_dfs_list.append(forecast_df)

#          # Concatenate all individual forecast DataFrames
#          if forecast_dfs_list:
#              all_targets_combined_forecast_df = pd.concat(forecast_dfs_list, axis=1)
#              all_targets_combined_forecast_df.index.name = 'Date' # Rename index to 'Date'

#              print("\nCombined Prophet with Covariates Forecasts (Future Dates) DataFrame Head:")
#              print(all_targets_combined_forecast_df.head())
#              print("\nCombined Prophet with Covariates Forecasts (Future Dates) DataFrame Tail:")
#              print(all_targets_combined_forecast_df.tail())
#              print("\nCombined Prophet with Covariates Forecasts (Future Dates) DataFrame Shape:", all_targets_combined_forecast_df.shape)
#          else:
#              print("\nNo successful Prophet with Covariates forecasts generated to combine.")


#     # Combine historical data with future forecasts for target commodities
#     combined_targets_historical_forecast_df = pd.DataFrame()

#     if 'prices_df' in locals() and not all_targets_combined_forecast_df.empty:
#         print("\nCombining historical target data with Prophet with Covariates forecasts...")

#         # Get historical data for target commodities that had successful forecasts
#         successful_forecast_commodities = [res['Commodity'] for res in prophet_covariates_forecast_results if res['Status'] == 'Success' and not res['Forecast DataFrame'].empty]

#         if successful_forecast_commodities:
#              # Filter prices_df to include only the successfully forecasted target commodities
#              # Ensure prices_df has 'Date' as index before filtering
#              if isinstance(prices_df.index, pd.DatetimeIndex) and prices_df.index.name == 'Date':
#                       historical_targets_df = prices_df[successful_forecast_commodities].dropna().copy()
#                       historical_targets_df.index.name = 'Date' # Ensure consistent index name

#                       # Combine historical and forecast data
#                       combined_targets_historical_forecast_df = pd.concat([historical_targets_df, all_targets_combined_forecast_df], axis=0, join='outer')

#                       # Sort by date
#                       combined_targets_historical_forecast_df.sort_index(inplace=True)


#                       print("\nCombined Targets Historical and Prophet with Covariates Forecasts DataFrame Head:")
#                       print(combined_targets_historical_forecast_df.head())
#                       print("\nCombined Targets Historical and Prophet with Covariates Forecasts DataFrame Tail:")
#                       print(combined_targets_historical_forecast_df.tail())
#                       print("\nCombined Targets Historical and Prophet with Covariates Forecasts DataFrame Shape:", combined_targets_historical_forecast_df.shape)
#              else:
#                   print("\nError: prices_df index is not a DatetimeIndex named 'Date'. Cannot combine historical data.")
#         else:
#               print("\nNo successful forecasts to combine with historical target data.")

#     else:
#         print("\nCannot combine historical target data with forecasts: Historical data ('prices_df') or combined forecasts ('all_targets_combined_forecast_df') are not available.")


# # --- End: Prophet Forecasting with Covariates Module ---

# # ==============================================================================
# # Store and Visualise Prophet output
# # ==============================================================================

# import pandas as pd
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings

# if 'combined_targets_historical_forecast_df' not in locals() or combined_targets_historical_forecast_df.empty:
#     print (f"saving {combined_targets_historical_forecast_df} to gcs")
#     gcs_prefix_fb_prophet_covariates_forecast = 'forecast_data/fb_prophet_covariates_forecast.csv'
#     save_to_gcs(
#         df=combined_targets_historical_forecast_df,
#         gcs_prefix = gcs_prefix_fb_prophet_covariates_forecast,
#         validate=False
#         )
#     print(f"saved {combined_targets_historical_forecast_df} to gcs")
# else:
#     print("combined_targets_historical_forecast_df is empty. Not saving.")

# #--- Forecasts and History saved---

# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_targets_historical_forecast_df is available from previous steps

# def plot_prophet_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and Prophet forecast with uncertainty intervals
#     for a specific commodity from the combined DataFrame.

#     Args:
#         df_combined (pd.DataFrame): DataFrame containing combined historical and
#                                     forecast data with 'Date' as index and
#                                     columns for commodity, commodity_lower, and
#                                     commodity_upper forecasts.
#         commodity_name (str): The name of the target commodity to plot.
#     """
#     if commodity_name not in df_combined.columns:
#         print(f"Error: Commodity '{commodity_name}' not found in the DataFrame.")
#         return

#     # Construct the names of the forecast columns
#     forecast_col = commodity_name
#     lower_col = f"{commodity_name}_lower"
#     upper_col = f"{commodity_name}_upper"

#     # Check if forecast columns exist
#     if lower_col not in df_combined.columns or upper_col not in df_combined.columns:
#         print(f"Warning: Uncertainty interval columns ('{lower_col}' or '{upper_col}') not found for '{commodity_name}'. Plotting forecast only.")
#         plot_uncertainty = False
#     else:
#         plot_uncertainty = True

#     plt.figure(figsize=(12, 3))

#     # Plot historical data (non-NaN values in the forecast column)
#     # We can infer historical points where the forecast column has NaN or is the same as the historical value
#     # A simpler approach for plotting is to use the original historical data if available,
#     # but since combined_targets_historical_forecast_df has both, we can plot all points and rely on NaN handling
#     historical_dates = df_combined[df_combined[forecast_col].notna()].index # Use non-NaN in forecast column as a proxy for dates to plot
#     plt.plot(historical_dates, df_combined.loc[historical_dates, forecast_col], label='Historical Data', color='blue')


#     # Plot the entire forecast line (including historical period if present)
#     plt.plot(df_combined.index, df_combined[forecast_col], label='Prophet Forecast', color='red', linestyle='--')

#     # Plot uncertainty intervals if available
#     if plot_uncertainty:
#         plt.fill_between(df_combined.index, df_combined[lower_col], df_combined[upper_col], color='red', alpha=0.2, label='Forecast Uncertainty Interval (80%)')


#     plt.title(f'Prophet Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_targets_historical_forecast_df is already loaded

# # List of target commodities (assuming this list is available)
# # If not available, you can get it from the columns of combined_targets_historical_forecast_df
# # excluding the uncertainty interval columns
# if 'target_commodities' in locals() and target_commodities:
#     commodities_to_visualize = target_commodities[:5] # Visualize the first 5 target commodities
# else:
#     # Infer potential target commodities from the combined DataFrame if target_commodities list is not defined
#     # Exclude columns ending with _lower or _upper and the index name
#     if 'combined_targets_historical_forecast_df' in locals() and not combined_targets_historical_forecast_df.empty:
#         all_cols = combined_targets_historical_forecast_df.columns.tolist()
#         commodities_to_visualize = [col for col in all_cols if not col.endswith('_lower') and not col.endswith('_upper')][:5]
#         print(f"Inferred commodities to visualize: {commodities_to_visualize}")
#     else:
#         print("Error: combined_targets_historical_forecast_df not found or is empty. Cannot visualize.")
#         commodities_to_visualize = []


# if 'combined_targets_historical_forecast_df' in locals() and not combined_targets_historical_forecast_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_prophet_forecast(combined_targets_historical_forecast_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_targets_historical_forecast_df is not available or empty, or no commodities to visualize.")

# """## GOOGLE TIMESFM

# > Ensure Python 3.11 (use runtime change in colab)


# """

# !pip install --upgrade --quiet timesfm
# !pip install --upgrade --quiet openpyxl
# #!pip install --upgrade --quiet dask[dataframe] -U -q --user
# !pip install dask[dataframe]==2024.12.1

# # tuples of (import name, install name, min_version)
# packages = [('timesfm', 'timesfm'),]

# import importlib
# install = False
# for package in packages:
#     if not importlib.util.find_spec(package[0]):
#         print(f'installing package {package[1]}')
#         install = True
#         !pip install {package[1]} -U -q --user
#     elif len(package) == 3:
#         if importlib.metadata.version(package[0]) < package[2]:
#             print(f'updating package {package[1]}')
#             install = True
#             !pip install {package[1]} -U -q --user

# import os
# os.environ['JAX_PLATFORMS'] = 'cpu'

# import timesfm
# import numpy as np
# from google.cloud import bigquery
# import pandas as pd
# from matplotlib import pyplot as plt

# from google.colab import userdata
# from google.colab import drive
# drive.mount("/content/drive", force_remount=True)

# import timesfm

# # For PyTorch
# tfm = timesfm.TimesFm(
#       hparams=timesfm.TimesFmHparams(
#           backend="cpu",
#           per_core_batch_size=32,
#           horizon_len=24, # Increased forecast horizon
#           input_patch_len=32,
#           output_patch_len=128,
#           num_layers=50,
#           model_dims=1280,
#           use_positional_embedding=False,
#       ),
#       checkpoint=timesfm.TimesFmCheckpoint(
#           huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
#   )

# type(tfm._model)




# ## Google Colab Set Up
# import gspread
# from google.colab import auth, drive
# from google.auth import default
# import asyncio, aiohttp

# # # Authenticate and create the gspread client
# auth.authenticate_user()
# creds, _ = default()

# # INPUT
# df = prices_df.copy() # Work on a copy to avoid modifying prices_df for other parts of the notebook
# print(df.shape)
# print(df.head(2))

# # Ensure the 'date' column (string format) is handled if it still exists and is not the index.
# # The prices_df now has 'Date' as index (datetime) and 'date' as a string column.
# # For melting, we want to use the datetime index.
# # Let's create a 'ds' column from the index and then melt.

# melted_df = df.reset_index().rename(columns={'Date': 'ds'}) # Make 'Date' index a column named 'ds'
# # Now drop the redundant 'date' string column if it exists, before melting.
# if 'date' in melted_df.columns:
#     melted_df = melted_df.drop(columns=['date'])

# # Melt the DataFrame using 'ds' as the identifier
# melted_df = pd.melt(melted_df, id_vars=['ds'], var_name='unique_id', value_name='y')

# # Ensure 'ds' is timezone-naive as TimesFM expects this sometimes
# melted_df['ds'] = pd.to_datetime(melted_df['ds']) # Ensure it's datetime
# melted_df['ds'] = melted_df['ds'].dt.tz_localize(None) # Convert to timezone-naive if it somehow became localized

# # Clean and convert 'y' column to numeric
# melted_df['y'] = melted_df['y'].astype(str).str.replace(r'[^0-9.-]', '', regex=True) # Retained only digits, period, and hyphen
# melted_df['y'] = pd.to_numeric(melted_df['y'], errors='coerce')  # 'coerce' handles errors by setting invalid values to NaN

# # Drop rows where 'y' is NaN after conversion
# melted_df = melted_df.dropna(subset=['y'])

# print(melted_df.head())

# forecast_df = tfm.forecast_on_df(
#         inputs=melted_df,
#         freq="w",  # Adjust frequency if your data isn't daily
#         value_name='y',
#         num_jobs=8  # Adjust for parallel processing if desired
#     )

# print (forecast_df.shape)
# print (forecast_df.head())

# import matplotlib.pyplot as plt

# combined_df_list = []
# # Iterate through the first 5 unique commodities
# for commodity in melted_df['unique_id'].unique()[:5]:
#     history = melted_df[melted_df['unique_id'] == commodity].set_index('ds')
#     horizon = forecast_df[forecast_df['unique_id'] == commodity].set_index('ds')[:12] # Limit to the first 12 time periods

#     plt.figure(figsize = (12,3))
#     plt.plot(history['y'], linestyle = '-', color = 'blue')
#     plt.plot(horizon['timesfm'], linestyle = '--', color = 'red')
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.4'], horizon['timesfm-q-0.6'], color = 'green', alpha = 1)
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.3'], horizon['timesfm-q-0.7'], color = 'green', alpha = 0.75)
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.2'], horizon['timesfm-q-0.8'], color = 'green', alpha = 0.5)
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.1'], horizon['timesfm-q-0.9'], color = 'green', alpha = 0.25)
#     plt.title(commodity)
#     plt.show()

# forecast_wide_df = forecast_df.copy()
# forecast_wide_df = forecast_wide_df.pivot(index='ds', columns='unique_id', values='timesfm')
# forecast_wide_df = forecast_wide_df.rename_axis('AssessDate').reset_index()
# forecast_wide_df = forecast_wide_df.rename(columns={'AssessDate': 'date'})

# forecast_wide_df.head()
# forecast_wide_df.to_csv('./forecasts.csv', index=False)

# history_wide_df = melted_df.copy()

# # Drop duplicates before pivoting
# history_wide_df = history_wide_df.drop_duplicates(subset=['ds', 'unique_id'])
# history_wide_df = history_wide_df.pivot(index='ds', columns='unique_id', values='y')
# history_wide_df = history_wide_df.rename_axis('date').reset_index()

# history_wide_df.head()
# history_wide_df.to_csv('./history.csv', index=False)

# timesfm_combined_df= pd.concat([history_wide_df, forecast_wide_df], axis=0)
# print (timesfm_combined_df.shape)

# if not timesfm_combined_df.empty:
#     print(f"saving {timesfm_combined_df} to gcs")
#     gcs_prefix_timesfm_combined = 'forecast_data/timesfm_combined.csv'
#     save_to_gcs(
#         df=timesfm_combined_df,
#         gcs_prefix = gcs_prefix_timesfm_combined,
#         validate=False
#         )
#     print(f"saved {timesfm_combined_df} to gcs")
# else:
#     print("timesfm_combined_df is empty. Not saving.")

# """## KAN FORECASTER
# Forecast prices using KAN with "prices_df" as historical data.
# """

# # Commented out IPython magic to ensure Python compatibility.
# # %%capture
# # !pip install pykan

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# from kan import KAN
# import torch

# # 1. Identify the numerical commodity columns
# all_columns = prices_df.columns.tolist()
# commodity_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(prices_df[col])]

# print("Identified commodity columns for KAN data preparation:")
# print(commodity_columns)

# # Define sequence length (number of past time steps to use for prediction)
# sequence_length = 20 # Example: use the past 30 days to predict the next day

# # Initialize lists to store sequences and targets for all commodities
# all_sequences = []
# all_targets = []

# # Initialize a dictionary to store scalers for each commodity
# scalers = {}

# # 2. Iterate through each identified commodity column
# print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
# for commodity in commodity_columns:
#     print(f"  Processing commodity: {commodity}")

#     # 3. For each commodity, extract the time series data
#     series = prices_df[commodity].dropna()

#     if len(series) < sequence_length + 1:
#         print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#         continue # Skip to the next commodity

#     # 4. Scale the time series data
#     scaler = RobustScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#     scalers[commodity] = scaler # Store the scaler for this commodity

#     # 5. Create input sequences (X) and corresponding target values (y)
#     sequences = []
#     targets = []
#     for i in range(len(scaled_series) - sequence_length):
#         seq = scaled_series[i:i + sequence_length]
#         target = scaled_series[i + sequence_length]
#         sequences.append(seq)
#         targets.append(target)

#     # 6. Store the prepared input sequences and target values for the current commodity
#     all_sequences.extend(sequences)
#     all_targets.extend(targets)
#     print(f"    Generated {len(sequences)} sequences for {commodity}.")


# # 7. Combine the input sequences and target values into PyTorch tensors
# if not all_sequences:
#     print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#     # Initialize tensors as empty if no data is processed
#     X_tensor = torch.empty(0, sequence_length, dtype=torch.float32)
#     y_tensor = torch.empty(0, dtype=torch.float32)

# else:
#     X_tensor = torch.tensor(all_sequences, dtype=torch.float32)
#     y_tensor = torch.tensor(all_targets, dtype=torch.float32)
#     print("\nCombined sequences and targets into PyTorch tensors.")
#     print("Shape of X_tensor:", X_tensor.shape)
#     print("Shape of y_tensor:", y_tensor.shape)


# # Store the scalers dictionary as it will be needed for inverse transformation of forecasts
# # Assuming 'scalers' is already defined and populated
# if 'scalers' in locals():
#     print("\nScalers for each commodity stored in 'scalers' dictionary.")

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from kan import KAN # Assuming kan library is installed and imported in a previous cell

# # Ensure X_tensor and y_tensor are available from the previous data preparation step
# if 'X_tensor' not in locals() or 'y_tensor' not in locals() or X_tensor.shape[0] == 0:
#     print("Error: X_tensor or y_tensor is not available or is empty. Cannot create Dataset and DataLoader.")
#     # Initialize necessary variables to prevent errors in later steps
#     train_loader = None
#     model = None

# else:
#     print("X_tensor and y_tensor are available. Proceeding with Dataset, DataLoader, and Model definition.")

#     # 1. Define a PyTorch Dataset class
#     class TimeSeriesDataset(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             return self.X[idx], self.y[idx]

#     # Instantiate the Dataset
#     # Assuming X_tensor and y_tensor are already created from the data preparation step
#     dataset = TimeSeriesDataset(X_tensor, y_tensor)
#     print(f"\nDataset created with {len(dataset)} samples.")

#     # 2. Create a PyTorch DataLoader
#     batch_size = 32 # Define your desired batch size
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     print(f"DataLoader created with batch size {batch_size}.")
#     print(f"Number of batches in DataLoader: {len(train_loader)}")


#     # 3. Define the KAN model architecture
#     # Assuming sequence_length is available from the data preparation step
#     if 'sequence_length' not in locals():
#         print("Error: 'sequence_length' is not defined. Cannot define KAN model architecture.")
#         model = None # Ensure model is None if sequence_length is missing
#     else:
#         print(f"\nDefining KAN model architecture with input dimension {sequence_length}...")
#         # The KAN model takes a flattened input, so the input dimension is sequence_length
#         # The output dimension is 1 for predicting a single next value
#         # Added an extra layer with 8 nodes
#         model = KAN(width=[sequence_length, 16, 8, 1]) # Example architecture: input -> 16 KAN nodes -> 8 KAN nodes -> 1 output

#         print("KAN model architecture defined.")

#         # 4. Instantiate the defined KAN model
#         # model is already instantiated by the kan.KAN() call above
#         print("\nKAN model instantiated.")
#         print("Model structure:")
#         print(model)

# import torch.optim as optim
# import torch.nn as nn

# # Ensure model and train_loader are available from the previous step
# if 'model' not in locals() or model is None:
#     print("Error: KAN model is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the model is not defined
#     # No need to initialize other variables as training won't start

# elif 'train_loader' not in locals() or train_loader is None:
#     print("Error: train_loader is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the train_loader is not defined
#     # No need to initialize other variables as training won't start

# else:
#     print("\nProceeding with KAN model training.")

#     # 1. Define the loss function (e.g., Mean Squared Error)
#     criterion = nn.MSELoss()
#     print("Loss function defined (MSELoss).")

#     # 2. Define the optimizer (e.g., Adam)
#     # Updated learning rate as requested by the user
#     optimizer = optim.Adam(model.parameters(), lr=0.00002) # Define your desired learning rate
#     print("Optimizer defined (Adam with learning rate 0.00002).")

#     # 3. Set the number of training epochs
#     num_epochs = 10 # Define your desired number of epochs
#     print(f"Number of training epochs set to: {num_epochs}")

#     # 4. Implement the training loop
#     print("\nStarting KAN model training...")

#     # Move model to the appropriate device if available (CPU or GPU)
#     # Check if CUDA is available and use GPU if it is, otherwise use CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     print(f"Using device: {device}")


#     for epoch in range(num_epochs):
#         model.train() # Set the model to training mode
#         running_loss = 0.0

#         for i, data in enumerate(train_loader):
#             inputs, targets = data

#             # Move data to the device
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(inputs)

#             # Calculate the loss
#             # Ensure targets have the same shape as outputs (usually [batch_size, 1])
#             # If targets is [batch_size], unsqueeze it to [batch_size, 1]
#             if targets.ndim == 1:
#                  targets = targets.unsqueeze(1)

#             loss = criterion(outputs, targets)

#             # Backward pass
#             loss.backward()

#             # Update the weights
#             optimizer.step()

#             # Accumulate the loss
#             running_loss += loss.item()

#         # Print the loss periodically
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

#     print("\nKAN model training complete.")

# import torch
# import os

# # Define the path to save the model on Google Drive
# # Ensure you have mounted Google Drive in a previous cell (e.g., cell Nv0Vvz_e5Ma2)
# model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_kan_model.pth' # You can change the filename

# # Ensure the directory exists
# os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model' not in locals() or model is None:
#     print("Error: KAN model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model.state_dict(), model_save_path)
#         print(f"KAN model state dictionary saved successfully to: {model_save_path}")
#     except Exception as e:
#         print(f"Error saving KAN model to Google Drive: {e}")

# import torch
# import pandas as pd
# import numpy as np

# # Ensure prices_df, commodity_columns, sequence_length, and scalers are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#     print("Error: 'prices_df' index is not a DatetimeIndex named 'Date'. Cannot proceed.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot prepare forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot prepare forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif 'scalers' not in locals() or not scalers:
#     print("Error: 'scalers' dictionary not found or is empty. Cannot scale forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# else:
#     print("\nPreparing forecast input sequences for KAN model...")

#     # Initialize an empty dictionary to store the last sequence for each commodity
#     forecast_input_sequences = {}

#     # Iterate through each identified commodity column
#     print(f"Extracting and scaling the last {sequence_length} data points for each commodity...")
#     for commodity in commodity_columns:
#         # Extract the time series data for the current commodity and drop missing values
#         series = prices_df[commodity].dropna()

#         # Check if the series has enough data points
#         if len(series) < sequence_length:
#             print(f"  Warning: Skipping {commodity}. Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#             continue # Skip to the next commodity

#         # Get the last `sequence_length` data points
#         last_sequence = series[-sequence_length:]

#         # Scale the last sequence using the corresponding scaler
#         if commodity in scalers:
#             scaler = scalers[commodity]
#             # Reshape the sequence to be a 2D array for scaling (MinMaxScaler expects 2D input)
#             scaled_sequence = scaler.transform(last_sequence.values.reshape(-1, 1)).flatten()

#             # Convert the scaled sequence to a PyTorch tensor
#             # Reshape to [1, sequence_length] as the model expects a batch of sequences
#             sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)

#             # Store the tensor in the dictionary
#             forecast_input_sequences[commodity] = sequence_tensor
#             print(f"  Prepared input sequence for {commodity}. Shape: {sequence_tensor.shape}")

#         else:
#              print(f"  Warning: Scaler not found for {commodity}. Cannot prepare forecast input sequence.")


#     # Print the number of commodities for which input sequences were prepared
#     print(f"\nPrepared input sequences for {len(forecast_input_sequences)} commodities.")

# import torch
# import pandas as pd
# import numpy as np
# from kan import KAN # Assuming kan library is installed and imported

# # Ensure model, forecast_input_sequences, sequence_length, scalers, and commodity_columns are available
# if 'model' not in locals() or model is None:
#     print("Error: KAN model is not defined. Cannot generate forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'forecast_input_sequences' not in locals() or not forecast_input_sequences:
#     print("Error: 'forecast_input_sequences' is not available or is empty. Cannot generate forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot generate forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'scalers' not in locals() or not scalers:
#     print("Error: 'scalers' dictionary not found or is empty. Cannot inverse scale forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot iterate through commodities for forecasting.")
#     kan_forecasts = {} # Initialize as empty
# else:
#     print("\nGenerating KAN future forecasts for each commodity...")

#     # Define the number of steps to forecast
#     forecast_steps = 24 # This should align with other forecasters if possible
#     print(f"Forecasting {forecast_steps} steps into the future.")

#     # Move the model to the appropriate device (CPU or GPU) if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval() # Set the model to evaluation mode

#     # Initialize a dictionary to store the generated forecasts for each commodity
#     kan_forecasts = {}

#     # Iterate through each commodity for which an input sequence was prepared
#     print(f"\nGenerating forecasts for {len(forecast_input_sequences)} commodities...")
#     with torch.no_grad(): # Disable gradient calculation during inference
#         for commodity in forecast_input_sequences.keys():
#             print(f"  Forecasting for: {commodity}")
#             # Get the initial input sequence tensor for the current commodity
#             current_sequence = forecast_input_sequences[commodity].to(device) # Move sequence to device

#             # Get the corresponding scaler for inverse scaling
#             if commodity not in scalers:
#                  print(f"    Warning: Scaler not found for {commodity}. Skipping forecast.")
#                  continue # Skip to the next commodity
#             scaler = scalers[commodity]

#             # Initialize a list to store the unscaled forecast values for this commodity
#             commodity_forecast_values = []

#             # Generate the forecast step by step
#             for _ in range(forecast_steps):
#                 # Get the KAN model's prediction for the next step (scaled value)
#                 predicted_scaled_value = model(current_sequence).item() # Get the scalar value

#                 # Inverse scale the predicted value to the original price scale
#                 # The scaler expects a 2D array, so reshape the scalar
#                 predicted_original_value = scaler.inverse_transform([[predicted_scaled_value]])[0][0]

#                 # Store the inverse-scaled forecast value
#                 commodity_forecast_values.append(predicted_original_value)

#                 # Update the input sequence for the next prediction
#                 # Remove the oldest value and append the new scaled prediction
#                 new_sequence = torch.cat((current_sequence[:, 1:], torch.tensor([[predicted_scaled_value]], dtype=torch.float32).to(device)), dim=1)
#                 current_sequence = new_sequence # Use the new sequence for the next step

#             # Store the generated forecast values for the current commodity
#             kan_forecasts[commodity] = commodity_forecast_values
#             print(f"    Generated {len(commodity_forecast_values)} forecast values for {commodity}.")

#     print("\nKAN forecasting complete.")

# # You now have the generated forecasts in the 'kan_forecasts' dictionary.

# """**Reasoning**:
# Combine the generated KAN forecasts with the historical data and save the combined DataFrame to a Google Sheet.


# """

# import pandas as pd
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings

# # Ensure prices_df (original data) and kan_forecasts (forecast dictionary) are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot combine data.")
#     # Finish the task with failure
# elif 'kan_forecasts' not in locals() or not kan_forecasts:
#     print("Error: 'kan_forecasts' dictionary not found or is empty. Cannot combine data.")
#     # Finish the task with failure
# else:
#     print("\nCombining original data and KAN forecasts...")

#     # Prepare the original data: select commodity columns and ensure DatetimeIndex
#     # Assuming the index of prices_df is already the correct DatetimeIndex ('Date')
#     # and numeric columns are identified in commodity_columns list
#     historical_df = prices_df[commodity_columns].copy()

#     # Prepare the forecast data:
#     # Create a DataFrame from the kan_forecasts dictionary
#     if kan_forecasts:
#         # Get the last date from the historical data to start the forecast index from the next day
#         last_historical_date = prices_df.index.max()

#         # Determine the frequency of the historical data
#         freq = pd.infer_freq(prices_df.index)
#         if freq is None:
#              # Fallback to 'D' if frequency cannot be inferred (assuming daily data)
#              freq = 'D'
#              print(f"Warning: Could not infer frequency from historical data. Assuming '{freq}'.")
#         else:
#             print(f"Inferred historical data frequency: {freq}")


#         # Generate future dates starting from the day after the last historical date
#         forecast_dates = pd.date_range(start=last_historical_date, periods=forecast_steps + 1, freq=freq)[1:]

#         # Create a DataFrame from the forecast dictionary, using the generated future dates as the index
#         forecast_df = pd.DataFrame(kan_forecasts, index=forecast_dates)

#         # Rename the index to 'Date' for consistency
#         forecast_df.index.name = 'Date'

#     else:
#         print("No KAN forecasts were generated to combine.")
#         forecast_df = pd.DataFrame() # Create an empty DataFrame if no forecasts

#     # Combine the historical data and the forecast data
#     # Use outer join to include all dates from both historical and forecast periods
#     # The index from both DataFrames (DatetimeIndex) will be used for alignment
#     combined_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')

#     # Sort by date to ensure chronological order
#     combined_df.sort_index(inplace=True)

#     print("\nCombined KAN DataFrame Head:")
#     print(combined_df.head(2))
#     print("\nCombined KAN DataFrame Tail:")
#     print(combined_df.tail(2))
#     print("\nCombined KAN DataFrame Shape:", combined_df.shape)

#     if not combined_df.empty:
#         print(f"saving {combined_df} to gcs")
#         gcs_prefix_kan_forecast = 'forecast_data/kan_forecasts.csv'
#         save_to_gcs(
#             df=combined_df,
#             gcs_prefix=gcs_prefix_kan_forecast,
#             validate=False)
#         print(f"saved {combined_df} to gcs")
#     else:
#         print("combined_df is empty. No data to save.")

# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_df is available from the previous step

# def plot_kan_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and KAN forecast for a specific commodity
#     from the combined DataFrame.

#     Args:
#         df_combined (pd.DataFrame): DataFrame containing combined historical and
#                                     forecast data with 'Date' as index.
#         commodity_name (str): The name of the commodity to plot.
#     """
#     if commodity_name not in df_combined.columns:
#         print(f"Error: Commodity '{commodity_name}' not found in the DataFrame.")
#         return

#     plt.figure(figsize=(12, 3))

#     # Plot historical data (non-NaN values in the historical period)
#     # Assuming historical data ends before the forecast period starts
#     # Find the index where the forecast starts (first non-NaN value in the forecast period)
#     # Or simply plot all available data in the combined df
#     plt.plot(df_combined.index, df_combined[commodity_name], label=f'{commodity_name} Data & KAN Forecast')


#     # Add a vertical line at the end of the historical data for visual separation
#     # Assuming the end of historical data is the last date before the forecast starts
#     # We can infer this from the first non-NaN value in the forecast period in the combined df
#     first_forecast_date = df_combined[df_combined[commodity_name].index > prices_df.index.max()].first_valid_index()
#     if first_forecast_date:
#         plt.axvline(prices_df.index.max(), color='red', linestyle='--', label='Forecast Start')


#     plt.title(f'KAN Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_df is already loaded

# # Get a list of commodities that were forecasted
# # We can get this from the keys of the kan_forecasts dictionary
# if 'kan_forecasts' in locals() and kan_forecasts:
#     commodities_to_visualize = list(kan_forecasts.keys())[:10] # Visualize the first 5 forecasted commodities
#     print(f"\nVisualizing KAN forecasts for the following commodities: {commodities_to_visualize}")
# else:
#     print("Error: 'kan_forecasts' dictionary not found or is empty. Cannot visualize forecasts.")
#     commodities_to_visualize = []


# if 'combined_df' in locals() and not combined_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_kan_forecast(combined_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_df is not available or empty, or no commodities to visualize.")

# """## GRU FORECASTER
# Build a GRU-based time series forecaster using the data in the `prices_df` DataFrame, generate forecasts, and save the results to a Google Sheet.
# """

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # Ensure prices_df is available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare GRU data.")
#     # Initialize tensors and scalers as empty if data is not available
#     X_tensor_gru = torch.empty(0, dtype=torch.float32)
#     y_tensor_gru = torch.empty(0, dtype=torch.float32)
#     scalers_gru = {}

# else:
#     # 1. Identify the numerical commodity columns
#     all_columns = prices_df.columns.tolist()
#     # Exclude non-numeric and potentially non-relevant columns like 'date' if it wasn't dropped
#     cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
#                        'day_of_week_sin', 'day_of_week_cos',
#                        'month_sin', 'month_cos',
#                        'year_sin', 'year_cos']
#     commodity_columns = [col for col in all_columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]

#     print("Identified commodity columns for GRU data preparation:")
#     print(commodity_columns)

#     # 2. Define sequence length (number of past time steps to use for prediction)
#     sequence_length = 30 # Example: use the past 30 days to predict the next day

#     # 3. Initialize lists to store sequences and targets for all commodities
#     all_sequences = []
#     all_targets = []

#     # 4. Initialize a dictionary to store scalers for each commodity
#     scalers_gru = {}

#     # 5. Iterate through each identified commodity column
#     print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
#     for commodity in commodity_columns:
#         print(f"  Processing commodity: {commodity}")

#         # Extract the time series data for the current commodity and drop missing values
#         series = prices_df[commodity].dropna()

#         if len(series) < sequence_length + 1:
#             print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#             continue # Skip to the next commodity

#         # Scale the time series data
#         # Using MinMaxScaler as an example, can be changed to StandardScaler or RobustScaler
#         scaler = MinMaxScaler()
#         scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#         scalers_gru[commodity] = scaler # Store the scaler for this commodity

#         # Create input sequences (X) and corresponding target values (y)
#         sequences = []
#         targets = []
#         for i in range(len(scaled_series) - sequence_length):
#             seq = scaled_series[i:i + sequence_length]
#             target = scaled_series[i + sequence_length]
#             sequences.append(seq)
#             targets.append(target)

#         # Store the prepared input sequences and target values for the current commodity
#         all_sequences.extend(sequences)
#         all_targets.extend(targets)
#         print(f"    Generated {len(sequences)} sequences for {commodity}.")


#     # 6. Combine the input sequences and target values into PyTorch tensors
#     if not all_sequences:
#         print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#         X_tensor_gru = torch.empty(0, dtype=torch.float32) # Initialize as empty tensor
#         y_tensor_gru = torch.empty(0, dtype=torch.float32) # Initialize as empty tensor
#         # scalers_gru is already initialized as an empty dictionary

#     else:
#         X_tensor_gru = torch.tensor(all_sequences, dtype=torch.float32)
#         y_tensor_gru = torch.tensor(all_targets, dtype=torch.float32)
#         print("\nCombined sequences and targets into PyTorch tensors.")
#         print("Shape of X_tensor_gru:", X_tensor_gru.shape)
#         print("Shape of y_tensor_gru:", y_tensor_gru.shape)

#     # 7. Confirm that the scalers dictionary is stored
#     if 'scalers_gru' in locals():
#         print("\nScalers for each commodity stored in 'scalers_gru' dictionary.")

# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# # Ensure X_tensor_gru and y_tensor_gru are available from the previous data preparation step
# if 'X_tensor_gru' not in locals() or 'y_tensor_gru' not in locals() or X_tensor_gru.shape[0] == 0:
#     print("Error: X_tensor_gru or y_tensor_gru is not available or is empty. Cannot create Dataset and DataLoader.")
#     # Initialize necessary variables to prevent errors in later steps
#     train_loader_gru = None
#     model_gru = None

# else:
#     print("X_tensor_gru and y_tensor_gru are available. Proceeding with Dataset, DataLoader, and Model definition.")

#     # 1. Define a PyTorch Dataset class for GRU
#     class TimeSeriesDatasetGRU(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             # GRU typically expects input shape (batch_size, sequence_length, input_dim)
#             # Since we are forecasting a single time series value at a time, input_dim is 1
#             # We need to unsqueeze the sequence tensor to add the input_dim dimension
#             return self.X[idx].unsqueeze(-1), self.y[idx] # Add last dimension for input_dim

#     # Instantiate the Dataset
#     dataset_gru = TimeSeriesDatasetGRU(X_tensor_gru, y_tensor_gru)
#     print(f"\nDataset for GRU created with {len(dataset_gru)} samples.")

#     # 2. Create a PyTorch DataLoader
#     batch_size = 64 # Define your desired batch size
#     train_loader_gru = DataLoader(dataset_gru, batch_size=batch_size, shuffle=True)
#     print(f"DataLoader for GRU created with batch size {batch_size}.")
#     print(f"Number of batches in DataLoader: {len(train_loader_gru)}")


#     # 3. Define the GRU model architecture
#     # Assuming sequence_length is available from the data preparation step
#     if 'sequence_length' not in locals():
#         print("Error: 'sequence_length' is not defined. Cannot define GRU model architecture.")
#         model_gru = None # Ensure model_gru is None if sequence_length is missing
#     else:
#         print(f"\nDefining GRU model architecture with sequence length {sequence_length}...")
#         # GRU model parameters - these are example values and may need tuning
#         input_dim = 1       # We are forecasting a single time series value
#         hidden_dim = 128    # Number of features in the hidden state
#         layer_dim = 3       # Number of recurrent layers
#         output_dim = 1      # We are predicting a single next value

#         class GRUModel(nn.Module):
#             def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#                 super().__init__()
#                 self.hidden_dim = hidden_dim
#                 self.layer_dim = layer_dim

#                 # GRU layers
#                 self.gru = nn.GRU(
#                     input_dim, hidden_dim, layer_dim, batch_first=True # batch_first=True means input is (batch_size, sequence_length, input_dim)
#                 )

#                 # Fully connected layer for output
#                 self.fc = nn.Linear(hidden_dim, output_dim)

#             def forward(self, x):
#                 # x shape: (batch_size, sequence_length, input_dim)

#                 # Initialize hidden state with zeros
#                 h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

#                 # We don't need to return the hidden state for this forecasting task, only the output
#                 # out shape: (batch_size, sequence_length, hidden_dim)
#                 # hn shape: (layer_dim, batch_size, hidden_dim)
#                 out, hn = self.gru(x, h0.detach())

#                 # We only need the output from the last time step to predict the next value
#                 last_step_out = out[:, -1, :] # Shape: (batch_size, hidden_dim)

#                 # Pass the output of the last time step through the fully connected layer
#                 prediction = self.fc(last_step_out) # Shape: (batch_size, output_dim)

#                 return prediction

#         model_gru = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

#         print("GRU model architecture defined.")

#         # 4. Instantiate the defined GRU model
#         # model_gru is already instantiated by the GRUModel() call above
#         print("\nGRU model instantiated.")
#         print("Model structure:")
#         print(model_gru)

# import torch.optim as optim
# import torch.nn as nn

# # Ensure model_gru and train_loader_gru are available from the previous step
# if 'model_gru' not in locals() or model_gru is None:
#     print("Error: GRU model is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the model is not defined
#     # No need to initialize other variables as training won't start

# elif 'train_loader_gru' not in locals() or train_loader_gru is None:
#     print("Error: train_loader_gru is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the train_loader_gru is not defined
#     # No need to initialize other variables as training won't start

# else:
#     print("\nProceeding with GRU model training.")

#     # 1. Define the loss function (e.g., Mean Squared Error)
#     criterion = nn.MSELoss()
#     print("Loss function defined (MSELoss).")

#     # 2. Define the optimizer (e.g., Adam)
#     # Define your desired learning rate - MAMBA might benefit from different rates
#     learning_rate = 0.00001
#     optimizer = optim.Adam(model_gru.parameters(), lr=learning_rate)
#     print(f"Optimizer defined (Adam with learning rate {learning_rate}).")

#     # 3. Set the number of training epochs
#     num_epochs = 10 # Define your desired number of epochs - might need more or fewer
#     print(f"Number of training epochs set to: {num_epochs}")

#     # 4. Implement the training loop
#     print("\nStarting GRU model training...")

#     # Move model to the appropriate device if available (CPU or GPU)
#     # Check if CUDA is available and use GPU if it is, otherwise use CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_gru.to(device)
#     print(f"Using device: {device}")


#     for epoch in range(num_epochs):
#         model_gru.train() # Set the model to training mode
#         running_loss = 0.0

#         for i, data in enumerate(train_loader_gru):
#             inputs, targets = data

#             # Move data to the device
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model_gru(inputs)

#             # Calculate the loss
#             # Ensure targets have the same shape as outputs (usually [batch_size, 1])
#             # If targets is [batch_size], unsqueeze it to [batch_size, 1]
#             if targets.ndim == 1:
#                  targets = targets.unsqueeze(1)

#             loss = criterion(outputs, targets)

#             # Backward pass
#             loss.backward()

#             # Update the weights
#             optimizer.step()

#             # Accumulate the loss
#             running_loss += loss.item()

#         # Print the loss periodically
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader_gru):.4f}")

#     print("\nGRU model training complete.")

# import torch
# import os

# # Define the path to save the model (e.g., in Google Drive)
# # Ensure you have mounted Google Drive in a previous cell if saving there
# gru_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_gru_model.pth' # You can change the filename and path

# # Ensure the directory exists
# os.makedirs(os.path.dirname(gru_model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model_gru' not in locals() or model_gru is None:
#     print("Error: GRU model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model_gru.state_dict(), gru_model_save_path)
#         print(f"GRU model state dictionary saved successfully to: {gru_model_save_path}")
#     except Exception as e:
#         print(f"Error saving GRU model: {e}")

# import torch
# import pandas as pd
# import numpy as np
# import os

# # Ensure GRU model architecture is defined (assuming it was defined in a previous cell like kacL81qmSp_t)
# # Ensure prices_df, commodity_columns, sequence_length, and scalers_gru are available

# # Define the path to the saved model
# gru_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_gru_model.pth' # This should match the save path

# # Check if the saved model file exists
# if not os.path.exists(gru_model_save_path):
#     print(f"Error: Saved GRU model not found at {gru_model_save_path}. Cannot generate forecasts using the saved model.")
#     gru_forecasts = {} # Initialize as empty
# elif 'GRUModel' not in locals():
#     print("Error: GRUModel architecture is not defined. Cannot load the saved model.")
#     gru_forecasts = {} # Initialize as empty
# elif 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare forecast input sequences.")
#     gru_forecasts = {} # Initialize as empty
# elif not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#     print("Error: 'prices_df' index is not a DatetimeIndex named 'Date'. Cannot proceed.")
#     gru_forecasts = {} # Initialize as empty
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot prepare forecast input sequences.")
#     gru_forecasts = {} # Initialize as empty
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot generate forecasts.")
#     gru_forecasts = {} # Initialize as empty
# elif 'scalers_gru' not in locals() or not scalers_gru:
#     print("Error: 'scalers_gru' dictionary not found or is empty. Cannot inverse scale forecasts.")
#     gru_forecasts = {} # Initialize as empty

# else:
#     print("\nLoading the saved GRU model and generating future forecasts...")

#     # Instantiate the GRU model architecture
#     # Ensure input_dim, hidden_dim, layer_dim, output_dim are defined as in the model definition cell (kacL81qmSp_t)
#     # Assuming these variables are available in the environment from previous cell executions
#     try:
#         # Re-instantiate the model with the same architecture parameters used for training
#         # These parameters should be available from the model definition cell (kacL81qmSp_t)
#         # If not explicitly defined globally, you might need to hardcode them or ensure that cell is run first.
#         # Assuming input_dim=1, hidden_dim=128, layer_dim=3, output_dim=1 based on previous successful run
#         input_dim = 1
#         hidden_dim = 128
#         layer_dim = 3
#         output_dim = 1

#         model_gru = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
#         print("GRU model architecture instantiated for loading.")

#         # Load the saved state dictionary
#         model_gru.load_state_dict(torch.load(gru_model_save_path))
#         print(f"GRU model state dictionary loaded from {gru_model_save_path}.")

#         # Move the model to the appropriate device (CPU or GPU) if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_gru.to(device)
#         model_gru.eval() # Set the model to evaluation mode
#         print(f"Using device for forecasting: {device}")


#     except Exception as e:
#         print(f"Error loading GRU model or setting up for forecasting: {e}")
#         gru_forecasts = {} # Initialize as empty
#         model_gru = None # Ensure model_gru is None if loading fails


#     # Initialize a dictionary to store the generated forecasts for each commodity
#     gru_forecasts = {}

#     if model_gru is not None:
#          # Define the number of steps to forecast
#          forecast_steps = 60 # This should align with other forecasters if possible
#          print(f"\nGenerating {forecast_steps} steps into the future for each commodity...")

#          # Iterate through each commodity for which an input sequence can be prepared
#          print(f"Generating forecasts for {len(commodity_columns)} commodities...")
#          with torch.no_grad(): # Disable gradient calculation during inference
#              for commodity in commodity_columns:
#                  print(f"  Forecasting for: {commodity}")

#                  # Get the time series for the current commodity and drop missing values
#                  series = prices_df[commodity].dropna()

#                  # Check if the series has enough data points for the initial sequence
#                  if len(series) < sequence_length:
#                      print(f"    Warning: Skipping {commodity}. Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#                      continue # Skip to the next commodity

#                  # Get the last `sequence_length` data points as the initial input sequence
#                  last_sequence = series[-sequence_length:]

#                  # Get the corresponding scaler for this commodity
#                  if commodity not in scalers_gru:
#                       print(f"    Warning: Scaler not found for {commodity}. Skipping forecast.")
#                       continue # Skip to the next commodity
#                  scaler = scalers_gru[commodity]

#                  # Scale the last sequence using the corresponding scaler
#                  # Reshape the sequence to be a 2D array for scaling
#                  scaled_sequence = scaler.transform(last_sequence.values.reshape(-1, 1)).flatten()

#                  # Convert the scaled sequence to a PyTorch tensor
#                  # Reshape to [1, sequence_length, input_dim] as the model expects
#                  sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # Add batch and input_dim dimensions and move to device

#                  # Initialize a list to store the unscaled forecast values for this commodity
#                  commodity_forecast_values = []

#                  # Generate the forecast step by step
#                  for _ in range(forecast_steps):
#                      # Get the GRU model's prediction for the next step (scaled value)
#                      predicted_scaled_value = model_gru(sequence_tensor).item() # Get the scalar value

#                      # Inverse scale the predicted value to the original price scale
#                      # The scaler expects a 2D array, so reshape the scalar
#                      predicted_original_value = scaler.inverse_transform([[predicted_scaled_value]])[0][0]

#                      # Store the inverse-scaled forecast value
#                      commodity_forecast_values.append(predicted_original_value)

#                      # Update the input sequence for the next prediction
#                      # Remove the oldest value and append the new scaled prediction
#                      # The new prediction needs to be reshaped to match the input tensor shape
#                      new_sequence_tensor = torch.cat((sequence_tensor[:, 1:, :], torch.tensor([[predicted_scaled_value]], dtype=torch.float32).unsqueeze(0).to(device)), dim=1)
#                      sequence_tensor = new_sequence_tensor # Use the new sequence for the next step

#                  # Store the generated forecast values for the current commodity
#                  gru_forecasts[commodity] = commodity_forecast_values
#                  print(f"    Generated {len(commodity_forecast_values)} forecast values for {commodity}.")

#          print("\nGRU forecasting complete.")

#     else:
#         print("GRU model is not available. Cannot generate forecasts.")


# # You now have the generated forecasts in the 'gru_forecasts' dictionary.

# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during data manipulation and saving

# # Ensure prices_df (original data) and gru_forecasts (forecast dictionary) are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot combine data.")
#     # Finish the task with failure
# elif 'gru_forecasts' not in locals() or not gru_forecasts:
#     print("Error: 'gru_forecasts' dictionary not found or is empty. Cannot combine data.")
#     # Finish the task with failure
# else:
#     print("\nCombining original data and GRU forecasts...")

#     # Prepare the original data: select commodity columns and ensure DatetimeIndex
#     # Assuming the index of prices_df is already the correct DatetimeIndex ('Date')
#     # and numeric columns are identified in commodity_columns list
#     # Ensure commodity_columns is available; if not, try to infer from prices_df
#     if 'commodity_columns' not in locals() or not commodity_columns:
#          print("Warning: 'commodity_columns' not found. Attempting to infer from prices_df.")
#          all_cols = prices_df.columns.tolist()
#          cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
#                             'day_of_week_sin', 'day_of_week_cos',
#                             'month_sin', 'month_cos',
#                             'year_sin', 'year_cos']
#          commodity_columns = [col for col in all_cols if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]
#          if not commodity_columns:
#              print("Error: Could not identify numeric commodity columns from prices_df.")
#              # Finish the task with failure
#              combined_gru_df = pd.DataFrame() # Initialize as empty
#          else:
#              print(f"Inferred commodity columns: {commodity_columns}")
#              historical_df = prices_df[commodity_columns].copy()
#     # Corrected indentation for this else block
#     else:
#          historical_df = prices_df[commodity_columns].copy()

#     # Ensure historical_df has the correct DatetimeIndex ('Date')
#     historical_df.index.name = 'Date'


#     # Prepare the forecast data:
#     # Create a DataFrame from the gru_forecasts dictionary
#     if gru_forecasts:
#         # Get the last date from the historical data to start the forecast index from the next day
#         last_historical_date = prices_df.index.max()

#         # Determine the frequency of the historical data
#         # Ensure prices_df index is DatetimeIndex before inferring frequency
#         if isinstance(prices_df.index, pd.DatetimeIndex):
#             freq = pd.infer_freq(prices_df.index)
#             if freq is None:
#                  # Fallback to 'D' if frequency cannot be inferred (assuming daily data)
#                  freq = 'D'
#                  print(f"Warning: Could not infer frequency from historical data. Assuming '{freq}'.")
#             else:
#                 print(f"Inferred historical data frequency: {freq}")

#             # Define the number of steps to forecast (from the gru_forecasts dictionary)
#             # Assuming all forecast lists in the dictionary have the same length
#             forecast_steps = len(list(gru_forecasts.values())[0]) if gru_forecasts else 0

#             if forecast_steps > 0:
#                  # Generate future dates starting from the day after the last historical date
#                  # Use periods = forecast_steps + 1 and slice from 1 to exclude the last historical date itself
#                  forecast_dates = pd.date_range(start=last_historical_date, periods=forecast_steps + 1, freq=freq)[1:]

#                  # Create a DataFrame from the forecast dictionary, using the generated future dates as the index
#                  forecast_df = pd.DataFrame(gru_forecasts, index=forecast_dates)

#                  # Rename the index to 'Date' for consistency
#                  forecast_df.index.name = 'Date'
#             else:
#                  print("No forecast steps defined based on gru_forecasts dictionary.")
#                  forecast_df = pd.DataFrame() # Create an empty DataFrame

#         else:
#              print("Error: prices_df index is not a DatetimeIndex. Cannot generate forecast dates.")
#              forecast_df = pd.DataFrame() # Create an empty DataFrame


#     else:
#         print("No GRU forecasts were generated to combine.")
#         forecast_df = pd.DataFrame() # Create an empty DataFrame if no forecasts

#     # Combine the historical data and the forecast data
#     # Use outer join to include all dates from both historical and forecast periods
#     # The index from both DataFrames (DatetimeIndex) will be used for alignment
#     combined_gru_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')

#     # Sort by date to ensure chronological order
#     combined_gru_df.sort_index(inplace=True)

#     print("\nCombined GRU DataFrame Head:")
#     print(combined_gru_df.head(2))
#     print("\nCombined GRU DataFrame Tail:")
#     print(combined_gru_df.tail(2))
#     print("\nCombined GRU DataFrame Shape:", combined_gru_df.shape)


# #===============================================================================
# # Save the Combined GRU Data to Google Sheets
# #===============================================================================
# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during model fitting

# # Ensure combined_gru_df is available
# if 'combined_gru_df' not in locals() or combined_gru_df.empty:
#     print("Error: 'combined_gru_df' DataFrame not found or is empty. Cannot save GRU forecasts.")
# else:
#     # Ensure gc (gspread client) and sh (Google Sheet object) are available from previous cells
#     # Assuming gc and sh are already initialized and authenticated
#     if 'sh' not in locals():
#         print("Google Sheet object 'sh' not found. Attempting to authenticate and open sheet.")
#         try:
#             auth.authenticate_user()
#             creds, _ = default()
#             gc = gspread.authorize(creds)
#             sh = gc.open('Quant_Calc_Main')
#             print("Authenticated and opened Google Sheet: Quant_Calc_Main")
#         except Exception as e:
#             print(f"Failed to authenticate or open Google Sheet: {e}")
#             sh = None # Ensure sh is None if fallback fails

#     if sh and not combined_gru_df.empty:
#         try:
#             print("\nAttempting to save combined GRU forecast results to Google Sheets...")
#             sheet_title = 'GRU_FORECAST' # Specify the desired sheet title

#             # Check if the worksheet with the specified title already exists
#             try:
#                 worksheet_gru_forecast = sh.worksheet(sheet_title)
#                 print(f"Worksheet '{sheet_title}' already exists. Clearing existing data.")
#                 worksheet_gru_forecast.clear() # Clear existing data
#             except gspread.WorksheetNotFound:
#                 # If not found, create a new one
#                 # Estimate the number of rows and columns needed
#                 num_rows = combined_gru_df.shape[0] + 1 # Data rows + header
#                 num_cols = combined_gru_df.shape[1] + 1 # Add 1 for the index ('Date')

#                 # gspread has a column limit, check if it's exceeded (usually 256)
#                 max_gspread_cols = 256
#                 if num_cols > max_gspread_cols:
#                      print(f"Warning: Number of columns ({num_cols}) exceeds Google Sheets limit ({max_gspread_cols}). Saving only the first {max_gspread_cols} columns.")
#                      num_cols_to_save = max_gspread_cols
#                      # Include the index column + limited data columns
#                      combined_df_limited = combined_gru_df.reset_index().iloc[:, :num_cols_to_save].copy()
#                 else:
#                      num_cols_to_save = num_cols
#                      combined_df_limited = combined_gru_df.reset_index().copy()


#                 worksheet_gru_forecast = sh.add_worksheet(title=sheet_title, rows=num_rows, cols=num_cols_to_save)
#                 print(f"Worksheet '{sheet_title}' created.")

#             # Convert the DataFrame to a list of lists for gspread, including headers
#             # Ensure 'Date' column is string and format it nicely
#             # Use .dt accessor now that we've ensured it's datetime index
#             if 'Date' in combined_df_limited.columns:
#                  if pd.api.types.is_datetime64_any_dtype(combined_df_limited['Date']):
#                       combined_df_limited['Date'] = combined_df_limited['Date'].dt.strftime('%Y-%m-%d')
#                  combined_df_limited['Date'].fillna('', inplace=True) # Fill any potential NaT dates with an empty string or placeholder if needed


#             # Convert all other data columns to string, handling potential NaN values
#             # Replace NaN with empty string for cleaner representation in Google Sheets
#             data_to_save = [combined_df_limited.columns.tolist()] + combined_df_limited.fillna('').astype(str).values.tolist()


#             # Update the worksheet with the data
#             # gspread expects a list of lists where each inner list is a row
#             worksheet_gru_forecast.update(data_to_save)
#             print(f"Combined historical data and GRU forecasts saved to worksheet '{sheet_title}'.")

#         except Exception as e:
#             print(f"Error saving combined GRU forecast results to Google Sheet: {e}")
#     else:
#         print("Google Sheet object 'sh' is not available. Cannot save results.")


# #===============================================================================
# # Visualize GRU Forecasts
# #===============================================================================
# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_gru_df is available from the previous step

# def plot_gru_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and GRU forecast for a specific commodity
#     from the combined DataFrame.

#     Args:
#         df_combined (pd.DataFrame): DataFrame containing combined historical and
#                                     forecast data with 'Date' as index.
#         commodity_name (str): The name of the commodity to plot.
#     """
#     if commodity_name not in df_combined.columns:
#         print(f"Error: Commodity '{commodity_name}' not found in the DataFrame.")
#         return

#     plt.figure(figsize=(12, 3))

#     # Plot historical data (non-NaN values in the historical period)
#     # Assuming historical data ends before the forecast period starts
#     # Find the index where the forecast starts (first non-NaN value in the forecast period)
#     # We can infer this from the first non-NaN value in the combined df after the end of original prices_df
#     if 'prices_df' in locals() and not prices_df.empty:
#         last_historical_date = prices_df.index.max()
#         historical_data_to_plot = df_combined[df_combined.index <= last_historical_date][commodity_name].dropna()
#         forecast_data_to_plot = df_combined[df_combined.index > last_historical_date][commodity_name].dropna()

#         plt.plot(historical_data_to_plot.index, historical_data_to_plot.values, label='Historical Data', color='blue')
#         plt.plot(forecast_data_to_plot.index, forecast_data_to_plot.values, label='GRU Forecast', color='red', linestyle='--')

#         # Add a vertical line at the end of the historical data
#         if last_historical_date:
#             plt.axvline(last_historical_date, color='red', linestyle='--', label='Forecast Start')

#     else:
#         # If original prices_df is not available, just plot all data in combined_gru_df
#         print("Warning: Original 'prices_df' not found. Plotting all combined data.")
#         plt.plot(df_combined.index, df_combined[commodity_name], label=f'{commodity_name} Data & GRU Forecast')


#     plt.title(f'GRU Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_gru_df is already loaded

# # Get a list of commodities that were forecasted
# # We can get this from the keys of the gru_forecasts dictionary
# if 'gru_forecasts' in locals() and gru_forecasts:
#     commodities_to_visualize = list(gru_forecasts.keys())[:5] # Visualize the first 5 forecasted commodities
#     print(f"\nVisualizing GRU forecasts for the following commodities: {commodities_to_visualize}")
# else:
#     print("Error: 'gru_forecasts' dictionary not found or is empty. Cannot visualize forecasts.")
#     commodities_to_visualize = []


# if 'combined_gru_df' in locals() and not combined_gru_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_gru_forecast(combined_gru_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_gru_df is not available or empty, or no commodities to visualize.")

# """**Reasoning**:
# Define the architecture of the MAMBA model and create the PyTorch DataLoader for the prepared time series data.

# ## MAMBA FORECASTER

# Implement a MAMBA forecaster for the time series data in `prices_df`, train it, generate forecasts, combine the forecasts with the historical data, save the result to a Google Sheet, and visualize the forecasts.
# """

# # Commented out IPython magic to ensure Python compatibility.
# # %%capture
# # !pip install mambapy einops

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # 1. Identify the numerical commodity columns
# all_columns = prices_df.columns.tolist()
# commodity_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(prices_df[col])]

# print("Identified commodity columns for MAMBA data preparation:")
# print(commodity_columns)

# # 2. Define sequence length (number of past time steps to use for prediction)
# sequence_length = 30 # Example: use the past 30 days to predict the next day

# # 3. Initialize lists to store sequences and targets for all commodities
# all_sequences = []
# all_targets = []

# # 4. Initialize a dictionary to store scalers for each commodity
# scalers_mamba = {}

# # 5. Iterate through each identified commodity column
# print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
# for commodity in commodity_columns:
#     print(f"  Processing commodity: {commodity}")

#     # 3. For each commodity, extract the time series data
#     series = prices_df[commodity].dropna()

#     if len(series) < sequence_length + 1:
#         print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#         continue # Skip to the next commodity

#     # 4. Scale the time series data
#     scaler = MinMaxScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#     scalers_mamba[commodity] = scaler # Store the scaler for this commodity

#     # 5. Create input sequences (X) and corresponding target values (y)
#     sequences = []
#     targets = []
#     for i in range(len(scaled_series) - sequence_length):
#         seq = scaled_series[i:i + sequence_length]
#         target = scaled_series[i + sequence_length]
#         sequences.append(seq)
#         targets.append(target)

#     # 6. Store the prepared input sequences and target values for the current commodity
#     all_sequences.extend(sequences)
#     all_targets.extend(targets)
#     print(f"    Generated {len(sequences)} sequences for {commodity}.")


# # 7. Combine the input sequences and target values into PyTorch tensors
# if not all_sequences:
#     print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#     # Initialize tensors as empty if no data is processed
#     X_tensor_mamba = torch.empty(0, sequence_length, dtype=torch.float32)
#     y_tensor_mamba = torch.empty(0, dtype=torch.float32)

# else:
#     X_tensor_mamba = torch.tensor(all_sequences, dtype=torch.float32)
#     y_tensor_mamba = torch.tensor(all_targets, dtype=torch.float32)
#     print("\nCombined sequences and targets into PyTorch tensors.")
#     print("Shape of X_tensor_mamba:", X_tensor_mamba.shape)
#     print("Shape of y_tensor_mamba:", y_tensor_mamba.shape)


# # 8. Confirm that the scalers dictionary is stored
# if 'scalers_mamba' in locals():
#     print("\nScalers for each commodity stored in 'scalers_mamba' dictionary.")

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from mambapy.mamba import Mamba, MambaConfig # Import MambaConfig

# # Ensure X_tensor_mamba and y_tensor_mamba are available from the previous data preparation step
# if 'X_tensor_mamba' not in locals() or 'y_tensor_mamba' not in locals() or X_tensor_mamba.shape[0] == 0:
#     print("Error: X_tensor_mamba or y_tensor_mamba is not available or is empty. Cannot create Dataset and DataLoader.")
#     # Initialize necessary variables to prevent errors in later steps
#     train_loader_mamba = None
#     model_mamba = None

# else:
#     print("X_tensor_mamba and y_tensor_mamba are available. Proceeding with Dataset, DataLoader, and Model definition.")

#     # 1. Define a PyTorch Dataset class for MAMBA
#     class TimeSeriesDatasetMamba(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             # MAMBA expects input shape (batch_size, sequence_length, input_dim)
#             # Since we are forecasting a single time series value at a time, input_dim is 1
#             # We need to unsqueeze the sequence tensor to add the input_dim dimension
#             return self.X[idx].unsqueeze(-1), self.y[idx] # Add last dimension for input_dim

#     # Instantiate the Dataset
#     dataset_mamba = TimeSeriesDatasetMamba(X_tensor_mamba, y_tensor_mamba)
#     print(f"\nDataset for MAMBA created with {len(dataset_mamba)} samples.")

#     # 2. Create a PyTorch DataLoader
#     batch_size = 32 # Define your desired batch size (can be the same as KAN)
#     train_loader_mamba = DataLoader(dataset_mamba, batch_size=batch_size, shuffle=True)
#     print(f"DataLoader for MAMBA created with batch size {batch_size}.")
#     print(f"Number of batches in DataLoader: {len(train_loader_mamba)}")


#     # 3. Define the MAMBA model architecture
#     # Assuming sequence_length is available from the data preparation step
#     if 'sequence_length' not in locals():
#         print("Error: 'sequence_length' is not defined. Cannot define MAMBA model architecture.")
#         model_mamba = None # Ensure model_mamba is None if sequence_length is missing
#     else:
#         print(f"\nDefining MAMBA model architecture with sequence length {sequence_length}...")

#         # Define the configuration for the Mamba model using MambaConfig
#         # Adjust d_model and n_layers as needed for your specific task and data complexity
#         # d_model: The dimension of the model embeddings (must match the last dimension of the input tensor)
#         # n_layers: The number of Mamba blocks
#         # Increased n_layers from 2 to 4
#         mamba_config = MambaConfig(d_model=1, n_layers=4) # Increased complexity

#         # Instantiate the base Mamba model with the defined configuration
#         try:
#              base_mamba_model = Mamba(mamba_config) # Instantiate with MambaConfig
#              print("Base Mamba model instantiated with config.")

#              # Add a final linear layer to map MAMBA output to the prediction for the next time step
#              # The output of Mamba is (batch_size, sequence_length, d_model)
#              # We are interested in the last time step's output for prediction
#              class MambaForecaster(nn.Module):
#                  def __init__(self, mamba_model, output_dim=1):
#                      super().__init__()
#                      self.mamba = mamba_model
#                      # The input dimension to the linear layer is the d_model of the Mamba output
#                      self.fc = nn.Linear(mamba_model.config.d_model, output_dim) # Use mamba_model.config.d_model

#                  def forward(self, x):
#                      # x shape: (batch_size, sequence_length, input_dim) where input_dim is 1
#                      mamba_output = self.mamba(x)
#                      # We need the output from the last time step to predict the next value
#                      last_step_output = mamba_output[:, -1, :] # Shape: (batch_size, mamba_model.config.d_model)
#                      prediction = self.fc(last_step_output) # Shape: (batch_size, output_dim)
#                      return prediction

#              # Instantiate the MambaForecaster
#              model_mamba = MambaForecaster(base_mamba_model, output_dim=1)


#              print("MAMBA model architecture defined.")

#              # 4. Instantiate the defined MAMBA model
#              # model_mamba is already instantiated by the MambaForecaster call above if successful
#              if model_mamba is not None:
#                  print("\nMAMBA model instantiated.")
#                  print("Model structure:")
#                  print(model_mamba)
#              else:
#                  print("\nMAMBA model instantiation failed.")

#         except Exception as e:
#              print(f"Error instantiating Mamba model with config: {e}")
#              model_mamba = None # Set to None if model creation fails

# import torch.optim as optim
# import torch.nn as nn

# # Ensure model_mamba and train_loader_mamba are available from the previous step
# if 'model_mamba' not in locals() or model_mamba is None:
#     print("Error: MAMBA model is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the model is not defined
#     # No need to initialize other variables as training won't start

# elif 'train_loader_mamba' not in locals() or train_loader_mamba is None:
#     print("Error: train_loader_mamba is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the train_loader_mamba is not defined
#     # No need to initialize other variables as training won't start

# else:
#     print("\nProceeding with MAMBA model training.")

#     # 1. Define the loss function (e.g., Mean Squared Error)
#     criterion = nn.MSELoss()
#     print("Loss function defined (MSELoss).")

#     # 2. Define the optimizer (e.g., Adam)
#     # Define your desired learning rate - MAMBA might benefit from different rates
#     learning_rate = 0.0001 # Example learning rate, might need tuning
#     optimizer = optim.Adam(model_mamba.parameters(), lr=learning_rate)
#     print(f"Optimizer defined (Adam with learning rate {learning_rate}).")

#     # 3. Set the number of training epochs
#     num_epochs = 10 # Define your desired number of epochs - might need more or fewer
#     print(f"Number of training epochs set to: {num_epochs}")

#     # 4. Implement the training loop
#     print("\nStarting MAMBA model training...")

#     # Move model to the appropriate device if available (CPU or GPU)
#     # Check if CUDA is available and use GPU if it is, otherwise use CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_mamba.to(device)
#     print(f"Using device: {device}")


#     for epoch in range(num_epochs):
#         model_mamba.train() # Set the model to training mode
#         running_loss = 0.0

#         for i, data in enumerate(train_loader_mamba):
#             inputs, targets = data

#             # Move data to the device
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model_mamba(inputs)

#             # Calculate the loss
#             # Ensure targets have the same shape as outputs (usually [batch_size, 1])
#             # If targets is [batch_size], unsqueeze it to [batch_size, 1]
#             if targets.ndim == 1:
#                  targets = targets.unsqueeze(1)

#             loss = criterion(outputs, targets)

#             # Backward pass
#             loss.backward()

#             # Update the weights
#             optimizer.step()

#             # Accumulate the loss
#             running_loss += loss.item()

#         # Print the loss periodically
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader_mamba):.4f}")

#     print("\nMAMBA model training complete.")

# import torch
# import os

# # Define the path to save the model (e.g., in Google Drive)
# # Ensure you have mounted Google Drive in a previous cell (e.g., cell Nv0Vvz_e5Ma2 or SzxtDdaQvm4n) if saving there
# # This path should be consistent when loading the model later
# mamba_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_mamba_model.pth' # You can change the filename and path

# # Ensure the directory exists
# os.makedirs(os.path.dirname(mamba_model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model_mamba' not in locals() or model_mamba is None:
#     print("Error: MAMBA model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model_mamba.state_dict(), mamba_model_save_path)
#         print(f"MAMBA model state dictionary saved successfully to: {mamba_model_save_path}")
#     except Exception as e:
#         print(f"Error saving MAMBA model: {e}")

# import torch
# import pandas as pd
# import numpy as np
# import os
# from mambapy.mamba import Mamba, MambaConfig # Import MambaConfig

# # Define the path to the saved model
# mamba_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_mamba_model.pth' # This should match the save path

# # Ensure MAMBA model architecture classes (Mamba, MambaConfig, MambaForecaster, etc.) are defined as in the model definition cell (cd723792)
# # Ensure prices_df, commodity_columns, sequence_length, and scalers_mamba are available

# # Check if the saved model file exists
# if not os.path.exists(mamba_model_save_path):
#     print(f"Error: Saved MAMBA model not found at {mamba_model_save_path}. Cannot generate forecasts using the saved model.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'Mamba' not in locals() or 'MambaConfig' not in locals() or 'MambaForecaster' not in locals():
#      print("Error: MAMBA model architecture classes (Mamba, MambaConfig, MambaForecaster) are not defined. Cannot load the saved model.")
#      mamba_forecasts = {} # Initialize as empty
# elif 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare forecast input sequences.")
#     mamba_forecasts = {} # Initialize as empty
# elif not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#     print("Error: 'prices_df' index is not a DatetimeIndex named 'Date'. Cannot proceed.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot prepare forecast input sequences.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot generate forecasts.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'scalers_mamba' not in locals() or not scalers_mamba:
#     print("Error: 'scalers_mamba' dictionary not found or is empty. Cannot inverse scale forecasts.")
#     mamba_forecasts = {} # Initialize as empty

# else:
#     print("\nLoading the saved MAMBA model and generating future forecasts...")

#     # Instantiate the MAMBA model architecture
#     # Ensure the same MambaConfig is used as during training (d_model=1, n_layers=4 based on previous step)
#     try:
#         mamba_config = MambaConfig(d_model=1, n_layers=4)
#         base_mamba_model = Mamba(mamba_config)
#         # Instantiate the MambaForecaster with the base Mamba model
#         model_mamba_loaded = MambaForecaster(base_mamba_model, output_dim=1) # Assuming MambaForecaster expects base model and output_dim

#         print("MAMBA model architecture instantiated for loading.")

#         # Load the saved state dictionary
#         model_mamba_loaded.load_state_dict(torch.load(mamba_model_save_path))
#         print(f"MAMBA model state dictionary loaded from {mamba_model_save_path}.")

#         # Move the model to the appropriate device (CPU or GPU) if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_mamba_loaded.to(device)
#         model_mamba_loaded.eval() # Set the model to evaluation mode
#         print(f"Using device for forecasting: {device}")


#     except Exception as e:
#         print(f"Error loading MAMBA model or setting up for forecasting: {e}")
#         mamba_forecasts = {} # Initialize as empty
#         model_mamba_loaded = None # Ensure model_mamba_loaded is None if loading fails


#     # Initialize a dictionary to store the generated forecasts for each commodity
#     mamba_forecasts = {}

#     if model_mamba_loaded is not None:
#          # Define the number of steps to forecast
#          forecast_steps = 60 # This should align with other forecasters if possible
#          print(f"\nGenerating {forecast_steps} steps into the future for each commodity...")

#          # Iterate through each commodity for which an input sequence can be prepared
#          print(f"Generating forecasts for {len(commodity_columns)} commodities...")
#          with torch.no_grad(): # Disable gradient calculation during inference
#              for commodity in commodity_columns:
#                  print(f"  Forecasting for: {commodity}")

#                  # Get the time series for the current commodity and drop missing values
#                  series = prices_df[commodity].dropna()

#                  # Check if the series has enough data points for the initial sequence
#                  if len(series) < sequence_length:
#                      print(f"    Warning: Skipping {commodity}. Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#                      continue # Skip to the next commodity

#                  # Get the last `sequence_length` data points as the initial input sequence
#                  last_sequence = series[-sequence_length:]

#                  # Get the corresponding scaler for this commodity
#                  if commodity not in scalers_mamba:
#                       print(f"    Warning: Scaler not found for {commodity}. Skipping forecast.")
#                       continue # Skip to the next commodity
#                  scaler = scalers_mamba[commodity]

#                  # Scale the last sequence using the corresponding scaler
#                  # Reshape the sequence to be a 2D array for scaling
#                  scaled_sequence = scaler.transform(last_sequence.values.reshape(-1, 1)).flatten()

#                  # Convert the scaled sequence to a PyTorch tensor
#                  # Reshape to [1, sequence_length, input_dim] as the model expects
#                  # input_dim is 1 for single time series
#                  sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # Add batch and input_dim dimensions and move to device

#                  # Initialize a list to store the unscaled forecast values for this commodity
#                  commodity_forecast_values = []

#                  # Generate the forecast step by step
#                  for _ in range(forecast_steps):
#                      # Get the MAMBA model's prediction for the next step (scaled value)
#                      predicted_scaled_value = model_mamba_loaded(sequence_tensor).item() # Get the scalar value

#                      # Inverse scale the predicted value to the original price scale
#                      # The scaler expects a 2D array, so reshape the scalar
#                      predicted_original_value = scaler.inverse_transform([[predicted_scaled_value]])[0][0]

#                      # Store the inverse-scaled forecast value
#                      commodity_forecast_values.append(predicted_original_value)

#                      # Update the input sequence for the next prediction
#                      # Remove the oldest value and append the new scaled prediction
#                      # The new prediction needs to be reshaped to match the input tensor shape
#                      new_sequence_tensor = torch.cat((sequence_tensor[:, 1:, :], torch.tensor([[predicted_scaled_value]], dtype=torch.float32).unsqueeze(0).to(device)), dim=1)
#                      sequence_tensor = new_sequence_tensor # Use the new sequence for the next step

#                  # Store the generated forecast values for the current commodity
#                  mamba_forecasts[commodity] = commodity_forecast_values
#                  print(f"    Generated {len(commodity_forecast_values)} forecast values for {commodity}.")

#          print("\nMAMBA forecasting complete.")

#     else:
#         print("MAMBA model is not available. Cannot generate forecasts.")


# # You now have the generated forecasts in the 'mamba_forecasts' dictionary.

# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during data manipulation and saving

# # Ensure prices_df (original data) and mamba_forecasts (forecast dictionary) are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot combine data.")
#     # Finish the task with failure
# elif 'mamba_forecasts' not in locals() or not mamba_forecasts:
#     print("Error: 'mamba_forecasts' dictionary not found or is empty. Cannot combine data.")
#     # Finish the task with failure
# else:
#     print("\nCombining original data and MAMBA forecasts...")

#     # Prepare the original data: select commodity columns and ensure DatetimeIndex
#     # Assuming the index of prices_df is already the correct DatetimeIndex ('Date')
#     # and numeric columns are identified in commodity_columns list
#     # Ensure commodity_columns is available; if not, try to infer from prices_df
#     if 'commodity_columns' not in locals() or not commodity_columns:
#          print("Warning: 'commodity_columns' not found. Attempting to infer from prices_df.")
#          all_cols = prices_df.columns.tolist()
#          cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
#                             'day_of_week_sin', 'day_of_week_cos',
#                             'month_sin', 'month_cos',
#                             'year_sin', 'year_cos']
#          commodity_columns = [col for col in all_cols if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]
#          if not commodity_columns:
#              print("Error: Could not identify numeric commodity columns from prices_df.")
#              # Finish the task with failure
#              combined_mamba_df = pd.DataFrame() # Initialize as empty
#          else:
#              print(f"Inferred commodity columns: {commodity_columns}")
#              historical_df = prices_df[commodity_columns].copy()
#     # Corrected indentation for this else block
#     else:
#          historical_df = prices_df[commodity_columns].copy()

#     # Ensure historical_df has the correct DatetimeIndex ('Date')
#     historical_df.index.name = 'Date'


#     # Prepare the forecast data:
#     # Create a DataFrame from the mamba_forecasts dictionary
#     if mamba_forecasts:
#         # Get the last date from the historical data to start the forecast index from the next day
#         last_historical_date = prices_df.index.max()

#         # Determine the frequency of the historical data
#         # Ensure prices_df index is DatetimeIndex before inferring frequency
#         if isinstance(prices_df.index, pd.DatetimeIndex):
#             freq = pd.infer_freq(prices_df.index)
#             if freq is None:
#                  # Fallback to 'D' if frequency cannot be inferred (assuming daily data)
#                  freq = 'D'
#                  print(f"Warning: Could not infer frequency from historical data. Assuming '{freq}'.")
#             else:
#                 print(f"Inferred historical data frequency: {freq}")

#             # Define the number of steps to forecast (from the mamba_forecasts dictionary)
#             # Assuming all forecast lists in the dictionary have the same length
#             forecast_steps = len(list(mamba_forecasts.values())[0]) if mamba_forecasts else 0

#             if forecast_steps > 0:
#                  # Generate future dates starting from the day after the last historical date
#                  # Use periods = forecast_steps + 1 and slice from 1 to exclude the last historical date itself
#                  forecast_dates = pd.date_range(start=last_historical_date, periods=forecast_steps + 1, freq=freq)[1:]

#                  # Create a DataFrame from the forecast dictionary, using the generated future dates as the index
#                  forecast_df = pd.DataFrame(mamba_forecasts, index=forecast_dates)

#                  # Rename the index to 'Date' for consistency
#                  forecast_df.index.name = 'Date'
#             else:
#                  print("No forecast steps defined based on mamba_forecasts dictionary.")
#                  forecast_df = pd.DataFrame() # Create an empty DataFrame

#         else:
#              print("Error: prices_df index is not a DatetimeIndex. Cannot generate forecast dates.")
#              forecast_df = pd.DataFrame() # Create an empty DataFrame


#     else:
#         print("No MAMBA forecasts were generated to combine.")
#         forecast_df = pd.DataFrame() # Create an empty DataFrame if no forecasts

#     # Combine the historical data and the forecast data
#     # Use outer join to include all dates from both historical and forecast periods
#     # The index from both DataFrames (DatetimeIndex) will be used for alignment
#     combined_mamba_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')

#     # Sort by date to ensure chronological order
#     combined_mamba_df.sort_index(inplace=True)

#     print("\nCombined MAMBA DataFrame Head:")
#     print(combined_mamba_df.head(2))
#     print("\nCombined MAMBA DataFrame Tail:")
#     print(combined_mamba_df.tail(2))
#     print("\nCombined MAMBA DataFrame Shape:", combined_mamba_df.shape)


# #===============================================================================
# # Save the Combined MAMBA Data to Google Sheets
# #===============================================================================
# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during model fitting

# # Ensure combined_mamba_df is available
# if 'combined_mamba_df' not in locals() or combined_mamba_df.empty:
#     print("Error: 'combined_mamba_df' DataFrame not found or is empty. Cannot save MAMBA forecasts.")
# else:
#     # Ensure gc (gspread client) and sh (Google Sheet object) are available from previous cells
#     # Assuming gc and sh are already initialized and authenticated
#     if 'sh' not in locals():
#         print("Google Sheet object 'sh' not found. Attempting to authenticate and open sheet.")
#         try:
#             auth.authenticate_user()
#             creds, _ = default()
#             gc = gspread.authorize(creds)
#             sh = gc.open('Quant_Calc_Main')
#             print("Authenticated and opened Google Sheet: Quant_Calc_Main")
#         except Exception as e:
#             print(f"Failed to authenticate or open Google Sheet: {e}")
#             sh = None # Ensure sh is None if fallback fails

#     if sh and not combined_mamba_df.empty:
#         try:
#             print("\nAttempting to save combined MAMBA forecast results to Google Sheets...")
#             sheet_title = 'MAMBA_FORECAST' # Specify the desired sheet title

#             # Check if the worksheet with the specified title already exists
#             try:
#                 worksheet_mamba_forecast = sh.worksheet(sheet_title)
#                 print(f"Worksheet '{sheet_title}' already exists. Clearing existing data.")
#                 worksheet_mamba_forecast.clear() # Clear existing data
#             except gspread.WorksheetNotFound:
#                 # If not found, create a new one
#                 # Estimate the number of rows and columns needed
#                 num_rows = combined_mamba_df.shape[0] + 1 # Data rows + header
#                 num_cols = combined_mamba_df.shape[1] + 1 # Add 1 for the index ('Date')

#                 # gspread has a column limit, check if it's exceeded (usually 256)
#                 max_gspread_cols = 256
#                 if num_cols > max_gspread_cols:
#                      print(f"Warning: Number of columns ({num_cols}) exceeds Google Sheets limit ({max_gspread_cols}). Saving only the first {max_gspread_cols} columns.")
#                      num_cols_to_save = max_gspread_cols
#                      # Include the index column + limited data columns
#                      combined_df_limited = combined_mamba_df.reset_index().iloc[:, :num_cols_to_save].copy()
#                 else:
#                      num_cols_to_save = num_cols
#                      combined_df_limited = combined_mamba_df.reset_index().copy()


#                 worksheet_mamba_forecast = sh.add_worksheet(title=sheet_title, rows=num_rows, cols=num_cols_to_save)
#                 print(f"Worksheet '{sheet_title}' created.")

#             # Convert the DataFrame to a list of lists for gspread, including headers
#             # Ensure 'Date' column is string and format it nicely
#             # Use .dt accessor now that we've ensured it's datetime index
#             if 'Date' in combined_df_limited.columns:
#                  if pd.api.types.is_datetime64_any_dtype(combined_df_limited['Date']):
#                       combined_df_limited['Date'] = combined_df_limited['Date'].dt.strftime('%Y-%m-%d')
#                  combined_df_limited['Date'].fillna('', inplace=True) # Fill any potential NaT dates with an empty string or placeholder if needed


#             # Convert all other data columns to string, handling potential NaN values
#             # Replace NaN with empty string for cleaner representation in Google Sheets
#             data_to_save = [combined_df_limited.columns.tolist()] + combined_df_limited.fillna('').astype(str).values.tolist()


#             # Update the worksheet with the data
#             # gspread expects a list of lists where each inner list is a row
#             worksheet_mamba_forecast.update(data_to_save)
#             print(f"Combined historical data and MAMBA forecasts saved to worksheet '{sheet_title}'.")

#         except Exception as e:
#             print(f"Error saving combined MAMBA forecast results to Google Sheet: {e}")
#     else:
#         print("Google Sheet object 'sh' is not available. Cannot save results.")

# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_mamba_df is available from the previous step

# def plot_mamba_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and MAMBA forecast for a specific commodity
#     from the combined DataFrame.

#     Args:
#         df_combined (pd.DataFrame): DataFrame containing combined historical and
#                                     forecast data with 'Date' as index.
#         commodity_name (str): The name of the commodity to plot.
#     """
#     if commodity_name not in df_combined.columns:
#         print(f"Error: Commodity '{commodity_name}' not found in the DataFrame.")
#         return

#     plt.figure(figsize=(12, 3))

#     # Plot historical data (non-NaN values in the historical period)
#     # Assuming historical data ends before the forecast period starts
#     # Find the index where the forecast starts (first non-NaN value in the forecast period)
#     # We can infer this from the first non-NaN value in the combined df after the end of original prices_df
#     if 'prices_df' in locals() and not prices_df.empty:
#         last_historical_date = prices_df.index.max()
#         historical_data_to_plot = df_combined[df_combined.index <= last_historical_date][commodity_name].dropna()
#         forecast_data_to_plot = df_combined[df_combined.index > last_historical_date][commodity_name].dropna()

#         plt.plot(historical_data_to_plot.index, historical_data_to_plot.values, label='Historical Data', color='blue')
#         plt.plot(forecast_data_to_plot.index, forecast_data_to_plot.values, label='MAMBA Forecast', color='red', linestyle='--')

#         # Add a vertical line at the end of the historical data
#         if last_historical_date:
#             plt.axvline(last_historical_date, color='red', linestyle='--', label='Forecast Start')

#     else:
#         # If original prices_df is not available, just plot all data in combined_mamba_df
#         print("Warning: Original 'prices_df' not found. Plotting all combined data.")
#         plt.plot(df_combined.index, df_combined[commodity_name], label=f'{commodity_name} Data & MAMBA Forecast')


#     plt.title(f'MAMBA Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_mamba_df is already loaded

# # Get a list of commodities that were forecasted
# # We can get this from the keys of the mamba_forecasts dictionary
# if 'mamba_forecasts' in locals() and mamba_forecasts:
#     commodities_to_visualize = list(mamba_forecasts.keys())[:5] # Visualize the first 5 forecasted commodities
#     print(f"\nVisualizing MAMBA forecasts for the following commodities: {commodities_to_visualize}")
# else:
#     print("Error: 'mamba_forecasts' dictionary not found or is empty. Cannot visualize forecasts.")
#     commodities_to_visualize = []


# if 'combined_mamba_df' in locals() and not combined_mamba_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_mamba_forecast(combined_mamba_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_mamba_df is not available or empty, or no commodities to visualize.")

# """## Summary:

# ### Data Analysis Key Findings
# *   The `mambapy` library was successfully installed.
# *   An error occurred in the MAMBA forecaster section where the data preparation step failed due to the `prices_df` variable not being defined, indicating an interruption in the execution flow.

# ### Insights or Next Steps
# *   The immediate next step is to re-run the data preparation cell to ensure `prices_df` and other essential variables for the MAMBA forecaster are correctly initialized before proceeding with further analysis.

# # Task
# #===============================================================================

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # 1. Identify the numerical commodity columns
# all_columns = prices_df.columns.tolist()
# commodity_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(prices_df[col])]

# print("Identified commodity columns for MAMBA data preparation:")
# print(commodity_columns)

# # 2. Define sequence length (number of past time steps to use for prediction)
# sequence_length = 30 # Example: use the past 30 days to predict the next day

# # 3. Initialize lists to store sequences and targets for all commodities
# all_sequences = []
# all_targets = []

# # 4. Initialize a dictionary to store scalers for each commodity
# scalers_mamba = {}

# # 5. Iterate through each identified commodity column
# print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
# for commodity in commodity_columns:
#     print(f"  Processing commodity: {commodity}")

#     # 3. For each commodity, extract the time series data
#     series = prices_df[commodity].dropna()

#     if len(series) < sequence_length + 1:
#         print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#         continue # Skip to the next commodity

#     # 4. Scale the time series data
#     scaler = MinMaxScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#     scalers_mamba[commodity] = scaler # Store the scaler for this commodity

#     # 5. Create input sequences (X) and corresponding target values (y)
#     sequences = []
#     targets = []
#     for i in range(len(scaled_series) - sequence_length):
#         seq = scaled_series[i:i + sequence_length]
#         target = scaled_series[i + sequence_length]
#         sequences.append(seq)
#         targets.append(target)

#     # 6. Store the prepared input sequences and target values for the current commodity
#     all_sequences.extend(sequences)
#     all_targets.extend(targets)
#     print(f"    Generated {len(sequences)} sequences for {commodity}.")


# # 7. Combine the input sequences and target values into PyTorch tensors
# if not all_sequences:
#     print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#     # Initialize tensors as empty if no data is processed
#     X_tensor_mamba = torch.empty(0, sequence_length, dtype=torch.float32)
#     y_tensor_mamba = torch.empty(0, dtype=torch.float32)

# else:
#     X_tensor_mamba = torch.tensor(all_sequences, dtype=torch.float32)
#     y_tensor_mamba = torch.tensor(all_targets, dtype=torch.float32)
#     print("\nCombined sequences and targets into PyTorch tensors.")
#     print("Shape of X_tensor_mamba:", X_tensor_mamba.shape)
#     print("Shape of y_tensor_mamba:", y_tensor_mamba.shape)


# # 8. Confirm that the scalers dictionary is stored
# if 'scalers_mamba' in locals():
#     print("\nScalers for each commodity stored in 'scalers_mamba' dictionary.")



    

# # Define the path to save the model (e.g., in Google Drive)
# # Ensure you have mounted Google Drive in a previous cell (e.g., cell SzxtDdaQvm4n) if saving there
# mamba_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_mamba_model.pth' # You can change the filename and path

# # Ensure the directory exists
# os.makedirs(os.path.dirname(mamba_model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model_mamba' not in locals() or model_mamba is None:
#     print("Error: MAMBA model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model_mamba.state_dict(), mamba_model_save_path)
#         print(f"MAMBA model state dictionary saved successfully to: {mamba_model_save_path}")
#     except Exception as e:
#         print(f"Error saving MAMBA model: {e}")
# #===============================================================================