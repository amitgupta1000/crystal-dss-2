import io
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.auth import default
from google.cloud import storage

from src.file_utils import save_dataframe_to_gcs

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message="Non-invertible starting seasonal moving average",
)
warnings.filterwarnings(
    "ignore",
    message="Non-stationary starting autoregressive",
)

creds, _ = default()
storage_client = storage.Client(credentials=creds)
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "crystal-dss")


def load_seasonality_study(
    gcs_path: str = "stats_studies_data/seasonality/seasonality_results_20260123.csv",
    bucket_name: str = BUCKET_NAME,
) -> Dict[str, int]:
    """Fetch commodity seasonal period hints from GCS."""
    try:
        print(f"\nLoading seasonality study from GCS: {gcs_path}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        data = blob.download_as_bytes()
        seasonality_df = pd.read_csv(io.BytesIO(data))
        seasonal_only = seasonality_df[seasonality_df['Is Seasonal for Period'] == True].copy()
        if seasonal_only.empty:
            print("  Warning: No seasonal periods found in study")
            return {}
        best_periods = seasonal_only.loc[
            seasonal_only.groupby('Commodity')['Seasonality Strength (STL)'].idxmax()
        ]
        seasonality_map = dict(
            zip(best_periods['Commodity'], best_periods['Period'].astype(int))
        )
        print(f"  ✓ Loaded seasonality data for {len(seasonality_map)} commodities")
        return seasonality_map
    except Exception as exc:
        print(f"  ✗ Failed to load seasonality study: {exc}")
        print("  Falling back to default m_values")
        return {}


def get_primary_seasonal_m(
    period: float,
    *,
    series_length: Optional[int] = None,
    max_seasonal_period: Optional[int] = None,
) -> List[int]:
    """Return candidate seasonal periods capped by history length."""
    if max_seasonal_period is None:
        env_cap = os.getenv("ARIMA_MAX_SEASONAL_PERIOD")
        try:
            max_seasonal_period = max(1, int(env_cap)) if env_cap is not None else 120
        except ValueError:
            max_seasonal_period = 120
    try:
        seasonal_candidate = max(1, int(round(period)))
    except Exception:
        seasonal_candidate = 1
    if max_seasonal_period:
        seasonal_candidate = min(seasonal_candidate, max_seasonal_period)
    if series_length is not None:
        max_supported = max(1, series_length // 2)
        if max_supported <= 1:
            seasonal_candidate = 1
        else:
            seasonal_candidate = min(seasonal_candidate, max_supported)
    m_values = {1}
    if seasonal_candidate > 1:
        m_values.add(seasonal_candidate)
    return sorted(m_values)


def build_commodity_m_mapping(
    commodity_columns: Iterable[str],
    seasonality_map: Dict[str, int],
    *,
    series_length_map: Optional[Dict[str, int]] = None,
    max_seasonal_period: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Map each commodity to the seasonal periods to evaluate."""
    commodity_m_map: Dict[str, List[int]] = {}
    default_m_values = [1]
    for commodity in commodity_columns:
        series_len = None
        if series_length_map:
            series_len = series_length_map.get(commodity)
        if commodity in seasonality_map:
            period = seasonality_map[commodity]
            m_values = get_primary_seasonal_m(
                period,
                series_length=series_len,
                max_seasonal_period=max_seasonal_period,
            )
            original_period = None
            try:
                original_period = int(round(period))
            except Exception:
                original_period = None
            seasonal_only = [m for m in m_values if m > 1]
            if original_period and seasonal_only:
                capped_value = seasonal_only[-1]
                if capped_value != original_period:
                    print(
                        f"  -> {commodity}: capped seasonal period from {original_period} "
                        f"to {capped_value} (max={max_seasonal_period or '120'}, len={series_len})"
                    )
            elif original_period and not seasonal_only:
                print(
                    f"  -> {commodity}: insufficient history for seasonal period {original_period}; using m=1"
                )
            commodity_m_map[commodity] = m_values
        else:
            commodity_m_map[commodity] = default_m_values
    return commodity_m_map


def choose_seasonal_probe_commodities(
    series_length_map: Dict[str, int],
    eligible_commodities: Optional[Iterable[str]] = None,
    default_probe_size: int = 3,
) -> List[str]:
    """Select representative commodities for SARIMAX probing."""
    if not series_length_map:
        return []
    manual = os.getenv("SARIMAX_PROBE_INCLUDE")
    if manual:
        requested = [c.strip() for c in manual.split(',') if c.strip()]
        if eligible_commodities is not None:
            requested = [c for c in requested if c in eligible_commodities]
        return requested
    try:
        probe_size_env = os.getenv("SARIMAX_PROBE_SIZE")
        probe_size = max(0, int(probe_size_env)) if probe_size_env else default_probe_size
    except ValueError:
        probe_size = default_probe_size
    candidates: List[Tuple[str, int]] = []
    for name, length in series_length_map.items():
        if length and (eligible_commodities is None or name in eligible_commodities):
            candidates.append((name, length))
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in candidates[:probe_size]]


def derive_extrapolated_seasonal_configs(
    arima_eval_df: pd.DataFrame,
    probe_commodities: Iterable[str],
    *,
    max_configs: int = 1,
) -> List[Dict[str, object]]:
    """Summarize top probe seasonal configs for extrapolation."""
    if not probe_commodities or arima_eval_df.empty:
        return []
    probe_mask = arima_eval_df['Commodity'].isin(probe_commodities)
    success_mask = arima_eval_df['Status'].isin(['Success', 'Success-Extrapolated'])
    seasonal_mask = arima_eval_df['m'] > 1
    filtered = arima_eval_df[probe_mask & success_mask & seasonal_mask].copy()
    filtered = filtered[filtered['seasonal_order'].notna()]
    if filtered.empty:
        return []
    filtered['order_tuple'] = filtered['order'].apply(
        lambda val: tuple(val) if isinstance(val, (list, tuple)) else tuple(val)
    )
    filtered['seasonal_order_tuple'] = filtered['seasonal_order'].apply(
        lambda val: tuple(val) if isinstance(val, (list, tuple)) else tuple(val)
    )
    grouped = (
        filtered.groupby(['order_tuple', 'seasonal_order_tuple'])['AIC']
        .mean()
        .reset_index()
        .rename(columns={'AIC': 'mean_aic'})
        .sort_values('mean_aic')
    )
    try:
        max_configs = max(1, int(max_configs))
    except (TypeError, ValueError):
        max_configs = 1
    configs = []
    for _, row in grouped.head(max_configs).iterrows():
        order_tuple = tuple(row['order_tuple'])
        seasonal_tuple = tuple(row['seasonal_order_tuple'])
        if len(seasonal_tuple) < 4 or seasonal_tuple[3] <= 0:
            continue
        configs.append(
            {
                'order': order_tuple,
                'seasonal_order': seasonal_tuple,
                'mean_aic': float(row['mean_aic']),
            }
        )
    return configs


def _commodity_auto_arima_worker(task: Tuple[str, np.ndarray, List[str], List[int]]):
    """Run auto_arima for one commodity (executed in workers)."""
    commodity, values, index_iso, m_values_for_commodity = task
    import pandas as _pd
    import pickle as _pickle
    import pmdarima as pm
    from pmdarima import auto_arima

    results = []
    best_map = {}
    s = _pd.Series(values, index=_pd.to_datetime(index_iso))
    for d in [0, 1]:
        for m in m_values_for_commodity:
            try:
                if m == 1:
                    model = auto_arima(
                        s,
                        d=d,
                        start_p=0,
                        max_p=2,
                        start_q=0,
                        max_q=2,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_order=None,
                        trace=False,
                        n_fits=50,
                    )
                else:
                    model = auto_arima(
                        s,
                        d=d,
                        start_p=0,
                        max_p=2,
                        start_q=0,
                        max_q=2,
                        seasonal=True,
                        m=m,
                        start_P=0,
                        max_P=1,
                        start_Q=0,
                        max_Q=1,
                        D=None,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_order=None,
                        trace=False,
                        n_fits=50,
                    )
                order = model.order
                seasonal_order = (
                    model.seasonal_order if hasattr(model, 'seasonal_order') else (0, 0, 0, 0)
                )
                aic = float(model.aic())
                results.append(
                    {
                        'Commodity': commodity,
                        'd': d,
                        'm': m,
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'AIC': aic,
                        'Status': 'Success',
                        'Error Message': None,
                    }
                )
                try:
                    best_bytes = _pickle.dumps(model)
                except Exception:
                    best_bytes = None
                best_map[(d, m)] = (order, seasonal_order, best_bytes, aic)
            except Exception as exc:
                results.append(
                    {
                        'Commodity': commodity,
                        'd': d,
                        'm': m,
                        'order': None,
                        'seasonal_order': None,
                        'AIC': float('nan'),
                        'Status': 'Failed',
                        'Error Message': str(exc),
                    }
                )
    return (commodity, results, best_map)


def _gcs_save_model(
    model_fit,
    commodity: str,
    *,
    prefix: str = 'models/arima/',
    d_override: Optional[int] = None,
    m_override: Optional[int] = None,
    bucket_name: str = BUCKET_NAME,
) -> Optional[str]:
    try:
        d_val = d_override
        m_val = m_override
        if m_val is None:
            try:
                so = getattr(model_fit, 'seasonal_order', None)
                if so and len(so) == 4 and so[3]:
                    m_val = int(so[3])
                else:
                    m_val = 1
            except Exception:
                m_val = 1
        if d_val is None:
            try:
                order = getattr(model_fit, 'order', None)
                if callable(order):
                    order = order()
                if isinstance(order, (list, tuple)) and len(order) >= 2:
                    d_val = int(order[1])
            except Exception:
                d_val = None
        suffix = (
            f"_d{d_val}_m{m_val}" if (d_val is not None or m_val is not None) else "_dNone_mNone"
        )
        key = prefix + re.sub(r"\s+", '_', commodity) + suffix + '.pkl'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        data = pickle.dumps(model_fit)
        blob.upload_from_string(data, content_type='application/octet-stream')
        return key
    except Exception as exc:
        print(f"  Failed to save model for {commodity} to GCS: {exc}")
        return None


def _is_flatline(
    fvals,
    hist_vals,
    *,
    abs_std_thresh: float = 1e-6,
    rel_range_thresh: float = 1e-3,
) -> bool:
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


def download_model(
    commodity: str,
    *,
    preferred_d: int = 1,
    preferred_m: Optional[int] = None,
    prefix: str = 'models/arima/',
    bucket_name: str = BUCKET_NAME,
):
    """Retrieve a saved ARIMA/SARIMA model from GCS."""
    norm = re.sub(r"\s+", '_', commodity)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    def _find_blob_for_d_m(d, m=None):
        candidates = []
        for b in blobs:
            name = b.name
            if norm in name and f"_d{d}_" in name and name.endswith('.pkl'):
                if m is not None:
                    if f"_m{m}" in name:
                        return b
                else:
                    candidates.append(b)
        return candidates[0] if candidates else None

    m_tries = [preferred_m] if preferred_m is not None else [1, 75, 125, 250, 375]
    d_tries = [preferred_d, 0 if preferred_d == 1 else 1]

    for d_try in d_tries:
        for m_try in m_tries:
            blob = _find_blob_for_d_m(d_try, m_try)
            if blob is None:
                continue
            try:
                data = blob.download_as_bytes()
                obj = pickle.loads(data)
                return (d_try, obj)
            except Exception as exc:
                print(f"  Failed to download/unpickle {blob.name}: {exc}")
                continue
    return (None, None)


def generate_and_save_combined_arima_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: Iterable[str],
    *,
    forecast_steps: int = 250,
    gcs_prefix: str = 'forecast_data/arima_forecast.csv',
    conf_interval_05: bool = True,
    conf_interval_10: bool = True,
    bucket_name: str = BUCKET_NAME,
):
    """Produce ARIMA forecasts and persist combined results to GCS."""
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="No supported index is available.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="No supported index is available.*",
    )
    forecast_series_dict: Dict[str, pd.Series] = {}
    conf_lower_05_dict: Dict[str, pd.Series] = {}
    conf_upper_05_dict: Dict[str, pd.Series] = {}
    conf_lower_10_dict: Dict[str, pd.Series] = {}
    conf_upper_10_dict: Dict[str, pd.Series] = {}
    arima_forecast_results = []
    for commodity in commodity_columns:
        series = prices_df[commodity].dropna()
        if series.empty:
            arima_forecast_results.append(
                {'Commodity': commodity, 'Status': 'Skipped', 'Error Message': 'Empty series'}
            )
            continue
        d_used, model = download_model(commodity, bucket_name=bucket_name)
        if model is None:
            d_used, model = download_model(commodity, preferred_d=0, bucket_name=bucket_name)
        if model is None:
            arima_forecast_results.append(
                {
                    'Commodity': commodity,
                    'Status': 'NoModel',
                    'Error Message': 'No saved ARIMA model found',
                }
            )
            print(f"  {commodity}: No saved model found")
            continue
        print(f"\n  {commodity}: Initial model loaded with d={d_used}")
        try:
            conf_int_05 = None
            conf_int_10 = None
            if hasattr(model, 'forecast'):
                fvals = model.forecast(forecast_steps)
            elif hasattr(model, 'predict'):
                fvals = model.predict(forecast_steps)
            else:
                arima_forecast_results.append(
                    {'Commodity': commodity, 'Status': 'NoForecastMethod'}
                )
                continue
            if hasattr(model, 'get_forecast'):
                try:
                    forecast_obj = model.get_forecast(steps=forecast_steps)
                    if conf_interval_05:
                        conf_int_05 = forecast_obj.conf_int(alpha=0.05)
                    if conf_interval_10:
                        conf_int_10 = forecast_obj.conf_int(alpha=0.10)
                except Exception as exc:
                    print(f"  Could not get confidence intervals for {commodity}: {exc}")
            elif hasattr(model, 'predict'):
                try:
                    if conf_interval_05:
                        _, conf_int_05 = model.predict(
                            n_periods=forecast_steps,
                            return_conf_int=True,
                            alpha=0.05,
                        )
                    if conf_interval_10:
                        _, conf_int_10 = model.predict(
                            n_periods=forecast_steps,
                            return_conf_int=True,
                            alpha=0.10,
                        )
                except Exception as exc:
                    print(f"  Could not get confidence intervals for {commodity}: {exc}")
        except Exception as exc:
            arima_forecast_results.append(
                {
                    'Commodity': commodity,
                    'Status': 'ForecastFailed',
                    'Error Message': str(exc),
                }
            )
            continue
        if _is_flatline(fvals, series.values) and d_used != 0:
            print(f"    → Flatline detected! Attempting to switch from d={d_used} to d=0")
            d0, model0 = download_model(commodity, preferred_d=0, bucket_name=bucket_name)
            if model0 is not None:
                try:
                    if hasattr(model0, 'forecast'):
                        fvals = model0.forecast(forecast_steps)
                    elif hasattr(model0, 'predict'):
                        fvals = model0.predict(forecast_steps)
                    d_used = 0
                    model = model0
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
                                _, conf_int_05 = model.predict(
                                    n_periods=forecast_steps,
                                    return_conf_int=True,
                                    alpha=0.05,
                                )
                            if conf_interval_10:
                                _, conf_int_10 = model.predict(
                                    n_periods=forecast_steps,
                                    return_conf_int=True,
                                    alpha=0.10,
                                )
                        except Exception:
                            pass
                    print("    → Successfully switched to d=0 model")
                except Exception:
                    print(f"    → Failed to switch to d=0 model, keeping d={d_used}")
            else:
                print(f"    → No d=0 model available, keeping d={d_used}")
        elif _is_flatline(fvals, series.values):
            print(f"    → Flatline detected (already using d={d_used})")
        else:
            print(f"    → Forecast looks good with d={d_used}")
        try:
            freq = pd.infer_freq(series.index)
        except Exception:
            freq = None
        if freq is None:
            freq = 'D'
        last_date = series.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]
        try:
            forecast_series = pd.Series(np.array(fvals).astype(float), index=forecast_dates)
            forecast_series_dict[commodity] = forecast_series
            if conf_int_05 is not None:
                try:
                    conf_int_05_array = np.array(conf_int_05)
                    if conf_int_05_array.ndim == 2:
                        conf_lower_05_dict[f"{commodity}_lower_05"] = pd.Series(
                            conf_int_05_array[:, 0], index=forecast_dates
                        )
                        conf_upper_05_dict[f"{commodity}_upper_05"] = pd.Series(
                            conf_int_05_array[:, 1], index=forecast_dates
                        )
                    elif conf_int_05_array.ndim == 1:
                        print(
                            f"  Warning: Unexpected 1D confidence interval structure for {commodity}"
                        )
                except Exception as exc:
                    print(f"  Could not store 5% confidence intervals for {commodity}: {exc}")
            if conf_int_10 is not None:
                try:
                    conf_int_10_array = np.array(conf_int_10)
                    if conf_int_10_array.ndim == 2:
                        conf_lower_10_dict[f"{commodity}_lower_10"] = pd.Series(
                            conf_int_10_array[:, 0], index=forecast_dates
                        )
                        conf_upper_10_dict[f"{commodity}_upper_10"] = pd.Series(
                            conf_int_10_array[:, 1], index=forecast_dates
                        )
                    elif conf_int_10_array.ndim == 1:
                        print(
                            f"  Warning: Unexpected 1D confidence interval structure for {commodity}"
                        )
                except Exception as exc:
                    print(f"  Could not store 10% confidence intervals for {commodity}: {exc}")
            arima_forecast_results.append(
                {
                    'Commodity': commodity,
                    'Status': 'Success',
                    'd_used': d_used,
                    'Has_CI_05': conf_int_05 is not None,
                    'Has_CI_10': conf_int_10 is not None,
                    'Forecast_Start': forecast_dates[0],
                    'Forecast_End': forecast_dates[-1],
                }
            )
            print(f"    ✓ Forecast complete using ARIMA model with d={d_used}")
        except Exception as exc:
            print(f"    ✗ Failed to build forecast series: {exc}")
            arima_forecast_results.append(
                {
                    'Commodity': commodity,
                    'Status': 'SeriesBuildFailed',
                    'Error Message': str(exc),
                }
            )
    arima_forecast_df = pd.DataFrame(arima_forecast_results)
    print("\n" + "=" * 80)
    print("ARIMA Forecast Summary:")
    print("=" * 80)
    print(arima_forecast_df.to_string())
    print("=" * 80)
    historical_df = prices_df[list(commodity_columns)].copy()
    all_forecast_dict: Dict[str, pd.Series] = {}
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
    combined_df_to_save = combined_df.reset_index()
    print(f"\nCombined DataFrame shape: {combined_df.shape}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Historical data points: {len(historical_df)}")
    print(f"Forecast data points: {len(forecast_wide_df)}")
    print(
        f"\nSaving combined ARIMA forecast results to GCS prefix: {gcs_prefix}"
    )
    save_dataframe_to_gcs(
        df=combined_df_to_save,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )
    return combined_df, gcs_prefix


def run_model_training(
    prices_df: pd.DataFrame,
    commodity_columns: Iterable[str],
    *,
    bucket_name: str = BUCKET_NAME,
):
    """Fit ARIMA/SARIMAX models and persist best selections to GCS."""
    print("\n" + "=" * 80)
    print("Running ARIMA Grid Search and Model Training")
    print("=" * 80)
    sarimax_choice = input("\nEnable seasonal SARIMAX exploration? (y/N): ").strip().lower()
    enable_sarimax = sarimax_choice in ("y", "yes")
    print("\n" + "=" * 80)
    print("ARIMA PARAMETER CONFIGURATION")
    print("=" * 80)
    if enable_sarimax:
        print("Running seasonal SARIMAX probe workflow alongside non-seasonal baselines")
        print("d range: [0, 1] | max_p=3 | max_q=3")
        print("Seasonal periods derived from seasonality study; probes will extrapolate configs")
    else:
        print("Running non-seasonal ARIMA (seasonal exploration disabled)")
        print("d range: [0, 1] | max_p=3 | max_q=3")
        print("\nBENEFITS: Faster convergence and lower risk of auto_arima hangs")
        print("          Seasonality can be re-enabled once stability is confirmed")
    series_length_map = {
        commodity: len(prices_df[commodity].dropna())
        for commodity in commodity_columns
    }
    commodity_m_map = {commodity: [1] for commodity in commodity_columns}
    probe_commodities: List[str] = []
    rest_commodities: List[str] = []
    extrapolated_config_cap = 0
    if enable_sarimax:
        seasonality_map = load_seasonality_study(bucket_name=bucket_name)
        commodity_m_map = build_commodity_m_mapping(
            commodity_columns,
            seasonality_map,
            series_length_map=series_length_map,
        )
        eligible_for_probe = [
            commodity
            for commodity in commodity_columns
            if len(commodity_m_map.get(commodity, [1])) > 1
        ]
        probe_commodities = choose_seasonal_probe_commodities(
            series_length_map,
            eligible_commodities=eligible_for_probe,
        )
        rest_commodities = [c for c in commodity_columns if c not in probe_commodities]
        try:
            extrapolated_config_cap = max(
                1, int(os.getenv("SARIMAX_MAX_CONFIGS", "1"))
            )
        except ValueError:
            extrapolated_config_cap = 1
        if probe_commodities:
            print(
                f"\nSeasonal SARIMAX probes ({len(probe_commodities)}): {', '.join(probe_commodities)}"
            )
            print(
                "Using probe results to extrapolate up to "
                f"{extrapolated_config_cap} seasonal configuration(s) to remaining commodities"
            )
        else:
            print(
                "\nNo eligible commodities selected for seasonal probing; proceeding non-seasonal for all."
            )
    else:
        print(
            "\nSeasonal SARIMAX exploration disabled; proceeding non-seasonal for all commodities."
        )
    tasks = []
    probe_set = set(probe_commodities)
    for commodity in commodity_columns:
        series = prices_df[commodity].dropna()
        if series.empty:
            print(f"  Skipping {commodity}: empty after dropna")
            continue
        if commodity in probe_set:
            m_values_for_commodity = commodity_m_map.get(commodity, [1])
        else:
            m_values_for_commodity = [1]
        tasks.append(
            (
                commodity,
                series.values,
                series.index.astype(str).tolist(),
                m_values_for_commodity,
            )
        )
    arima_evaluation_results = []
    best_models: Dict[str, Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...], object, float]]] = {}
    cpu_based_workers = max(1, (os.cpu_count() or 1) - 1)
    configured_cap = int(os.getenv("ARIMA_MAX_WORKERS", "8"))
    max_workers = max(1, min(cpu_based_workers, configured_cap))
    print(
        f"\nRunning per-commodity auto_arima using {max_workers} processes (cap={configured_cap})..."
    )
    import pickle as _pickle
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_commodity_auto_arima_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            commodity, eval_rows, best_map_ser = fut.result()
            arima_evaluation_results.extend(eval_rows)
            best_models[commodity] = {}
            for (d_val, m_val), (order, seasonal_order, best_bytes, aic) in best_map_ser.items():
                try:
                    model_fit = _pickle.loads(best_bytes) if best_bytes is not None else None
                except Exception:
                    model_fit = None
                if model_fit is not None:
                    best_models[commodity][(d_val, m_val)] = (
                        order,
                        seasonal_order,
                        model_fit,
                        aic,
                    )
    arima_evaluation_df = pd.DataFrame(arima_evaluation_results)
    seasonal_configs = derive_extrapolated_seasonal_configs(
        arima_evaluation_df,
        probe_commodities,
        max_configs=extrapolated_config_cap,
    )
    extrapolated_rows = []
    if seasonal_configs and rest_commodities:
        print("\nExtrapolated seasonal configuration candidates:")
        for idx, cfg in enumerate(seasonal_configs, start=1):
            print(
                f"  {idx}. order={cfg['order']} seasonal_order={cfg['seasonal_order']} "
                f"(probe mean AIC={cfg['mean_aic']:.2f})"
            )
        try:
            import pmdarima as pm
        except ImportError:
            pm = None
            print(
                "  Warning: pmdarima not available for extrapolated seasonal fitting."
            )
        if pm is not None:
            for cfg in seasonal_configs:
                order = tuple(cfg['order'])
                seasonal_order = tuple(cfg['seasonal_order'])
                if len(order) < 3 or len(seasonal_order) < 4:
                    continue
                m_key = seasonal_order[3]
                if m_key <= 1:
                    continue
                d_key = order[1]
                for commodity in rest_commodities:
                    series_len = series_length_map.get(commodity, 0)
                    if series_len <= m_key * 2:
                        extrapolated_rows.append(
                            {
                                'Commodity': commodity,
                                'd': d_key,
                                'm': m_key,
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'AIC': float('nan'),
                                'Status': 'Skipped-Extrapolated',
                                'Error Message': f'Insufficient history for m={m_key}',
                                'Source': 'Extrapolated',
                            }
                        )
                        continue
                    commodity_models = best_models.setdefault(commodity, {})
                    if (d_key, m_key) in commodity_models:
                        continue
                    series = prices_df[commodity].dropna()
                    if series.empty:
                        extrapolated_rows.append(
                            {
                                'Commodity': commodity,
                                'd': d_key,
                                'm': m_key,
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'AIC': float('nan'),
                                'Status': 'Skipped-Extrapolated',
                                'Error Message': 'Empty series',
                                'Source': 'Extrapolated',
                            }
                        )
                        continue
                    try:
                        model = pm.arima.ARIMA(
                            order=order,
                            seasonal_order=seasonal_order,
                            suppress_warnings=True,
                            error_action='ignore',
                            maxiter=200,
                        )
                        model.fit(series)
                        aic_val = float(model.aic())
                        commodity_models[(d_key, m_key)] = (
                            order,
                            seasonal_order,
                            model,
                            aic_val,
                        )
                        extrapolated_rows.append(
                            {
                                'Commodity': commodity,
                                'd': d_key,
                                'm': m_key,
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'AIC': aic_val,
                                'Status': 'Success-Extrapolated',
                                'Error Message': None,
                                'Source': 'Extrapolated',
                            }
                        )
                    except Exception as exc:
                        err_msg = str(exc)
                        if len(err_msg) > 160:
                            err_msg = err_msg[:157] + '...'
                        extrapolated_rows.append(
                            {
                                'Commodity': commodity,
                                'd': d_key,
                                'm': m_key,
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'AIC': float('nan'),
                                'Status': 'Failed-Extrapolated',
                                'Error Message': err_msg,
                                'Source': 'Extrapolated',
                            }
                        )
    elif probe_commodities:
        print("\nNo seasonal configurations from probes qualified for extrapolation.")
    if extrapolated_rows:
        extrapolated_df = pd.DataFrame(extrapolated_rows)
        arima_evaluation_df = pd.concat(
            [arima_evaluation_df, extrapolated_df],
            ignore_index=True,
        )
    if 'Source' not in arima_evaluation_df.columns:
        arima_evaluation_df['Source'] = arima_evaluation_df.get('Source', None)
    print("\nARIMA Evaluation Results DataFrame Head:")
    print(arima_evaluation_df.head())
    arima_fitted_models = {}
    print("\nSaving selected best-fit ARIMA/SARIMA models to GCS...")
    for commodity, dmap in best_models.items():
        arima_fitted_models[commodity] = {}
        for (d_val, m_val), (order, seasonal_order, model_fit, aic) in dmap.items():
            try:
                key = _gcs_save_model(
                    model_fit,
                    commodity,
                    d_override=d_val,
                    m_override=m_val,
                    bucket_name=bucket_name,
                )
                if key:
                    model_type = "SARIMA" if m_val > 1 else "ARIMA"
                    print(
                        f"  Saved {commodity} {model_type} d={d_val} m={m_val} "
                        f"order={order} seasonal={seasonal_order} AIC={aic:.2f} -> {key}"
                    )
                arima_fitted_models[commodity][(d_val, m_val)] = model_fit
            except Exception as exc:
                print(
                    f"  Failed to save selected model for {commodity} d={d_val} m={m_val}: {exc}"
                )
    print("\n" + "=" * 80)
    print("Model Training Complete")
    print("=" * 80)
    return {
        'evaluation_df': arima_evaluation_df,
        'fitted_models': arima_fitted_models,
    }
