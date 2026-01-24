"""Prophet forecasting utilities aligned with ARIMA workflow."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from prophet import Prophet

from .file_utils import save_dataframe_to_gcs

_DEFAULT_BUCKET = os.getenv("GCS_BUCKET_NAME", "crystal-dss")
_CONF_RATIO_90_TO_95 = 1.645 / 1.96


def _prepare_prophet_frame(series: pd.Series) -> pd.DataFrame:
    """Return DataFrame with Prophet-friendly schema."""
    frame = pd.DataFrame({"ds": series.index, "y": series.values})
    frame.dropna(subset=["ds", "y"], inplace=True)
    frame["ds"] = pd.to_datetime(frame["ds"], utc=False)
    frame.sort_values("ds", inplace=True)
    frame.drop_duplicates(subset=["ds"], keep="last", inplace=True)
    frame["y"] = frame["y"].astype(float)
    return frame


def _infer_frequency(index: pd.Index) -> str:
    """Best-effort frequency inference with daily fallback."""
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    return freq or "D"


def _scale_intervals_to_90(
    yhat: pd.Series,
    lower_95: pd.Series,
    upper_95: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Approximate 90% bounds by scaling 95% deltas under near-normal assumption."""
    upper_delta = (upper_95 - yhat).clip(lower=0).fillna(0)
    lower_delta = (yhat - lower_95).clip(lower=0).fillna(0)
    factor = _CONF_RATIO_90_TO_95
    lower_90 = yhat - lower_delta * factor
    upper_90 = yhat + upper_delta * factor
    return lower_90.rename(None), upper_90.rename(None)


def _prophet_forecast_single(
    commodity: str,
    series: pd.Series,
    forecast_steps: int,
    conf_interval_05: bool,
    conf_interval_10: bool,
) -> Dict[str, pd.Series]:
    """Fit Prophet for one series and return forecast + interval bands."""
    frame = _prepare_prophet_frame(series)
    if frame.empty or len(frame) < 5:
        raise ValueError("insufficient observations for Prophet forecast")

    freq = _infer_frequency(frame["ds"])
    last_date = frame["ds"].max()

    model = Prophet(interval_width=0.95, daily_seasonality=False)
    model.fit(frame)

    future = model.make_future_dataframe(periods=forecast_steps, freq=freq)
    forecast = model.predict(future)

    future_forecast = forecast[forecast["ds"] > last_date].copy()
    if future_forecast.empty:
        raise ValueError("no future rows produced by Prophet")

    yhat = pd.Series(future_forecast["yhat"].values, index=future_forecast["ds"], name=commodity)

    bands: Dict[str, pd.Series] = {"forecast": yhat}

    if conf_interval_05:
        lower_95 = pd.Series(
            future_forecast["yhat_lower"].values,
            index=future_forecast["ds"],
            name=f"{commodity}_lower_05",
        )
        upper_95 = pd.Series(
            future_forecast["yhat_upper"].values,
            index=future_forecast["ds"],
            name=f"{commodity}_upper_05",
        )
        bands["lower_05"] = lower_95
        bands["upper_05"] = upper_95

        if conf_interval_10:
            lower_90, upper_90 = _scale_intervals_to_90(yhat, lower_95, upper_95)
            lower_90.rename(f"{commodity}_lower_10", inplace=True)
            upper_90.rename(f"{commodity}_upper_10", inplace=True)
            bands["lower_10"] = lower_90
            bands["upper_10"] = upper_90
    elif conf_interval_10:
        model_90 = Prophet(interval_width=0.90, daily_seasonality=False)
        model_90.fit(frame)
        forecast_90 = model_90.predict(future)
        future_forecast_90 = forecast_90[forecast_90["ds"] > last_date]
        lower_90 = pd.Series(
            future_forecast_90["yhat_lower"].values,
            index=future_forecast_90["ds"],
            name=f"{commodity}_lower_10",
        )
        upper_90 = pd.Series(
            future_forecast_90["yhat_upper"].values,
            index=future_forecast_90["ds"],
            name=f"{commodity}_upper_10",
        )
        bands["lower_10"] = lower_90
        bands["upper_10"] = upper_90

    return bands


def generate_and_save_prophet_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: Iterable[str],
    *,
    forecast_steps: int = 250,
    gcs_prefix: str = "forecast_data/fb_prophet_forecast.csv",
    conf_interval_05: bool = True,
    conf_interval_10: bool = True,
    bucket_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Generate Prophet forecasts for all commodities and persist to GCS."""
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df must contain historical data")

    bucket = bucket_name or _DEFAULT_BUCKET

    historical_df = prices_df[list(commodity_columns)].copy()
    forecast_series: Dict[str, pd.Series] = {}
    lower_05_series: Dict[str, pd.Series] = {}
    upper_05_series: Dict[str, pd.Series] = {}
    lower_10_series: Dict[str, pd.Series] = {}
    upper_10_series: Dict[str, pd.Series] = {}
    run_summaries = []

    for commodity in commodity_columns:
        series = prices_df[commodity].dropna()
        if series.empty:
            run_summaries.append({"Commodity": commodity, "Status": "Skipped", "Reason": "Empty series"})
            continue

        try:
            bands = _prophet_forecast_single(
                commodity,
                series,
                forecast_steps,
                conf_interval_05,
                conf_interval_10,
            )
            forecast_series[commodity] = bands["forecast"]
            if "lower_05" in bands:
                lower_05_series[bands["lower_05"].name] = bands["lower_05"]
                upper_05_series[bands["upper_05"].name] = bands["upper_05"]
            if "lower_10" in bands:
                lower_10_series[bands["lower_10"].name] = bands["lower_10"]
                upper_10_series[bands["upper_10"].name] = bands["upper_10"]
            run_summaries.append({"Commodity": commodity, "Status": "Success"})
        except Exception as exc:
            run_summaries.append({"Commodity": commodity, "Status": "Failed", "Reason": str(exc)})
            continue

    summary_df = pd.DataFrame(run_summaries)
    if not summary_df.empty:
        print("\nProphet forecasting summary:")
        print(summary_df.to_string(index=False))

    combined_forecast_frames: Dict[str, pd.Series] = {}
    combined_forecast_frames.update(forecast_series)
    combined_forecast_frames.update(lower_05_series)
    combined_forecast_frames.update(upper_05_series)
    combined_forecast_frames.update(lower_10_series)
    combined_forecast_frames.update(upper_10_series)

    if combined_forecast_frames:
        forecast_wide_df = pd.DataFrame(combined_forecast_frames)
    else:
        forecast_wide_df = pd.DataFrame(columns=historical_df.columns)

    combined_df = pd.concat([historical_df, forecast_wide_df], axis=0, join="outer")
    combined_df.sort_index(inplace=True)
    combined_df.index.name = "Date"
    combined_df_to_save = combined_df.reset_index()

    print(f"\nSaving combined Prophet forecast results to GCS prefix: {gcs_prefix}")
    save_dataframe_to_gcs(
        df=combined_df_to_save,
        bucket_name=bucket,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )

    return combined_df, gcs_prefix


__all__ = ["generate_and_save_prophet_forecast"]

