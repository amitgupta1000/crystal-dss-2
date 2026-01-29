"""Prophet forecasting with external regressors."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from prophet import Prophet

from .fbprophet import _DEFAULT_BUCKET
from .fbprophet import _infer_frequency, _prepare_prophet_frame, _scale_intervals_to_90
from .file_utils import save_dataframe_to_gcs

_DEFAULT_BUCKET = os.getenv("GCS_BUCKET_NAME", "crystal-dss")


def _filter_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Return only columns that exist in df with numeric dtype."""
    unique_cols = []
    seen = set()
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            unique_cols.append(col)
    return unique_cols


def _forecast_exogenous_series(
    prices_df: pd.DataFrame,
    regressors: Iterable[str],
    forecast_steps: int,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Forecast each regressor individually to supply future exogenous values."""
    forecasts: Dict[str, pd.Series] = {}
    summaries: List[Dict[str, str]] = []

    for name in regressors:
        series = prices_df[name].dropna()
        if series.empty or len(series) < 5:
            summaries.append({"Series": name, "Status": "Skipped", "Reason": "Insufficient history"})
            continue

        try:
            frame = _prepare_prophet_frame(series)
            if frame.empty:
                summaries.append({"Series": name, "Status": "Skipped", "Reason": "No valid rows"})
                continue

            freq = _infer_frequency(frame["ds"])
            last_date = frame["ds"].max()

            base_model = Prophet(interval_width=0.80, daily_seasonality=False)
            base_model.fit(frame)

            future = base_model.make_future_dataframe(periods=forecast_steps, freq=freq)
            forecast = base_model.predict(future)

            future_block = forecast[forecast["ds"] > last_date].copy()
            if future_block.empty:
                future_block = forecast.tail(forecast_steps).copy()

            future_block["ds"] = pd.to_datetime(future_block["ds"]).dt.tz_localize(None)
            future_block.drop_duplicates(subset="ds", keep="last", inplace=True)
            future_block = future_block.tail(forecast_steps)

            forecasts[name] = pd.Series(
                future_block["yhat"].values,
                index=future_block["ds"],
                name=name,
            )
            summaries.append({"Series": name, "Status": "Success"})
        except Exception as exc:  # pragma: no cover - defensive
            summaries.append({"Series": name, "Status": "Failed", "Reason": str(exc)})

    future_df = pd.DataFrame(forecasts) if forecasts else pd.DataFrame()
    if not future_df.empty:
        future_index = pd.to_datetime(future_df.index)
        if getattr(future_index, "tz", None) is not None:
            future_index = future_index.tz_localize(None)
        future_df.index = future_index
    future_df.index.name = "Date"
    future_df.sort_index(inplace=True)
    return future_df, summaries


def _build_regressor_frame(
    history: pd.DataFrame,
    future: pd.DataFrame,
    index: pd.DatetimeIndex,
    regressor_names: Sequence[str],
) -> pd.DataFrame:
    """Align historical and future regressors to the requested index."""
    frames = []
    if not history.empty:
        frames.append(history[regressor_names])
    if not future.empty:
        available = [name for name in regressor_names if name in future.columns]
        if available:
            frames.append(future[available])

    aligned = pd.concat(frames, axis=0) if frames else pd.DataFrame(columns=regressor_names)
    aligned = aligned.reindex(index)

    for name in regressor_names:
        if name not in aligned.columns:
            last_known = history[name].dropna().iloc[-1] if name in history.columns and not history[name].dropna().empty else 0.0
            aligned[name] = last_known

    aligned = aligned[regressor_names]
    if not aligned.empty:
        aligned = aligned.fillna(method="ffill").fillna(method="bfill")
    return aligned


def _prophet_with_covariates_single(
    prices_df: pd.DataFrame,
    target: str,
    regressors: Sequence[str],
    future_regressors: pd.DataFrame,
    forecast_steps: int,
    conf_interval_05: bool,
    conf_interval_10: bool,
) -> Dict[str, pd.Series]:
    """Fit Prophet with regressors for one target commodity."""
    working_prices = prices_df.copy()
    if working_prices.index.has_duplicates:
        working_prices = working_prices.loc[~working_prices.index.duplicated(keep="last")]

    series = working_prices[target].dropna()
    if series.empty or len(series) < 10:
        raise ValueError("insufficient observations for target forecast")

    frame = _prepare_prophet_frame(series)
    if frame.empty:
        raise ValueError("no valid rows after preprocessing")

    frame.set_index("ds", inplace=True)
    freq = _infer_frequency(frame.index)
    regressor_names = [name for name in regressors if name != target]

    if not regressor_names:
        raise ValueError("no usable regressors supplied")

    reg_history_source = working_prices[regressor_names]
    if reg_history_source.index.has_duplicates:
        reg_history_source = reg_history_source.loc[~reg_history_source.index.duplicated(keep="last")]
    reg_history = reg_history_source.reindex(frame.index)
    reg_history = reg_history.fillna(method="ffill").fillna(method="bfill")

    train_df = frame.join(reg_history)
    train_df.dropna(inplace=True)
    if train_df.empty:
        raise ValueError("training data empty after aligning regressors")

    model = Prophet(
        interval_width=0.95,
        daily_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.75,
    )
    for reg_name in regressor_names:
        model.add_regressor(reg_name)

    train_df_reset = train_df.reset_index().rename(columns={"index": "ds"})
    model.fit(train_df_reset)

    future_dates = model.make_future_dataframe(periods=forecast_steps, freq=freq)
    future_dates.set_index("ds", inplace=True)
    aligned_regressors = _build_regressor_frame(
        history=reg_history,
        future=future_regressors,
        index=future_dates.index,
        regressor_names=regressor_names,
    )

    if aligned_regressors.isnull().any().any():
        aligned_regressors = aligned_regressors.fillna(method="ffill").fillna(method="bfill")

    future_dates[regressor_names] = aligned_regressors[regressor_names]
    future_with_regressors = future_dates.reset_index()

    forecast = model.predict(future_with_regressors)
    forecast_future = forecast[forecast["ds"] > frame.index.max()].copy()
    if forecast_future.empty:
        forecast_future = forecast.tail(forecast_steps).copy()

    forecast_future["ds"] = pd.to_datetime(forecast_future["ds"]).dt.tz_localize(None)
    forecast_future.drop_duplicates(subset="ds", keep="last", inplace=True)
    forecast_future = forecast_future.tail(forecast_steps)
    if forecast_future.empty:
        raise ValueError("model produced no future rows")

    yhat = pd.Series(forecast_future["yhat"].values, index=forecast_future["ds"], name=target)
    bands: Dict[str, pd.Series] = {"forecast": yhat}

    if conf_interval_05:
        lower_95 = pd.Series(
            forecast_future["yhat_lower"].values,
            index=forecast_future["ds"],
            name=f"{target}_lower_05",
        )
        upper_95 = pd.Series(
            forecast_future["yhat_upper"].values,
            index=forecast_future["ds"],
            name=f"{target}_upper_05",
        )
        bands["lower_05"] = lower_95
        bands["upper_05"] = upper_95

        if conf_interval_10:
            lower_90, upper_90 = _scale_intervals_to_90(yhat, lower_95, upper_95)
            lower_90.rename(f"{target}_lower_10", inplace=True)
            upper_90.rename(f"{target}_upper_10", inplace=True)
            bands["lower_10"] = lower_90
            bands["upper_10"] = upper_90
    elif conf_interval_10:
        model_90 = Prophet(
            interval_width=0.90,
            daily_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.75,
        )
        for reg_name in regressor_names:
            model_90.add_regressor(reg_name)
        model_90.fit(train_df_reset)
        forecast_90 = model_90.predict(future_with_regressors)
        forecast_future_90 = forecast_90[forecast_90["ds"] > frame.index.max()]
        lower_90 = pd.Series(
            forecast_future_90["yhat_lower"].values,
            index=forecast_future_90["ds"],
            name=f"{target}_lower_10",
        )
        upper_90 = pd.Series(
            forecast_future_90["yhat_upper"].values,
            index=forecast_future_90["ds"],
            name=f"{target}_upper_10",
        )
        bands["lower_10"] = lower_90
        bands["upper_10"] = upper_90

    return bands


def generate_and_save_prophet_covariate_forecast(
    prices_df: pd.DataFrame,
    target_commodities: Optional[Iterable[str]] = None,
    other_commodities: Optional[Iterable[str]] = None,
    *,
    forecast_steps: int = 250,
    gcs_prefix: str = "forecast_data/fb_prophet_covariates_forecast.csv",
    conf_interval_05: bool = True,
    conf_interval_10: bool = True,
    bucket_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Generate Prophet forecasts using external regressors and persist them."""
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df must contain historical data")

    prices_df = prices_df.copy()
    if prices_df.index.has_duplicates:
        prices_df = prices_df[~prices_df.index.duplicated(keep="last")]

    bucket = bucket_name or _DEFAULT_BUCKET

    working_prices_df = prices_df.copy()
    if working_prices_df.index.has_duplicates:
        working_prices_df = working_prices_df.loc[~working_prices_df.index.duplicated(keep="last")]

    numeric_columns = _filter_numeric_columns(working_prices_df, list(working_prices_df.columns))

    target_list = list(target_commodities) if target_commodities is not None else numeric_columns
    valid_targets = _filter_numeric_columns(working_prices_df, target_list)

    regressor_input = list(other_commodities) if other_commodities is not None else numeric_columns
    regressor_pool = _filter_numeric_columns(working_prices_df, regressor_input)

    if not valid_targets:
        raise ValueError("no valid target commodities available")
    if not regressor_pool:
        raise ValueError("no valid exogenous regressors available")

    future_regressors, regressor_summary = _forecast_exogenous_series(
        prices_df=working_prices_df,
        regressors=regressor_pool,
        forecast_steps=forecast_steps,
    )

    if regressor_summary:
        print("\nProphet covariate forecasting summary:")
        print(pd.DataFrame(regressor_summary).to_string(index=False))

    forecasts: Dict[str, pd.Series] = {}
    lower_05: Dict[str, pd.Series] = {}
    upper_05: Dict[str, pd.Series] = {}
    lower_10: Dict[str, pd.Series] = {}
    upper_10: Dict[str, pd.Series] = {}
    run_summaries: List[Dict[str, str]] = []

    for target in valid_targets:
        try:
            target_regressors = [name for name in regressor_pool if name != target]
            if not target_regressors:
                run_summaries.append({"Commodity": target, "Status": "Failed", "Reason": "No regressors available"})
                continue
            bands = _prophet_with_covariates_single(
                prices_df=working_prices_df,
                target=target,
                regressors=target_regressors,
                future_regressors=future_regressors,
                forecast_steps=forecast_steps,
                conf_interval_05=conf_interval_05,
                conf_interval_10=conf_interval_10,
            )
            forecasts[target] = bands["forecast"]
            if "lower_05" in bands:
                lower_05[bands["lower_05"].name] = bands["lower_05"]
                upper_05[bands["upper_05"].name] = bands["upper_05"]
            if "lower_10" in bands:
                lower_10[bands["lower_10"].name] = bands["lower_10"]
                upper_10[bands["upper_10"].name] = bands["upper_10"]
            run_summaries.append({"Commodity": target, "Status": "Success"})
        except Exception as exc:  # pragma: no cover - defensive
            run_summaries.append({"Commodity": target, "Status": "Failed", "Reason": str(exc)})

    if run_summaries:
        print("\nProphet target forecasting summary:")
        print(pd.DataFrame(run_summaries).to_string(index=False))

    historical_df = working_prices_df[valid_targets].copy()
    combined_frames: Dict[str, pd.Series] = {}
    combined_frames.update(forecasts)
    combined_frames.update(lower_05)
    combined_frames.update(upper_05)
    combined_frames.update(lower_10)
    combined_frames.update(upper_10)
    forecast_wide = pd.DataFrame(combined_frames) if combined_frames else pd.DataFrame(columns=historical_df.columns)
    if not forecast_wide.empty:
        future_index = pd.to_datetime(forecast_wide.index)
        if getattr(future_index, "tz", None) is not None:
            future_index = future_index.tz_localize(None)
        forecast_wide.index = future_index
    combined_df = pd.concat([historical_df, forecast_wide], axis=0, join="outer")
    combined_df.sort_index(inplace=True)
    combined_df.index.name = "Date"

    if not forecast_wide.empty:
        horizon_msg = (
            f"Prophet covariate forecast horizon: "
            f"{forecast_wide.index.min().date()} -> {forecast_wide.index.max().date()}"
            f" ({forecast_wide.shape[0]} steps)"
        )
        print(f"\n{horizon_msg}")
    combined_to_save = combined_df.reset_index()

    print(f"\nSaving Prophet covariate forecast results to GCS prefix: {gcs_prefix}")
    save_dataframe_to_gcs(
        df=combined_to_save,
        bucket_name=bucket,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )

    return combined_df, gcs_prefix


__all__ = ["generate_and_save_prophet_covariate_forecast"]
