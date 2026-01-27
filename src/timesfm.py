"""TimesFM forecasting helper aligned with the DSS workflow."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

try:
    from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
except ImportError as exc:  # pragma: no cover - clearer guidance for operators
    raise ImportError(
        "timesfm package is required; install with `pip install timesfm`."
    ) from exc

from .file_utils import save_dataframe_to_gcs

_DEFAULT_BUCKET = os.getenv("GCS_BUCKET_NAME", "crystal-dss")
_TIMESFM_REPO = os.getenv("TIMESFM_REPO_ID", "google/timesfm-2.0-500m-pytorch")


def _infer_frequency(index: pd.Index) -> str:
    """Best-effort frequency inference with daily fallback."""
    try:
        freq = pd.infer_freq(index)
    except Exception:  # pragma: no cover - pandas-specific edge cases
        freq = None
    return freq or "D"


def _prepare_long_frame(prices_df: pd.DataFrame, commodities: Sequence[str]) -> pd.DataFrame:
    """Transform wide price data into TimesFM-friendly long format."""
    working = prices_df[commodities].copy()
    working.sort_index(inplace=True)
    reset_df = working.reset_index()
    ds_column = reset_df.columns[0]
    reset_df.rename(columns={ds_column: "ds"}, inplace=True)
    melted = reset_df.melt(id_vars="ds", var_name="unique_id", value_name="y")
    melted.dropna(subset=["y"], inplace=True)
    melted["ds"] = pd.to_datetime(melted["ds"], utc=False)
    melted["ds"] = melted["ds"].dt.tz_localize(None)
    melted.sort_values(["unique_id", "ds"], inplace=True)
    melted.reset_index(drop=True, inplace=True)
    return melted


def _quantile_column_map(columns: Sequence[str]) -> Dict[float, str]:
    """Return mapping from quantile value to TimesFM column name."""
    mapping: Dict[float, str] = {}
    prefix = "timesfm-q-"
    for col in columns:
        if not col.startswith(prefix):
            continue
        try:
            quantile = float(col.replace(prefix, ""))
        except ValueError:
            continue
        mapping[round(quantile, 4)] = col
    return mapping


def _select_quantile_column(mapping: Dict[float, str], target: float) -> Optional[str]:
    """Pick the available quantile column closest to the requested target."""
    if not mapping:
        return None
    closest = min(mapping.keys(), key=lambda value: abs(value - target))
    return mapping.get(closest)


def _pivot_forecast_values(forecast_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot a forecast column into wide format keyed by unique_id."""
    if column not in forecast_df.columns:
        return pd.DataFrame()
    wide = forecast_df.pivot(index="ds", columns="unique_id", values=column)
    wide.sort_index(inplace=True)
    wide.index = pd.to_datetime(wide.index, utc=False).tz_localize(None)
    return wide


def _build_timesfm_model(forecast_steps: int) -> TimesFm:
    """Instantiate a TimesFM model for the requested horizon."""
    hparams = TimesFmHparams(backend="cpu", horizon_len=forecast_steps)
    checkpoint = TimesFmCheckpoint(huggingface_repo_id=_TIMESFM_REPO)
    return TimesFm(hparams=hparams, checkpoint=checkpoint)


def generate_and_save_timesfm_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: Iterable[str],
    *,
    forecast_steps: int = 250,
    gcs_prefix: str = "forecast_data/timesfm_forecast.csv",
    conf_interval_05: bool = True,
    conf_interval_10: bool = True,
    bucket_name: Optional[str] = None,
    num_jobs: int = 4,
) -> Tuple[pd.DataFrame, str]:
    """Generate TimesFM forecasts and persist combined output to GCS."""
    if prices_df is None or prices_df.empty:
        raise ValueError("prices_df must contain historical data")

    commodity_list = list(commodity_columns)
    if not commodity_list:
        raise ValueError("commodity_columns must list at least one series")

    bucket = bucket_name or _DEFAULT_BUCKET

    history_df = prices_df[commodity_list].copy()
    history_df.sort_index(inplace=True)
    if history_df.index.has_duplicates:
        history_df = history_df.loc[~history_df.index.duplicated(keep="last")]

    long_frame = _prepare_long_frame(history_df, commodity_list)
    freq = _infer_frequency(history_df.index)

    model = _build_timesfm_model(forecast_steps)
    quantiles = sorted({0.05, 0.10, 0.90, 0.95})
    forecast_df = model.forecast_on_df(
        inputs=long_frame,
        freq=freq,
        value_name="y",
        quantiles=quantiles,
        num_jobs=num_jobs,
    )

    forecast_df = forecast_df.copy()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], utc=False).dt.tz_localize(None)

    quantile_map = _quantile_column_map(forecast_df.columns)
    lower_05_col = _select_quantile_column(quantile_map, 0.05)
    upper_05_col = _select_quantile_column(quantile_map, 0.95)
    lower_10_col = _select_quantile_column(quantile_map, 0.10)
    upper_10_col = _select_quantile_column(quantile_map, 0.90)

    point_wide = _pivot_forecast_values(forecast_df, "timesfm")
    lower_05_wide = _pivot_forecast_values(forecast_df, lower_05_col) if lower_05_col else pd.DataFrame()
    upper_05_wide = _pivot_forecast_values(forecast_df, upper_05_col) if upper_05_col else pd.DataFrame()
    lower_10_wide = _pivot_forecast_values(forecast_df, lower_10_col) if lower_10_col else pd.DataFrame()
    upper_10_wide = _pivot_forecast_values(forecast_df, upper_10_col) if upper_10_col else pd.DataFrame()

    combined_frames: Dict[str, pd.Series] = {}

    if not point_wide.empty:
        for commodity in point_wide.columns:
            series = point_wide[commodity].iloc[:forecast_steps]
            combined_frames[commodity] = series.rename(commodity)

    if conf_interval_05 and not lower_05_wide.empty and not upper_05_wide.empty:
        for commodity in lower_05_wide.columns:
            lower_series = lower_05_wide[commodity].iloc[:forecast_steps]
            upper_series = upper_05_wide[commodity].iloc[:forecast_steps]
            combined_frames[f"{commodity}_lower_05"] = lower_series.rename(f"{commodity}_lower_05")
            combined_frames[f"{commodity}_upper_05"] = upper_series.rename(f"{commodity}_upper_05")

    if conf_interval_10 and not lower_10_wide.empty and not upper_10_wide.empty:
        for commodity in lower_10_wide.columns:
            lower_series = lower_10_wide[commodity].iloc[:forecast_steps]
            upper_series = upper_10_wide[commodity].iloc[:forecast_steps]
            combined_frames[f"{commodity}_lower_10"] = lower_series.rename(f"{commodity}_lower_10")
            combined_frames[f"{commodity}_upper_10"] = upper_series.rename(f"{commodity}_upper_10")

    forecast_wide = pd.DataFrame(combined_frames) if combined_frames else pd.DataFrame(columns=history_df.columns)
    forecast_wide.index.name = "Date"

    combined_df = pd.concat([history_df, forecast_wide], axis=0, join="outer")
    combined_df.sort_index(inplace=True)
    combined_df.index.name = "Date"

    if not forecast_wide.empty:
        horizon_msg = (
            f"TimesFM forecast horizon: {forecast_wide.index.min().date()} -> "
            f"{forecast_wide.index.max().date()} ({forecast_wide.shape[0]} steps)"
        )
        print(f"\n{horizon_msg}")

    combined_to_save = combined_df.reset_index()
    print(f"\nSaving TimesFM forecast results to GCS prefix: {gcs_prefix}")
    save_dataframe_to_gcs(
        df=combined_to_save,
        bucket_name=bucket,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )

    return combined_df, gcs_prefix


__all__ = ["generate_and_save_timesfm_forecast"]
