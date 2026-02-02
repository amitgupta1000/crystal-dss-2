"""TimesFM forecasting helper aligned with the DSS workflow."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

try:
    from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
except ImportError as exc:  # pragma: no cover - clearer guidance for operators
    raise ImportError(
        "timesfm package is required; install with `pip install timesfm`."
    ) from exc

from .file_utils import save_dataframe_to_gcs, filter_and_fill_series

_DEFAULT_BUCKET = os.getenv("GCS_BUCKET_NAME", "crystal-dss")
_TIMESFM_REPO = os.getenv("TIMESFM_REPO_ID", "google/timesfm-2.0-500m-pytorch")


def _infer_frequency(index: pd.Index) -> str:
    """Best-effort frequency inference with daily fallback."""
    try:
        freq = pd.infer_freq(index)
    except Exception:  # pragma: no cover - pandas-specific edge cases
        freq = None
    if not freq:
        return "D"

    # Normalize pandas frequency strings to a single-letter code expected
    # by downstream code / TimesFM examples (e.g. 'W-SUN' -> 'W').
    # Map common pandas freq codes to simplified letters.
    code = freq.upper()
    if code.startswith("W"):
        return "W"
    if code.startswith("M"):
        return "M"
    if code.startswith("Q"):
        return "Q"
    if code.startswith("A") or code.startswith("Y"):
        return "Y"
    if code.startswith("D") or code.startswith("B"):
        return "D"
    if code.startswith("H"):
        return "H"
    if code.startswith("T") or code.startswith("MIN"):
        return "T"
    if code.startswith("S"):
        return "S"

    # Default to daily if unknown
    return "D"


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
    backend = os.getenv("TIMESFM_BACKEND", "cpu")

    # Use explicit hparams when the well-known 500m checkpoint is requested.
    # The README for that checkpoint requires these five parameters to match
    # the pretrained architecture.
    hparams_kwargs = {"backend": backend, "horizon_len": forecast_steps}
    if "500m" in _TIMESFM_REPO or _TIMESFM_REPO.endswith("timesfm-2.0-500m-pytorch"):
        hparams_kwargs.update(
            {
                "input_patch_len": 32,
                "output_patch_len": 128,
                "num_layers": 50,
                "model_dims": 1280,
                "use_positional_embedding": False,
            }
        )

    hparams = TimesFmHparams(**hparams_kwargs)
    checkpoint = TimesFmCheckpoint(huggingface_repo_id=_TIMESFM_REPO)
    try:
        return TimesFm(hparams=hparams, checkpoint=checkpoint)
    except Exception as exc:  # pragma: no cover - runtime mismatch between weights and code
        # Attempt to surface a helpful error message when checkpoint and installed
        # `timesfm` package versions/architectures mismatch (common cause of
        # unexpected state_dict keys when loading pretrained weights).
        import importlib

        try:
            timesfm_mod = importlib.import_module("timesfm")
            timesfm_version = getattr(timesfm_mod, "__version__", "unknown")
        except Exception:
            timesfm_version = "(unable to determine installed timesfm version)"

        hint_lines = [
            f"Failed to instantiate TimesFm using checkpoint: {_TIMESFM_REPO}",
            f"Installed timesfm package version: {timesfm_version}",
            "This usually means the checkpoint architecture doesn't match the installed",
            "timesfm package. Possible remedies:",
            "  1) Set the environment variable TIMESFM_REPO_ID to a repo compatible",
            "     with your installed `timesfm` package (for example an older checkpoint).",
            "  2) Update/downgrade the `timesfm` package so its model code matches",
            "     the checkpoint downloaded (edit requirements.txt and reinstall).",
            "  3) Avoid using the pretrained checkpoint by constructing TimesFm without",
            "     the checkpoint (not recommended if you need pretrained weights).",
            "If you'd like, I can try to modify the code to fall back more gracefully.",
        ]

        detailed = "\n".join(hint_lines)
        raise RuntimeError(f"TimesFm model instantiation failed: {exc}\n\n{detailed}") from exc


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

    # Filter commodities and fill missing values
    keep_cols, filled_df, dropped = filter_and_fill_series(prices_df, commodity_list, min_non_nulls=1000)
    if not keep_cols:
        raise ValueError("No commodity series have >=1000 non-null values; cannot generate TimesFM forecast")
    if dropped:
        print(f"TimesFM: Dropped {len(dropped)} commodities due to insufficient data: {dropped[:10]}{'...' if len(dropped)>10 else ''}")

    history_df = filled_df[keep_cols].copy()
    history_df.sort_index(inplace=True)
    if history_df.index.has_duplicates:
        history_df = history_df.loc[~history_df.index.duplicated(keep="last")]

    long_frame = _prepare_long_frame(history_df, commodity_list)
    freq = _infer_frequency(history_df.index)

    model = _build_timesfm_model(forecast_steps)
    quantiles = sorted({0.05, 0.10, 0.90, 0.95})
    # Some installed `timesfm` versions may not accept the `quantiles` kwarg
    # on `forecast_on_df`. Attempt with quantiles and fall back to calling
    # without if the installed package raises a TypeError about the argument.
    try:
        forecast_df = model.forecast_on_df(
            inputs=long_frame,
            freq=freq,
            value_name="y",
            quantiles=quantiles,
            num_jobs=num_jobs,
        )
        quantiles_supported = True
    except TypeError as exc:  # pragma: no cover - depends on installed timesfm
        msg = str(exc)
        if "quantiles" in msg or "unexpected" in msg:
            print("TimesFM: installed `timesfm` does not support `quantiles` on forecast_on_df; retrying without quantiles.")
            try:
                forecast_df = model.forecast_on_df(
                    inputs=long_frame,
                    freq=freq,
                    value_name="y",
                    num_jobs=num_jobs,
                )
                quantiles_supported = False
            except Exception:
                raise
        else:
            raise

    forecast_df = forecast_df.copy()
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], utc=False).dt.tz_localize(None)
    print (forecast_df.tail(5))
    
    quantile_map = _quantile_column_map(forecast_df.columns)
    if not quantile_map:
        if quantiles and not quantiles_supported:
            print("TimesFM: quantile columns not present in forecast output; returning point forecasts only.")
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

    # Fallback: if requested confidence intervals are enabled but TimesFM did
    # not provide quantiles, estimate bands from historical one-step changes.
    if quantile_map == {}:
        if conf_interval_05 and (lower_05_wide.empty or upper_05_wide.empty):
            for commodity in point_wide.columns:
                try:
                    diffs = history_df[commodity].diff().dropna().values
                    if diffs.size == 0:
                        continue
                    l95 = np.percentile(diffs, 2.5)
                    u95 = np.percentile(diffs, 97.5)
                    lower_series = point_wide[commodity].iloc[:forecast_steps] + l95
                    upper_series = point_wide[commodity].iloc[:forecast_steps] + u95
                    combined_frames[f"{commodity}_lower_05"] = lower_series.rename(f"{commodity}_lower_05")
                    combined_frames[f"{commodity}_upper_05"] = upper_series.rename(f"{commodity}_upper_05")
                except Exception:
                    continue

        if conf_interval_10 and (lower_10_wide.empty or upper_10_wide.empty):
            for commodity in point_wide.columns:
                try:
                    diffs = history_df[commodity].diff().dropna().values
                    if diffs.size == 0:
                        continue
                    l90 = np.percentile(diffs, 5.0)
                    u90 = np.percentile(diffs, 95.0)
                    lower_series = point_wide[commodity].iloc[:forecast_steps] + l90
                    upper_series = point_wide[commodity].iloc[:forecast_steps] + u90
                    combined_frames[f"{commodity}_lower_10"] = lower_series.rename(f"{commodity}_lower_10")
                    combined_frames[f"{commodity}_upper_10"] = upper_series.rename(f"{commodity}_upper_10")
                except Exception:
                    continue

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
