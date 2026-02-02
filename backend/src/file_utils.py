import os
import pandas as pd
from google.cloud import storage
from google.api_core import exceptions
import io
from contextlib import contextmanager

@contextmanager
def _suppress_fds():
    import sys as _sys, os as _os

    _sys.stdout.flush(); _sys.stderr.flush()
    devnull_fd = _os.open(_os.devnull, _os.O_RDWR)
    try:
        old_stdout = _os.dup(1)
        old_stderr = _os.dup(2)
        _os.dup2(devnull_fd, 1)
        _os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            _sys.stdout.flush(); _sys.stderr.flush()
        except Exception:
            pass
        _os.dup2(old_stdout, 1)
        _os.dup2(old_stderr, 2)
        _os.close(old_stdout)
        _os.close(old_stderr)
        _os.close(devnull_fd)


def upload_excel_file(source=None, prompt="Select Excel workbook (.xlsx/.xls) containing BA_RAW, PLATTS_RAW, and ICIS_RAW sheets"):
    import io as _io

    if source is not None:
        if isinstance(source, dict):
            return source
        if isinstance(source, (str, os.PathLike)):
            path = str(source)
            with open(path, 'rb') as f:
                return {os.path.basename(path): f.read()}
        if hasattr(source, 'read'):
            data = source.read()
            if isinstance(data, str):
                data = data.encode()
            name = getattr(source, 'name', 'uploaded_file')
            return {os.path.basename(str(name)): data}

    try:
        import tkinter as _tk
        from tkinter import filedialog as _filedialog
        root = _tk.Tk()
        root.withdraw()
        file_path = _filedialog.askopenfilename(title=prompt, filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        root.destroy()
        if file_path:
            with open(file_path, 'rb') as f:
                return {os.path.basename(file_path): f.read()}
    except Exception:
        pass

    try:
        path = input(f"{prompt} — enter local file path (or leave blank to cancel): ").strip()
        if path:
            with open(path, 'rb') as f:
                return {os.path.basename(path): f.read()}
    except Exception:
        pass

    raise RuntimeError("No file uploaded or available. Provide 'source' argument or run in Colab/desktop environment.")


def save_dataframe_to_gcs(df: pd.DataFrame, bucket_name: str, gcs_prefix: str, validate_rows: bool = True, include_index: bool = False):
    client = storage.Client()

    if validate_rows:
        df_str = df.astype(str)
        nan_rows = df.isnull().any(axis=1)
        blank_rows = (df_str.apply(lambda x: x.str.strip() == '', axis=1)).any(axis=1)
        rows_to_keep = ~(nan_rows | blank_rows)
        df_validated = df[rows_to_keep].copy()

        num_skipped = len(df) - len(df_validated)
        if num_skipped > 0:
            print(f"Skipped {num_skipped} rows due to NaN or entirely blank values.")
        df = df_validated

    if df.empty:
        print("DataFrame is empty after validation. No file will be uploaded.")
        return

    try:
        bucket = client.get_bucket(bucket_name)
    except exceptions.NotFound:
        print(f"Error: Bucket '{bucket_name}' not found. Please ensure the bucket name is correct.")
        return
    except exceptions.Forbidden:
        print(f"Error: Permission denied for bucket '{bucket_name}'. Please check your GCP permissions.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while accessing bucket '{bucket_name}': {e}")
        return

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=include_index)

    try:
        blob = bucket.blob(gcs_prefix)
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        gs_path = f"gs://{bucket_name}/{gcs_prefix}"
        print(f"DataFrame successfully saved to {gs_path}")
        return gs_path
    except exceptions.Forbidden:
        print(f"Error: Permission denied when uploading to gs://{bucket_name}/{gcs_prefix}. Please check your GCP permissions.")
    except Exception as e:
        print(f"An unexpected error occurred while uploading to GCS: {e}")
    return None


def download_latest_csv_from_gcs(bucket_name: str = 'crystal-dss', gcs_prefix: str = 'cleaned_data') -> pd.DataFrame:
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
    except exceptions.NotFound:
        raise RuntimeError(f"Bucket '{bucket_name}' not found")
    except exceptions.Forbidden:
        raise RuntimeError(f"Permission denied for bucket '{bucket_name}'")
    except Exception as e:
        raise RuntimeError(f"Error accessing bucket '{bucket_name}': {e}")

    search_prefix = gcs_prefix.rstrip('/') + '/'

    blobs = list(bucket.list_blobs(prefix=search_prefix))
    csv_blobs = [b for b in blobs if b.name.lower().endswith('.csv')]
    if not csv_blobs:
        raise RuntimeError(f"No CSV files found under gs://{bucket_name}/{search_prefix}")

    latest_blob = max(csv_blobs, key=lambda b: getattr(b, 'updated', None) or b.time_created)

    try:
        data_bytes = latest_blob.download_as_bytes()
    except Exception as e:
        raise RuntimeError(f"Failed to download '{latest_blob.name}': {e}")

    try:
        df = pd.read_csv(io.BytesIO(data_bytes))
    except Exception as e:
        try:
            df = pd.read_csv(io.StringIO(data_bytes.decode('utf-8')))
        except Exception:
            raise RuntimeError(f"Failed to parse CSV from '{latest_blob.name}': {e}")

    return df


def filter_and_fill_series(
    prices_df: pd.DataFrame,
    commodity_columns: list,
    *,
    min_non_nulls: int = 1000,
) -> tuple[list, pd.DataFrame, list]:
    """Select commodity series with at least `min_non_nulls` non-null values and
    fill remaining gaps via backfill then forward-fill.

    Returns:
        keep_cols: list of columns kept
        df_filled: DataFrame with kept columns and NaNs filled
        dropped: list of columns dropped due to insufficient data
    """
    counts = prices_df[commodity_columns].notna().sum()
    keep_cols = [c for c in commodity_columns if counts.get(c, 0) >= min_non_nulls]
    dropped = [c for c in commodity_columns if c not in keep_cols]

    if not keep_cols:
        return [], pd.DataFrame(), dropped

    df = prices_df[keep_cols].copy()
    pre_nans = int(df.isna().sum().sum())
    if pre_nans > 0:
        # backfill then forward-fill to preserve most recent history
        df = df.bfill().ffill()

    return keep_cols, df, dropped
