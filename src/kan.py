"""
KAN (Kolmogorov-Arnold Network) based time series forecasting module.
Trains KAN models on commodity price data and generates multi-step forecasts.
"""

import io
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google.auth import default
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.file_utils import save_dataframe_to_gcs

warnings.filterwarnings("ignore")

# Initialize GCS client
creds, _ = default()
storage_client = storage.Client(credentials=creds)
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "crystal-dss")

# Hidden layer configuration for KAN: comma-separated env var, default 32,16
_raw = os.getenv("KAN_HIDDEN_UNITS", "32,16")
try:
    KAN_HIDDEN_UNITS = tuple(int(x) for x in _raw.split(',') if x.strip())
    if not KAN_HIDDEN_UNITS:
        KAN_HIDDEN_UNITS = (32, 16)
except Exception:
    KAN_HIDDEN_UNITS = (32, 16)

# Check for pykan availability
try:
    from kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("Warning: 'pykan' library not installed. KAN forecasting will not be available.")
    print("Install with: pip install pykan")


class TimeSeriesDatasetKAN(Dataset):
    """PyTorch Dataset for KAN time series."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # KAN expects flattened input
        return self.X[idx], self.y[idx]


def prepare_sequences(
    series: pd.Series,
    sequence_length: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Prepare sequences for KAN training/forecasting.
    
    Returns:
        X_tensor: Input sequences
        y_tensor: Target values
        scaler: Fitted scaler for inverse transformation
    """
    series = series.dropna()
    
    if len(series) < sequence_length + 1:
        raise ValueError(
            f"Insufficient data ({len(series)} points) for sequence length {sequence_length}"
        )
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequences = []
    targets = []
    for i in range(len(scaled_series) - sequence_length):
        seq = scaled_series[i : i + sequence_length]
        target = scaled_series[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    X_tensor = torch.tensor(sequences, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return X_tensor, y_tensor, scaler


def diagnose_kan_fit(
    series: pd.Series,
    sequence_length: int = 30,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.00001,
    hidden_units: Tuple[int, ...] = KAN_HIDDEN_UNITS,
    device: Optional[torch.device] = None,
):
    """Run diagnostics for KAN fit on a single series.

    Prints training-set fit diagnostics and the model prediction for the
    last input sequence so you can see why the first forecast step may
    be disconnected from the history.
    """
    if not KAN_AVAILABLE:
        print("KAN not available in this environment. Install pykan to run diagnostics.")
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series = series.dropna()
    if len(series) < sequence_length + 1:
        print(f"Insufficient data ({len(series)}) for sequence length {sequence_length}")
        return None

    print(f"Running KAN diagnostics: seq_len={sequence_length}, epochs={num_epochs}, hidden={hidden_units}")

    X_tensor, y_tensor, scaler = prepare_sequences(series, sequence_length)

    # Train model (this will raise if KAN not available)
    model = train_kan_model(
        X_tensor,
        y_tensor,
        sequence_length=sequence_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
    )

    model.to(device)
    model.eval()

    # Predict on the entire training set (scaled space)
    with torch.no_grad():
        inputs = X_tensor.to(device)
        try:
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                preds = outputs[0].cpu().numpy().reshape(-1)
            else:
                preds = outputs.cpu().numpy().reshape(-1)
        except Exception as exc:
            print(f"  ✗ Failed to run model on training inputs: {exc}")
            return None

    # Compare scaled predictions vs targets
    y_scaled = y_tensor.cpu().numpy().reshape(-1)
    pred_scaled = preds

    # Metrics in scaled space
    mae_scaled = float(np.mean(np.abs(pred_scaled - y_scaled)))
    rmse_scaled = float(np.sqrt(np.mean((pred_scaled - y_scaled) ** 2)))

    # Inverse transform to original space for interpretability
    try:
        y_orig = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
        pred_clipped = np.clip(pred_scaled, 0.0, 1.0)
        pred_orig = scaler.inverse_transform(pred_clipped.reshape(-1, 1)).reshape(-1)
    except Exception:
        y_orig = None
        pred_orig = None

    print("\nTraining-set diagnostics:")
    print(f"  Samples: {len(y_scaled)}")
    print(f"  Target scaled: mean={y_scaled.mean():.4f}, min={y_scaled.min():.4f}, max={y_scaled.max():.4f}")
    print(f"  Pred scaled: mean={pred_scaled.mean():.4f}, min={pred_scaled.min():.4f}, max={pred_scaled.max():.4f}")
    print(f"  MAE (scaled): {mae_scaled:.6f}, RMSE (scaled): {rmse_scaled:.6f}")

    # Show last 5 target vs prediction (scaled and original)
    n = min(5, len(y_scaled))
    print("\nLast training targets vs preds (scaled):")
    for i in range(-n, 0):
        print(f"  target={y_scaled[i]:.6f}  pred={pred_scaled[i]:.6f}")

    if y_orig is not None:
        print("\nLast training targets vs preds (original scale):")
        for i in range(-n, 0):
            print(f"  target={y_orig[i]:.6f}  pred={pred_orig[i]:.6f}")

    # Inspect the last input sequence (the one used to generate the first forecast)
    last_sequence = series.values[-sequence_length:]
    last_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    last_seq_tensor = torch.tensor(last_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(last_seq_tensor)
        if isinstance(out, tuple):
            next_scaled = float(out[0].cpu().numpy().reshape(-1)[0])
        else:
            next_scaled = float(out.cpu().numpy().reshape(-1)[0])

    print("\nLast input sequence (scaled) summary:")
    print(f"  last_scaled: mean={last_scaled.mean():.4f}, min={last_scaled.min():.4f}, max={last_scaled.max():.4f}")
    print(f"  predicted next (scaled, raw) = {next_scaled:.6f}")
    print(f"  predicted next (scaled, clipped) = {np.clip(next_scaled,0,1):.6f}")
    try:
        pred_next_orig = scaler.inverse_transform([[np.clip(next_scaled, 0.0, 1.0)]])[0][0]
        print(f"  predicted next (original, clipped) = {pred_next_orig:.6f}")
        print(f"  last observed (original) = {last_sequence[-1]:.6f}")
    except Exception:
        pass

    return {
        'model': model,
        'scaler': scaler,
        'mae_scaled': mae_scaled,
        'rmse_scaled': rmse_scaled,
        'last_pred_scaled': next_scaled,
    }


def train_kan_model(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    sequence_length: int = 30,
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 5,
    hidden_units: Tuple[int, ...] = KAN_HIDDEN_UNITS,
    use_manual: bool = True,
) -> 'KAN':
    """Train a KAN model on the provided data.

    Default is to use a controlled manual PyTorch training loop via
    `_train_kan_manual` to avoid hidden package regularizers. Set
    `use_manual=False` to call the library `.fit()` API instead.
    """
    if not KAN_AVAILABLE:
        raise RuntimeError("KAN library not available. Install with: pip install pykan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on device: {device}")

    # Ensure targets shape
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)

    # Initialize model
    width = [sequence_length] + list(hidden_units) + [1]
    model = KAN(width=width, grid=5, k=3, seed=42, device=device)

    if use_manual:
        print(f"    Using manual training (num_epochs={num_epochs}, lr={learning_rate})")
        model = _train_kan_manual(
            model,
            X_tensor,
            y_tensor,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            device=device,
            clip_norm=1.0,
        )
        return model

    # Otherwise, attempt to use pykan's .fit()
    empty_input = torch.empty((0, X_tensor.shape[1]), dtype=torch.float32, device=device)
    empty_label = torch.empty((0, y_tensor.shape[1]), dtype=torch.float32, device=device)

    dataset = {
        'train_input': X_tensor.to(device),
        'train_label': y_tensor.to(device),
        'test_input': empty_input,
        'test_label': empty_label,
        'val_input': empty_input,
        'val_label': empty_label,
    }

    try:
        print(f"    Training KAN (pykan .fit) for {num_epochs} steps...")
        model.fit(
            dataset,
            opt="Adam",
            steps=num_epochs,
            lr=learning_rate,
            lamb=0.0,
            lamb_entropy=0.0,
        )
        print(f"    Training complete")
    except Exception as e:
        print(f"    Warning: pykan .fit() failed ({e}), falling back to manual training")
        model = _train_kan_manual(model, X_tensor, y_tensor, num_epochs, batch_size, learning_rate, patience, device)

    return model





def _train_kan_manual(
    model: 'KAN',
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    clip_norm: float = 1.0,
) -> 'KAN':
    """Fallback manual training if pykan .fit() fails."""
    # Create dataset and dataloader
    dataset = TimeSeriesDatasetKAN(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    # Use model.parameters() if available, otherwise get parameters from layers
    try:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    except:
        # pykan may have different parameter structure
        params = []
        for layer in model.act_fun:
            params.extend(list(layer.parameters()))
        optimizer = optim.Adam(params, lr=learning_rate)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Move model to device and set train mode
    try:
        model.to(device)
    except Exception:
        pass
    try:
        model.train()
    except Exception:
        pass

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Ensure targets have correct shape
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs may be tuple from pykan layers
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            try:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], clip_norm)
            except Exception:
                try:
                    params = [p for p in model.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(params, clip_norm)
                except Exception:
                    pass

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            try:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            except Exception:
                best_model_state = None
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

        # Early stopping triggered
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        try:
            model.load_state_dict(best_model_state)
            print(f"    Best loss: {best_loss:.4f}")
        except:
            print(f"    Final loss: {avg_loss:.4f}")
    
    return model


def forecast_with_kan(
    model: 'KAN',
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    forecast_steps: int,
    device: torch.device,
    *,
    blend_steps: int = 3,
) -> np.ndarray:
    """Generate multi-step forecast using trained KAN model.

    If `blend_steps` > 0, the first `blend_steps` forecasted values are
    linearly blended with the last observed value to avoid a sharp
    discontinuity at the forecast boundary. This is a pragmatic fallback
    while the model is being tuned.
    """

    model.eval()

    # Scale the initial sequence
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    forecast_values = []

    last_observed = float(last_sequence[-1])

    with torch.no_grad():
        for step in range(forecast_steps):
            # Predict next value
            output = model(sequence_tensor)
            # Handle potential tuple return (some KAN versions return (output, preacts, postacts, postspline))
            if isinstance(output, tuple):
                predicted_scaled = output[0].item()
            else:
                predicted_scaled = output.item()

            try:
                predicted_scaled = float(predicted_scaled)
            except Exception:
                predicted_scaled = float(np.asarray(predicted_scaled).item())

            # Clip predicted scaled value to valid [0, 1] range to avoid large extrapolation
            clipped = float(np.clip(predicted_scaled, 0.0, 1.0))

            # Inverse scale using clipped value
            predicted_original = scaler.inverse_transform([[clipped]])[0][0]

            # Apply blending/anchoring for the first few steps
            if blend_steps and step < blend_steps:
                # alpha ramps from 0 (use last observed) to 1 (use model)
                alpha = (step + 1) / float(blend_steps)
                blended = alpha * predicted_original + (1.0 - alpha) * last_observed
                forecast_values.append(blended)
            else:
                forecast_values.append(predicted_original)

            # Update sequence (shift window) using clipped scaled value
            new_val = torch.tensor([[clipped]], dtype=torch.float32).to(device)
            sequence_tensor = torch.cat((sequence_tensor[:, 1:], new_val), dim=1)

    return np.array(forecast_values)


def save_kan_model_to_gcs(
    model: 'KAN',
    commodity: str,
    *,
    sequence_length: int = 30,
    prefix: str = 'models/kan/',
    bucket_name: str = BUCKET_NAME,
    hidden_units: Tuple[int, ...] = KAN_HIDDEN_UNITS,
) -> Optional[str]:
    """Save trained KAN model to GCS."""
    try:
        # Create filename
        clean_name = re.sub(r"\s+", '_', commodity)
        hidden_part = "-".join(str(x) for x in hidden_units) if hidden_units else "none"
        key = f"{prefix}{clean_name}_seq{sequence_length}_h{hidden_part}.pth"
        
        # Save model state dict to bytes
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Upload to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        print(f"    ✓ Model saved to GCS: {key}")
        return key
    except Exception as exc:
        print(f"    ✗ Failed to save model for {commodity}: {exc}")
        return None


def load_kan_model_from_gcs(
    commodity: str,
    *,
    sequence_length: int = 30,
    prefix: str = 'models/kan/',
    bucket_name: str = BUCKET_NAME,
) -> Optional['KAN']:
    """Load trained KAN model from GCS."""
    
    if not KAN_AVAILABLE:
        print(f"    KAN library not available")
        return None
    
    try:
        clean_name = re.sub(r"\s+", '_', commodity)
        # Construct key using configured hidden unit sizes to match saved model filename
        hidden_part = "-".join(str(x) for x in KAN_HIDDEN_UNITS) if KAN_HIDDEN_UNITS else "none"
        key = f"{prefix}{clean_name}_seq{sequence_length}_h{hidden_part}.pth"
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        if not blob.exists():
            print(f"    Model not found in GCS: {key}")
            return None
        
        # Download model state dict
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Load into model with matching initialization parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Build model using configured hidden units
        width = [sequence_length] + list(KAN_HIDDEN_UNITS) + [1]
        model = KAN(width=width, grid=5, k=3, seed=42, device=device)
        model.load_state_dict(torch.load(buffer, map_location=device))
        model.eval()
        
        return model
    except Exception as exc:
        print(f"    ✗ Failed to load model for {commodity}: {exc}")
        return None


def _train_single_commodity(args):
    """
    Worker function for parallel training of a single commodity.
    Must be defined at module level for pickle compatibility.
    """
    (
        commodity,
        series_data,
        sequence_length,
        num_epochs,
        batch_size,
        learning_rate,
        bucket_name,
    ) = args
    
    try:
        print(f"  [{commodity}] Starting KAN training...")
        
        # Recreate series as pandas Series
        series = pd.Series(series_data)
        
        if len(series) < sequence_length + 10:
            print(f"  [{commodity}] Skipping: Insufficient data ({len(series)} points)")
            return (commodity, None, None, f"Insufficient data ({len(series)} points)")
        
        # Prepare sequences
        X_tensor, y_tensor, scaler = prepare_sequences(series, sequence_length)
        
        # Train model using the official training function
        model = train_kan_model(
            X_tensor,
            y_tensor,
            sequence_length=sequence_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=5
        )
        
        # Save model to GCS
        gcs_key = save_kan_model_to_gcs(
            model,
            commodity,
            sequence_length=sequence_length,
            bucket_name=bucket_name,
        )
        
        if gcs_key:
            print(f"  [{commodity}] ✓ Training complete and saved to GCS: {gcs_key}")
        else:
            print(f"  [{commodity}] ✓ Training complete (GCS save failed)")

        # Return the GCS key and scaler only. The KAN model object may contain
        # non-picklable callables (lambdas) which cause ProcessPool to fail when
        # transferring the object back to the parent. Save the model in the
        # worker and let the parent load it from GCS instead.
        return (commodity, gcs_key, scaler, "Success")
        
    except Exception as exc:
        print(f"  [{commodity}] ✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return (commodity, None, None, str(exc))


def generate_and_save_kan_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    forecast_steps: int,
    gcs_prefix: str,
    *,
    train_new_models: bool = False,
    sequence_length: int = 30,
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    conf_interval_05: bool = False,
    conf_interval_10: bool = False,
    bucket_name: str = BUCKET_NAME,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate KAN forecasts for all commodities and save to GCS.
    
    Args:
        prices_df: Historical price data with DatetimeIndex
        commodity_columns: List of commodity column names to forecast
        forecast_steps: Number of periods to forecast
        gcs_prefix: GCS path prefix for saving forecast
        train_new_models: If True, train new models; if False, load from GCS
        sequence_length: Number of historical steps for input sequence
        num_epochs: Training epochs per commodity (default 20 for KAN)
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        conf_interval_05: Generate 5% confidence intervals (not implemented for KAN)
        conf_interval_10: Generate 10% confidence intervals (not implemented for KAN)
        bucket_name: GCS bucket name
        
    Returns:
        Tuple of (forecast_dataframe, gcs_path)
    """
    
    if not KAN_AVAILABLE:
        raise RuntimeError(
            "KAN library not available. Install with: pip install pykan"
        )
    
    print("\n" + "=" * 80)
    print("KAN FORECASTING")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sequence length: {sequence_length}")
    print(f"KAN hidden units: {KAN_HIDDEN_UNITS}")
    print(f"Forecast steps: {forecast_steps}")
    
    # Determine if we need to train models
    commodities_to_train = []
    commodities_to_load = []
    
    if train_new_models:
        print("\nForce training new models for all commodities")
        commodities_to_train = commodity_columns
    else:
        # Check which models exist in GCS
        print("\nChecking for existing models in GCS...")
        for commodity in commodity_columns:
            try:
                series = prices_df[commodity].dropna()
                if len(series) < sequence_length + 10:
                    print(f"  [{commodity}] Skipping: Insufficient data")
                    continue
                
                # Try to load existing model
                model = load_kan_model_from_gcs(
                    commodity,
                    sequence_length=sequence_length,
                    bucket_name=bucket_name,
                )
                
                if model is None:
                    commodities_to_train.append(commodity)
                else:
                    commodities_to_load.append(commodity)
            except Exception:
                commodities_to_train.append(commodity)
        
        print(f"  Models to load: {len(commodities_to_load)}")
        print(f"  Models to train: {len(commodities_to_train)}")
    
    all_forecasts = {}
    
    # Train models in parallel if needed
    if commodities_to_train:
        print(f"\n{'='*80}")
        print(f"PARALLEL TRAINING ({len(commodities_to_train)} commodities)")
        print("="*80)
        
        # Prepare arguments for parallel training
        train_args = []
        for commodity in commodities_to_train:
            series = prices_df[commodity].dropna()
            series_data = series.values  # Convert to numpy for serialization
            train_args.append((
                commodity,
                series_data,
                sequence_length,
                num_epochs,
                batch_size,
                learning_rate,
                bucket_name,
            ))
        
        # Use ProcessPoolExecutor for parallel training
        max_workers = min(os.cpu_count() or 4, len(commodities_to_train), 8)
        print(f"Using {max_workers} parallel workers")
        
        trained_models = {}
        trained_scalers = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_train_single_commodity, args): args[0]
                      for args in train_args}

            for future in as_completed(futures):
                commodity, gcs_key, scaler, status = future.result()
                if gcs_key is not None and scaler is not None:
                    # Attempt to load the model back from GCS into this process.
                    model = load_kan_model_from_gcs(
                        commodity,
                        sequence_length=sequence_length,
                        bucket_name=bucket_name,
                    )
                    if model is not None:
                        trained_models[commodity] = model
                        trained_scalers[commodity] = scaler
                    else:
                        print(f"  [{commodity}] Warning: model saved to GCS but failed to load in parent")
                elif status != "Success":
                    print(f"  [{commodity}] Failed: {status}")
        
        print(f"\n✓ Parallel training complete: {len(trained_models)}/{len(commodities_to_train)} successful")
        
        # Generate forecasts for trained models
        print("\nGenerating forecasts for newly trained models...")
        for commodity in trained_models:
            try:
                series = prices_df[commodity].dropna()
                last_sequence = series.values[-sequence_length:]
                
                forecast_values = forecast_with_kan(
                    trained_models[commodity],
                    last_sequence,
                    trained_scalers[commodity],
                    forecast_steps,
                    device,
                )
                
                all_forecasts[commodity] = forecast_values
                print(f"  [{commodity}] ✓ Forecast generated")
            except Exception as exc:
                print(f"  [{commodity}] ✗ Forecast error: {exc}")
    
    # Load existing models and forecast
    if commodities_to_load:
        print(f"\n{'='*80}")
        print(f"LOADING EXISTING MODELS ({len(commodities_to_load)} commodities)")
        print("="*80)
        
        for commodity in commodities_to_load:
            try:
                series = prices_df[commodity].dropna()
                
                # Load model
                model = load_kan_model_from_gcs(
                    commodity,
                    sequence_length=sequence_length,
                    bucket_name=bucket_name,
                )
                
                if model is None:
                    print(f"  [{commodity}] ⚠ Model not found (skipping)")
                    continue
                
                # Prepare scaler for forecasting
                _, _, scaler = prepare_sequences(series, sequence_length)
                
                # Generate forecast
                last_sequence = series.values[-sequence_length:]
                forecast_values = forecast_with_kan(
                    model, last_sequence, scaler, forecast_steps, device
                )
                
                all_forecasts[commodity] = forecast_values
                print(f"  [{commodity}] ✓ Loaded and forecasted")
                
            except Exception as exc:
                print(f"  [{commodity}] ✗ Error: {exc}")
                continue
    
    if not all_forecasts:
        raise RuntimeError("No KAN forecasts were generated")
    
    # Build forecast dataframe
    print(f"\n{'='*80}")
    print("BUILDING COMBINED FORECAST DATAFRAME")
    print("="*80)
    
    last_date = prices_df.index.max()
    freq = pd.infer_freq(prices_df.index) or 'D'
    forecast_dates = pd.date_range(
        start=last_date, periods=forecast_steps + 1, freq=freq
    )[1:]
    
    forecast_df = pd.DataFrame(all_forecasts, index=forecast_dates)
    forecast_df.index.name = 'Date'
    
    print(f"\n✓ Forecast dataframe created")
    print(f"  Shape: {forecast_df.shape}")
    print(f"  Commodities: {len(all_forecasts)}")
    print(f"  Date range: {forecast_dates[0]} to {forecast_dates[-1]}")
    
    # Combine historical data with forecast
    print(f"\nCombining historical data with forecast...")
    historical_df = prices_df[list(all_forecasts.keys())].copy()
    combined_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')
    combined_df.sort_index(inplace=True)
    combined_df.index.name = 'Date'
    
    print(f"  Historical data points: {len(historical_df)}")
    print(f"  Forecast data points: {len(forecast_df)}")
    print(f"  Combined shape: {combined_df.shape}")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Save combined dataframe to GCS
    print(f"\nSaving combined historical + forecast to GCS: {gcs_prefix}")
    combined_df_to_save = combined_df.reset_index()
    save_dataframe_to_gcs(
        df=combined_df_to_save,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )
    
    print("="*80)
    print("KAN FORECASTING COMPLETE")
    print("="*80 + "\n")
    
    return combined_df, gcs_prefix
