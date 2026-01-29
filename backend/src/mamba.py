"""
Mamba (State Space Model) based time series forecasting module.
Trains Mamba models on commodity price data and generates multi-step forecasts.

Based on the official state-spaces/mamba repository:
https://github.com/state-spaces/mamba

Key Implementation Details:
- Uses official Mamba SSM module from mamba-ssm package
- Architecture: Input projection → Mamba block → Output projection
- Model configuration (matching official examples):
  * d_model=64: Model dimension (internal embedding dimension)
  * d_state=16: SSM state expansion factor (standard for Mamba)
  * d_conv=4: Local convolution width
  * expand=2: Block expansion factor (controls model capacity)
- Parallel training with ProcessPoolExecutor
- GCS model persistence at gs://crystal-dss/models/mamba/
- Early stopping with patience=5
- Default 30 epochs (similar to GRU)

Architecture Pattern:
1. MambaForecaster wraps the core Mamba module
2. Input: 1D time series → Linear(1, d_model)
3. Mamba SSM: (batch, seq_len, d_model) → (batch, seq_len, d_model)
4. Take last timestep: (batch, d_model)
5. Output: Linear(d_model, 1) → single price prediction
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

# Check for mamba-ssm availability
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: 'mamba-ssm' library not installed. Mamba forecasting will not be available.")
    print("Install with: pip install mamba-ssm")


class MambaForecaster(nn.Module):
    """
    Mamba-based time series forecaster.
    Wraps the Mamba SSM module with input/output projections for forecasting.
    """
    
    def __init__(self, sequence_length: int = 30, d_model: int = 64, d_state: int = 16, 
                 d_conv: int = 4, expand: int = 2, device='cpu'):
        """
        Args:
            sequence_length: Input sequence length
            d_model: Model dimension (internal dimension)
            d_state: SSM state expansion factor
            d_conv: Local convolution width
            expand: Block expansion factor
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Input projection: map from 1D time series to d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        
        # Mamba block (official state space model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Output projection: map from d_model back to 1D prediction
        self.output_proj = nn.Linear(d_model, 1)
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # x shape: (batch, seq_len)
        # Reshape to (batch, seq_len, 1) for input projection
        x = x.unsqueeze(-1)
        
        # Project to d_model dimensions: (batch, seq_len, d_model)
        x = self.input_proj(x)
        
        # Apply Mamba: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.mamba(x)
        
        # Take the last time step for prediction: (batch, d_model)
        x = x[:, -1, :]
        
        # Project to 1D output: (batch, 1)
        out = self.output_proj(x)
        
        return out


class TimeSeriesDatasetMamba(Dataset):
    """PyTorch Dataset for Mamba time series."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_sequences(
    series: pd.Series,
    sequence_length: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Prepare sequences for Mamba training/forecasting.
    
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


def train_mamba_model(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    sequence_length: int = 30,
    d_model: int = 64,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    patience: int = 5,
    device: str = 'cpu',
) -> MambaForecaster:
    """Train a Mamba model on the provided data with early stopping."""
    
    if not MAMBA_AVAILABLE:
        raise RuntimeError("Mamba library not available. Install with: pip install mamba-ssm")
    
    device_obj = torch.device(device)
    print(f"  Training on device: {device_obj}")
    
    # Create dataset and dataloader
    dataset = TimeSeriesDatasetMamba(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Mamba forecaster model
    model = MambaForecaster(
        sequence_length=sequence_length,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure targets have correct shape
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Early stopping triggered
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device_obj)
        print(f"    Best loss: {best_loss:.4f}")
    
    return model


def forecast_with_mamba(
    model: MambaForecaster,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    forecast_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Generate multi-step forecast using trained Mamba model."""
    
    model.eval()
    
    # Scale the initial sequence
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    forecast_values = []
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            # Predict next value
            predicted_scaled = model(sequence_tensor).item()
            
            # Inverse scale
            predicted_original = scaler.inverse_transform([[predicted_scaled]])[0][0]
            forecast_values.append(predicted_original)
            
            # Update sequence (shift window)
            new_val = torch.tensor([[predicted_scaled]], dtype=torch.float32).to(device)
            # Remove first element, append new prediction
            sequence_tensor = torch.cat((sequence_tensor[:, 1:], new_val.unsqueeze(1)), dim=1)
    
    return np.array(forecast_values)


def save_mamba_model_to_gcs(
    model: MambaForecaster,
    commodity: str,
    *,
    sequence_length: int = 30,
    prefix: str = 'models/mamba/',
    bucket_name: str = BUCKET_NAME,
) -> Optional[str]:
    """Save trained Mamba model to GCS."""
    try:
        # Create filename
        clean_name = re.sub(r"\s+", '_', commodity)
        key = f"{prefix}{clean_name}_seq{sequence_length}.pth"
        
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


def load_mamba_model_from_gcs(
    commodity: str,
    *,
    sequence_length: int = 30,
    d_model: int = 64,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    prefix: str = 'models/mamba/',
    bucket_name: str = BUCKET_NAME,
) -> Optional[MambaForecaster]:
    """Load trained Mamba model from GCS."""
    
    if not MAMBA_AVAILABLE:
        print(f"    Mamba library not available")
        return None
    
    try:
        clean_name = re.sub(r"\s+", '_', commodity)
        key = f"{prefix}{clean_name}_seq{sequence_length}.pth"
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        if not blob.exists():
            print(f"    Model not found in GCS: {key}")
            return None
        
        # Download model state dict
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Load into model with matching architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MambaForecaster(
            sequence_length=sequence_length,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            device=device
        )
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
        d_model,
        d_state,
        d_conv,
        expand,
        num_epochs,
        batch_size,
        learning_rate,
        bucket_name,
    ) = args
    
    try:
        print(f"  [{commodity}] Starting Mamba training...")
        
        # Recreate series as pandas Series
        series = pd.Series(series_data)
        
        if len(series) < sequence_length + 10:
            print(f"  [{commodity}] Skipping: Insufficient data ({len(series)} points)")
            return (commodity, None, None, f"Insufficient data ({len(series)} points)")
        
        # Prepare sequences
        X_tensor, y_tensor, scaler = prepare_sequences(series, sequence_length)
        
        # Train model (force CPU for parallel training to avoid CUDA issues)
        model = train_mamba_model(
            X_tensor,
            y_tensor,
            sequence_length=sequence_length,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=5,
            device='cpu'  # Use CPU for parallel training
        )
        
        # Save model to GCS
        gcs_key = save_mamba_model_to_gcs(
            model,
            commodity,
            sequence_length=sequence_length,
            bucket_name=bucket_name,
        )
        
        if gcs_key:
            print(f"  [{commodity}] ✓ Training complete and saved to GCS")
        else:
            print(f"  [{commodity}] ✓ Training complete (GCS save failed)")
        
        return (commodity, model, scaler, "Success")
        
    except Exception as exc:
        print(f"  [{commodity}] ✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return (commodity, None, None, str(exc))


def generate_and_save_mamba_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    forecast_steps: int,
    gcs_prefix: str,
    *,
    train_new_models: bool = False,
    sequence_length: int = 30,
    d_model: int = 64,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    conf_interval_05: bool = False,
    conf_interval_10: bool = False,
    bucket_name: str = BUCKET_NAME,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate Mamba forecasts for all commodities and save to GCS.
    
    Args:
        prices_df: Historical price data with DatetimeIndex
        commodity_columns: List of commodity column names to forecast
        forecast_steps: Number of periods to forecast
        gcs_prefix: GCS path prefix for saving forecast
        train_new_models: If True, train new models; if False, load from GCS
        sequence_length: Number of historical steps for input sequence
        d_model: Mamba model dimension (internal dimension)
        d_state: SSM state expansion factor
        d_conv: Local convolution width
        expand: Block expansion factor
        num_epochs: Training epochs per commodity (default 30)
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        conf_interval_05: Generate 5% confidence intervals (not implemented for Mamba)
        conf_interval_10: Generate 10% confidence intervals (not implemented for Mamba)
        bucket_name: GCS bucket name
        
    Returns:
        Tuple of (forecast_dataframe, gcs_path)
    """
    
    if not MAMBA_AVAILABLE:
        raise RuntimeError(
            "Mamba library not available. Install with: pip install mamba-ssm"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine which commodities need training
    commodities_to_train = []
    commodities_to_load = []
    
    if train_new_models:
        print("Training mode: Will train all models from scratch")
        commodities_to_train = commodity_columns
    else:
        print("Auto mode: Checking for existing models in GCS...")
        for commodity in commodity_columns:
            try:
                clean_name = re.sub(r"\s+", '_', commodity)
                key = f"models/mamba/{clean_name}_seq{sequence_length}.pth"
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(key)
                
                if blob.exists():
                    commodities_to_load.append(commodity)
                else:
                    commodities_to_train.append(commodity)
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
                d_model,
                d_state,
                d_conv,
                expand,
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
                commodity, model, scaler, status = future.result()
                if model is not None and scaler is not None:
                    trained_models[commodity] = model
                    trained_scalers[commodity] = scaler
                elif status != "Success":
                    print(f"  [{commodity}] Failed: {status}")
        
        print(f"\n✓ Parallel training complete: {len(trained_models)}/{len(commodities_to_train)} successful")
        
        # Generate forecasts for trained models
        print("\nGenerating forecasts for newly trained models...")
        for commodity in trained_models:
            try:
                series = prices_df[commodity].dropna()
                last_sequence = series.values[-sequence_length:]
                
                forecast_values = forecast_with_mamba(
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
                model = load_mamba_model_from_gcs(
                    commodity,
                    sequence_length=sequence_length,
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bucket_name=bucket_name,
                )
                
                if model is None:
                    print(f"  [{commodity}] ⚠ Model not found (skipping)")
                    continue
                
                # Prepare scaler for forecasting
                _, _, scaler = prepare_sequences(series, sequence_length)
                
                # Generate forecast
                last_sequence = series.values[-sequence_length:]
                forecast_values = forecast_with_mamba(
                    model, last_sequence, scaler, forecast_steps, device
                )
                
                all_forecasts[commodity] = forecast_values
                print(f"  [{commodity}] ✓ Forecast generated")
                
            except Exception as exc:
                print(f"  [{commodity}] ✗ Error: {exc}")
    
    # Build combined forecast dataframe
    print(f"\n{'='*80}")
    print("BUILDING COMBINED FORECAST DATAFRAME")
    print("="*80)
    
    if not all_forecasts:
        print("No forecasts generated!")
        return pd.DataFrame(), gcs_prefix
    
    # Create future dates
    last_date = prices_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    
    # Build forecast dataframe
    forecast_dict = {'Date': forecast_dates}
    for commodity, forecast in all_forecasts.items():
        forecast_dict[commodity] = forecast
    
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df.set_index('Date', inplace=True)
    
    # Combine historical and forecast
    historical_subset = prices_df[list(all_forecasts.keys())].copy()
    combined_df = pd.concat([historical_subset, forecast_df], axis=0)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Save to GCS
    print(f"\nSaving combined forecast to GCS: {gcs_prefix}")
    gcs_path = save_dataframe_to_gcs(combined_df, gcs_prefix, bucket_name)
    
    print(f"✓ Mamba forecast complete: {len(all_forecasts)} commodities, {forecast_steps} days")
    
    return combined_df, gcs_path
