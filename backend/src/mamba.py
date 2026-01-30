"""
Mamba (State Space Model) based time series forecasting module.
Trains a UNIFIED Mamba model on ALL commodities simultaneously (multivariate approach).

Based on the mambapy library:
https://github.com/kyscg/MambaPy

Key Implementation Details:
- MULTIVARIATE: One model learns from all commodities together
- Uses mambapy.mamba.Mamba with MambaConfig
- Architecture: d_model = num_commodities (e.g., 55), n_layers=4
- Model configuration:
  * d_model: Number of commodities (input features per timestep)
  * n_layers: Number of Mamba blocks stacked (default 4)
- Learns cross-commodity dynamics and causality relationships
- GCS model persistence at gs://crystal-dss/models/mamba/
- Early stopping with patience=5
- Default 30 epochs

Architecture Pattern (UNIFIED):
1. Input: (batch, seq_len, num_commodities) - all commodities at each timestep
2. Mamba blocks: Process multivariate sequences through n_layers
3. Take last timestep output: (batch, d_model)
4. Linear layer: Map to predictions for all commodities (batch, num_commodities)

Advantages over univariate:
- Learns commodity interdependencies and causal relationships
- More data efficient (one model vs N models)
- Leverages cross-commodity correlations discovered in causality analysis
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

# Check for mambapy availability
try:
    from mambapy.mamba import Mamba, MambaConfig
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: 'mambapy' library not installed. Mamba forecasting will not be available.")
    print("Install with: pip install mambapy")


class MambaForecaster(nn.Module):
    """
    Unified Mamba-based multivariate time series forecaster using mambapy.
    Trains on all commodities simultaneously to learn cross-commodity dynamics.
    """
    
    def __init__(self, num_commodities: int, n_layers: int = 4):
        """
        Args:
            num_commodities: Number of commodities (features) in the dataset
            n_layers: Number of Mamba blocks to stack
        """
        super().__init__()
        
        self.num_commodities = num_commodities
        self.n_layers = n_layers
        
        # Create MambaConfig: d_model = number of commodities
        mamba_config = MambaConfig(d_model=num_commodities, n_layers=n_layers)
        
        # Instantiate the base Mamba model
        self.mamba = Mamba(mamba_config)
        
        # Final linear layer: predict all commodities
        self.fc = nn.Linear(num_commodities, num_commodities)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_commodities)
            
        Returns:
            Output tensor of shape (batch_size, num_commodities)
        """
        # x shape: (batch, seq_len, num_commodities)
        
        # Pass through Mamba blocks
        mamba_output = self.mamba(x)  # Output: (batch, seq_len, num_commodities)
        
        # Take the last time step for prediction
        last_step_output = mamba_output[:, -1, :]  # Shape: (batch, num_commodities)
        
        # Project to output predictions for all commodities
        prediction = self.fc(last_step_output)  # Shape: (batch, num_commodities)
        
        return prediction


class TimeSeriesDatasetMamba(Dataset):
    """PyTorch Dataset for multivariate Mamba time series."""

    def __init__(self, X, y):
        """
        Args:
            X: Input sequences tensor (num_samples, sequence_length, num_commodities)
            y: Target values tensor (num_samples, num_commodities)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X already has shape (seq_len, num_commodities)
        # y has shape (num_commodities,)
        return self.X[idx], self.y[idx]


def prepare_multivariate_sequences(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    sequence_length: int = 21,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, MinMaxScaler]]:
    """
    Prepare multivariate sequences for unified Mamba training.
    All commodities are included in each sequence.
    
    Args:
        prices_df: DataFrame with all commodity prices
        commodity_columns: List of commodity column names
        sequence_length: Number of timesteps in each sequence
    
    Returns:
        X_tensor: Input sequences (num_samples, sequence_length, num_commodities)
        y_tensor: Target values (num_samples, num_commodities)
        scalers: Dict of fitted scalers per commodity for inverse transformation
    """
    # Select only commodity columns and drop rows with any NaN
    data = prices_df[commodity_columns].dropna()
    
    print(f"\n  Data transformation:")
    print(f"    Raw data shape: {prices_df.shape} (all columns including Date)")
    print(f"    Commodity columns selected: {len(commodity_columns)}")
    print(f"    After selecting commodities & dropping NaN: {data.shape}")
    
    if len(data) < sequence_length + 1:
        raise ValueError(
            f"Insufficient data ({len(data)} points) for sequence length {sequence_length}"
        )
    
    # Scale each commodity independently
    scalers = {}
    scaled_data = np.zeros_like(data.values, dtype=np.float32)
    
    for i, commodity in enumerate(commodity_columns):
        scaler = MinMaxScaler()
        scaled_data[:, i] = scaler.fit_transform(data[commodity].values.reshape(-1, 1)).flatten()
        scalers[commodity] = scaler
    
    # Create sequences: each sample contains all commodities at each timestep
    # SLIDING WINDOW APPROACH:
    # From timesteps [0, 1, 2, ..., N], we create:
    #   Sample 0: X = [0:21],   y = [21]
    #   Sample 1: X = [1:22],   y = [22]
    #   Sample 2: X = [2:23],   y = [23]
    #   ...
    #   Sample (N-21): X = [(N-21):(N)], y = [N]
    # Result: (N - sequence_length) samples, each with shape (sequence_length, num_commodities)
    
    sequences = []
    targets = []
    
    for i in range(len(scaled_data) - sequence_length):
        # Sequence: (sequence_length, num_commodities)
        seq = scaled_data[i : i + sequence_length, :]
        # Target: next timestep values for all commodities
        target = scaled_data[i + sequence_length, :]
        sequences.append(seq)
        targets.append(target)
    
    X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
    
    print(f"\n  Sequence creation (sliding window with sequence_length={sequence_length}):")
    print(f"    Created {len(sequences)} training samples from {len(data)} timesteps")
    print(f"    Lost samples: {len(data) - len(sequences)} (need {sequence_length} history for first prediction)")
    print(f"    Final tensor shapes:")
    print(f"      X_tensor: {X_tensor.shape} = (samples, sequence_length, commodities)")
    print(f"      y_tensor: {y_tensor.shape} = (samples, commodities)")
    print(f"\n  Example: To predict day 22, we use days [1-21] as input")
    print(f"           Each input has all {len(commodity_columns)} commodity prices at each of the 21 days")
    
    return X_tensor, y_tensor, scalers


def train_unified_mamba_model(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    num_commodities: int,
    n_layers: int = 4,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 5,
    device: str = 'cpu',
) -> MambaForecaster:
    """
    Train a unified Mamba model on multivariate data with early stopping.
    
    Args:
        X_tensor: Input sequences (num_samples, sequence_length, num_commodities)
        y_tensor: Target values (num_samples, num_commodities)
        num_commodities: Number of commodities in the dataset
        n_layers: Number of Mamba blocks
        num_epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        
    Returns:
        Trained MambaForecaster model
    """
    
    if not MAMBA_AVAILABLE:
        raise RuntimeError("Mamba library not available. Install with: pip install mambapy")
    
    device_obj = torch.device(device)
    print(f"  Training unified model on device: {device_obj}")
    print(f"  Input shape: {X_tensor.shape}")
    print(f"  Output shape: {y_tensor.shape}")
    
    # Create dataset and dataloader
    dataset = TimeSeriesDatasetMamba(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"  Dataset: {len(dataset)} samples, {len(train_loader)} batches")
    
    # Initialize unified Mamba forecaster model
    model = MambaForecaster(num_commodities=num_commodities, n_layers=n_layers)
    model.to(device_obj)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Model architecture: {num_commodities} commodities, {n_layers} layers")
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    import time
    model.train()
    training_start = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)
            
            # inputs shape: (batch, seq_len, num_commodities)
            # targets shape: (batch, num_commodities)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
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
            improvement = "✓ (improved)"
        else:
            patience_counter += 1
            improvement = f"(no improvement x{patience_counter})"
        
        # Print every epoch to see actual progress
        if (epoch + 1) % 1 == 0:
            print(f"    Epoch [{epoch+1:3d}/{num_epochs}], Loss: {avg_loss:.6f} {improvement}")
        
        # Early stopping triggered
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    training_time = time.time() - training_start
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device_obj)
        print(f"    Best loss: {best_loss:.6f}")
        print(f"    Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
    
    return model


def forecast_multivariate_mamba(
    model: MambaForecaster,
    last_sequences: np.ndarray,
    scalers: Dict[str, MinMaxScaler],
    commodity_columns: List[str],
    forecast_steps: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Generate multi-step forecast for all commodities using unified Mamba model.
    
    Args:
        model: Trained unified MambaForecaster
        last_sequences: Last sequence_length x num_commodities array from historical data
        scalers: Dict of fitted scalers per commodity for inverse transformation
        commodity_columns: List of commodity names
        forecast_steps: Number of steps to forecast
        device: torch device
        
    Returns:
        Dict mapping commodity names to forecast arrays
    """
    
    model.eval()
    
    # Scale the initial sequences for all commodities
    scaled_sequences = np.zeros_like(last_sequences, dtype=np.float32)
    for i, commodity in enumerate(commodity_columns):
        scaler = scalers[commodity]
        scaled_sequences[:, i] = scaler.transform(last_sequences[:, i].reshape(-1, 1)).flatten()
    
    # Convert to tensor with shape (1, seq_len, num_commodities)
    sequence_tensor = torch.tensor(scaled_sequences, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Store forecasts for each commodity
    all_forecasts = {commodity: [] for commodity in commodity_columns}
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            # Predict next timestep for all commodities
            predicted_scaled = model(sequence_tensor).cpu().numpy()[0]  # Shape: (num_commodities,)
            
            # Inverse scale each commodity prediction
            for i, commodity in enumerate(commodity_columns):
                scaler = scalers[commodity]
                predicted_original = scaler.inverse_transform([[predicted_scaled[i]]])[0][0]
                all_forecasts[commodity].append(predicted_original)
            
            # Update sequence: remove first timestep, append new predictions
            new_timestep = torch.tensor(predicted_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            # new_timestep shape: (1, 1, num_commodities)
            sequence_tensor = torch.cat((sequence_tensor[:, 1:, :], new_timestep), dim=1)
    
    # Convert lists to arrays
    return {commodity: np.array(forecasts) for commodity, forecasts in all_forecasts.items()}


def save_unified_mamba_model_to_gcs(
    model: MambaForecaster,
    *,
    num_commodities: int,
    n_layers: int = 4,
    sequence_length: int = 21,
    prefix: str = 'models/mamba/',
    bucket_name: str = BUCKET_NAME,
) -> Optional[str]:
    """Save trained unified Mamba model to GCS."""
    try:
        # Create filename with architecture details
        key = f"{prefix}unified_mamba_seq{sequence_length}_n{num_commodities}_l{n_layers}.pth"
        
        # Save model state dict to bytes
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Upload to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        print(f"    ✓ Unified model saved to GCS: {key}")
        return key
    except Exception as exc:
        print(f"    ✗ Failed to save unified model: {exc}")
        return None


def load_unified_mamba_model_from_gcs(
    *,
    num_commodities: int,
    n_layers: int = 4,
    sequence_length: int = 21,
    prefix: str = 'models/mamba/',
    bucket_name: str = BUCKET_NAME,
) -> Optional[MambaForecaster]:
    """
    Load trained unified Mamba model from GCS.
    
    Args:
        num_commodities: Number of commodities (must match saved model)
        n_layers: Number of Mamba layers (must match saved model)
        sequence_length: Sequence length (for filename)
        prefix: GCS path prefix
        bucket_name: GCS bucket name
        
    Returns:
        Loaded MambaForecaster or None if not found
    """
    
    if not MAMBA_AVAILABLE:
        print(f"    Mamba library not available")
        return None
    
    try:
        key = f"{prefix}unified_mamba_seq{sequence_length}_n{num_commodities}_l{n_layers}.pth"
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        if not blob.exists():
            print(f"    Unified model not found in GCS: {key}")
            return None
        
        # Download model state dict
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Load into model with matching architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MambaForecaster(num_commodities=num_commodities, n_layers=n_layers)
        model.load_state_dict(torch.load(buffer, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"    ✓ Loaded unified model from GCS: {key}")
        return model
    except Exception as exc:
        print(f"    ✗ Failed to load unified model: {exc}")
        return None


def generate_and_save_mamba_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    forecast_steps: int,
    gcs_prefix: str,
    *,
    train_new_model: bool = False,
    sequence_length: int = 21,
    n_layers: int = 4,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    conf_interval_05: bool = False,
    conf_interval_10: bool = False,
    bucket_name: str = BUCKET_NAME,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate unified Mamba forecasts for ALL commodities simultaneously and save to GCS.
    Trains one multivariate model that learns cross-commodity dynamics.
    
    Args:
        prices_df: Historical price data with DatetimeIndex
        commodity_columns: List of commodity column names to forecast
        forecast_steps: Number of periods to forecast
        gcs_prefix: GCS path prefix for saving forecast
        train_new_model: If True, train new model; if False, try to load from GCS
        sequence_length: Number of historical steps for input sequence
        n_layers: Number of Mamba blocks (default 4)
        num_epochs: Training epochs (default 30)
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
            "Mamba library not available. Install with: pip install mambapy"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_commodities = len(commodity_columns)
    print(f"\n{'='*80}")
    print(f"UNIFIED MAMBA FORECASTING")
    print(f"{'='*80}")
    print(f"  Commodities: {num_commodities}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Layers: {n_layers}")
    print(f"  Forecast horizon: {forecast_steps} days")
    
    model = None
    scalers = None
    
    # Try to load existing model if not training new
    if not train_new_model:
        print(f"\n{'='*80}")
        print("ATTEMPTING TO LOAD EXISTING MODEL")
        print("="*80)
        
        model = load_unified_mamba_model_from_gcs(
            num_commodities=num_commodities,
            n_layers=n_layers,
            sequence_length=sequence_length,
            bucket_name=bucket_name,
        )
        
        if model is not None:
            # Still need to prepare scalers for forecasting
            print("  Preparing scalers for loaded model...")
            try:
                _, _, scalers = prepare_multivariate_sequences(
                    prices_df, commodity_columns, sequence_length
                )
                print("  ✓ Scalers prepared")
            except Exception as exc:
                print(f"  ✗ Failed to prepare scalers: {exc}")
                model = None
    
    # Train new model if requested or if loading failed
    if model is None:
        print(f"\n{'='*80}")
        print("TRAINING UNIFIED MAMBA MODEL")
        print("="*80)
        
        try:
            # Prepare multivariate sequences
            print("  Preparing multivariate sequences...")
            X_tensor, y_tensor, scalers = prepare_multivariate_sequences(
                prices_df, commodity_columns, sequence_length
            )
            print(f"  ✓ Prepared {len(X_tensor)} sequences")
            print(f"  ✓ Input shape: {X_tensor.shape}")
            print(f"  ✓ Target shape: {y_tensor.shape}")
            
            # Train unified model
            print("\n  Training unified model on all commodities...")
            model = train_unified_mamba_model(
                X_tensor,
                y_tensor,
                num_commodities=num_commodities,
                n_layers=n_layers,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=5,
                device=str(device),
            )
            
            # Save model to GCS
            print("\n  Saving unified model to GCS...")
            gcs_key = save_unified_mamba_model_to_gcs(
                model,
                num_commodities=num_commodities,
                n_layers=n_layers,
                sequence_length=sequence_length,
                bucket_name=bucket_name,
            )
            
            if gcs_key:
                print(f"  ✓ Model saved to: {gcs_key}")
            
        except Exception as exc:
            print(f"  ✗ Training failed: {exc}")
            import traceback
            traceback.print_exc()
            raise
    
    # Generate forecasts
    print(f"\n{'='*80}")
    print("GENERATING FORECASTS")
    print("="*80)
    
    try:
        # Get last sequence_length rows for all commodities
        last_data = prices_df[commodity_columns].dropna().tail(sequence_length)
        last_sequences = last_data.values  # Shape: (sequence_length, num_commodities)
        
        print(f"  Using last {len(last_sequences)} timesteps from historical data")
        
        # Generate forecasts for all commodities
        all_forecasts = forecast_multivariate_mamba(
            model,
            last_sequences,
            scalers,
            commodity_columns,
            forecast_steps,
            device,
        )
        
        print(f"  ✓ Generated forecasts for {len(all_forecasts)} commodities")
        
    except Exception as exc:
        print(f"  ✗ Forecasting failed: {exc}")
        import traceback
        traceback.print_exc()
        raise
    
    # Build combined forecast dataframe
    print(f"\n{'='*80}")
    print("BUILDING COMBINED FORECAST DATAFRAME")
    print("="*80)
    
    # Create future dates (matching KAN approach)
    last_date = prices_df.index.max()
    freq = pd.infer_freq(prices_df.index) or 'D'
    forecast_dates = pd.date_range(
        start=last_date, periods=forecast_steps + 1, freq=freq
    )[1:]
    
    # Build forecast dataframe
    forecast_df = pd.DataFrame(all_forecasts, index=forecast_dates)
    forecast_df.index.name = 'Date'
    
    print(f"\n✓ Forecast dataframe created")
    print(f"  Shape: {forecast_df.shape}")
    print(f"  Commodities: {len(all_forecasts)}")
    print(f"  Date range: {forecast_dates[0]} to {forecast_dates[-1]}")
    
    # Combine historical and forecast
    print(f"\nCombining historical data with forecast...")
    historical_df = prices_df[commodity_columns].copy()
    combined_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')
    combined_df.sort_index(inplace=True)
    combined_df.index.name = 'Date'
    
    print(f"  Historical data points: {len(historical_df)}")
    print(f"  Forecast data points: {len(forecast_df)}")
    print(f"  Combined shape: {combined_df.shape}")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    # Save to GCS
    print(f"\nSaving combined historical + forecast to GCS: {gcs_prefix}")
    combined_df_to_save = combined_df.reset_index()
    gcs_path = save_dataframe_to_gcs(
        df=combined_df_to_save,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )
    
    print(f"\n{'='*80}")
    print(f"✓ UNIFIED MAMBA FORECAST COMPLETE")
    print(f"{'='*80}")
    print(f"  Commodities: {len(commodity_columns)}")
    print(f"  Forecast horizon: {forecast_steps} days")
    print(f"  GCS path: {gcs_path}")
    
    # Return the combined dataframe with Date as column (matching KAN behavior)
    return combined_df_to_save, gcs_path
