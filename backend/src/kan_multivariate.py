"""
KAN (Kolmogorov-Arnold Network) based MULTIVARIATE time series forecasting.
Trains a single unified KAN model on all commodities simultaneously to learn cross-commodity dynamics.
"""

import io
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google.auth import default
from google.cloud import storage
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.file_utils import save_dataframe_to_gcs

warnings.filterwarnings("ignore")

# Initialize GCS client
creds, _ = default()
storage_client = storage.Client(credentials=creds)
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "crystal-dss")

# Check for pykan availability
try:
    from kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("Warning: 'pykan' library not installed. KAN forecasting will not be available.")
    print("Install with: pip install pykan")


class MultivariateLegacyKANForecaster(nn.Module):
    """
    Multivariate KAN forecaster using pykan library.
    Takes sequences of shape (batch, sequence_length, num_commodities)
    and predicts next timestep for all commodities simultaneously.
    """
    
    def __init__(self, sequence_length: int, num_commodities: int, hidden_units: Tuple[int, ...] = (32, 16)):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_commodities = num_commodities
        self.hidden_units = hidden_units
        
        # Flatten input: sequence_length * num_commodities -> flattened features
        input_dim = sequence_length * num_commodities
        
        # KAN architecture: [input_dim, *hidden_units, num_commodities]
        width = [input_dim] + list(hidden_units) + [num_commodities]
        
        if not KAN_AVAILABLE:
            raise RuntimeError("pykan library not available. Install with: pip install pykan")
        
        # Initialize KAN model
        self.kan = KAN(width=width, grid=5, k=3, seed=42)
        
        print(f"  Initialized Multivariate KAN:")
        print(f"    Input: ({sequence_length} × {num_commodities}) = {input_dim} features")
        print(f"    Architecture: {width}")
        print(f"    Output: {num_commodities} commodities")
        
    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, num_commodities)
        Returns:
            (batch, num_commodities)
        """
        batch_size = x.shape[0]
        # Flatten: (batch, sequence_length, num_commodities) -> (batch, sequence_length * num_commodities)
        x_flat = x.reshape(batch_size, -1)
        
        # Pass through KAN
        out = self.kan(x_flat)
        
        # Handle tuple return from KAN
        if isinstance(out, tuple):
            out = out[0]
        
        return out


def prepare_multivariate_sequences(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    sequence_length: int = 21,
) -> Tuple[torch.Tensor, torch.Tensor, List[MinMaxScaler]]:
    """
    Prepare multivariate sequences for unified KAN training.
    
    Args:
        prices_df: DataFrame with DatetimeIndex and commodity columns
        commodity_columns: List of commodity names
        sequence_length: Number of historical timesteps to use
    
    Returns:
        X_tensor: (num_samples, sequence_length, num_commodities)
        y_tensor: (num_samples, num_commodities)
        scalers: List of MinMaxScaler (one per commodity)
    """
    print(f"\nPreparing multivariate sequences...")
    print(f"  Raw data shape: {prices_df.shape}")
    
    # Select commodity columns and drop NaN
    data = prices_df[commodity_columns].dropna()
    print(f"  After selecting {len(commodity_columns)} commodities & dropping NaN: {data.shape}")
    
    if len(data) < sequence_length + 1:
        raise ValueError(
            f"Insufficient data ({len(data)} rows) for sequence_length={sequence_length}"
        )
    
    # Scale each commodity independently
    scalers = []
    scaled_data = np.zeros_like(data.values, dtype=np.float32)
    
    for i, col in enumerate(commodity_columns):
        scaler = MinMaxScaler()
        scaled_data[:, i] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
        scalers.append(scaler)
    
    # Create sliding window sequences
    X_list = []
    y_list = []
    
    for i in range(len(scaled_data) - sequence_length):
        # Input: sequence_length timesteps of all commodities
        X_seq = scaled_data[i:i + sequence_length, :]  # (sequence_length, num_commodities)
        # Target: next timestep for all commodities
        y_next = scaled_data[i + sequence_length, :]  # (num_commodities,)
        
        X_list.append(X_seq)
        y_list.append(y_next)
    
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
    
    print(f"\n  Sequence creation (sliding window with sequence_length={sequence_length}):")
    print(f"    Created {X_tensor.shape[0]} training samples from {len(scaled_data)} timesteps")
    print(f"    Lost samples: {sequence_length} (need {sequence_length} history for first prediction)")
    print(f"    Final tensor shapes:")
    print(f"      X_tensor: {X_tensor.shape} = (samples, sequence_length, commodities)")
    print(f"      y_tensor: {y_tensor.shape} = (samples, commodities)")
    print(f"\n  Example: To predict day {sequence_length + 1}, we use days [1-{sequence_length}] as input")
    print(f"           Each input has all {len(commodity_columns)} commodity prices at each of the {sequence_length} days")
    
    return X_tensor, y_tensor, scalers


def train_unified_kan_model(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    sequence_length: int,
    num_commodities: int,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 5,
    hidden_units: Tuple[int, ...] = (32, 16),
) -> MultivariateLegacyKANForecaster:
    """
    Train unified multivariate KAN model.
    
    Args:
        X_tensor: (num_samples, sequence_length, num_commodities)
        y_tensor: (num_samples, num_commodities)
        sequence_length: Sequence length
        num_commodities: Number of commodities
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        patience: Early stopping patience
        hidden_units: Hidden layer sizes
    
    Returns:
        Trained MultivariateLegacyKANForecaster
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining unified KAN model on {device}")
    print(f"  Samples: {X_tensor.shape[0]}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Commodities: {num_commodities}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Initialize model
    model = MultivariateLegacyKANForecaster(
        sequence_length=sequence_length,
        num_commodities=num_commodities,
        hidden_units=hidden_units,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")
    
    # Prepare DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\n  Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Handle tuple return
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"    Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f} - Elapsed: {elapsed:.1f}s")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
                break
    
    total_time = time.time() - start_time
    print(f"\n  Training complete!")
    print(f"    Final loss: {best_loss:.6f}")
    print(f"    Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return model


def forecast_multivariate_kan(
    model: MultivariateLegacyKANForecaster,
    last_sequence: np.ndarray,
    scalers: List[MinMaxScaler],
    forecast_steps: int,
) -> np.ndarray:
    """
    Generate multi-step forecast for all commodities.
    
    Args:
        model: Trained MultivariateLegacyKANForecaster
        last_sequence: (sequence_length, num_commodities) - last observed scaled data
        scalers: List of MinMaxScaler for inverse transform
        forecast_steps: Number of steps to forecast
    
    Returns:
        forecast: (forecast_steps, num_commodities) in original scale
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Initialize with last sequence
    current_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    # Shape: (1, sequence_length, num_commodities)
    
    forecasts_scaled = []
    
    with torch.no_grad():
        for step in range(forecast_steps):
            # Predict next timestep
            pred = model(current_sequence)
            
            # Handle tuple return
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # pred shape: (1, num_commodities)
            pred_np = pred.cpu().numpy()[0]  # (num_commodities,)
            
            # Clip to valid range
            pred_np = np.clip(pred_np, 0.0, 1.0)
            forecasts_scaled.append(pred_np)
            
            # Update sequence: remove oldest, append newest
            # current_sequence: (1, sequence_length, num_commodities)
            # pred: (1, num_commodities)
            pred_reshaped = pred.unsqueeze(1)  # (1, 1, num_commodities)
            current_sequence = torch.cat([current_sequence[:, 1:, :], pred_reshaped], dim=1)
    
    forecasts_scaled = np.array(forecasts_scaled)  # (forecast_steps, num_commodities)
    
    # Inverse transform each commodity
    forecasts_original = np.zeros_like(forecasts_scaled)
    for i, scaler in enumerate(scalers):
        forecasts_original[:, i] = scaler.inverse_transform(
            forecasts_scaled[:, i].reshape(-1, 1)
        ).flatten()
    
    return forecasts_original


def save_unified_kan_model_to_gcs(
    model: MultivariateLegacyKANForecaster,
    scalers: List[MinMaxScaler],
    commodity_columns: List[str],
    sequence_length: int,
    bucket_name: str = BUCKET_NAME,
) -> str:
    """
    Save unified KAN model and scalers to GCS.
    
    Returns:
        GCS key where model was saved
    """
    num_commodities = len(commodity_columns)
    hidden_str = "_".join(map(str, model.hidden_units))
    
    gcs_key = f"models/kan_multivariate/unified_kan_seq{sequence_length}_n{num_commodities}_h{hidden_str}.pth"
    
    print(f"\nSaving unified KAN model to GCS...")
    print(f"  Path: gs://{bucket_name}/{gcs_key}")
    
    # Prepare state dict
    state_dict = {
        'model_state_dict': model.state_dict(),
        'scalers': scalers,
        'commodity_columns': commodity_columns,
        'sequence_length': sequence_length,
        'num_commodities': num_commodities,
        'hidden_units': model.hidden_units,
    }
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    
    # Upload to GCS
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_key)
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    
    print(f"  ✓ Model saved successfully")
    
    return gcs_key


def load_unified_kan_model_from_gcs(
    sequence_length: int,
    num_commodities: int,
    hidden_units: Tuple[int, ...] = (32, 16),
    bucket_name: str = BUCKET_NAME,
) -> Optional[Tuple[MultivariateLegacyKANForecaster, List[MinMaxScaler], List[str]]]:
    """
    Load unified KAN model from GCS.
    
    Returns:
        (model, scalers, commodity_columns) or None if not found
    """
    hidden_str = "_".join(map(str, hidden_units))
    gcs_key = f"models/kan_multivariate/unified_kan_seq{sequence_length}_n{num_commodities}_h{hidden_str}.pth"
    
    print(f"\nLoading unified KAN model from GCS...")
    print(f"  Path: gs://{bucket_name}/{gcs_key}")
    
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_key)
        
        if not blob.exists():
            print(f"  ✗ Model not found")
            return None
        
        # Download model
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Load state dict
        state_dict = torch.load(buffer, map_location='cpu')
        
        # Reconstruct model
        model = MultivariateLegacyKANForecaster(
            sequence_length=state_dict['sequence_length'],
            num_commodities=state_dict['num_commodities'],
            hidden_units=state_dict['hidden_units'],
        )
        model.load_state_dict(state_dict['model_state_dict'])
        
        scalers = state_dict['scalers']
        commodity_columns = state_dict['commodity_columns']
        
        print(f"  ✓ Model loaded successfully")
        print(f"    Commodities: {num_commodities}")
        print(f"    Sequence length: {sequence_length}")
        
        return model, scalers, commodity_columns
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return None


def generate_and_save_kan_multivariate_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    forecast_steps: int,
    gcs_prefix: str,
    *,
    train_new_model: bool = False,
    sequence_length: int = 21,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_units: Tuple[int, ...] = (32, 16),
    bucket_name: str = BUCKET_NAME,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate KAN forecasts using unified multivariate approach and save to GCS.
    
    Args:
        prices_df: Historical price data with DatetimeIndex
        commodity_columns: List of commodity column names to forecast
        forecast_steps: Number of periods to forecast
        gcs_prefix: GCS path prefix for saving forecast
        train_new_model: If True, train new model; if False, load from GCS
        sequence_length: Number of historical steps for input sequence
        num_epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        hidden_units: Hidden layer architecture
        bucket_name: GCS bucket name
        
    Returns:
        Tuple of (forecast_dataframe, gcs_path)
    """
    
    if not KAN_AVAILABLE:
        raise RuntimeError(
            "KAN library not available. Install with: pip install pykan"
        )
    
    print("\n" + "=" * 80)
    print("KAN MULTIVARIATE FORECASTING")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Commodities: {len(commodity_columns)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Hidden units: {hidden_units}")
    print(f"Forecast steps: {forecast_steps}")
    
    # Try to load existing model
    model = None
    scalers = None
    loaded_commodity_columns = None
    
    if not train_new_model:
        result = load_unified_kan_model_from_gcs(
            sequence_length=sequence_length,
            num_commodities=len(commodity_columns),
            hidden_units=hidden_units,
            bucket_name=bucket_name,
        )
        if result is not None:
            model, scalers, loaded_commodity_columns = result
            
            # Verify commodity columns match
            if set(loaded_commodity_columns) != set(commodity_columns):
                print(f"  ⚠ Warning: Loaded model commodities don't match. Retraining...")
                model = None
    
    # Train new model if needed
    if model is None:
        print("\nTraining new unified KAN model...")
        
        # Prepare multivariate sequences
        X_tensor, y_tensor, scalers = prepare_multivariate_sequences(
            prices_df,
            commodity_columns,
            sequence_length=sequence_length,
        )
        
        # Train model
        model = train_unified_kan_model(
            X_tensor,
            y_tensor,
            sequence_length=sequence_length,
            num_commodities=len(commodity_columns),
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=5,
            hidden_units=hidden_units,
        )
        
        # Save model to GCS
        save_unified_kan_model_to_gcs(
            model,
            scalers,
            commodity_columns,
            sequence_length,
            bucket_name=bucket_name,
        )
    
    # Generate forecasts
    print("\n" + "=" * 80)
    print("GENERATING FORECASTS")
    print("=" * 80)
    
    model = model.to(device)
    model.eval()
    
    # Prepare last sequence for forecasting
    data = prices_df[commodity_columns].dropna()
    last_sequence_original = data.iloc[-sequence_length:].values  # (sequence_length, num_commodities)
    
    # Scale using trained scalers
    last_sequence_scaled = np.zeros_like(last_sequence_original, dtype=np.float32)
    for i, scaler in enumerate(scalers):
        last_sequence_scaled[:, i] = scaler.transform(
            last_sequence_original[:, i].reshape(-1, 1)
        ).flatten()
    
    # Generate forecast
    forecasts = forecast_multivariate_kan(
        model,
        last_sequence_scaled,
        scalers,
        forecast_steps,
    )
    
    # Create forecast DataFrame
    last_date = prices_df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    
    forecast_df = pd.DataFrame(
        forecasts,
        index=forecast_dates,
        columns=commodity_columns,
    )
    forecast_df.index.name = 'Date'
    
    # Add Date column for saving
    combined_df_to_save = forecast_df.reset_index()
    
    print(f"\n  Generated forecasts:")
    print(f"    Shape: {forecast_df.shape}")
    print(f"    Date range: {forecast_dates[0]} to {forecast_dates[-1]}")
    
    # Save to GCS
    print(f"\nSaving forecast to GCS...")
    save_dataframe_to_gcs(
        df=combined_df_to_save,
        bucket_name=bucket_name,
        gcs_prefix=gcs_prefix,
        validate_rows=False,
    )
    print(f"  ✓ Forecast saved to: gs://{bucket_name}/{gcs_prefix}")
    
    return combined_df_to_save, gcs_prefix
