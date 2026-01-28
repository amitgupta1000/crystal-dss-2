"""
GRU-based time series forecasting module.
Trains a GRU model on commodity price data and generates multi-step forecasts.
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


class TimeSeriesDatasetGRU(Dataset):
    """PyTorch Dataset for GRU time series."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # GRU expects input shape (batch_size, sequence_length, input_dim)
        return self.X[idx].unsqueeze(-1), self.y[idx]


class GRUModel(nn.Module):
    """GRU model for time series forecasting."""

    def __init__(self, input_dim=1, hidden_dim=128, layer_dim=3, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0.detach())
        last_step_out = out[:, -1, :]  # Get last time step
        prediction = self.fc(last_step_out)
        return prediction


def prepare_sequences(
    series: pd.Series,
    sequence_length: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Prepare sequences for GRU training/forecasting.
    
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


def train_gru_model(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    hidden_dim: int = 128,
    layer_dim: int = 3,
    patience: int = 5,
) -> GRUModel:
    """Train a GRU model on the provided data with early stopping."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on device: {device}")
    
    # Create dataset and dataloader
    dataset = TimeSeriesDatasetGRU(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = GRUModel(
        input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=1
    )
    model.to(device)
    
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
            inputs, targets = inputs.to(device), targets.to(device)
            
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
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Early stopping triggered
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"    Best loss: {best_loss:.4f}")
    
    return model


def forecast_with_gru(
    model: GRUModel,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    forecast_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Generate multi-step forecast using trained GRU model."""
    
    model.eval()
    
    # Scale the initial sequence
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    sequence_tensor = (
        torch.tensor(scaled_sequence, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(device)
    )
    
    forecast_values = []
    
    with torch.no_grad():
        for _ in range(forecast_steps):
            # Predict next value
            predicted_scaled = model(sequence_tensor).item()
            
            # Inverse scale
            predicted_original = scaler.inverse_transform([[predicted_scaled]])[0][0]
            forecast_values.append(predicted_original)
            
            # Update sequence
            new_val = torch.tensor(
                [[predicted_scaled]], dtype=torch.float32
            ).unsqueeze(0).to(device)
            sequence_tensor = torch.cat((sequence_tensor[:, 1:, :], new_val), dim=1)
    
    return np.array(forecast_values)


def save_gru_model_to_gcs(
    model: GRUModel,
    commodity: str,
    *,
    sequence_length: int = 30,
    prefix: str = 'models/gru/',
    bucket_name: str = BUCKET_NAME,
) -> Optional[str]:
    """Save trained GRU model to GCS."""
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


def load_gru_model_from_gcs(
    commodity: str,
    *,
    sequence_length: int = 30,
    hidden_dim: int = 128,
    layer_dim: int = 3,
    prefix: str = 'models/gru/',
    bucket_name: str = BUCKET_NAME,
) -> Optional[GRUModel]:
    """Load trained GRU model from GCS."""
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
        
        # Load into model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GRUModel(input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=1)
        model.load_state_dict(torch.load(buffer, map_location=device))
        model.to(device)
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
        hidden_dim,
        layer_dim,
        bucket_name,
    ) = args
    
    try:
        print(f"  [{commodity}] Starting training...")
        
        # Recreate series as pandas Series
        series = pd.Series(series_data)
        
        if len(series) < sequence_length + 10:
            print(f"  [{commodity}] Skipping: Insufficient data ({len(series)} points)")
            return (commodity, None, None, f"Insufficient data ({len(series)} points)")
        
        # Prepare sequences
        X_tensor, y_tensor, scaler = prepare_sequences(series, sequence_length)
        
        # Train model (force CPU for parallel training)
        device = torch.device("cpu")
        
        dataset = TimeSeriesDatasetGRU(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = GRUModel(
            input_dim=1, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=1
        )
        model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        patience = 5
        
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save to GCS
        gcs_key = save_gru_model_to_gcs(
            model,
            commodity,
            sequence_length=sequence_length,
            bucket_name=bucket_name,
        )
        
        print(f"  [{commodity}] ✓ Training complete (best loss: {best_loss:.4f})")
        
        return (commodity, model, scaler, "Success")
        
    except Exception as exc:
        print(f"  [{commodity}] ✗ Error: {exc}")
        return (commodity, None, None, str(exc))


def generate_and_save_gru_forecast(
    prices_df: pd.DataFrame,
    commodity_columns: List[str],
    forecast_steps: int,
    gcs_prefix: str,
    *,
    train_new_models: bool = False,
    sequence_length: int = 30,
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    hidden_dim: int = 128,
    layer_dim: int = 3,
    conf_interval_05: bool = False,
    conf_interval_10: bool = False,
    bucket_name: str = BUCKET_NAME,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate GRU forecasts for all commodities and save to GCS.
    
    Args:
        prices_df: Historical price data with DatetimeIndex
        commodity_columns: List of commodity column names to forecast
        forecast_steps: Number of periods to forecast
        gcs_prefix: GCS path prefix for saving forecast
        train_new_models: If True, train new models; if False, load from GCS
        sequence_length: Number of historical steps for input sequence
        num_epochs: Training epochs per commodity
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        hidden_dim: GRU hidden dimension
        layer_dim: Number of GRU layers
        conf_interval_05: Generate 5% confidence intervals (not implemented for GRU)
        conf_interval_10: Generate 10% confidence intervals (not implemented for GRU)
        bucket_name: GCS bucket name
        
    Returns:
        Tuple of (forecast_dataframe, gcs_path)
    """
    
    print("\n" + "=" * 80)
    print("GRU FORECASTING")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sequence length: {sequence_length}")
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
                model = load_gru_model_from_gcs(
                    commodity,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                    layer_dim=layer_dim,
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
                hidden_dim,
                layer_dim,
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
                
                forecast_values = forecast_with_gru(
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
                model = load_gru_model_from_gcs(
                    commodity,
                    sequence_length=sequence_length,
                    hidden_dim=hidden_dim,
                    layer_dim=layer_dim,
                    bucket_name=bucket_name,
                )
                
                if model is None:
                    print(f"  [{commodity}] ⚠ Model not found (skipping)")
                    continue
                
                # Prepare scaler for forecasting
                _, _, scaler = prepare_sequences(series, sequence_length)
                
                # Generate forecast
                last_sequence = series.values[-sequence_length:]
                forecast_values = forecast_with_gru(
                    model, last_sequence, scaler, forecast_steps, device
                )
                
                all_forecasts[commodity] = forecast_values
                print(f"  [{commodity}] ✓ Loaded and forecasted")
                
            except Exception as exc:
                print(f"  [{commodity}] ✗ Error: {exc}")
                continue
    
    if not all_forecasts:
        raise RuntimeError("No GRU forecasts were generated")
    
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
    print("GRU FORECASTING COMPLETE")
    print("="*80 + "\n")
    
    return combined_df, gcs_prefix
