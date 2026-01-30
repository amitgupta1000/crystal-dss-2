"""
Quick test script for unified multivariate Mamba forecasting.
Tests the new architecture with synthetic data.
"""

import torch
import numpy as np
import pandas as pd
from src.mamba import (
    MambaForecaster,
    prepare_multivariate_sequences,
    train_unified_mamba_model,
    forecast_multivariate_mamba,
    MAMBA_AVAILABLE
)

def test_multivariate_mamba():
    """Test unified Mamba with synthetic multivariate data."""
    
    if not MAMBA_AVAILABLE:
        print("❌ mambapy not available. Install with: pip install mambapy")
        return
    
    print("="*80)
    print("TESTING UNIFIED MULTIVARIATE MAMBA")
    print("="*80)
    
    # Create synthetic multivariate time series data
    np.random.seed(42)
    num_timesteps = 200
    num_commodities = 5
    sequence_length = 30
    
    print(f"\nGenerating synthetic data:")
    print(f"  - Timesteps: {num_timesteps}")
    print(f"  - Commodities: {num_commodities}")
    print(f"  - Sequence length: {sequence_length}")
    
    # Create correlated time series to simulate commodity relationships
    dates = pd.date_range('2023-01-01', periods=num_timesteps, freq='D')
    base_trend = np.linspace(100, 150, num_timesteps)
    
    data = {}
    for i in range(num_commodities):
        # Each commodity has base trend + noise + influence from previous commodity
        noise = np.random.randn(num_timesteps) * 5
        if i > 0:
            # Add influence from previous commodity (simulating causality)
            data[f'Commodity_{i}'] = base_trend + noise + 0.3 * data[f'Commodity_{i-1}']
        else:
            data[f'Commodity_{i}'] = base_trend + noise
    
    df = pd.DataFrame(data, index=dates)
    commodity_columns = list(df.columns)
    
    print(f"\n✓ Created synthetic data with shape: {df.shape}")
    
    # Test 1: Prepare multivariate sequences
    print(f"\n{'='*80}")
    print("TEST 1: Prepare Multivariate Sequences")
    print("="*80)
    
    try:
        X_tensor, y_tensor, scalers = prepare_multivariate_sequences(
            df, commodity_columns, sequence_length
        )
        print(f"✓ X_tensor shape: {X_tensor.shape}")
        print(f"✓ y_tensor shape: {y_tensor.shape}")
        print(f"✓ Expected: ({num_timesteps - sequence_length}, {sequence_length}, {num_commodities})")
        print(f"✓ Scalers created: {len(scalers)}")
        
        assert X_tensor.shape == (num_timesteps - sequence_length, sequence_length, num_commodities)
        assert y_tensor.shape == (num_timesteps - sequence_length, num_commodities)
        assert len(scalers) == num_commodities
        print("\n✅ Sequence preparation successful!")
    except Exception as exc:
        print(f"\n❌ Sequence preparation failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Create model architecture
    print(f"\n{'='*80}")
    print("TEST 2: Model Architecture")
    print("="*80)
    
    try:
        model = MambaForecaster(num_commodities=num_commodities, n_layers=2)
        print(f"✓ Model created:")
        print(f"  - Input dimension: {num_commodities}")
        print(f"  - Output dimension: {num_commodities}")
        print(f"  - Layers: 2")
        
        # Test forward pass
        test_input = X_tensor[:4]  # Batch of 4
        output = model(test_input)
        print(f"\n✓ Forward pass:")
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        assert output.shape == (4, num_commodities)
        print("\n✅ Model architecture correct!")
    except Exception as exc:
        print(f"\n❌ Model architecture test failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Training (short)
    print(f"\n{'='*80}")
    print("TEST 3: Training (5 epochs)")
    print("="*80)
    
    try:
        trained_model = train_unified_mamba_model(
            X_tensor,
            y_tensor,
            num_commodities=num_commodities,
            n_layers=2,
            num_epochs=5,
            batch_size=16,
            learning_rate=0.001,
            patience=10,
            device='cpu'
        )
        print("\n✅ Training successful!")
    except Exception as exc:
        print(f"\n❌ Training failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Forecasting
    print(f"\n{'='*80}")
    print("TEST 4: Multivariate Forecasting")
    print("="*80)
    
    try:
        device = torch.device('cpu')
        last_sequences = df.tail(sequence_length).values
        
        forecasts = forecast_multivariate_mamba(
            trained_model,
            last_sequences,
            scalers,
            commodity_columns,
            forecast_steps=10,
            device=device
        )
        
        print(f"✓ Generated forecasts for {len(forecasts)} commodities")
        for commodity, forecast in forecasts.items():
            print(f"  - {commodity}: {len(forecast)} steps, range [{forecast.min():.2f}, {forecast.max():.2f}]")
        
        assert len(forecasts) == num_commodities
        assert all(len(f) == 10 for f in forecasts.values())
        print("\n✅ Forecasting successful!")
    except Exception as exc:
        print(f"\n❌ Forecasting failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nUnified multivariate Mamba is working correctly!")
    print("Ready to use with real commodity data.")

if __name__ == "__main__":
    test_multivariate_mamba()
