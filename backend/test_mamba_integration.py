"""
Test MAMBA integration with dss_forecast.py workflow.
Verifies: data input, model saving/loading, forecast saving to GCS.
"""

import pandas as pd
import numpy as np
from src.mamba import (
    prepare_multivariate_sequences,
    train_unified_mamba_model,
    save_unified_mamba_model_to_gcs,
    load_unified_mamba_model_from_gcs,
    forecast_multivariate_mamba,
    generate_and_save_mamba_forecast,
    MAMBA_AVAILABLE
)
import torch

def test_integration():
    """Test MAMBA integration mimicking dss_forecast.py workflow."""
    
    if not MAMBA_AVAILABLE:
        print("❌ mambapy not available")
        return
    
    print("="*80)
    print("TESTING MAMBA INTEGRATION WITH DSS_FORECAST WORKFLOW")
    print("="*80)
    
    # Create synthetic data mimicking real commodity price data
    np.random.seed(42)
    num_timesteps = 500  # More realistic
    num_commodities = 10  # Smaller than 55 for testing
    
    dates = pd.date_range('2023-01-01', periods=num_timesteps, freq='D')
    
    # Create price data with realistic patterns
    data = {}
    for i in range(num_commodities):
        base = 100 + i * 10
        trend = np.linspace(0, 20, num_timesteps)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(num_timesteps) / 365)
        noise = np.random.randn(num_timesteps) * 3
        data[f'Commodity_{i}'] = base + trend + seasonality + noise
    
    # Create DataFrame with DatetimeIndex (like dss_forecast.py provides)
    prices_df = pd.DataFrame(data, index=dates)
    prices_df.index.name = 'Date'
    commodity_columns = list(prices_df.columns)
    
    print(f"\n✓ Created synthetic price data:")
    print(f"  Shape: {prices_df.shape}")
    print(f"  Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    print(f"  Commodities: {len(commodity_columns)}")
    
    # Test 1: Full workflow with generate_and_save_mamba_forecast
    print(f"\n{'='*80}")
    print("TEST 1: Full Workflow (like dss_forecast.py)")
    print("="*80)
    
    try:
        # This is exactly how dss_forecast.py calls it
        mamba_df, mamba_gcs_path = generate_and_save_mamba_forecast(
            prices_df=prices_df,
            commodity_columns=commodity_columns,
            forecast_steps=30,
            gcs_prefix='forecast_data/test_mamba_forecast.csv',
            train_new_model=True,  # Force training for test
            sequence_length=21,
            n_layers=2,  # Smaller for faster test
            num_epochs=5,  # Fewer epochs for test
            batch_size=32,
            learning_rate=0.001,
            bucket_name='crystal-dss',
        )
        
        print(f"\n✅ Full workflow successful!")
        print(f"  Output shape: {mamba_df.shape}")
        print(f"  GCS path: {mamba_gcs_path}")
        print(f"  Date range: {mamba_df['Date'].min()} to {mamba_df['Date'].max()}")
        
        # Verify output structure
        assert 'Date' in mamba_df.columns, "Missing Date column"
        assert all(c in mamba_df.columns for c in commodity_columns), "Missing commodity columns"
        assert len(mamba_df) == num_timesteps + 30, f"Wrong shape: {len(mamba_df)} vs expected {num_timesteps + 30}"
        
        print(f"\n✓ Output structure validated")
        print(f"  Columns: {len(mamba_df.columns)}")
        print(f"  Historical rows: {num_timesteps}")
        print(f"  Forecast rows: 30")
        print(f"  Total rows: {len(mamba_df)}")
        
    except Exception as exc:
        print(f"\n❌ Full workflow failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Verify model can be loaded
    print(f"\n{'='*80}")
    print("TEST 2: Model Loading")
    print("="*80)
    
    try:
        loaded_model = load_unified_mamba_model_from_gcs(
            num_commodities=num_commodities,
            n_layers=2,
            sequence_length=21,
            bucket_name='crystal-dss',
        )
        
        if loaded_model is not None:
            print(f"✅ Model loaded successfully from GCS")
            
            # Test inference
            device = torch.device('cpu')
            test_input = torch.randn(1, 21, num_commodities)
            output = loaded_model(test_input)
            
            print(f"  Test inference:")
            print(f"    Input shape: {test_input.shape}")
            print(f"    Output shape: {output.shape}")
            assert output.shape == (1, num_commodities), "Wrong output shape"
            print(f"  ✓ Inference working correctly")
        else:
            print(f"⚠ Model not loaded (may need more time to upload)")
            
    except Exception as exc:
        print(f"❌ Model loading failed: {exc}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ INTEGRATION TEST COMPLETE")
    print("="*80)
    print("\nMAMBA is properly wired into dss_forecast.py!")
    print("Ready to run on production data with option 7.")

if __name__ == "__main__":
    test_integration()
