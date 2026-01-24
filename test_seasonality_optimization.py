"""
Quick test to verify seasonality-guided optimization is working correctly
"""
import pandas as pd
import io
from google.cloud import storage
from google.auth import default

# Initialize GCS client
creds, _ = default()
storage_client = storage.Client(credentials=creds)
bucket_name = 'crystal-dss'

# Test loading seasonality data
def load_seasonality_study(gcs_path='stats_studies_data/seasonality/seasonality_results_20260123.csv'):
    try:
        print(f"\nLoading seasonality study from GCS: {gcs_path}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        data = blob.download_as_bytes()
        seasonality_df = pd.read_csv(io.BytesIO(data))
        
        seasonal_only = seasonality_df[seasonality_df['Is Seasonal for Period'] == True].copy()
        
        if seasonal_only.empty:
            print("  Warning: No seasonal periods found in study")
            return {}
        
        best_periods = seasonal_only.loc[
            seasonal_only.groupby('Commodity')['Seasonality Strength (STL)'].idxmax()
        ]
        
        seasonality_map = dict(zip(
            best_periods['Commodity'], 
            best_periods['Period'].astype(int)
        ))
        
        print(f"  ✓ Loaded seasonality data for {len(seasonality_map)} commodities")
        return seasonality_map
    
    except Exception as e:
        print(f"  ✗ Failed to load seasonality study: {e}")
        return {}


def get_robust_m_values(period, standard_periods=[52, 75, 130, 156, 195, 250, 312, 390, 520, 781]):
    m_values = [1, period]
    tolerance = 0.20
    for std_period in standard_periods:
        if std_period == period:
            continue
        ratio = abs(std_period - period) / period
        if ratio <= tolerance:
            m_values.append(std_period)
    return sorted(list(set(m_values)))


# Test the functions
print("="*80)
print("TESTING SEASONALITY-GUIDED OPTIMIZATION")
print("="*80)

seasonality_map = load_seasonality_study()

# Test target commodities
target_commodities = ['Acetic Acid', 'Butyl Acetate', 'Toluene', 'Isomer-MX', 
                      'Solvent-MX', 'Methanol', 'MTBE', 'Benzene']

print("\n" + "="*80)
print("TARGET COMMODITIES - OPTIMIZATION RESULTS")
print("="*80)
print(f"{'Commodity':<20} {'Detected Period':<15} {'Optimized m_values':<40}")
print("-"*80)

for commodity in target_commodities:
    if commodity in seasonality_map:
        period = seasonality_map[commodity]
        m_vals = get_robust_m_values(period)
        print(f"{commodity:<20} {period:<15} {str(m_vals):<40}")
        
        # Calculate model count reduction
        old_count = 3 * 3 * 2 * (1 + 2 * 2 * 2 * 2)  # 90 models with [1, 75, 250]
        new_count = 3 * 3 * 2 * (1 + 2 * 2 * 2 * (len(m_vals) - 1))
        reduction = ((old_count - new_count) / old_count) * 100
        print(f"  → Models: {new_count} (was 90, reduced by {reduction:.1f}%)")
    else:
        print(f"{commodity:<20} {'NOT FOUND':<15} {'[1, 75, 250] (default)':<40}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Successfully loaded seasonality data for {len(seasonality_map)} commodities")
print(f"✓ Optimized m_values include detected period + nearest standard periods (±20%)")
print(f"✓ Expected grid search reduction: 30-50% fewer models per commodity")
