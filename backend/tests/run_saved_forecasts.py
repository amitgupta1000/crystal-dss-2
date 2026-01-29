import warnings
from google.auth import default
from google.cloud import storage
import pickle
import pandas as pd
import numpy as np
import sys
import re

# Import helpers from main module
import os, sys
sys.path.insert(0, os.getcwd())
from dss_forecaster import download_model, _is_flatline, commodity_columns, prices_df

creds, _ = default()
client = storage.Client(credentials=creds)

forecast_steps = 52
results = []

# Suppress statsmodels warning about unsupported index for prediction
warnings.filterwarnings("ignore", category=FutureWarning, message="No supported index is available.*")
warnings.filterwarnings("ignore", category=Warning, message="No supported index is available.*")

for commodity in commodity_columns:
    series = prices_df[commodity].dropna()
    if series.empty:
        print(f"Skipping {commodity}: empty series")
        results.append({'Commodity': commodity, 'Status': 'Skipped', 'Reason': 'Empty series'})
        continue

    # try preferred d=1 first
    d_used, model = download_model(commodity, preferred_d=1)
    if model is None:
        d_used, model = download_model(commodity, preferred_d=0)
    if model is None:
        print(f"No saved model found for {commodity}")
        results.append({'Commodity': commodity, 'Status': 'NoModel'})
        continue

    tried_switch = False
    try:
        if hasattr(model, 'forecast'):
            fvals = model.forecast(forecast_steps)
        elif hasattr(model, 'predict'):
            fvals = model.predict(forecast_steps)
        else:
            print(f"Model for {commodity} has no forecast/predict")
            results.append({'Commodity': commodity, 'Status': 'NoForecastMethod'})
            continue
    except Exception as e:
        print(f"Forecast failed for {commodity} with d={d_used}: {e}")
        results.append({'Commodity': commodity, 'Status': 'ForecastFailed', 'Error': str(e)})
        continue

    if _is_flatline(fvals, series.values):
        # force d=0
        if d_used != 0:
            print(f"Flatline detected for {commodity} with d={d_used}; forcing d=0")
            d0, model0 = download_model(commodity, preferred_d=0)
            tried_switch = True
            if model0 is not None:
                try:
                    if hasattr(model0, 'forecast'):
                        fvals = model0.forecast(forecast_steps)
                    elif hasattr(model0, 'predict'):
                        fvals = model0.predict(forecast_steps)
                    d_used = 0
                except Exception as e:
                    print(f"Fallback forecast failed for {commodity} d=0: {e}")

    results.append({'Commodity': commodity, 'Status': 'Success', 'd_used': d_used, 'FlatlineSwitched': tried_switch, 'FirstForecastValues': list(np.array(fvals)[:5])})

res_df = pd.DataFrame(results)
print(res_df)

# Optionally save to CSV
res_df.to_csv('tests/saved_model_forecasts_summary.csv', index=False)
print('Saved summary to tests/saved_model_forecasts_summary.csv')
