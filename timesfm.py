
# !pip install --upgrade --quiet timesfm
# !pip install --upgrade --quiet openpyxl
# #!pip install --upgrade --quiet dask[dataframe] -U -q --user
# !pip install dask[dataframe]==2024.12.1

# # tuples of (import name, install name, min_version)
# packages = [('timesfm', 'timesfm'),]

# import importlib
# install = False
# for package in packages:
#     if not importlib.util.find_spec(package[0]):
#         print(f'installing package {package[1]}')
#         install = True
#         !pip install {package[1]} -U -q --user
#     elif len(package) == 3:
#         if importlib.metadata.version(package[0]) < package[2]:
#             print(f'updating package {package[1]}')
#             install = True
#             !pip install {package[1]} -U -q --user

# import os
# os.environ['JAX_PLATFORMS'] = 'cpu'

# import timesfm
# import numpy as np
# from google.cloud import bigquery
# import pandas as pd
# from matplotlib import pyplot as plt

# from google.colab import userdata
# from google.colab import drive
# drive.mount("/content/drive", force_remount=True)

# import timesfm

# # For PyTorch
# tfm = timesfm.TimesFm(
#       hparams=timesfm.TimesFmHparams(
#           backend="cpu",
#           per_core_batch_size=32,
#           horizon_len=24, # Increased forecast horizon
#           input_patch_len=32,
#           output_patch_len=128,
#           num_layers=50,
#           model_dims=1280,
#           use_positional_embedding=False,
#       ),
#       checkpoint=timesfm.TimesFmCheckpoint(
#           huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
#   )

# type(tfm._model)




# ## Google Colab Set Up
# import gspread
# from google.colab import auth, drive
# from google.auth import default
# import asyncio, aiohttp

# # # Authenticate and create the gspread client
# auth.authenticate_user()
# creds, _ = default()

# # INPUT
# df = prices_df.copy() # Work on a copy to avoid modifying prices_df for other parts of the notebook
# print(df.shape)
# print(df.head(2))

# # Ensure the 'date' column (string format) is handled if it still exists and is not the index.
# # The prices_df now has 'Date' as index (datetime) and 'date' as a string column.
# # For melting, we want to use the datetime index.
# # Let's create a 'ds' column from the index and then melt.

# melted_df = df.reset_index().rename(columns={'Date': 'ds'}) # Make 'Date' index a column named 'ds'
# # Now drop the redundant 'date' string column if it exists, before melting.
# if 'date' in melted_df.columns:
#     melted_df = melted_df.drop(columns=['date'])

# # Melt the DataFrame using 'ds' as the identifier
# melted_df = pd.melt(melted_df, id_vars=['ds'], var_name='unique_id', value_name='y')

# # Ensure 'ds' is timezone-naive as TimesFM expects this sometimes
# melted_df['ds'] = pd.to_datetime(melted_df['ds']) # Ensure it's datetime
# melted_df['ds'] = melted_df['ds'].dt.tz_localize(None) # Convert to timezone-naive if it somehow became localized

# # Clean and convert 'y' column to numeric
# melted_df['y'] = melted_df['y'].astype(str).str.replace(r'[^0-9.-]', '', regex=True) # Retained only digits, period, and hyphen
# melted_df['y'] = pd.to_numeric(melted_df['y'], errors='coerce')  # 'coerce' handles errors by setting invalid values to NaN

# # Drop rows where 'y' is NaN after conversion
# melted_df = melted_df.dropna(subset=['y'])

# print(melted_df.head())

# forecast_df = tfm.forecast_on_df(
#         inputs=melted_df,
#         freq="w",  # Adjust frequency if your data isn't daily
#         value_name='y',
#         num_jobs=8  # Adjust for parallel processing if desired
#     )

# print (forecast_df.shape)
# print (forecast_df.head())

# import matplotlib.pyplot as plt

# combined_df_list = []
# # Iterate through the first 5 unique commodities
# for commodity in melted_df['unique_id'].unique()[:5]:
#     history = melted_df[melted_df['unique_id'] == commodity].set_index('ds')
#     horizon = forecast_df[forecast_df['unique_id'] == commodity].set_index('ds')[:12] # Limit to the first 12 time periods

#     plt.figure(figsize = (12,3))
#     plt.plot(history['y'], linestyle = '-', color = 'blue')
#     plt.plot(horizon['timesfm'], linestyle = '--', color = 'red')
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.4'], horizon['timesfm-q-0.6'], color = 'green', alpha = 1)
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.3'], horizon['timesfm-q-0.7'], color = 'green', alpha = 0.75)
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.2'], horizon['timesfm-q-0.8'], color = 'green', alpha = 0.5)
#     plt.fill_between(horizon.index, horizon['timesfm-q-0.1'], horizon['timesfm-q-0.9'], color = 'green', alpha = 0.25)
#     plt.title(commodity)
#     plt.show()

# forecast_wide_df = forecast_df.copy()
# forecast_wide_df = forecast_wide_df.pivot(index='ds', columns='unique_id', values='timesfm')
# forecast_wide_df = forecast_wide_df.rename_axis('AssessDate').reset_index()
# forecast_wide_df = forecast_wide_df.rename(columns={'AssessDate': 'date'})

# forecast_wide_df.head()
# forecast_wide_df.to_csv('./forecasts.csv', index=False)

# history_wide_df = melted_df.copy()

# # Drop duplicates before pivoting
# history_wide_df = history_wide_df.drop_duplicates(subset=['ds', 'unique_id'])
# history_wide_df = history_wide_df.pivot(index='ds', columns='unique_id', values='y')
# history_wide_df = history_wide_df.rename_axis('date').reset_index()

# history_wide_df.head()
# history_wide_df.to_csv('./history.csv', index=False)

# timesfm_combined_df= pd.concat([history_wide_df, forecast_wide_df], axis=0)
# print (timesfm_combined_df.shape)

# if not timesfm_combined_df.empty:
#     print(f"saving {timesfm_combined_df} to gcs")
#     gcs_prefix_timesfm_combined = 'forecast_data/timesfm_combined.csv'
#     save_to_gcs(
#         df=timesfm_combined_df,
#         gcs_prefix = gcs_prefix_timesfm_combined,
#         validate=False
#         )
#     print(f"saved {timesfm_combined_df} to gcs")
# else:
#     print("timesfm_combined_df is empty. Not saving.")
