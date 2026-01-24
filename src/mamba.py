# ## MAMBA FORECASTER

# Implement a MAMBA forecaster for the time series data in `prices_df`, train it, generate forecasts, combine the forecasts with the historical data, save the result to a Google Sheet, and visualize the forecasts.
# """

# # Commented out IPython magic to ensure Python compatibility.
# # %%capture
# # !pip install mambapy einops

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # 1. Identify the numerical commodity columns
# all_columns = prices_df.columns.tolist()
# commodity_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(prices_df[col])]

# print("Identified commodity columns for MAMBA data preparation:")
# print(commodity_columns)

# # 2. Define sequence length (number of past time steps to use for prediction)
# sequence_length = 30 # Example: use the past 30 days to predict the next day

# # 3. Initialize lists to store sequences and targets for all commodities
# all_sequences = []
# all_targets = []

# # 4. Initialize a dictionary to store scalers for each commodity
# scalers_mamba = {}

# # 5. Iterate through each identified commodity column
# print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
# for commodity in commodity_columns:
#     print(f"  Processing commodity: {commodity}")

#     # 3. For each commodity, extract the time series data
#     series = prices_df[commodity].dropna()

#     if len(series) < sequence_length + 1:
#         print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#         continue # Skip to the next commodity

#     # 4. Scale the time series data
#     scaler = MinMaxScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#     scalers_mamba[commodity] = scaler # Store the scaler for this commodity

#     # 5. Create input sequences (X) and corresponding target values (y)
#     sequences = []
#     targets = []
#     for i in range(len(scaled_series) - sequence_length):
#         seq = scaled_series[i:i + sequence_length]
#         target = scaled_series[i + sequence_length]
#         sequences.append(seq)
#         targets.append(target)

#     # 6. Store the prepared input sequences and target values for the current commodity
#     all_sequences.extend(sequences)
#     all_targets.extend(targets)
#     print(f"    Generated {len(sequences)} sequences for {commodity}.")


# # 7. Combine the input sequences and target values into PyTorch tensors
# if not all_sequences:
#     print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#     # Initialize tensors as empty if no data is processed
#     X_tensor_mamba = torch.empty(0, sequence_length, dtype=torch.float32)
#     y_tensor_mamba = torch.empty(0, dtype=torch.float32)

# else:
#     X_tensor_mamba = torch.tensor(all_sequences, dtype=torch.float32)
#     y_tensor_mamba = torch.tensor(all_targets, dtype=torch.float32)
#     print("\nCombined sequences and targets into PyTorch tensors.")
#     print("Shape of X_tensor_mamba:", X_tensor_mamba.shape)
#     print("Shape of y_tensor_mamba:", y_tensor_mamba.shape)


# # 8. Confirm that the scalers dictionary is stored
# if 'scalers_mamba' in locals():
#     print("\nScalers for each commodity stored in 'scalers_mamba' dictionary.")

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from mambapy.mamba import Mamba, MambaConfig # Import MambaConfig

# # Ensure X_tensor_mamba and y_tensor_mamba are available from the previous data preparation step
# if 'X_tensor_mamba' not in locals() or 'y_tensor_mamba' not in locals() or X_tensor_mamba.shape[0] == 0:
#     print("Error: X_tensor_mamba or y_tensor_mamba is not available or is empty. Cannot create Dataset and DataLoader.")
#     # Initialize necessary variables to prevent errors in later steps
#     train_loader_mamba = None
#     model_mamba = None

# else:
#     print("X_tensor_mamba and y_tensor_mamba are available. Proceeding with Dataset, DataLoader, and Model definition.")

#     # 1. Define a PyTorch Dataset class for MAMBA
#     class TimeSeriesDatasetMamba(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             # MAMBA expects input shape (batch_size, sequence_length, input_dim)
#             # Since we are forecasting a single time series value at a time, input_dim is 1
#             # We need to unsqueeze the sequence tensor to add the input_dim dimension
#             return self.X[idx].unsqueeze(-1), self.y[idx] # Add last dimension for input_dim

#     # Instantiate the Dataset
#     dataset_mamba = TimeSeriesDatasetMamba(X_tensor_mamba, y_tensor_mamba)
#     print(f"\nDataset for MAMBA created with {len(dataset_mamba)} samples.")

#     # 2. Create a PyTorch DataLoader
#     batch_size = 32 # Define your desired batch size (can be the same as KAN)
#     train_loader_mamba = DataLoader(dataset_mamba, batch_size=batch_size, shuffle=True)
#     print(f"DataLoader for MAMBA created with batch size {batch_size}.")
#     print(f"Number of batches in DataLoader: {len(train_loader_mamba)}")


#     # 3. Define the MAMBA model architecture
#     # Assuming sequence_length is available from the data preparation step
#     if 'sequence_length' not in locals():
#         print("Error: 'sequence_length' is not defined. Cannot define MAMBA model architecture.")
#         model_mamba = None # Ensure model_mamba is None if sequence_length is missing
#     else:
#         print(f"\nDefining MAMBA model architecture with sequence length {sequence_length}...")

#         # Define the configuration for the Mamba model using MambaConfig
#         # Adjust d_model and n_layers as needed for your specific task and data complexity
#         # d_model: The dimension of the model embeddings (must match the last dimension of the input tensor)
#         # n_layers: The number of Mamba blocks
#         # Increased n_layers from 2 to 4
#         mamba_config = MambaConfig(d_model=1, n_layers=4) # Increased complexity

#         # Instantiate the base Mamba model with the defined configuration
#         try:
#              base_mamba_model = Mamba(mamba_config) # Instantiate with MambaConfig
#              print("Base Mamba model instantiated with config.")

#              # Add a final linear layer to map MAMBA output to the prediction for the next time step
#              # The output of Mamba is (batch_size, sequence_length, d_model)
#              # We are interested in the last time step's output for prediction
#              class MambaForecaster(nn.Module):
#                  def __init__(self, mamba_model, output_dim=1):
#                      super().__init__()
#                      self.mamba = mamba_model
#                      # The input dimension to the linear layer is the d_model of the Mamba output
#                      self.fc = nn.Linear(mamba_model.config.d_model, output_dim) # Use mamba_model.config.d_model

#                  def forward(self, x):
#                      # x shape: (batch_size, sequence_length, input_dim) where input_dim is 1
#                      mamba_output = self.mamba(x)
#                      # We need the output from the last time step to predict the next value
#                      last_step_output = mamba_output[:, -1, :] # Shape: (batch_size, mamba_model.config.d_model)
#                      prediction = self.fc(last_step_output) # Shape: (batch_size, output_dim)
#                      return prediction

#              # Instantiate the MambaForecaster
#              model_mamba = MambaForecaster(base_mamba_model, output_dim=1)


#              print("MAMBA model architecture defined.")

#              # 4. Instantiate the defined MAMBA model
#              # model_mamba is already instantiated by the MambaForecaster call above if successful
#              if model_mamba is not None:
#                  print("\nMAMBA model instantiated.")
#                  print("Model structure:")
#                  print(model_mamba)
#              else:
#                  print("\nMAMBA model instantiation failed.")

#         except Exception as e:
#              print(f"Error instantiating Mamba model with config: {e}")
#              model_mamba = None # Set to None if model creation fails

# import torch.optim as optim
# import torch.nn as nn

# # Ensure model_mamba and train_loader_mamba are available from the previous step
# if 'model_mamba' not in locals() or model_mamba is None:
#     print("Error: MAMBA model is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the model is not defined
#     # No need to initialize other variables as training won't start

# elif 'train_loader_mamba' not in locals() or train_loader_mamba is None:
#     print("Error: train_loader_mamba is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the train_loader_mamba is not defined
#     # No need to initialize other variables as training won't start

# else:
#     print("\nProceeding with MAMBA model training.")

#     # 1. Define the loss function (e.g., Mean Squared Error)
#     criterion = nn.MSELoss()
#     print("Loss function defined (MSELoss).")

#     # 2. Define the optimizer (e.g., Adam)
#     # Define your desired learning rate - MAMBA might benefit from different rates
#     learning_rate = 0.0001 # Example learning rate, might need tuning
#     optimizer = optim.Adam(model_mamba.parameters(), lr=learning_rate)
#     print(f"Optimizer defined (Adam with learning rate {learning_rate}).")

#     # 3. Set the number of training epochs
#     num_epochs = 10 # Define your desired number of epochs - might need more or fewer
#     print(f"Number of training epochs set to: {num_epochs}")

#     # 4. Implement the training loop
#     print("\nStarting MAMBA model training...")

#     # Move model to the appropriate device if available (CPU or GPU)
#     # Check if CUDA is available and use GPU if it is, otherwise use CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_mamba.to(device)
#     print(f"Using device: {device}")


#     for epoch in range(num_epochs):
#         model_mamba.train() # Set the model to training mode
#         running_loss = 0.0

#         for i, data in enumerate(train_loader_mamba):
#             inputs, targets = data

#             # Move data to the device
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model_mamba(inputs)

#             # Calculate the loss
#             # Ensure targets have the same shape as outputs (usually [batch_size, 1])
#             # If targets is [batch_size], unsqueeze it to [batch_size, 1]
#             if targets.ndim == 1:
#                  targets = targets.unsqueeze(1)

#             loss = criterion(outputs, targets)

#             # Backward pass
#             loss.backward()

#             # Update the weights
#             optimizer.step()

#             # Accumulate the loss
#             running_loss += loss.item()

#         # Print the loss periodically
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader_mamba):.4f}")

#     print("\nMAMBA model training complete.")

# import torch
# import os

# # Define the path to save the model (e.g., in Google Drive)
# # Ensure you have mounted Google Drive in a previous cell (e.g., cell Nv0Vvz_e5Ma2 or SzxtDdaQvm4n) if saving there
# # This path should be consistent when loading the model later
# mamba_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_mamba_model.pth' # You can change the filename and path

# # Ensure the directory exists
# os.makedirs(os.path.dirname(mamba_model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model_mamba' not in locals() or model_mamba is None:
#     print("Error: MAMBA model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model_mamba.state_dict(), mamba_model_save_path)
#         print(f"MAMBA model state dictionary saved successfully to: {mamba_model_save_path}")
#     except Exception as e:
#         print(f"Error saving MAMBA model: {e}")

# import torch
# import pandas as pd
# import numpy as np
# import os
# from mambapy.mamba import Mamba, MambaConfig # Import MambaConfig

# # Define the path to the saved model
# mamba_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_mamba_model.pth' # This should match the save path

# # Ensure MAMBA model architecture classes (Mamba, MambaConfig, MambaForecaster, etc.) are defined as in the model definition cell (cd723792)
# # Ensure prices_df, commodity_columns, sequence_length, and scalers_mamba are available

# # Check if the saved model file exists
# if not os.path.exists(mamba_model_save_path):
#     print(f"Error: Saved MAMBA model not found at {mamba_model_save_path}. Cannot generate forecasts using the saved model.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'Mamba' not in locals() or 'MambaConfig' not in locals() or 'MambaForecaster' not in locals():
#      print("Error: MAMBA model architecture classes (Mamba, MambaConfig, MambaForecaster) are not defined. Cannot load the saved model.")
#      mamba_forecasts = {} # Initialize as empty
# elif 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare forecast input sequences.")
#     mamba_forecasts = {} # Initialize as empty
# elif not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#     print("Error: 'prices_df' index is not a DatetimeIndex named 'Date'. Cannot proceed.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot prepare forecast input sequences.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot generate forecasts.")
#     mamba_forecasts = {} # Initialize as empty
# elif 'scalers_mamba' not in locals() or not scalers_mamba:
#     print("Error: 'scalers_mamba' dictionary not found or is empty. Cannot inverse scale forecasts.")
#     mamba_forecasts = {} # Initialize as empty

# else:
#     print("\nLoading the saved MAMBA model and generating future forecasts...")

#     # Instantiate the MAMBA model architecture
#     # Ensure the same MambaConfig is used as during training (d_model=1, n_layers=4 based on previous step)
#     try:
#         mamba_config = MambaConfig(d_model=1, n_layers=4)
#         base_mamba_model = Mamba(mamba_config)
#         # Instantiate the MambaForecaster with the base Mamba model
#         model_mamba_loaded = MambaForecaster(base_mamba_model, output_dim=1) # Assuming MambaForecaster expects base model and output_dim

#         print("MAMBA model architecture instantiated for loading.")

#         # Load the saved state dictionary
#         model_mamba_loaded.load_state_dict(torch.load(mamba_model_save_path))
#         print(f"MAMBA model state dictionary loaded from {mamba_model_save_path}.")

#         # Move the model to the appropriate device (CPU or GPU) if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_mamba_loaded.to(device)
#         model_mamba_loaded.eval() # Set the model to evaluation mode
#         print(f"Using device for forecasting: {device}")


#     except Exception as e:
#         print(f"Error loading MAMBA model or setting up for forecasting: {e}")
#         mamba_forecasts = {} # Initialize as empty
#         model_mamba_loaded = None # Ensure model_mamba_loaded is None if loading fails


#     # Initialize a dictionary to store the generated forecasts for each commodity
#     mamba_forecasts = {}

#     if model_mamba_loaded is not None:
#          # Define the number of steps to forecast
#          forecast_steps = 60 # This should align with other forecasters if possible
#          print(f"\nGenerating {forecast_steps} steps into the future for each commodity...")

#          # Iterate through each commodity for which an input sequence can be prepared
#          print(f"Generating forecasts for {len(commodity_columns)} commodities...")
#          with torch.no_grad(): # Disable gradient calculation during inference
#              for commodity in commodity_columns:
#                  print(f"  Forecasting for: {commodity}")

#                  # Get the time series for the current commodity and drop missing values
#                  series = prices_df[commodity].dropna()

#                  # Check if the series has enough data points for the initial sequence
#                  if len(series) < sequence_length:
#                      print(f"    Warning: Skipping {commodity}. Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#                      continue # Skip to the next commodity

#                  # Get the last `sequence_length` data points as the initial input sequence
#                  last_sequence = series[-sequence_length:]

#                  # Get the corresponding scaler for this commodity
#                  if commodity not in scalers_mamba:
#                       print(f"    Warning: Scaler not found for {commodity}. Skipping forecast.")
#                       continue # Skip to the next commodity
#                  scaler = scalers_mamba[commodity]

#                  # Scale the last sequence using the corresponding scaler
#                  # Reshape the sequence to be a 2D array for scaling
#                  scaled_sequence = scaler.transform(last_sequence.values.reshape(-1, 1)).flatten()

#                  # Convert the scaled sequence to a PyTorch tensor
#                  # Reshape to [1, sequence_length, input_dim] as the model expects
#                  # input_dim is 1 for single time series
#                  sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # Add batch and input_dim dimensions and move to device

#                  # Initialize a list to store the unscaled forecast values for this commodity
#                  commodity_forecast_values = []

#                  # Generate the forecast step by step
#                  for _ in range(forecast_steps):
#                      # Get the MAMBA model's prediction for the next step (scaled value)
#                      predicted_scaled_value = model_mamba_loaded(sequence_tensor).item() # Get the scalar value

#                      # Inverse scale the predicted value to the original price scale
#                      # The scaler expects a 2D array, so reshape the scalar
#                      predicted_original_value = scaler.inverse_transform([[predicted_scaled_value]])[0][0]

#                      # Store the inverse-scaled forecast value
#                      commodity_forecast_values.append(predicted_original_value)

#                      # Update the input sequence for the next prediction
#                      # Remove the oldest value and append the new scaled prediction
#                      # The new prediction needs to be reshaped to match the input tensor shape
#                      new_sequence_tensor = torch.cat((sequence_tensor[:, 1:, :], torch.tensor([[predicted_scaled_value]], dtype=torch.float32).unsqueeze(0).to(device)), dim=1)
#                      sequence_tensor = new_sequence_tensor # Use the new sequence for the next step

#                  # Store the generated forecast values for the current commodity
#                  mamba_forecasts[commodity] = commodity_forecast_values
#                  print(f"    Generated {len(commodity_forecast_values)} forecast values for {commodity}.")

#          print("\nMAMBA forecasting complete.")

#     else:
#         print("MAMBA model is not available. Cannot generate forecasts.")


# # You now have the generated forecasts in the 'mamba_forecasts' dictionary.

# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during data manipulation and saving

# # Ensure prices_df (original data) and mamba_forecasts (forecast dictionary) are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot combine data.")
#     # Finish the task with failure
# elif 'mamba_forecasts' not in locals() or not mamba_forecasts:
#     print("Error: 'mamba_forecasts' dictionary not found or is empty. Cannot combine data.")
#     # Finish the task with failure
# else:
#     print("\nCombining original data and MAMBA forecasts...")

#     # Prepare the original data: select commodity columns and ensure DatetimeIndex
#     # Assuming the index of prices_df is already the correct DatetimeIndex ('Date')
#     # and numeric columns are identified in commodity_columns list
#     # Ensure commodity_columns is available; if not, try to infer from prices_df
#     if 'commodity_columns' not in locals() or not commodity_columns:
#          print("Warning: 'commodity_columns' not found. Attempting to infer from prices_df.")
#          all_cols = prices_df.columns.tolist()
#          cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
#                             'day_of_week_sin', 'day_of_week_cos',
#                             'month_sin', 'month_cos',
#                             'year_sin', 'year_cos']
#          commodity_columns = [col for col in all_cols if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]
#          if not commodity_columns:
#              print("Error: Could not identify numeric commodity columns from prices_df.")
#              # Finish the task with failure
#              combined_mamba_df = pd.DataFrame() # Initialize as empty
#          else:
#              print(f"Inferred commodity columns: {commodity_columns}")
#              historical_df = prices_df[commodity_columns].copy()
#     # Corrected indentation for this else block
#     else:
#          historical_df = prices_df[commodity_columns].copy()

#     # Ensure historical_df has the correct DatetimeIndex ('Date')
#     historical_df.index.name = 'Date'


#     # Prepare the forecast data:
#     # Create a DataFrame from the mamba_forecasts dictionary
#     if mamba_forecasts:
#         # Get the last date from the historical data to start the forecast index from the next day
#         last_historical_date = prices_df.index.max()

#         # Determine the frequency of the historical data
#         # Ensure prices_df index is DatetimeIndex before inferring frequency
#         if isinstance(prices_df.index, pd.DatetimeIndex):
#             freq = pd.infer_freq(prices_df.index)
#             if freq is None:
#                  # Fallback to 'D' if frequency cannot be inferred (assuming daily data)
#                  freq = 'D'
#                  print(f"Warning: Could not infer frequency from historical data. Assuming '{freq}'.")
#             else:
#                 print(f"Inferred historical data frequency: {freq}")

#             # Define the number of steps to forecast (from the mamba_forecasts dictionary)
#             # Assuming all forecast lists in the dictionary have the same length
#             forecast_steps = len(list(mamba_forecasts.values())[0]) if mamba_forecasts else 0

#             if forecast_steps > 0:
#                  # Generate future dates starting from the day after the last historical date
#                  # Use periods = forecast_steps + 1 and slice from 1 to exclude the last historical date itself
#                  forecast_dates = pd.date_range(start=last_historical_date, periods=forecast_steps + 1, freq=freq)[1:]

#                  # Create a DataFrame from the forecast dictionary, using the generated future dates as the index
#                  forecast_df = pd.DataFrame(mamba_forecasts, index=forecast_dates)

#                  # Rename the index to 'Date' for consistency
#                  forecast_df.index.name = 'Date'
#             else:
#                  print("No forecast steps defined based on mamba_forecasts dictionary.")
#                  forecast_df = pd.DataFrame() # Create an empty DataFrame

#         else:
#              print("Error: prices_df index is not a DatetimeIndex. Cannot generate forecast dates.")
#              forecast_df = pd.DataFrame() # Create an empty DataFrame


#     else:
#         print("No MAMBA forecasts were generated to combine.")
#         forecast_df = pd.DataFrame() # Create an empty DataFrame if no forecasts

#     # Combine the historical data and the forecast data
#     # Use outer join to include all dates from both historical and forecast periods
#     # The index from both DataFrames (DatetimeIndex) will be used for alignment
#     combined_mamba_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')

#     # Sort by date to ensure chronological order
#     combined_mamba_df.sort_index(inplace=True)

#     print("\nCombined MAMBA DataFrame Head:")
#     print(combined_mamba_df.head(2))
#     print("\nCombined MAMBA DataFrame Tail:")
#     print(combined_mamba_df.tail(2))
#     print("\nCombined MAMBA DataFrame Shape:", combined_mamba_df.shape)


# #===============================================================================
# # Save the Combined MAMBA Data to Google Sheets
# #===============================================================================
# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during model fitting

# # Ensure combined_mamba_df is available
# if 'combined_mamba_df' not in locals() or combined_mamba_df.empty:
#     print("Error: 'combined_mamba_df' DataFrame not found or is empty. Cannot save MAMBA forecasts.")
# else:
#     # Ensure gc (gspread client) and sh (Google Sheet object) are available from previous cells
#     # Assuming gc and sh are already initialized and authenticated
#     if 'sh' not in locals():
#         print("Google Sheet object 'sh' not found. Attempting to authenticate and open sheet.")
#         try:
#             auth.authenticate_user()
#             creds, _ = default()
#             gc = gspread.authorize(creds)
#             sh = gc.open('Quant_Calc_Main')
#             print("Authenticated and opened Google Sheet: Quant_Calc_Main")
#         except Exception as e:
#             print(f"Failed to authenticate or open Google Sheet: {e}")
#             sh = None # Ensure sh is None if fallback fails

#     if sh and not combined_mamba_df.empty:
#         try:
#             print("\nAttempting to save combined MAMBA forecast results to Google Sheets...")
#             sheet_title = 'MAMBA_FORECAST' # Specify the desired sheet title

#             # Check if the worksheet with the specified title already exists
#             try:
#                 worksheet_mamba_forecast = sh.worksheet(sheet_title)
#                 print(f"Worksheet '{sheet_title}' already exists. Clearing existing data.")
#                 worksheet_mamba_forecast.clear() # Clear existing data
#             except gspread.WorksheetNotFound:
#                 # If not found, create a new one
#                 # Estimate the number of rows and columns needed
#                 num_rows = combined_mamba_df.shape[0] + 1 # Data rows + header
#                 num_cols = combined_mamba_df.shape[1] + 1 # Add 1 for the index ('Date')

#                 # gspread has a column limit, check if it's exceeded (usually 256)
#                 max_gspread_cols = 256
#                 if num_cols > max_gspread_cols:
#                      print(f"Warning: Number of columns ({num_cols}) exceeds Google Sheets limit ({max_gspread_cols}). Saving only the first {max_gspread_cols} columns.")
#                      num_cols_to_save = max_gspread_cols
#                      # Include the index column + limited data columns
#                      combined_df_limited = combined_mamba_df.reset_index().iloc[:, :num_cols_to_save].copy()
#                 else:
#                      num_cols_to_save = num_cols
#                      combined_df_limited = combined_mamba_df.reset_index().copy()


#                 worksheet_mamba_forecast = sh.add_worksheet(title=sheet_title, rows=num_rows, cols=num_cols_to_save)
#                 print(f"Worksheet '{sheet_title}' created.")

#             # Convert the DataFrame to a list of lists for gspread, including headers
#             # Ensure 'Date' column is string and format it nicely
#             # Use .dt accessor now that we've ensured it's datetime index
#             if 'Date' in combined_df_limited.columns:
#                  if pd.api.types.is_datetime64_any_dtype(combined_df_limited['Date']):
#                       combined_df_limited['Date'] = combined_df_limited['Date'].dt.strftime('%Y-%m-%d')
#                  combined_df_limited['Date'].fillna('', inplace=True) # Fill any potential NaT dates with an empty string or placeholder if needed


#             # Convert all other data columns to string, handling potential NaN values
#             # Replace NaN with empty string for cleaner representation in Google Sheets
#             data_to_save = [combined_df_limited.columns.tolist()] + combined_df_limited.fillna('').astype(str).values.tolist()


#             # Update the worksheet with the data
#             # gspread expects a list of lists where each inner list is a row
#             worksheet_mamba_forecast.update(data_to_save)
#             print(f"Combined historical data and MAMBA forecasts saved to worksheet '{sheet_title}'.")

#         except Exception as e:
#             print(f"Error saving combined MAMBA forecast results to Google Sheet: {e}")
#     else:
#         print("Google Sheet object 'sh' is not available. Cannot save results.")

# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_mamba_df is available from the previous step

# def plot_mamba_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and MAMBA forecast for a specific commodity
#     from the combined DataFrame.

#     Args:
#         df_combined (pd.DataFrame): DataFrame containing combined historical and
#                                     forecast data with 'Date' as index.
#         commodity_name (str): The name of the commodity to plot.
#     """
#     if commodity_name not in df_combined.columns:
#         print(f"Error: Commodity '{commodity_name}' not found in the DataFrame.")
#         return

#     plt.figure(figsize=(12, 3))

#     # Plot historical data (non-NaN values in the historical period)
#     # Assuming historical data ends before the forecast period starts
#     # Find the index where the forecast starts (first non-NaN value in the forecast period)
#     # We can infer this from the first non-NaN value in the combined df after the end of original prices_df
#     if 'prices_df' in locals() and not prices_df.empty:
#         last_historical_date = prices_df.index.max()
#         historical_data_to_plot = df_combined[df_combined.index <= last_historical_date][commodity_name].dropna()
#         forecast_data_to_plot = df_combined[df_combined.index > last_historical_date][commodity_name].dropna()

#         plt.plot(historical_data_to_plot.index, historical_data_to_plot.values, label='Historical Data', color='blue')
#         plt.plot(forecast_data_to_plot.index, forecast_data_to_plot.values, label='MAMBA Forecast', color='red', linestyle='--')

#         # Add a vertical line at the end of the historical data
#         if last_historical_date:
#             plt.axvline(last_historical_date, color='red', linestyle='--', label='Forecast Start')

#     else:
#         # If original prices_df is not available, just plot all data in combined_mamba_df
#         print("Warning: Original 'prices_df' not found. Plotting all combined data.")
#         plt.plot(df_combined.index, df_combined[commodity_name], label=f'{commodity_name} Data & MAMBA Forecast')


#     plt.title(f'MAMBA Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_mamba_df is already loaded

# # Get a list of commodities that were forecasted
# # We can get this from the keys of the mamba_forecasts dictionary
# if 'mamba_forecasts' in locals() and mamba_forecasts:
#     commodities_to_visualize = list(mamba_forecasts.keys())[:5] # Visualize the first 5 forecasted commodities
#     print(f"\nVisualizing MAMBA forecasts for the following commodities: {commodities_to_visualize}")
# else:
#     print("Error: 'mamba_forecasts' dictionary not found or is empty. Cannot visualize forecasts.")
#     commodities_to_visualize = []


# if 'combined_mamba_df' in locals() and not combined_mamba_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_mamba_forecast(combined_mamba_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_mamba_df is not available or empty, or no commodities to visualize.")

# """## Summary:

# ### Data Analysis Key Findings
# *   The `mambapy` library was successfully installed.
# *   An error occurred in the MAMBA forecaster section where the data preparation step failed due to the `prices_df` variable not being defined, indicating an interruption in the execution flow.

# ### Insights or Next Steps
# *   The immediate next step is to re-run the data preparation cell to ensure `prices_df` and other essential variables for the MAMBA forecaster are correctly initialized before proceeding with further analysis.

# # Task
# #===============================================================================

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # 1. Identify the numerical commodity columns
# all_columns = prices_df.columns.tolist()
# commodity_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(prices_df[col])]

# print("Identified commodity columns for MAMBA data preparation:")
# print(commodity_columns)

# # 2. Define sequence length (number of past time steps to use for prediction)
# sequence_length = 30 # Example: use the past 30 days to predict the next day

# # 3. Initialize lists to store sequences and targets for all commodities
# all_sequences = []
# all_targets = []

# # 4. Initialize a dictionary to store scalers for each commodity
# scalers_mamba = {}

# # 5. Iterate through each identified commodity column
# print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
# for commodity in commodity_columns:
#     print(f"  Processing commodity: {commodity}")

#     # 3. For each commodity, extract the time series data
#     series = prices_df[commodity].dropna()

#     if len(series) < sequence_length + 1:
#         print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#         continue # Skip to the next commodity

#     # 4. Scale the time series data
#     scaler = MinMaxScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#     scalers_mamba[commodity] = scaler # Store the scaler for this commodity

#     # 5. Create input sequences (X) and corresponding target values (y)
#     sequences = []
#     targets = []
#     for i in range(len(scaled_series) - sequence_length):
#         seq = scaled_series[i:i + sequence_length]
#         target = scaled_series[i + sequence_length]
#         sequences.append(seq)
#         targets.append(target)

#     # 6. Store the prepared input sequences and target values for the current commodity
#     all_sequences.extend(sequences)
#     all_targets.extend(targets)
#     print(f"    Generated {len(sequences)} sequences for {commodity}.")


# # 7. Combine the input sequences and target values into PyTorch tensors
# if not all_sequences:
#     print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#     # Initialize tensors as empty if no data is processed
#     X_tensor_mamba = torch.empty(0, sequence_length, dtype=torch.float32)
#     y_tensor_mamba = torch.empty(0, dtype=torch.float32)

# else:
#     X_tensor_mamba = torch.tensor(all_sequences, dtype=torch.float32)
#     y_tensor_mamba = torch.tensor(all_targets, dtype=torch.float32)
#     print("\nCombined sequences and targets into PyTorch tensors.")
#     print("Shape of X_tensor_mamba:", X_tensor_mamba.shape)
#     print("Shape of y_tensor_mamba:", y_tensor_mamba.shape)


# # 8. Confirm that the scalers dictionary is stored
# if 'scalers_mamba' in locals():
#     print("\nScalers for each commodity stored in 'scalers_mamba' dictionary.")



    

# # Define the path to save the model (e.g., in Google Drive)
# # Ensure you have mounted Google Drive in a previous cell (e.g., cell SzxtDdaQvm4n) if saving there
# mamba_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_mamba_model.pth' # You can change the filename and path

# # Ensure the directory exists
# os.makedirs(os.path.dirname(mamba_model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model_mamba' not in locals() or model_mamba is None:
#     print("Error: MAMBA model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model_mamba.state_dict(), mamba_model_save_path)
#         print(f"MAMBA model state dictionary saved successfully to: {mamba_model_save_path}")
#     except Exception as e:
#         print(f"Error saving MAMBA model: {e}")
# #===============================================================================