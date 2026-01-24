
# """## GRU FORECASTER
# Build a GRU-based time series forecaster using the data in the `prices_df` DataFrame, generate forecasts, and save the results to a Google Sheet.
# """

# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # Ensure prices_df is available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare GRU data.")
#     # Initialize tensors and scalers as empty if data is not available
#     X_tensor_gru = torch.empty(0, dtype=torch.float32)
#     y_tensor_gru = torch.empty(0, dtype=torch.float32)
#     scalers_gru = {}

# else:
#     # 1. Identify the numerical commodity columns
#     all_columns = prices_df.columns.tolist()
#     # Exclude non-numeric and potentially non-relevant columns like 'date' if it wasn't dropped
#     cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
#                        'day_of_week_sin', 'day_of_week_cos',
#                        'month_sin', 'month_cos',
#                        'year_sin', 'year_cos']
#     commodity_columns = [col for col in all_columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]

#     print("Identified commodity columns for GRU data preparation:")
#     print(commodity_columns)

#     # 2. Define sequence length (number of past time steps to use for prediction)
#     sequence_length = 30 # Example: use the past 30 days to predict the next day

#     # 3. Initialize lists to store sequences and targets for all commodities
#     all_sequences = []
#     all_targets = []

#     # 4. Initialize a dictionary to store scalers for each commodity
#     scalers_gru = {}

#     # 5. Iterate through each identified commodity column
#     print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
#     for commodity in commodity_columns:
#         print(f"  Processing commodity: {commodity}")

#         # Extract the time series data for the current commodity and drop missing values
#         series = prices_df[commodity].dropna()

#         if len(series) < sequence_length + 1:
#             print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#             continue # Skip to the next commodity

#         # Scale the time series data
#         # Using MinMaxScaler as an example, can be changed to StandardScaler or RobustScaler
#         scaler = MinMaxScaler()
#         scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#         scalers_gru[commodity] = scaler # Store the scaler for this commodity

#         # Create input sequences (X) and corresponding target values (y)
#         sequences = []
#         targets = []
#         for i in range(len(scaled_series) - sequence_length):
#             seq = scaled_series[i:i + sequence_length]
#             target = scaled_series[i + sequence_length]
#             sequences.append(seq)
#             targets.append(target)

#         # Store the prepared input sequences and target values for the current commodity
#         all_sequences.extend(sequences)
#         all_targets.extend(targets)
#         print(f"    Generated {len(sequences)} sequences for {commodity}.")


#     # 6. Combine the input sequences and target values into PyTorch tensors
#     if not all_sequences:
#         print("\nNo sequences were generated. Cannot create PyTorch tensors.")
#         X_tensor_gru = torch.empty(0, dtype=torch.float32) # Initialize as empty tensor
#         y_tensor_gru = torch.empty(0, dtype=torch.float32) # Initialize as empty tensor
#         # scalers_gru is already initialized as an empty dictionary

#     else:
#         X_tensor_gru = torch.tensor(all_sequences, dtype=torch.float32)
#         y_tensor_gru = torch.tensor(all_targets, dtype=torch.float32)
#         print("\nCombined sequences and targets into PyTorch tensors.")
#         print("Shape of X_tensor_gru:", X_tensor_gru.shape)
#         print("Shape of y_tensor_gru:", y_tensor_gru.shape)

#     # 7. Confirm that the scalers dictionary is stored
#     if 'scalers_gru' in locals():
#         print("\nScalers for each commodity stored in 'scalers_gru' dictionary.")

# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# # Ensure X_tensor_gru and y_tensor_gru are available from the previous data preparation step
# if 'X_tensor_gru' not in locals() or 'y_tensor_gru' not in locals() or X_tensor_gru.shape[0] == 0:
#     print("Error: X_tensor_gru or y_tensor_gru is not available or is empty. Cannot create Dataset and DataLoader.")
#     # Initialize necessary variables to prevent errors in later steps
#     train_loader_gru = None
#     model_gru = None

# else:
#     print("X_tensor_gru and y_tensor_gru are available. Proceeding with Dataset, DataLoader, and Model definition.")

#     # 1. Define a PyTorch Dataset class for GRU
#     class TimeSeriesDatasetGRU(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             # GRU typically expects input shape (batch_size, sequence_length, input_dim)
#             # Since we are forecasting a single time series value at a time, input_dim is 1
#             # We need to unsqueeze the sequence tensor to add the input_dim dimension
#             return self.X[idx].unsqueeze(-1), self.y[idx] # Add last dimension for input_dim

#     # Instantiate the Dataset
#     dataset_gru = TimeSeriesDatasetGRU(X_tensor_gru, y_tensor_gru)
#     print(f"\nDataset for GRU created with {len(dataset_gru)} samples.")

#     # 2. Create a PyTorch DataLoader
#     batch_size = 64 # Define your desired batch size
#     train_loader_gru = DataLoader(dataset_gru, batch_size=batch_size, shuffle=True)
#     print(f"DataLoader for GRU created with batch size {batch_size}.")
#     print(f"Number of batches in DataLoader: {len(train_loader_gru)}")


#     # 3. Define the GRU model architecture
#     # Assuming sequence_length is available from the data preparation step
#     if 'sequence_length' not in locals():
#         print("Error: 'sequence_length' is not defined. Cannot define GRU model architecture.")
#         model_gru = None # Ensure model_gru is None if sequence_length is missing
#     else:
#         print(f"\nDefining GRU model architecture with sequence length {sequence_length}...")
#         # GRU model parameters - these are example values and may need tuning
#         input_dim = 1       # We are forecasting a single time series value
#         hidden_dim = 128    # Number of features in the hidden state
#         layer_dim = 3       # Number of recurrent layers
#         output_dim = 1      # We are predicting a single next value

#         class GRUModel(nn.Module):
#             def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#                 super().__init__()
#                 self.hidden_dim = hidden_dim
#                 self.layer_dim = layer_dim

#                 # GRU layers
#                 self.gru = nn.GRU(
#                     input_dim, hidden_dim, layer_dim, batch_first=True # batch_first=True means input is (batch_size, sequence_length, input_dim)
#                 )

#                 # Fully connected layer for output
#                 self.fc = nn.Linear(hidden_dim, output_dim)

#             def forward(self, x):
#                 # x shape: (batch_size, sequence_length, input_dim)

#                 # Initialize hidden state with zeros
#                 h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

#                 # We don't need to return the hidden state for this forecasting task, only the output
#                 # out shape: (batch_size, sequence_length, hidden_dim)
#                 # hn shape: (layer_dim, batch_size, hidden_dim)
#                 out, hn = self.gru(x, h0.detach())

#                 # We only need the output from the last time step to predict the next value
#                 last_step_out = out[:, -1, :] # Shape: (batch_size, hidden_dim)

#                 # Pass the output of the last time step through the fully connected layer
#                 prediction = self.fc(last_step_out) # Shape: (batch_size, output_dim)

#                 return prediction

#         model_gru = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

#         print("GRU model architecture defined.")

#         # 4. Instantiate the defined GRU model
#         # model_gru is already instantiated by the GRUModel() call above
#         print("\nGRU model instantiated.")
#         print("Model structure:")
#         print(model_gru)

# import torch.optim as optim
# import torch.nn as nn

# # Ensure model_gru and train_loader_gru are available from the previous step
# if 'model_gru' not in locals() or model_gru is None:
#     print("Error: GRU model is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the model is not defined
#     # No need to initialize other variables as training won't start

# elif 'train_loader_gru' not in locals() or train_loader_gru is None:
#     print("Error: train_loader_gru is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the train_loader_gru is not defined
#     # No need to initialize other variables as training won't start

# else:
#     print("\nProceeding with GRU model training.")

#     # 1. Define the loss function (e.g., Mean Squared Error)
#     criterion = nn.MSELoss()
#     print("Loss function defined (MSELoss).")

#     # 2. Define the optimizer (e.g., Adam)
#     # Define your desired learning rate - MAMBA might benefit from different rates
#     learning_rate = 0.00001
#     optimizer = optim.Adam(model_gru.parameters(), lr=learning_rate)
#     print(f"Optimizer defined (Adam with learning rate {learning_rate}).")

#     # 3. Set the number of training epochs
#     num_epochs = 10 # Define your desired number of epochs - might need more or fewer
#     print(f"Number of training epochs set to: {num_epochs}")

#     # 4. Implement the training loop
#     print("\nStarting GRU model training...")

#     # Move model to the appropriate device if available (CPU or GPU)
#     # Check if CUDA is available and use GPU if it is, otherwise use CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_gru.to(device)
#     print(f"Using device: {device}")


#     for epoch in range(num_epochs):
#         model_gru.train() # Set the model to training mode
#         running_loss = 0.0

#         for i, data in enumerate(train_loader_gru):
#             inputs, targets = data

#             # Move data to the device
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model_gru(inputs)

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
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader_gru):.4f}")

#     print("\nGRU model training complete.")

# import torch
# import os

# # Define the path to save the model (e.g., in Google Drive)
# # Ensure you have mounted Google Drive in a previous cell if saving there
# gru_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_gru_model.pth' # You can change the filename and path

# # Ensure the directory exists
# os.makedirs(os.path.dirname(gru_model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model_gru' not in locals() or model_gru is None:
#     print("Error: GRU model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model_gru.state_dict(), gru_model_save_path)
#         print(f"GRU model state dictionary saved successfully to: {gru_model_save_path}")
#     except Exception as e:
#         print(f"Error saving GRU model: {e}")

# import torch
# import pandas as pd
# import numpy as np
# import os

# # Ensure GRU model architecture is defined (assuming it was defined in a previous cell like kacL81qmSp_t)
# # Ensure prices_df, commodity_columns, sequence_length, and scalers_gru are available

# # Define the path to the saved model
# gru_model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_gru_model.pth' # This should match the save path

# # Check if the saved model file exists
# if not os.path.exists(gru_model_save_path):
#     print(f"Error: Saved GRU model not found at {gru_model_save_path}. Cannot generate forecasts using the saved model.")
#     gru_forecasts = {} # Initialize as empty
# elif 'GRUModel' not in locals():
#     print("Error: GRUModel architecture is not defined. Cannot load the saved model.")
#     gru_forecasts = {} # Initialize as empty
# elif 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare forecast input sequences.")
#     gru_forecasts = {} # Initialize as empty
# elif not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#     print("Error: 'prices_df' index is not a DatetimeIndex named 'Date'. Cannot proceed.")
#     gru_forecasts = {} # Initialize as empty
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot prepare forecast input sequences.")
#     gru_forecasts = {} # Initialize as empty
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot generate forecasts.")
#     gru_forecasts = {} # Initialize as empty
# elif 'scalers_gru' not in locals() or not scalers_gru:
#     print("Error: 'scalers_gru' dictionary not found or is empty. Cannot inverse scale forecasts.")
#     gru_forecasts = {} # Initialize as empty

# else:
#     print("\nLoading the saved GRU model and generating future forecasts...")

#     # Instantiate the GRU model architecture
#     # Ensure input_dim, hidden_dim, layer_dim, output_dim are defined as in the model definition cell (kacL81qmSp_t)
#     # Assuming these variables are available in the environment from previous cell executions
#     try:
#         # Re-instantiate the model with the same architecture parameters used for training
#         # These parameters should be available from the model definition cell (kacL81qmSp_t)
#         # If not explicitly defined globally, you might need to hardcode them or ensure that cell is run first.
#         # Assuming input_dim=1, hidden_dim=128, layer_dim=3, output_dim=1 based on previous successful run
#         input_dim = 1
#         hidden_dim = 128
#         layer_dim = 3
#         output_dim = 1

#         model_gru = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
#         print("GRU model architecture instantiated for loading.")

#         # Load the saved state dictionary
#         model_gru.load_state_dict(torch.load(gru_model_save_path))
#         print(f"GRU model state dictionary loaded from {gru_model_save_path}.")

#         # Move the model to the appropriate device (CPU or GPU) if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_gru.to(device)
#         model_gru.eval() # Set the model to evaluation mode
#         print(f"Using device for forecasting: {device}")


#     except Exception as e:
#         print(f"Error loading GRU model or setting up for forecasting: {e}")
#         gru_forecasts = {} # Initialize as empty
#         model_gru = None # Ensure model_gru is None if loading fails


#     # Initialize a dictionary to store the generated forecasts for each commodity
#     gru_forecasts = {}

#     if model_gru is not None:
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
#                  if commodity not in scalers_gru:
#                       print(f"    Warning: Scaler not found for {commodity}. Skipping forecast.")
#                       continue # Skip to the next commodity
#                  scaler = scalers_gru[commodity]

#                  # Scale the last sequence using the corresponding scaler
#                  # Reshape the sequence to be a 2D array for scaling
#                  scaled_sequence = scaler.transform(last_sequence.values.reshape(-1, 1)).flatten()

#                  # Convert the scaled sequence to a PyTorch tensor
#                  # Reshape to [1, sequence_length, input_dim] as the model expects
#                  sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) # Add batch and input_dim dimensions and move to device

#                  # Initialize a list to store the unscaled forecast values for this commodity
#                  commodity_forecast_values = []

#                  # Generate the forecast step by step
#                  for _ in range(forecast_steps):
#                      # Get the GRU model's prediction for the next step (scaled value)
#                      predicted_scaled_value = model_gru(sequence_tensor).item() # Get the scalar value

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
#                  gru_forecasts[commodity] = commodity_forecast_values
#                  print(f"    Generated {len(commodity_forecast_values)} forecast values for {commodity}.")

#          print("\nGRU forecasting complete.")

#     else:
#         print("GRU model is not available. Cannot generate forecasts.")


# # You now have the generated forecasts in the 'gru_forecasts' dictionary.

# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during data manipulation and saving

# # Ensure prices_df (original data) and gru_forecasts (forecast dictionary) are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot combine data.")
#     # Finish the task with failure
# elif 'gru_forecasts' not in locals() or not gru_forecasts:
#     print("Error: 'gru_forecasts' dictionary not found or is empty. Cannot combine data.")
#     # Finish the task with failure
# else:
#     print("\nCombining original data and GRU forecasts...")

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
#              combined_gru_df = pd.DataFrame() # Initialize as empty
#          else:
#              print(f"Inferred commodity columns: {commodity_columns}")
#              historical_df = prices_df[commodity_columns].copy()
#     # Corrected indentation for this else block
#     else:
#          historical_df = prices_df[commodity_columns].copy()

#     # Ensure historical_df has the correct DatetimeIndex ('Date')
#     historical_df.index.name = 'Date'


#     # Prepare the forecast data:
#     # Create a DataFrame from the gru_forecasts dictionary
#     if gru_forecasts:
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

#             # Define the number of steps to forecast (from the gru_forecasts dictionary)
#             # Assuming all forecast lists in the dictionary have the same length
#             forecast_steps = len(list(gru_forecasts.values())[0]) if gru_forecasts else 0

#             if forecast_steps > 0:
#                  # Generate future dates starting from the day after the last historical date
#                  # Use periods = forecast_steps + 1 and slice from 1 to exclude the last historical date itself
#                  forecast_dates = pd.date_range(start=last_historical_date, periods=forecast_steps + 1, freq=freq)[1:]

#                  # Create a DataFrame from the forecast dictionary, using the generated future dates as the index
#                  forecast_df = pd.DataFrame(gru_forecasts, index=forecast_dates)

#                  # Rename the index to 'Date' for consistency
#                  forecast_df.index.name = 'Date'
#             else:
#                  print("No forecast steps defined based on gru_forecasts dictionary.")
#                  forecast_df = pd.DataFrame() # Create an empty DataFrame

#         else:
#              print("Error: prices_df index is not a DatetimeIndex. Cannot generate forecast dates.")
#              forecast_df = pd.DataFrame() # Create an empty DataFrame


#     else:
#         print("No GRU forecasts were generated to combine.")
#         forecast_df = pd.DataFrame() # Create an empty DataFrame if no forecasts

#     # Combine the historical data and the forecast data
#     # Use outer join to include all dates from both historical and forecast periods
#     # The index from both DataFrames (DatetimeIndex) will be used for alignment
#     combined_gru_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')

#     # Sort by date to ensure chronological order
#     combined_gru_df.sort_index(inplace=True)

#     print("\nCombined GRU DataFrame Head:")
#     print(combined_gru_df.head(2))
#     print("\nCombined GRU DataFrame Tail:")
#     print(combined_gru_df.tail(2))
#     print("\nCombined GRU DataFrame Shape:", combined_gru_df.shape)


# #===============================================================================
# # Save the Combined GRU Data to Google Sheets
# #===============================================================================
# import pandas as pd
# import numpy as np
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings during model fitting

# # Ensure combined_gru_df is available
# if 'combined_gru_df' not in locals() or combined_gru_df.empty:
#     print("Error: 'combined_gru_df' DataFrame not found or is empty. Cannot save GRU forecasts.")
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

#     if sh and not combined_gru_df.empty:
#         try:
#             print("\nAttempting to save combined GRU forecast results to Google Sheets...")
#             sheet_title = 'GRU_FORECAST' # Specify the desired sheet title

#             # Check if the worksheet with the specified title already exists
#             try:
#                 worksheet_gru_forecast = sh.worksheet(sheet_title)
#                 print(f"Worksheet '{sheet_title}' already exists. Clearing existing data.")
#                 worksheet_gru_forecast.clear() # Clear existing data
#             except gspread.WorksheetNotFound:
#                 # If not found, create a new one
#                 # Estimate the number of rows and columns needed
#                 num_rows = combined_gru_df.shape[0] + 1 # Data rows + header
#                 num_cols = combined_gru_df.shape[1] + 1 # Add 1 for the index ('Date')

#                 # gspread has a column limit, check if it's exceeded (usually 256)
#                 max_gspread_cols = 256
#                 if num_cols > max_gspread_cols:
#                      print(f"Warning: Number of columns ({num_cols}) exceeds Google Sheets limit ({max_gspread_cols}). Saving only the first {max_gspread_cols} columns.")
#                      num_cols_to_save = max_gspread_cols
#                      # Include the index column + limited data columns
#                      combined_df_limited = combined_gru_df.reset_index().iloc[:, :num_cols_to_save].copy()
#                 else:
#                      num_cols_to_save = num_cols
#                      combined_df_limited = combined_gru_df.reset_index().copy()


#                 worksheet_gru_forecast = sh.add_worksheet(title=sheet_title, rows=num_rows, cols=num_cols_to_save)
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
#             worksheet_gru_forecast.update(data_to_save)
#             print(f"Combined historical data and GRU forecasts saved to worksheet '{sheet_title}'.")

#         except Exception as e:
#             print(f"Error saving combined GRU forecast results to Google Sheet: {e}")
#     else:
#         print("Google Sheet object 'sh' is not available. Cannot save results.")


# #===============================================================================
# # Visualize GRU Forecasts
# #===============================================================================
# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_gru_df is available from the previous step

# def plot_gru_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and GRU forecast for a specific commodity
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
#         plt.plot(forecast_data_to_plot.index, forecast_data_to_plot.values, label='GRU Forecast', color='red', linestyle='--')

#         # Add a vertical line at the end of the historical data
#         if last_historical_date:
#             plt.axvline(last_historical_date, color='red', linestyle='--', label='Forecast Start')

#     else:
#         # If original prices_df is not available, just plot all data in combined_gru_df
#         print("Warning: Original 'prices_df' not found. Plotting all combined data.")
#         plt.plot(df_combined.index, df_combined[commodity_name], label=f'{commodity_name} Data & GRU Forecast')


#     plt.title(f'GRU Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_gru_df is already loaded

# # Get a list of commodities that were forecasted
# # We can get this from the keys of the gru_forecasts dictionary
# if 'gru_forecasts' in locals() and gru_forecasts:
#     commodities_to_visualize = list(gru_forecasts.keys())[:5] # Visualize the first 5 forecasted commodities
#     print(f"\nVisualizing GRU forecasts for the following commodities: {commodities_to_visualize}")
# else:
#     print("Error: 'gru_forecasts' dictionary not found or is empty. Cannot visualize forecasts.")
#     commodities_to_visualize = []


# if 'combined_gru_df' in locals() and not combined_gru_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_gru_forecast(combined_gru_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_gru_df is not available or empty, or no commodities to visualize.")
