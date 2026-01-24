
# """## KAN FORECASTER
# Forecast prices using KAN with "prices_df" as historical data.
# """

# # Commented out IPython magic to ensure Python compatibility.
# # %%capture
# # !pip install pykan

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
# from kan import KAN
# import torch

# # 1. Identify the numerical commodity columns
# all_columns = prices_df.columns.tolist()
# commodity_columns = [col for col in all_columns if pd.api.types.is_numeric_dtype(prices_df[col])]

# print("Identified commodity columns for KAN data preparation:")
# print(commodity_columns)

# # Define sequence length (number of past time steps to use for prediction)
# sequence_length = 20 # Example: use the past 30 days to predict the next day

# # Initialize lists to store sequences and targets for all commodities
# all_sequences = []
# all_targets = []

# # Initialize a dictionary to store scalers for each commodity
# scalers = {}

# # 2. Iterate through each identified commodity column
# print(f"\nPreparing data sequences for {len(commodity_columns)} commodities with sequence length {sequence_length}...")
# for commodity in commodity_columns:
#     print(f"  Processing commodity: {commodity}")

#     # 3. For each commodity, extract the time series data
#     series = prices_df[commodity].dropna()

#     if len(series) < sequence_length + 1:
#         print(f"  Skipping {commodity}: Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#         continue # Skip to the next commodity

#     # 4. Scale the time series data
#     scaler = RobustScaler()
#     scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
#     scalers[commodity] = scaler # Store the scaler for this commodity

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
#     X_tensor = torch.empty(0, sequence_length, dtype=torch.float32)
#     y_tensor = torch.empty(0, dtype=torch.float32)

# else:
#     X_tensor = torch.tensor(all_sequences, dtype=torch.float32)
#     y_tensor = torch.tensor(all_targets, dtype=torch.float32)
#     print("\nCombined sequences and targets into PyTorch tensors.")
#     print("Shape of X_tensor:", X_tensor.shape)
#     print("Shape of y_tensor:", y_tensor.shape)


# # Store the scalers dictionary as it will be needed for inverse transformation of forecasts
# # Assuming 'scalers' is already defined and populated
# if 'scalers' in locals():
#     print("\nScalers for each commodity stored in 'scalers' dictionary.")

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from kan import KAN # Assuming kan library is installed and imported in a previous cell

# # Ensure X_tensor and y_tensor are available from the previous data preparation step
# if 'X_tensor' not in locals() or 'y_tensor' not in locals() or X_tensor.shape[0] == 0:
#     print("Error: X_tensor or y_tensor is not available or is empty. Cannot create Dataset and DataLoader.")
#     # Initialize necessary variables to prevent errors in later steps
#     train_loader = None
#     model = None

# else:
#     print("X_tensor and y_tensor are available. Proceeding with Dataset, DataLoader, and Model definition.")

#     # 1. Define a PyTorch Dataset class
#     class TimeSeriesDataset(Dataset):
#         def __init__(self, X, y):
#             self.X = X
#             self.y = y

#         def __len__(self):
#             return len(self.X)

#         def __getitem__(self, idx):
#             return self.X[idx], self.y[idx]

#     # Instantiate the Dataset
#     # Assuming X_tensor and y_tensor are already created from the data preparation step
#     dataset = TimeSeriesDataset(X_tensor, y_tensor)
#     print(f"\nDataset created with {len(dataset)} samples.")

#     # 2. Create a PyTorch DataLoader
#     batch_size = 32 # Define your desired batch size
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     print(f"DataLoader created with batch size {batch_size}.")
#     print(f"Number of batches in DataLoader: {len(train_loader)}")


#     # 3. Define the KAN model architecture
#     # Assuming sequence_length is available from the data preparation step
#     if 'sequence_length' not in locals():
#         print("Error: 'sequence_length' is not defined. Cannot define KAN model architecture.")
#         model = None # Ensure model is None if sequence_length is missing
#     else:
#         print(f"\nDefining KAN model architecture with input dimension {sequence_length}...")
#         # The KAN model takes a flattened input, so the input dimension is sequence_length
#         # The output dimension is 1 for predicting a single next value
#         # Added an extra layer with 8 nodes
#         model = KAN(width=[sequence_length, 16, 8, 1]) # Example architecture: input -> 16 KAN nodes -> 8 KAN nodes -> 1 output

#         print("KAN model architecture defined.")

#         # 4. Instantiate the defined KAN model
#         # model is already instantiated by the kan.KAN() call above
#         print("\nKAN model instantiated.")
#         print("Model structure:")
#         print(model)

# import torch.optim as optim
# import torch.nn as nn

# # Ensure model and train_loader are available from the previous step
# if 'model' not in locals() or model is None:
#     print("Error: KAN model is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the model is not defined
#     # No need to initialize other variables as training won't start

# elif 'train_loader' not in locals() or train_loader is None:
#     print("Error: train_loader is not defined. Cannot proceed with training.")
#     # Finish the task with failure if the train_loader is not defined
#     # No need to initialize other variables as training won't start

# else:
#     print("\nProceeding with KAN model training.")

#     # 1. Define the loss function (e.g., Mean Squared Error)
#     criterion = nn.MSELoss()
#     print("Loss function defined (MSELoss).")

#     # 2. Define the optimizer (e.g., Adam)
#     # Updated learning rate as requested by the user
#     optimizer = optim.Adam(model.parameters(), lr=0.00002) # Define your desired learning rate
#     print("Optimizer defined (Adam with learning rate 0.00002).")

#     # 3. Set the number of training epochs
#     num_epochs = 10 # Define your desired number of epochs
#     print(f"Number of training epochs set to: {num_epochs}")

#     # 4. Implement the training loop
#     print("\nStarting KAN model training...")

#     # Move model to the appropriate device if available (CPU or GPU)
#     # Check if CUDA is available and use GPU if it is, otherwise use CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     print(f"Using device: {device}")


#     for epoch in range(num_epochs):
#         model.train() # Set the model to training mode
#         running_loss = 0.0

#         for i, data in enumerate(train_loader):
#             inputs, targets = data

#             # Move data to the device
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(inputs)

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
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

#     print("\nKAN model training complete.")

# import torch
# import os

# # Define the path to save the model on Google Drive
# # Ensure you have mounted Google Drive in a previous cell (e.g., cell Nv0Vvz_e5Ma2)
# model_save_path = '/content/drive/MyDrive/Colab Notebooks/quant_kan_model.pth' # You can change the filename

# # Ensure the directory exists
# os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# # Ensure the model is available from the training step
# if 'model' not in locals() or model is None:
#     print("Error: KAN model is not defined. Cannot save the model.")
# else:
#     try:
#         # Save the model's state dictionary
#         torch.save(model.state_dict(), model_save_path)
#         print(f"KAN model state dictionary saved successfully to: {model_save_path}")
#     except Exception as e:
#         print(f"Error saving KAN model to Google Drive: {e}")

# import torch
# import pandas as pd
# import numpy as np

# # Ensure prices_df, commodity_columns, sequence_length, and scalers are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot prepare forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif not isinstance(prices_df.index, pd.DatetimeIndex) or prices_df.index.name != 'Date':
#     print("Error: 'prices_df' index is not a DatetimeIndex named 'Date'. Cannot proceed.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot prepare forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot prepare forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# elif 'scalers' not in locals() or not scalers:
#     print("Error: 'scalers' dictionary not found or is empty. Cannot scale forecast input sequences.")
#     forecast_input_sequences = {} # Initialize as empty to prevent errors
# else:
#     print("\nPreparing forecast input sequences for KAN model...")

#     # Initialize an empty dictionary to store the last sequence for each commodity
#     forecast_input_sequences = {}

#     # Iterate through each identified commodity column
#     print(f"Extracting and scaling the last {sequence_length} data points for each commodity...")
#     for commodity in commodity_columns:
#         # Extract the time series data for the current commodity and drop missing values
#         series = prices_df[commodity].dropna()

#         # Check if the series has enough data points
#         if len(series) < sequence_length:
#             print(f"  Warning: Skipping {commodity}. Insufficient data ({len(series)} data points) for sequence length {sequence_length}.")
#             continue # Skip to the next commodity

#         # Get the last `sequence_length` data points
#         last_sequence = series[-sequence_length:]

#         # Scale the last sequence using the corresponding scaler
#         if commodity in scalers:
#             scaler = scalers[commodity]
#             # Reshape the sequence to be a 2D array for scaling (MinMaxScaler expects 2D input)
#             scaled_sequence = scaler.transform(last_sequence.values.reshape(-1, 1)).flatten()

#             # Convert the scaled sequence to a PyTorch tensor
#             # Reshape to [1, sequence_length] as the model expects a batch of sequences
#             sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)

#             # Store the tensor in the dictionary
#             forecast_input_sequences[commodity] = sequence_tensor
#             print(f"  Prepared input sequence for {commodity}. Shape: {sequence_tensor.shape}")

#         else:
#              print(f"  Warning: Scaler not found for {commodity}. Cannot prepare forecast input sequence.")


#     # Print the number of commodities for which input sequences were prepared
#     print(f"\nPrepared input sequences for {len(forecast_input_sequences)} commodities.")

# import torch
# import pandas as pd
# import numpy as np
# from kan import KAN # Assuming kan library is installed and imported

# # Ensure model, forecast_input_sequences, sequence_length, scalers, and commodity_columns are available
# if 'model' not in locals() or model is None:
#     print("Error: KAN model is not defined. Cannot generate forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'forecast_input_sequences' not in locals() or not forecast_input_sequences:
#     print("Error: 'forecast_input_sequences' is not available or is empty. Cannot generate forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'sequence_length' not in locals():
#     print("Error: 'sequence_length' is not defined. Cannot generate forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'scalers' not in locals() or not scalers:
#     print("Error: 'scalers' dictionary not found or is empty. Cannot inverse scale forecasts.")
#     kan_forecasts = {} # Initialize as empty
# elif 'commodity_columns' not in locals() or not commodity_columns:
#     print("Error: 'commodity_columns' list not found or is empty. Cannot iterate through commodities for forecasting.")
#     kan_forecasts = {} # Initialize as empty
# else:
#     print("\nGenerating KAN future forecasts for each commodity...")

#     # Define the number of steps to forecast
#     forecast_steps = 24 # This should align with other forecasters if possible
#     print(f"Forecasting {forecast_steps} steps into the future.")

#     # Move the model to the appropriate device (CPU or GPU) if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval() # Set the model to evaluation mode

#     # Initialize a dictionary to store the generated forecasts for each commodity
#     kan_forecasts = {}

#     # Iterate through each commodity for which an input sequence was prepared
#     print(f"\nGenerating forecasts for {len(forecast_input_sequences)} commodities...")
#     with torch.no_grad(): # Disable gradient calculation during inference
#         for commodity in forecast_input_sequences.keys():
#             print(f"  Forecasting for: {commodity}")
#             # Get the initial input sequence tensor for the current commodity
#             current_sequence = forecast_input_sequences[commodity].to(device) # Move sequence to device

#             # Get the corresponding scaler for inverse scaling
#             if commodity not in scalers:
#                  print(f"    Warning: Scaler not found for {commodity}. Skipping forecast.")
#                  continue # Skip to the next commodity
#             scaler = scalers[commodity]

#             # Initialize a list to store the unscaled forecast values for this commodity
#             commodity_forecast_values = []

#             # Generate the forecast step by step
#             for _ in range(forecast_steps):
#                 # Get the KAN model's prediction for the next step (scaled value)
#                 predicted_scaled_value = model(current_sequence).item() # Get the scalar value

#                 # Inverse scale the predicted value to the original price scale
#                 # The scaler expects a 2D array, so reshape the scalar
#                 predicted_original_value = scaler.inverse_transform([[predicted_scaled_value]])[0][0]

#                 # Store the inverse-scaled forecast value
#                 commodity_forecast_values.append(predicted_original_value)

#                 # Update the input sequence for the next prediction
#                 # Remove the oldest value and append the new scaled prediction
#                 new_sequence = torch.cat((current_sequence[:, 1:], torch.tensor([[predicted_scaled_value]], dtype=torch.float32).to(device)), dim=1)
#                 current_sequence = new_sequence # Use the new sequence for the next step

#             # Store the generated forecast values for the current commodity
#             kan_forecasts[commodity] = commodity_forecast_values
#             print(f"    Generated {len(commodity_forecast_values)} forecast values for {commodity}.")

#     print("\nKAN forecasting complete.")

# # You now have the generated forecasts in the 'kan_forecasts' dictionary.

# """**Reasoning**:
# Combine the generated KAN forecasts with the historical data and save the combined DataFrame to a Google Sheet.


# """

# import pandas as pd
# import gspread
# from google.colab import auth
# from google.auth import default
# import warnings

# warnings.filterwarnings("ignore") # Suppress warnings

# # Ensure prices_df (original data) and kan_forecasts (forecast dictionary) are available
# if 'prices_df' not in locals():
#     print("Error: 'prices_df' DataFrame not found. Cannot combine data.")
#     # Finish the task with failure
# elif 'kan_forecasts' not in locals() or not kan_forecasts:
#     print("Error: 'kan_forecasts' dictionary not found or is empty. Cannot combine data.")
#     # Finish the task with failure
# else:
#     print("\nCombining original data and KAN forecasts...")

#     # Prepare the original data: select commodity columns and ensure DatetimeIndex
#     # Assuming the index of prices_df is already the correct DatetimeIndex ('Date')
#     # and numeric columns are identified in commodity_columns list
#     historical_df = prices_df[commodity_columns].copy()

#     # Prepare the forecast data:
#     # Create a DataFrame from the kan_forecasts dictionary
#     if kan_forecasts:
#         # Get the last date from the historical data to start the forecast index from the next day
#         last_historical_date = prices_df.index.max()

#         # Determine the frequency of the historical data
#         freq = pd.infer_freq(prices_df.index)
#         if freq is None:
#              # Fallback to 'D' if frequency cannot be inferred (assuming daily data)
#              freq = 'D'
#              print(f"Warning: Could not infer frequency from historical data. Assuming '{freq}'.")
#         else:
#             print(f"Inferred historical data frequency: {freq}")


#         # Generate future dates starting from the day after the last historical date
#         forecast_dates = pd.date_range(start=last_historical_date, periods=forecast_steps + 1, freq=freq)[1:]

#         # Create a DataFrame from the forecast dictionary, using the generated future dates as the index
#         forecast_df = pd.DataFrame(kan_forecasts, index=forecast_dates)

#         # Rename the index to 'Date' for consistency
#         forecast_df.index.name = 'Date'

#     else:
#         print("No KAN forecasts were generated to combine.")
#         forecast_df = pd.DataFrame() # Create an empty DataFrame if no forecasts

#     # Combine the historical data and the forecast data
#     # Use outer join to include all dates from both historical and forecast periods
#     # The index from both DataFrames (DatetimeIndex) will be used for alignment
#     combined_df = pd.concat([historical_df, forecast_df], axis=0, join='outer')

#     # Sort by date to ensure chronological order
#     combined_df.sort_index(inplace=True)

#     print("\nCombined KAN DataFrame Head:")
#     print(combined_df.head(2))
#     print("\nCombined KAN DataFrame Tail:")
#     print(combined_df.tail(2))
#     print("\nCombined KAN DataFrame Shape:", combined_df.shape)

#     if not combined_df.empty:
#         print(f"saving {combined_df} to gcs")
#         gcs_prefix_kan_forecast = 'forecast_data/kan_forecasts.csv'
#         save_to_gcs(
#             df=combined_df,
#             gcs_prefix=gcs_prefix_kan_forecast,
#             validate=False)
#         print(f"saved {combined_df} to gcs")
#     else:
#         print("combined_df is empty. No data to save.")

# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming combined_df is available from the previous step

# def plot_kan_forecast(df_combined, commodity_name):
#     """
#     Visualizes the historical data and KAN forecast for a specific commodity
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
#     # Or simply plot all available data in the combined df
#     plt.plot(df_combined.index, df_combined[commodity_name], label=f'{commodity_name} Data & KAN Forecast')


#     # Add a vertical line at the end of the historical data for visual separation
#     # Assuming the end of historical data is the last date before the forecast starts
#     # We can infer this from the first non-NaN value in the forecast period in the combined df
#     first_forecast_date = df_combined[df_combined[commodity_name].index > prices_df.index.max()].first_valid_index()
#     if first_forecast_date:
#         plt.axvline(prices_df.index.max(), color='red', linestyle='--', label='Forecast Start')


#     plt.title(f'KAN Forecast for {commodity_name}')
#     plt.xlabel('Date')
#     plt.ylabel(commodity_name)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Example usage:
# # Assuming combined_df is already loaded

# # Get a list of commodities that were forecasted
# # We can get this from the keys of the kan_forecasts dictionary
# if 'kan_forecasts' in locals() and kan_forecasts:
#     commodities_to_visualize = list(kan_forecasts.keys())[:10] # Visualize the first 5 forecasted commodities
#     print(f"\nVisualizing KAN forecasts for the following commodities: {commodities_to_visualize}")
# else:
#     print("Error: 'kan_forecasts' dictionary not found or is empty. Cannot visualize forecasts.")
#     commodities_to_visualize = []


# if 'combined_df' in locals() and not combined_df.empty and commodities_to_visualize:
#     for commodity in commodities_to_visualize:
#         plot_kan_forecast(combined_df, commodity)
# else:
#     print("Cannot visualize forecasts: combined_df is not available or empty, or no commodities to visualize.")
