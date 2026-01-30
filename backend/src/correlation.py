import pandas as pd
import numpy as np

__all__ = [
	"compute_correlations",
	"build_filtered_correlation_matrix",
	"compute_rolling_correlations",
]


def compute_correlations(prices_df: pd.DataFrame, window: int = 12, interpolate_method: str = 'time') -> dict:
	prices_df_numeric = prices_df.copy()
	average_prices_df = prices_df_numeric.rolling(window=window).mean()

	numerical_cols_for_correlations_average = average_prices_df.select_dtypes(include=np.number).columns.tolist()
	average_df = average_prices_df[numerical_cols_for_correlations_average]

	try:
		average_df = average_df.interpolate(method=interpolate_method).dropna()
	except Exception:
		average_df = average_df.dropna()

	correlation_matrix = average_df.corr()

	# Build pairwise long-form (upper triangle, excluding diagonal)
	try:
		mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)
		pairwise = correlation_matrix.where(mask)
		pairwise_corr_df = pairwise.stack().reset_index()
		pairwise_corr_df.columns = ['Commodity_A', 'Commodity_B', 'Correlation']
	except Exception:
		pairwise_corr_df = pd.DataFrame(columns=['Commodity_A', 'Commodity_B', 'Correlation'])

	# Diagnostics
	try:
		col_non_na_counts = average_df.notna().sum()
		col_unique_counts = average_df.nunique(dropna=True)
		col_variances = average_df.var()
		corr_non_na = correlation_matrix.notna().sum()

		diagnostics = pd.DataFrame({
			'non_na_count': col_non_na_counts,
			'unique_count': col_unique_counts,
			'variance': col_variances,
			'finite_correlations': corr_non_na
		})
		diagnostics['likely_omitted'] = diagnostics['finite_correlations'] <= 1
	except Exception:
		diagnostics = pd.DataFrame()

	return {
		'average_df': average_df,
		'correlation_matrix': correlation_matrix,
		'pairwise_corr_df': pairwise_corr_df,
		'diagnostics': diagnostics
	}


def compute_rolling_correlations(prices_df: pd.DataFrame, correlation_window: int, interpolate_method: str = 'time') -> pd.DataFrame:
	"""
	Computes rolling correlations over time to analyze correlation drift.

	Args:
		prices_df (pd.DataFrame): DataFrame with datetime index and commodity prices as columns.
		correlation_window (int): The window size (number of periods) for the rolling correlation calculation.
		interpolate_method (str): Method for interpolation.

	Returns:
		pd.DataFrame: A long-form DataFrame with columns ['Date', 'Commodity_A', 'Commodity_B', 'Correlation'].
	"""
	prices_df_numeric = prices_df.select_dtypes(include=np.number)

	# Interpolate once to handle missing values
	try:
		prices_df_interpolated = prices_df_numeric.interpolate(method=interpolate_method)
	except Exception:
		prices_df_interpolated = prices_df_numeric

	# Get the rolling window iterator
	rolling_windows = prices_df_interpolated.rolling(window=correlation_window)

	all_rolling_correlations = []

	# Manually iterate through each window to compute correlations efficiently
	for window_df in rolling_windows:
		if len(window_df) < correlation_window:
			continue

		# Calculate correlation matrix for the current window
		corr_matrix = window_df.corr()
		
		# Get the date for this window (the last date in the window)
		window_date = window_df.index[-1]
		
		# Unstack to a long format, but only keep the upper triangle to avoid duplicates
		corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().reset_index()
		corr_pairs.columns = ['Commodity_A', 'Commodity_B', 'Correlation']
		corr_pairs['Date'] = window_date
		all_rolling_correlations.append(corr_pairs)

	# Concatenate all results into a single DataFrame
	return pd.concat(all_rolling_correlations, ignore_index=True) if all_rolling_correlations else pd.DataFrame()


def build_filtered_correlation_matrix(pairwise_corr_df: pd.DataFrame, correlation_matrix: pd.DataFrame, target_commodities: list, other_commodities: list) -> pd.DataFrame:
	all_commodities = set()
	if isinstance(pairwise_corr_df, pd.DataFrame) and not pairwise_corr_df.empty:
		all_commodities = set(pairwise_corr_df['Commodity_A']).union(set(pairwise_corr_df['Commodity_B']))
	elif isinstance(correlation_matrix, pd.DataFrame) and not correlation_matrix.empty:
		all_commodities = set(correlation_matrix.index.tolist())

	valid_target_commodities = [c for c in target_commodities if c in all_commodities]
	valid_other_commodities = [c for c in other_commodities if c in all_commodities]

	if not valid_target_commodities or not valid_other_commodities:
		return pd.DataFrame()

	filtered_correlation_df = pd.DataFrame(index=valid_target_commodities, columns=valid_other_commodities, dtype=float)

	if isinstance(pairwise_corr_df, pd.DataFrame) and not pairwise_corr_df.empty:
		for t in valid_target_commodities:
			for o in valid_other_commodities:
				if t == o:
					filtered_correlation_df.loc[t, o] = 0.0
					continue
				mask_ab = (pairwise_corr_df['Commodity_A'] == t) & (pairwise_corr_df['Commodity_B'] == o)
				mask_ba = (pairwise_corr_df['Commodity_A'] == o) & (pairwise_corr_df['Commodity_B'] == t)
				if mask_ab.any():
					filtered_correlation_df.loc[t, o] = pairwise_corr_df.loc[mask_ab, 'Correlation'].iloc[0]
				elif mask_ba.any():
					filtered_correlation_df.loc[t, o] = pairwise_corr_df.loc[mask_ba, 'Correlation'].iloc[0]
				else:
					try:
						filtered_correlation_df.loc[t, o] = correlation_matrix.loc[t, o]
					except Exception:
						filtered_correlation_df.loc[t, o] = np.nan
	else:
		for t in valid_target_commodities:
			for o in valid_other_commodities:
				try:
					if t == o:
						filtered_correlation_df.loc[t, o] = 0.0
					else:
						filtered_correlation_df.loc[t, o] = correlation_matrix.loc[t, o]
				except Exception:
					filtered_correlation_df.loc[t, o] = np.nan

	filtered_correlation_df.index.name = 'Target'
	filtered_correlation_df.columns.name = 'Other'

	return filtered_correlation_df
