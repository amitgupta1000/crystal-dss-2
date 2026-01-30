import os
import contextlib
import pandas as pd
import numpy as np
import logging
import json
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

def prepare_causality_data(prices_df: pd.DataFrame, adf_significance_level=0.05):
    """
    Prepares data for causality testing by ensuring stationarity through differencing.
    """
    logger = logging.getLogger(__name__)
    numerical_df = prices_df.select_dtypes(include=np.number).copy()
    stationary_df = pd.DataFrame(index=prices_df.index)

    for col in numerical_df.columns:
        series = numerical_df[col].dropna()
        # Test for stationarity on the original series
        adf_result = adfuller(series)
        p_value = adf_result[1]

        if p_value < adf_significance_level:
            # Series is already stationary
            stationary_df[col] = series
            logger.debug("Series '%s' is stationary (p=%.4f). No differencing needed.", col, p_value)
        else:
            # Difference the series to make it stationary
            diff_series = series.diff().dropna()
            adf_result_diff = adfuller(diff_series)
            p_value_diff = adf_result_diff[1]
            stationary_df[col] = diff_series
            if p_value_diff < adf_significance_level:
                logger.debug("Series '%s' made stationary with 1st difference (p=%.4f).", col, p_value_diff)
            else:
                logger.warning("Series '%s' may not be stationary after 1st difference (p=%.4f).", col, p_value_diff)

    # Drop columns that are all NaN after processing
    stationary_df.dropna(axis=1, how='all', inplace=True)

    # Scale the data
    scaler = StandardScaler()
    scaled_stationary_df = pd.DataFrame(scaler.fit_transform(stationary_df),
                                        columns=stationary_df.columns,
                                        index=stationary_df.index)
    return scaled_stationary_df


def run_granger_tests(stationary_df: pd.DataFrame, lags_to_test: list, alpha: float = 0.04):
    logger = logging.getLogger(__name__)
    logging.getLogger('statsmodels').setLevel(logging.ERROR)

    commodities_for_test = stationary_df.columns
    num_commodities_for_test = len(commodities_for_test)

    all_granger_test_results = []
    causality_adjacency_matrix_np = np.zeros((num_commodities_for_test, num_commodities_for_test))
    max_lag = max(lags_to_test) if lags_to_test else 0

    # Suppress stdout from grangercausalitytests
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        for i in range(num_commodities_for_test):
            for j in range(num_commodities_for_test):
                if i == j:
                    continue
                source_commodity = commodities_for_test[i]
                target_commodity = commodities_for_test[j]
                data_pair = stationary_df[[source_commodity, target_commodity]]
                data_pair = data_pair.dropna()

                try:
                    # Check if there's enough data for the test
                    if len(data_pair) <= max_lag:
                        logger.debug("Skipping Granger test for (%s -> %s): insufficient data (%d) for max lag (%d)",
                                     source_commodity, target_commodity, len(data_pair), max_lag)
                        continue

                    test_results = grangercausalitytests(data_pair, lags_to_test)

                    for lag in lags_to_test:
                        p_value_ftest = test_results[lag][0]['ssr_ftest'][1]
                        all_granger_test_results.append({
                            'Source': source_commodity,
                            'Target': target_commodity,
                            'Lag': lag,
                            'P-value (F-test)': p_value_ftest
                        })
                        # Note: Adjacency matrix will now reflect significance at *any* tested lag
                        if p_value_ftest < alpha:
                            causality_adjacency_matrix_np[i, j] = 1
                except Exception as e:
                    logger.warning("Granger test failed for (%s -> %s): %s",
                                   source_commodity, target_commodity, str(e))

    all_granger_test_results_df = pd.DataFrame(all_granger_test_results)
    return all_granger_test_results_df
