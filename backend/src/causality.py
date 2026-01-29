import os
import contextlib
from contextlib import contextmanager
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

@contextmanager
def _suppress_fds():
    import sys as _sys
    import os as _os
    _sys.stdout.flush(); _sys.stderr.flush()
    devnull_fd = _os.open(_os.devnull, _os.O_RDWR)
    try:
        old_stdout = _os.dup(1)
        old_stderr = _os.dup(2)
        _os.dup2(devnull_fd, 1)
        _os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            _sys.stdout.flush(); _sys.stderr.flush()
        except Exception:
            pass
        _os.dup2(old_stdout, 1)
        _os.dup2(old_stderr, 2)
        _os.close(old_stdout)
        _os.close(old_stderr)
        _os.close(devnull_fd)


def prepare_causality_data(prices_df: pd.DataFrame, window: int = 12):
    prices_df = prices_df.copy()
    average_prices_df = prices_df.rolling(window=window).mean().dropna()
    numerical_cols_average = average_prices_df.select_dtypes(include=np.number).columns.tolist()
    numerical_df_for_causality_average = average_prices_df[numerical_cols_average]
    numerical_df_diff_average = numerical_df_for_causality_average.diff().dropna()

    scaler_average = StandardScaler()
    numerical_df_diff_scaled_average = pd.DataFrame(scaler_average.fit_transform(numerical_df_diff_average),
                                                    columns=numerical_df_diff_average.columns,
                                                    index=numerical_df_diff_average.index)
    return numerical_df_diff_scaled_average


def run_granger_tests(numerical_df_diff_scaled_average: pd.DataFrame, max_lag: int = 10, alpha: float = 0.04):
    logging.getLogger('statsmodels').setLevel(logging.ERROR)

    commodities_for_test = numerical_df_diff_scaled_average.columns
    num_commodities_for_test = len(commodities_for_test)

    all_granger_test_results = []
    causality_adjacency_matrix_np = np.zeros((num_commodities_for_test, num_commodities_for_test))

    # Suppress high-level stdout/stderr and low-level FD writes
    with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull), _suppress_fds():
        for i in range(num_commodities_for_test):
            for j in range(num_commodities_for_test):
                if i == j:
                    continue
                source_commodity = commodities_for_test[i]
                target_commodity = commodities_for_test[j]
                data_pair = numerical_df_diff_scaled_average[[source_commodity, target_commodity]]

                try:
                    test_results = grangercausalitytests(data_pair, maxlag=max_lag, verbose=False)
                    for lag in range(1, max_lag + 1):
                        if lag in test_results:
                            p_value_ftest = test_results[lag][0]['ssr_ftest'][1]
                            all_granger_test_results.append({
                                'Source': source_commodity,
                                'Target': target_commodity,
                                'Lag': lag,
                                'P-value (F-test)': p_value_ftest
                            })
                            if p_value_ftest < alpha:
                                causality_adjacency_matrix_np[i, j] = 1
                except Exception:
                    pass

    all_granger_test_results_df = pd.DataFrame(all_granger_test_results)
    causality_adjacency_matrix_df = pd.DataFrame(causality_adjacency_matrix_np, index=commodities_for_test, columns=commodities_for_test)

    return all_granger_test_results_df, causality_adjacency_matrix_df
