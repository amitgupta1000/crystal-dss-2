import os
import sys
import time
import urllib.request
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import date
from src.file_utils import save_dataframe_to_gcs, upload_excel_file, download_latest_csv_from_gcs
from src.correlation import compute_correlations, build_filtered_correlation_matrix
from src.causality import prepare_causality_data, run_granger_tests
from src.regression import run_regression_models
from src.seasonality import run_seasonality_analysis

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

load_dotenv()

def main():
    # Define the target commodities
    target_commodities = [
        'Acetic Acid', 'Butyl Acetate', 'Toluene', 'Isomer-MX', 'Solvent-MX', 'Methanol',
        'MTBE', 'Benzene'
    ]

    # Define the other commodities to correlate against
    other_commodities = [
        'Gold', 'Silver', 'Copper', 'S&P 500', 'Shanghai Composite', 'USD Index',
        'Japanese Yen', 'US 10-Y BOND PRICE', 'Crude Oil', 'Natural Gas', 'Naphtha',
        'EDC', 'Ethylene', 'Propylene', 'N-Butanol', 'Paraxylene', 'OrthXylene',
        'Cyclohexane', 'Styrene', 'DEG', '2 EH', 'Acetic Acid', 'Butyl Acetate',
        'Toluene', 'Isomer-MX', 'Solvent-MX', 'Methanol', 'MTBE', 'Benzene'
    ]

    prices_df = download_latest_csv_from_gcs(bucket_name='crystal-dss', gcs_prefix='cleaned_data')
    logger.info('Downloaded latest cleaned DataFrame from GCS.')

    prices_df = prices_df.copy()

    # Convert 'date' column to datetime objects
    prices_df['date'] = pd.to_datetime(prices_df['date'], format='%d-%m-%y', errors='coerce')
    prices_df.set_index('date', inplace=True)
    prices_df.index.name = 'Date'
    prices_df.columns = prices_df.columns.str.strip()

    # Ensure data is in ascending order
    prices_df.sort_index(ascending=True, inplace=True)
    logger.info('Commodity Prices DataFrame sorted by Date (ascending). Shape: %s', prices_df.shape)

    # Drop columns and rows that are entirely NaN
    prices_df.dropna(axis=1, how='all', inplace=True)
    prices_df.dropna(axis=0, how='all', inplace=True)
    logger.info('Shape after dropping empty rows/columns: %s', prices_df.shape)

    # Correlation
    window = 12
    cor_results = compute_correlations(prices_df, window=window, interpolate_method='time')

    numerical_df_for_correlation_average = cor_results.get('average_df')
    correlation_matrix = cor_results.get('correlation_matrix')
    pairwise_corr_df = cor_results.get('pairwise_corr_df')
    diagnostics = cor_results.get('diagnostics')

    try:
        if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
            logger.info('Correlation column diagnostics (top 10):\n%s', diagnostics.sort_values(by=['finite_correlations', 'non_na_count'], ascending=False).head(10))
            try:
                save_dataframe_to_gcs(df=diagnostics.reset_index().rename(columns={'index': 'commodity'}), bucket_name='crystal-dss', gcs_prefix='stats_studies_data/correlation_column_diagnostics.csv', validate_rows=False)
                logger.info('Saved correlation column diagnostics to GCS: stats_studies_data/correlation_column_diagnostics.csv')
            except Exception:
                logger.exception('Failed saving correlation diagnostics to GCS')
    except Exception:
        logger.exception('Could not compute correlation diagnostics')

    # Save full correlation matrix to GCS
    gcs_prefix_correlation = 'stats_studies_data/correlation_matrix.csv'
    save_dataframe_to_gcs(df=correlation_matrix, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_correlation, validate_rows=False, include_index=True)
    logger.info('Saved full correlation matrix to GCS: %s', gcs_prefix_correlation)

    # Build filtered correlation matrix using the correlation module
    try:
        filtered_correlation_df = build_filtered_correlation_matrix(pairwise_corr_df, correlation_matrix, target_commodities, other_commodities)
        if filtered_correlation_df.empty:
            logger.info('Filtered correlation matrix is empty — no valid targets/others found or no correlation data available.')
        else:
            logger.info('Filtered Correlation Matrix preview:\n%s', filtered_correlation_df.head(5))
            try:
                gcs_prefix_filtered_corr_matrix = 'stats_studies_data/filtered_correlation_matrix.csv'
                save_dataframe_to_gcs(df=filtered_correlation_df, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_filtered_corr_matrix, validate_rows=False, include_index=True)
                logger.info('Saved filtered correlation matrix to GCS: %s', gcs_prefix_filtered_corr_matrix)
            except Exception:
                logger.exception('Failed to save filtered correlation matrix to GCS')
    except Exception:
        logger.exception('Error building filtered correlation matrix')

    # Causality
    numerical_df_diff_scaled_average = prepare_causality_data(prices_df, window=window)
    alpha = 0.04
    max_lag = 10
    logger.info('Alpha (significance level): %s; Max lag: %s', alpha, max_lag)

    logger.info('Performing Granger causality tests...')
    all_granger_test_results_df, causality_adjacency_matrix_df = run_granger_tests(numerical_df_diff_scaled_average, max_lag=max_lag, alpha=alpha)
    logger.info('Head of DataFrame with ALL Granger Test Results:\n%s', all_granger_test_results_df.head() if not all_granger_test_results_df.empty else all_granger_test_results_df)

    # Save Granger results
    if isinstance(all_granger_test_results_df, pd.DataFrame) and not all_granger_test_results_df.empty:
        logger.info('Saving all_granger_test_results_df to GCS...')
        gcs_prefix_granger_results = 'stats_studies_data/all_granger_test_results.csv'
        save_dataframe_to_gcs(df=all_granger_test_results_df, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_granger_results, validate_rows=False)
        logger.info('Saved Granger results to GCS: %s', gcs_prefix_granger_results)
    else:
        logger.warning("'all_granger_test_results_df' not available or empty; skipping save")

    # Filter causality
    if 'all_granger_test_results_df' not in locals() or all_granger_test_results_df.empty:
        logger.warning("'all_granger_test_results_df' not found or empty. Cannot filter causality results.")
        filtered_causality_df = pd.DataFrame()
    else:
        logger.info('Filtering causality relationships to find sources for target commodities...')
        most_significant_lags = all_granger_test_results_df.loc[all_granger_test_results_df.groupby(['Source', 'Target'])['P-value (F-test)'].idxmin()].copy()
        if 'alpha' in locals():
            most_significant_lags = most_significant_lags[most_significant_lags['P-value (F-test)'] < alpha]
        else:
            logger.warning("'alpha' not defined. Using default 0.05")
            most_significant_lags = most_significant_lags[most_significant_lags['P-value (F-test)'] < 0.05]

        filtered_causality_df = most_significant_lags[most_significant_lags['Source'].isin(other_commodities) & most_significant_lags['Target'].isin(target_commodities)].copy()

        if not filtered_causality_df.empty:
            filtered_causality_df = filtered_causality_df.sort_values(by=['Target', 'P-value (F-test)']).reset_index(drop=True)
            logger.info('Filtered Causality Table (Sources for Target Commodities):\n%s', filtered_causality_df)
            gcs_prefix_casuality_matrix = 'stats_studies_data/filtered_causality_matrix.csv'
            save_dataframe_to_gcs(df=filtered_causality_df, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_casuality_matrix, validate_rows=False)
            logger.info('Saved filtered causality matrix to GCS: %s', gcs_prefix_casuality_matrix)
        else:
            logger.info('No significant filtered causal relationships found for target commodities.')

    # Regression
    prices_df_local = prices_df.copy()
    numerical_price_features_df = prices_df_local.select_dtypes(include=np.number)
    logger.info('Numeric features shape: %s', numerical_price_features_df.shape)

    numerical_price_features_diff_df = numerical_price_features_df.diff()
    logger.info('Differenced numeric features shape: %s', numerical_price_features_diff_df.shape)

    min_values_threshold = 300
    logger.info('Filtering commodities with at least %s non-missing values', min_values_threshold)
    non_missing_counts = numerical_price_features_diff_df.count()
    logger.info('Total number of commodities: %s', len(non_missing_counts))
    columns_to_keep = non_missing_counts[non_missing_counts >= min_values_threshold].index.tolist()
    numerical_price_features_diff_df = numerical_price_features_diff_df[columns_to_keep].copy()
    logger.info('Remaining commodities after filtering: %s', len(columns_to_keep))
    prices_df_filled = numerical_price_features_diff_df.fillna(method='ffill').fillna(method='bfill')
    commodities = numerical_price_features_diff_df.columns.tolist()

    target_commodities = [
        'Acetic Acid', 'Butyl Acetate', 'Toluene', 'Isomer-MX', 'Solvent-MX', 'Methanol',
        'MTBE', 'Benzene', 'Crude Oil', 'Natural Gas', 'Naphtha', 'EDC', 'Ethylene', 'Propylene'
    ]
    target_commodities = [c for c in target_commodities if c in numerical_price_features_diff_df.columns]
    logger.info('Target commodities present: %s', len(target_commodities))


    logger.info('Running regression models for target commodities...')
    regression_results = run_regression_models(numerical_price_features_diff_df, commodities, target_commodities, degrees_to_test=[1, 2, 3], top_n=8, max_features_for_poly=8)
    logger.info('Completed regressions: %s model results collected.', len(regression_results))

    regression_results_df = pd.DataFrame(regression_results)
    logger.info('Regression results summary:\n%s', regression_results_df[['Target Commodity', 'Degree', 'Features Used Count', 'Train Adj R2', 'Test Adj R2', 'Train RMSE', 'Test RMSE']].head())

    if not regression_results_df.empty:
        gcs_prefix_regressions_results = 'stats_studies_data/regression_results_matrix.csv'
        save_dataframe_to_gcs(df=regression_results_df, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_regressions_results, validate_rows=False)
        logger.info('Saved regression results to GCS: %s', gcs_prefix_regressions_results)
    else:
        logger.info('No regression results to save')

    # Seasonality

    logger.info('Running seasonality analysis...')
    seasonality_results_df_flattened, seasonality_summary_table_df = run_seasonality_analysis(prices_df, num_top_periods_to_analyze=6)

    if seasonality_results_df_flattened is not None and not seasonality_results_df_flattened.empty:
        logger.info('Saving seasonality_results_df_flattened to GCS...')
        gcs_prefix_seasonality = 'stats_studies_data/seasonality_results.csv'
        save_dataframe_to_gcs(df=seasonality_results_df_flattened, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_seasonality, validate_rows=False)
        logger.info('Saved seasonality results to GCS: %s', gcs_prefix_seasonality)
    else:
        logger.info("No seasonality detailed results to save")

    if seasonality_summary_table_df is not None and not seasonality_summary_table_df.empty:
        logger.info('Seasonality summary (top rows):\n%s', seasonality_summary_table_df.head())
        gcs_prefix_seasonality_summary = 'stats_studies_data/seasonality_summary.csv'
        save_dataframe_to_gcs(df=seasonality_summary_table_df, bucket_name='crystal-dss', gcs_prefix=gcs_prefix_seasonality_summary, validate_rows=False, include_index=True)
        logger.info('Saved seasonality summary to GCS: %s', gcs_prefix_seasonality_summary)
    else:
        logger.info('No seasonality summary table to save')


if __name__ == '__main__':
    main()

