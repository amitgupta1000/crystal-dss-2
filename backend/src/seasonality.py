import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from .file_utils import save_dataframe_to_gcs


def _is_datetime_indexed(df):
    return isinstance(df.index, pd.DatetimeIndex) and df.index.name == 'Date'


def run_seasonality_analysis(prices_df,
                             cols_to_exclude=None,
                             num_top_periods_to_analyze=6,
                             min_peak_prominence_ratio=0.1,
                             max_peaks_store=30,
                             gcs_bucket_name=None,
                             gcs_prefix=None):
    if cols_to_exclude is None:
        cols_to_exclude = ['date', 'day_of_week', 'month', 'year',
                           'day_of_week_sin', 'day_of_week_cos',
                           'month_sin', 'month_cos', 'year_sin', 'year_cos']

    if prices_df is None or prices_df.empty:
        return None, None

    if not _is_datetime_indexed(prices_df):
        return None, None

    all_columns = prices_df.columns.tolist()
    commodity_columns = [col for col in all_columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(prices_df[col])]
    if not commodity_columns:
        return None, None

    seasonality_analysis_results = {}

    for commodity in commodity_columns:
        commodity_series = prices_df[commodity].dropna()
        commodity_results = {
            'Commodity': commodity,
            'Overall Seasonality Summary': '',
            'Analysis Status': 'Success',
            'Error Message': None,
            'FFT Peaks': [],
            'Analyzed Periods': [],
            'Decomposition File Path': None
        }

        try:
            yf = fft(commodity_series.values)
            xf = fftfreq(len(commodity_series), d=1)
            power_spectrum = np.abs(yf) ** 2
            positive_frequencies = xf > 0
            frequencies = xf[positive_frequencies]
            powers = power_spectrum[positive_frequencies]

            peak_indices, _ = find_peaks(powers, prominence=np.mean(powers) * min_peak_prominence_ratio)
            if peak_indices.size > 0:
                peak_frequencies = frequencies[peak_indices]
                peak_powers = powers[peak_indices]
                combined_peaks = sorted(zip(peak_frequencies, peak_powers), key=lambda x: x[1], reverse=True)
                for i in range(min(max_peaks_store, len(combined_peaks))):
                    freq, power = combined_peaks[i]
                    period = 1 / freq if freq != 0 else np.inf
                    commodity_results['FFT Peaks'].append({'Frequency': float(freq), 'Period': float(period), 'Power': float(power)})
                commodity_results['Overall FFT Power'] = float(np.sum([p['Power'] for p in commodity_results['FFT Peaks'][:min(5, len(commodity_results['FFT Peaks']))]]))
        except Exception as e:
            commodity_results['Analysis Status'] = 'Partial Success (FFT Failed)'
            commodity_results['Error Message'] = f'FFT failed: {e}'
            commodity_results['FFT Peaks'] = []
            commodity_results['Overall FFT Power'] = np.nan

        periods_to_analyze_for_commodity = []
        if commodity_results['FFT Peaks']:
            top_peaks_periods = [peak['Period'] for peak in commodity_results['FFT Peaks'][:num_top_periods_to_analyze]]
            periods_to_analyze_for_commodity = sorted(list(set([int(p) for p in top_peaks_periods if np.isfinite(p) and p > 1])))

        if not periods_to_analyze_for_commodity:
            seasonality_analysis_results[commodity] = commodity_results
            continue

        # Store decomposition objects temporarily to find the best one later
        decompositions_by_period = {}

        for period in periods_to_analyze_for_commodity:
            period_analysis_result = {
                'Period': period,
                'Period Analysis Status': 'Success',
                'Period Error Message': None,
                'Seasonality Strength (STL)': np.nan,
                'Calculated Seasonality Score (STL Ratio)': np.nan,
                'ACF at Period Lag': np.nan,
                'PACF at Period Lag': np.nan,
                'FFT Power at Period': np.nan,
                'Is Seasonal for Period': False,
                'Is ACF Significant at Period': False,
                'Is PACF Significant at Period': False
            }

            # STL decomposition requires the series to be at least 2 * period long.
            min_length_for_stl = 2 * period
            if len(commodity_series) < min_length_for_stl:
                period_analysis_result['Period Analysis Status'] = 'Skipped (Insufficient Data for STL)'
                period_analysis_result['Period Error Message'] = f'Insufficient data for STL decomposition (min length: {min_length_for_stl}).'
                commodity_results['Analyzed Periods'].append(period_analysis_result)
                continue

            try:
                try:
                    stl = seasonal_decompose(commodity_series, model='additive', period=period, robust=True)
                except TypeError:
                    stl = seasonal_decompose(commodity_series, model='additive', period=period)

                # Store the decomposition components
                decompositions_by_period[period] = {
                    'trend': stl.trend,
                    'seasonal': stl.seasonal,
                    'resid': stl.resid
                }
                seasonal_component = stl.seasonal
                residual_component = stl.resid
                seasonality_plus_residual_std = np.nanstd(seasonal_component + residual_component)
                if seasonality_plus_residual_std > 1e-9:
                    seasonality_strength = np.nanstd(seasonal_component) / seasonality_plus_residual_std
                else:
                    seasonality_strength = 0.0
                period_analysis_result['Seasonality Strength (STL)'] = float(seasonality_strength)
                residual_std = np.nanstd(residual_component)
                if residual_std > 1e-9:
                    stl_ratio = np.nanstd(seasonal_component) / residual_std
                else:
                    stl_ratio = np.inf
                period_analysis_result['Calculated Seasonality Score (STL Ratio)'] = float(stl_ratio)
                if seasonality_strength > 0.1:
                    period_analysis_result['Is Seasonal for Period'] = True
            except Exception as e:
                period_analysis_result['Period Analysis Status'] = 'Failed (STL)'
                period_analysis_result['Period Error Message'] = f'STL failed: {e}'

            try:
                n_lags_acf_pacf = period
                if len(commodity_series) > n_lags_acf_pacf:
                    acf_values, conf_int_acf = acf(commodity_series.dropna(), nlags=n_lags_acf_pacf, alpha=0.05)
                    pacf_values, conf_int_pacf = pacf(commodity_series.dropna(), nlags=n_lags_acf_pacf, alpha=0.05)
                    period_lag_index = period
                    period_analysis_result['ACF at Period Lag'] = float(acf_values[period_lag_index])
                    period_analysis_result['PACF at Period Lag'] = float(pacf_values[period_lag_index])
                    # The confidence interval is centered at 0. A value is significant if it's outside the interval.
                    # For a symmetric interval [-c, c], this is equivalent to |value| > c.
                    if conf_int_acf is not None and period_lag_index < len(conf_int_acf) and abs(acf_values[period_lag_index]) > conf_int_acf[period_lag_index, 1]:
                        period_analysis_result['Is ACF Significant at Period'] = True
                    if conf_int_pacf is not None and period_lag_index < len(conf_int_pacf) and abs(pacf_values[period_lag_index]) > conf_int_pacf[period_lag_index, 1]:
                        period_analysis_result['Is PACF Significant at Period'] = True
            except Exception as e:
                period_analysis_result['Period Analysis Status'] = 'Failed (ACF/PACF)'
                period_analysis_result['Period Error Message'] = f'ACF/PACF failed: {e}'

            try:
                if commodity_results['FFT Peaks']:
                    fft_periods = np.array([peak['Period'] for peak in commodity_results['FFT Peaks']])
                    closest_peak_index = np.argmin(np.abs(fft_periods - period))
                    closest_peak = commodity_results['FFT Peaks'][closest_peak_index]
                    period_analysis_result['FFT Power at Period'] = float(closest_peak['Power'])
            except Exception:
                period_analysis_result['Period Analysis Status'] = 'Partial Success (FFT Power Failed)'
                period_analysis_result['Period Error Message'] = 'FFT power lookup failed.'

            commodity_results['Analyzed Periods'].append(period_analysis_result)

        # --- Save decomposition data for the best period ---
        best_period_info = None
        if commodity_results['Analyzed Periods']:
            successful_periods = [p for p in commodity_results['Analyzed Periods'] if 'FFT Power at Period' in p and pd.notna(p['FFT Power at Period'])]
            if successful_periods:
                best_period_info = max(successful_periods, key=lambda x: x['FFT Power at Period'])

        if best_period_info and gcs_bucket_name and gcs_prefix:
            best_period = best_period_info['Period']
            if best_period in decompositions_by_period:
                try:
                    decomposition_data = decompositions_by_period[best_period]
                    decomposition_df = pd.DataFrame({
                        'original': commodity_series,
                        'trend': decomposition_data['trend'],
                        'seasonal': decomposition_data['seasonal'],
                        'residual': decomposition_data['resid']
                    })
                    decomposition_df.index.name = 'Date'

                    file_name = f"decomposition_{commodity.replace(' ', '_')}_period{best_period}.csv"
                    gcs_file_path = os.path.join(gcs_prefix, file_name)

                    save_dataframe_to_gcs(df=decomposition_df, bucket_name=gcs_bucket_name, gcs_prefix=gcs_file_path, include_index=True)
                    commodity_results['Decomposition File Path'] = f"gs://{gcs_bucket_name}/{gcs_file_path}"
                except Exception as e:
                    commodity_results['Error Message'] = (commodity_results['Error Message'] or "") + f" | Failed to save decomposition data: {e}"
        # ----------------------------------------------------

        seasonal_periods_found = [p['Period'] for p in commodity_results['Analyzed Periods'] if p.get('Is Seasonal for Period')]
        significant_acf_periods = [p['Period'] for p in commodity_results['Analyzed Periods'] if p.get('Is ACF Significant at Period')]
        significant_pacf_periods = [p['Period'] for p in commodity_results['Analyzed Periods'] if p.get('Is PACF Significant at Period')]
        summary_parts = []
        if seasonal_periods_found:
            summary_parts.append(f"Evidence of seasonality found at analyzed periods (STL Strength > 0.1): {seasonal_periods_found}")
        if significant_acf_periods:
            summary_parts.append(f"Significant ACF at analyzed periods: {significant_acf_periods}")
        if significant_pacf_periods:
            summary_parts.append(f"Significant PACF at analyzed periods: {significant_pacf_periods}")
        if summary_parts:
            commodity_results['Overall Seasonality Summary'] = ". ".join(summary_parts)
        else:
            commodity_results['Overall Seasonality Summary'] = "No significant seasonality detected at the analyzed periods based on combined checks."

        if commodity_results['Analysis Status'] != 'Success':
            commodity_results['Overall Seasonality Summary'] = f"{commodity_results['Analysis Status']}. {commodity_results['Overall Seasonality Summary']}"
        elif any(p['Period Analysis Status'] != 'Success' for p in commodity_results['Analyzed Periods']):
            failed_periods = [p['Period'] for p in commodity_results['Analyzed Periods'] if p['Period Analysis Status'] != 'Success']
            commodity_results['Overall Seasonality Summary'] = f"Partial Success (Errors in periods {failed_periods}). {commodity_results['Overall Seasonality Summary']}"
            commodity_results['Analysis Status'] = 'Partial Success (Period Analysis Errors)'

        seasonality_analysis_results[commodity] = commodity_results

    seasonality_results_list_flattened = []
    for commodity, result in seasonality_analysis_results.items():
        base = {
            'Commodity': result.get('Commodity'),
            'Overall Seasonality Summary': result.get('Overall Seasonality Summary'),
            'Analysis Status': result.get('Analysis Status'),
            'Error Message': result.get('Error Message'),
            'Overall FFT Power': result.get('Overall FFT Power'),
            'FFT Peaks': result.get('FFT Peaks', []),
            'Decomposition File Path': result.get('Decomposition File Path')
        }
        if result.get('Analyzed Periods'):
            for period_result in result['Analyzed Periods']:
                flattened_row = base.copy()
                flattened_row.update(period_result)
                seasonality_results_list_flattened.append(flattened_row)
        else:
            seasonality_results_list_flattened.append(base)

    seasonality_results_df_flattened = pd.DataFrame(seasonality_results_list_flattened)

    if seasonality_results_df_flattened.empty:
        return seasonality_results_df_flattened, pd.DataFrame()

    summary_entries = []
    unique_commodities = seasonality_results_df_flattened['Commodity'].unique()
    for commodity in unique_commodities:
        commodity_entries = seasonality_results_df_flattened[seasonality_results_df_flattened['Commodity'] == commodity].copy()
        if commodity_entries.empty or commodity_entries['Period'].isnull().all():
            summary_entries.append({
                'Commodity': commodity,
                'Period 1': np.nan, 'Strength 1': np.nan, 'Ratio 1': np.nan, 'FFT Power 1': np.nan, 'Seasonal 1': False, 'Explanation 1': 'No seasonality data available.',
                'Period 2': np.nan, 'Strength 2': np.nan, 'Ratio 2': np.nan, 'FFT Power 2': np.nan, 'Seasonal 2': False, 'Explanation 2': '',
                'Period 3': np.nan, 'Strength 3': np.nan, 'Ratio 3': np.nan, 'FFT Power 3': np.nan, 'Seasonal 3': False, 'Explanation 3': ''
            })
            continue

        commodity_entries['FFT Power at Period'] = pd.to_numeric(commodity_entries['FFT Power at Period'], errors='coerce')
        sorted_entries = commodity_entries.sort_values(by='FFT Power at Period', ascending=False, na_position='last')
        top_3_patterns = sorted_entries.head(3)

        commodity_summary = {'Commodity': commodity}
        for i, (index, row) in enumerate(top_3_patterns.iterrows()):
            period_num = i + 1
            period = row.get('Period', np.nan)
            strength_stl = row.get('Seasonality Strength (STL)', np.nan)
            ratio_stl = row.get('Calculated Seasonality Score (STL Ratio)', np.nan)
            fft_power = row.get('FFT Power at Period', np.nan)
            is_seasonal = row.get('Is Seasonal for Period', False)
            explanation = f"Period: {period}" if pd.notna(period) else ""
            if pd.notna(strength_stl):
                explanation += f", Strength (STL): {strength_stl:.4f}"
            if pd.notna(ratio_stl):
                explanation += f", Ratio (STL): {ratio_stl:.4f}"
            if pd.notna(fft_power):
                explanation += f", FFT Power: {fft_power:.2f}"
            explanation += f". Identified as seasonal: {is_seasonal}."
            commodity_summary[f'Period {period_num}'] = period
            commodity_summary[f'Strength {period_num}'] = strength_stl
            commodity_summary[f'Ratio {period_num}'] = ratio_stl
            commodity_summary[f'FFT Power {period_num}'] = fft_power
            commodity_summary[f'Seasonal {period_num}'] = is_seasonal
            commodity_summary[f'Explanation {period_num}'] = explanation

        for p in range(len(top_3_patterns) + 1, 4):
            commodity_summary[f'Period {p}'] = np.nan
            commodity_summary[f'Strength {p}'] = np.nan
            commodity_summary[f'Ratio {p}'] = np.nan
            commodity_summary[f'FFT Power {p}'] = np.nan
            commodity_summary[f'Seasonal {p}'] = False
            commodity_summary[f'Explanation {p}'] = ''

        summary_entries.append(commodity_summary)

    seasonality_summary_table_df = pd.DataFrame(summary_entries)

    return seasonality_results_df_flattened, seasonality_summary_table_df
