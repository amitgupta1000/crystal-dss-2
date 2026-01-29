"""Quick standalone test for pmdarima.auto_arima on local data.
Usage: run this file in the workspace root where ALL_CLEAN_DATA.csv is located.
"""
import sys
import pandas as pd

def main():
    try:
        import pmdarima as pm
    except Exception as e:
        print("pmdarima is not installed or could not be imported:", e)
        print("Install with: pip install pmdarima")
        sys.exit(2)

    # Load local CSV (expects a 'date' column)
    try:
        df = pd.read_csv('ALL_CLEAN_DATA.csv', parse_dates=['date'], dayfirst=True, infer_datetime_format=True)
    except Exception as e:
        print('Failed to read ALL_CLEAN_DATA.csv:', e)
        sys.exit(3)

    # Set date index if present
    if 'date' in df.columns:
        df.set_index('date', inplace=True)

    # Pick the first numeric column to test
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print('No numeric columns found in ALL_CLEAN_DATA.csv to test ARIMA on.')
        sys.exit(4)

    col = numeric_cols[0]
    print('Testing auto_arima on column:', col)

    series = df[col].dropna()
    if series.empty:
        print(f'Series for {col} is empty after dropping NaNs.')
        sys.exit(5)

    try:
        model = pm.auto_arima(
            series,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            max_d=2,
            seasonal=False,
            stepwise=True,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            information_criterion='aic'
        )

        print('\nAuto-ARIMA finished. Model summary:')
        print(model.summary())

        n_periods = 12
        fc = model.predict(n_periods=n_periods)
        print(f'Forecast for next {n_periods} periods:')
        print(fc)

    except Exception as e:
        print('auto_arima failed:', e)
        sys.exit(6)

if __name__ == '__main__':
    main()
