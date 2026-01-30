import os
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from google.cloud import storage
from urllib.parse import urlparse
import joblib


def adjusted_r2_score(r2, n, k):
    if n - k - 1 == 0:
        return -np.inf
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_regression_models(numerical_price_features_diff_df,
                          commodities,
                          target_commodities,
                          bucket_name='crystal-dss',
                          models_prefix='models/regression',
                          degrees_to_test=None,
                          top_n=8,
                          max_features_for_poly=8):
    if degrees_to_test is None:
        degrees_to_test = [1, 2, 3]

    regression_results = []

    for target_commodity in target_commodities:
        all_potential_features = [col for col in commodities if col != target_commodity]
        all_potential_features = [f for f in all_potential_features if f in numerical_price_features_diff_df.columns]

        if not all_potential_features:
            continue

        data_full = numerical_price_features_diff_df[all_potential_features + [target_commodity]].dropna()
        if data_full.empty:
            continue

        for degree in degrees_to_test:
            current_features = []
            if degree == 1:
                current_features = all_potential_features
            else:
                correlations = data_full[all_potential_features].corrwith(data_full[target_commodity]).abs()
                top_correlated_features = correlations.nlargest(max_features_for_poly).index.tolist()
                current_features = top_correlated_features

            if not current_features:
                continue

            X = data_full[current_features]
            y = data_full[target_commodity]

            if len(X) < 2:
                continue

            test_size_val = min(0.2, (len(X) - 1) / len(X))
            if test_size_val == 0:
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, shuffle=False)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            poly = PolynomialFeatures(degree=degree)
            X_poly_train = poly.fit_transform(X_train_scaled)
            X_poly_test = poly.transform(X_test_scaled)

            k_train = X_poly_train.shape[1] - 1
            k_test = X_poly_test.shape[1] - 1

            if X_poly_train.shape[0] < k_train + 1:
                continue

            model = Ridge(alpha=1.0).fit(X_poly_train, y_train)

            y_pred_train = model.predict(X_poly_train)
            r2_train = r2_score(y_train, y_pred_train)
            adj_r2_train = adjusted_r2_score(r2_train, X_train.shape[0], k_train)
            rmse_train = calculate_rmse(y_train, y_pred_train)

            y_pred_test = model.predict(X_poly_test)
            r2_test = r2_score(y_test, y_pred_test)
            adj_r2_test = adjusted_r2_score(r2_test, X_test.shape[0], k_test)
            rmse_test = calculate_rmse(y_test, y_pred_test)

            coef = pd.Series(model.coef_, index=poly.get_feature_names_out(input_features=current_features))
            coef_for_ranking = coef[coef.index != '1']
            top_coef_sorted = coef_for_ranking.abs().sort_values(ascending=False).head(top_n)
            top_coefficients_with_values = coef[top_coef_sorted.index].to_dict()

            # --- Save model and transformers for scenario analysis ---
            model_id = f"{target_commodity}_deg{degree}"
            
            # Create metadata for full reconstruction
            metadata = {
                'target_commodity': target_commodity,
                'degree': degree,
                'alpha': 1.0,  # Ridge alpha parameter
                'features': current_features,
                'feature_count': len(current_features),
                'train_adj_r2': adj_r2_train,
                'test_adj_r2': adj_r2_test,
                'train_rmse': rmse_train,
                'test_rmse': rmse_test,
                'intercept': float(model.intercept_),
            }
            
            artifacts = {
                'model': model,
                'scaler': scaler,
                'poly': poly,
                'metadata': metadata,
            }
            artifact_paths = {}

            # Save to Google Cloud Storage
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)

                for name, obj in artifacts.items():
                    blob_name = f"{models_prefix}/{model_id}_{name}.joblib"
                    blob = bucket.blob(blob_name)
                    with io.BytesIO() as buffer:
                        joblib.dump(obj, buffer)
                        buffer.seek(0)
                        blob.upload_from_file(buffer, content_type='application/octet-stream')
                    artifact_paths[name] = f"gs://{bucket_name}/{blob_name}"
            except Exception as e:
                print(f"Error uploading to GCS: {e}")
                continue  # Skip this model if upload fails
            # -------------------------------------------------------------

            result_entry = {
                'Target Commodity': target_commodity,
                'Degree': degree,
                'Model ID': model_id,
                'Features Used Count': len(current_features),
                'Train Adj R2': adj_r2_train,
                'Test Adj R2': adj_r2_test,
                'Train RMSE': rmse_train,
                'Test RMSE': rmse_test,
                'Intercept': float(model.intercept_),
                'Model Path': artifact_paths['model'],
                'Scaler Path': artifact_paths['scaler'],
                'Poly Path': artifact_paths['poly'],
                'Metadata Path': artifact_paths['metadata'],
            }

            for i, (feature_name, coefficient_value) in enumerate(top_coefficients_with_values.items()):
                result_entry[f'Top {i+1} Feature'] = feature_name
                result_entry[f'Top {i+1} Value'] = float(coefficient_value)

            regression_results.append(result_entry)

    return regression_results
