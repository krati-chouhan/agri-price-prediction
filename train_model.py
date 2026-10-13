"""
train_model.py
==============
Script to train, evaluate, and save the Machine Learning model for Crop Price Prediction.
Trains Random Forest Regressor and AdaBoost Regressor on crops_dataset.csv,
evaluates metrics (R2, MSE, MAE, RMSE), and serializes the best model.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'crops_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'random_forest_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'model_metrics.json')

def load_and_preprocess_data(dataset_path):
    print(f"[*] Loading dataset from: {dataset_path}")
    data = pd.read_csv(dataset_path)

    # Convert date column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%b-%y')

    # Extract Month and Year
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    # Label encoding for categorical columns
    le_location = LabelEncoder()
    le_crop = LabelEncoder()

    data['Location_Encoded'] = le_location.fit_transform(data['Location'])
    data['Crop_Encoded'] = le_crop.fit_transform(data['Crop'])

    mapping_info = {
        'crops': {str(k): int(v) for k, v in zip(le_crop.classes_, range(len(le_crop.classes_)))},
        'locations': {str(k): int(v) for k, v in zip(le_location.classes_, range(len(le_location.classes_)))}
    }

    feature_cols = ['Location_Encoded', 'Crop_Encoded', 'Rainfall', 'Temperature', 'Humidity', 'Month', 'Year']
    X = data[feature_cols].rename(columns={'Location_Encoded': 'Location', 'Crop_Encoded': 'Crop'})
    y = data['Crop_Price']

    return X, y, data, mapping_info

def evaluate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    return {'r2': round(r2, 4), 'mse': round(mse, 4), 'mae': round(mae, 4), 'rmse': round(rmse, 4)}

def train_and_save():
    X, y, raw_data, mapping_info = load_and_preprocess_data(DATASET_PATH)

    print(f"[*] Total dataset records: {len(X)}")
    print(f"[*] Features: {list(X.columns)}")
    print(f"[*] Crop classes: {mapping_info['crops']}")
    print(f"[*] Location classes: {mapping_info['locations']}")

    # Stratified split to ensure balanced representation across crop and location
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=raw_data[['Location', 'Crop']]
    )

    print("\n[*] Training Random Forest Regressor (300 estimators)...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_metrics(rf_model, X_test, y_test)
    print(f"    Random Forest -> R2: {rf_metrics['r2']}, RMSE: {rf_metrics['rmse']}, MAE: {rf_metrics['mae']}")

    print("\n[*] Training AdaBoost Regressor...")
    ab_model = AdaBoostRegressor(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )
    ab_model.fit(X_train, y_train)
    ab_metrics = evaluate_metrics(ab_model, X_test, y_test)
    print(f"    AdaBoost      -> R2: {ab_metrics['r2']}, RMSE: {ab_metrics['rmse']}, MAE: {ab_metrics['mae']}")

    # Feature importances
    feature_importances = {
        col: round(float(imp), 4)
        for col, imp in zip(X.columns, rf_model.feature_importances_)
    }

    # Save model
    print(f"\n[*] Saving trained Random Forest model to: {MODEL_PATH}")
    joblib.dump(rf_model, MODEL_PATH)

    # Save metrics JSON
    metrics_data = {
        'best_model': 'Random Forest Regressor',
        'random_forest': rf_metrics,
        'adaboost': ab_metrics,
        'feature_importances': feature_importances,
        'dataset_size': len(X),
        'features': list(X.columns),
        'mappings': mapping_info
    }

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[*] Saved metrics and info to: {METRICS_PATH}")

    # Also copy to subfolder if it exists
    subfolder = os.path.join(BASE_DIR, 'Crop-Price-Prediction-Minor-Project-main')
    if os.path.isdir(subfolder):
        sub_model_path = os.path.join(subfolder, 'random_forest_model.joblib')
        sub_metrics_path = os.path.join(subfolder, 'model_metrics.json')
        joblib.dump(rf_model, sub_model_path)
        with open(sub_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"[*] Synchronized model to subfolder: {sub_model_path}")

    print("\n[SUCCESS] Training completed successfully!")

if __name__ == '__main__':
    train_and_save()
