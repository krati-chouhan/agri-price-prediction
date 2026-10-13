"""
generate_pan_india_dataset.py
=============================
Generates comprehensive, realistic Pan-India agricultural market dataset
for major Indian States and Crops (featuring Pulses, Potato, Onion, Rice, Maize, etc.)
from 2019 to 2023, reflecting genuine APMC mandi price dynamics, monsoon cycles,
and seasonal harvest gluts.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Indian States with typical agro-climatic base parameters (Rainfall mm, Temp C, Humidity %)
STATE_CLIMATE = {
    'Maharashtra': {'rain_base': 1050, 'temp_base': 27.5, 'hum_base': 64, 'rain_std': 220},
    'Uttar Pradesh': {'rain_base': 950, 'temp_base': 26.0, 'hum_base': 65, 'rain_std': 190},
    'Madhya Pradesh': {'rain_base': 1020, 'temp_base': 26.8, 'hum_base': 58, 'rain_std': 200},
    'Andhra Pradesh': {'rain_base': 1120, 'temp_base': 28.5, 'hum_base': 68, 'rain_std': 210},
    'Punjab': {'rain_base': 680, 'temp_base': 24.5, 'hum_base': 56, 'rain_std': 150},
    'Karnataka': {'rain_base': 1180, 'temp_base': 26.5, 'hum_base': 66, 'rain_std': 220},
    'Gujarat': {'rain_base': 820, 'temp_base': 28.2, 'hum_base': 60, 'rain_std': 180},
    'West Bengal': {'rain_base': 1480, 'temp_base': 27.0, 'hum_base': 75, 'rain_std': 260},
    'Rajasthan': {'rain_base': 560, 'temp_base': 28.5, 'hum_base': 48, 'rain_std': 140},
    'Bihar': {'rain_base': 1220, 'temp_base': 26.5, 'hum_base': 69, 'rain_std': 230}
}

# Crop pricing baselines and seasonal harvest impact (multiplier by month 1-12)
# Prices in Rs per kg
CROP_SPECS = {
    'Potato': {
        'base_price': 22.0,
        'std': 4.5,
        'min_p': 11.0,
        'max_p': 42.0,
        # Harvest arrives Feb-April (low), stock depletes Aug-Nov (high)
        'month_multipliers': [1.05, 0.78, 0.72, 0.75, 0.88, 1.02, 1.15, 1.25, 1.35, 1.38, 1.22, 1.08]
    },
    'Onion': {
        'base_price': 34.0,
        'std': 9.5,
        'min_p': 16.0,
        'max_p': 78.0,
        # Rabi harvest in March-May (low), Pre-Kharif gap in Sept-Nov (peaks)
        'month_multipliers': [0.95, 0.88, 0.74, 0.70, 0.78, 0.92, 1.10, 1.28, 1.48, 1.55, 1.35, 1.05]
    },
    'Pulses': {
        'base_price': 88.0,
        'std': 14.0,
        'min_p': 58.0,
        'max_p': 152.0,
        # Chana arrives Mar-Apr, Tur arrives Dec-Jan; peak in Aug-Oct
        'month_multipliers': [0.90, 0.85, 0.82, 0.86, 0.95, 1.04, 1.14, 1.22, 1.25, 1.18, 1.02, 0.94]
    },
    'Rice': {
        'base_price': 52.0,
        'std': 8.0,
        'min_p': 30.0,
        'max_p': 82.0,
        # Kharif harvest in Nov-Dec (low), summer rise in May-July
        'month_multipliers': [0.94, 0.96, 0.98, 1.02, 1.06, 1.08, 1.05, 1.02, 1.00, 0.98, 0.88, 0.86]
    },
    'Maize': {
        'base_price': 25.0,
        'std': 3.8,
        'min_p': 16.0,
        'max_p': 38.0,
        # Harvest in Sept-Oct
        'month_multipliers': [0.98, 1.00, 1.02, 1.05, 1.08, 1.06, 1.04, 1.02, 0.86, 0.88, 0.92, 0.96]
    },
    'Chilli': {
        'base_price': 125.0,
        'std': 22.0,
        'min_p': 72.0,
        'max_p': 195.0,
        # Harvest in Jan-Mar, peak storage sale in July-Oct
        'month_multipliers': [0.86, 0.82, 0.80, 0.88, 0.98, 1.08, 1.18, 1.24, 1.22, 1.15, 1.02, 0.92]
    },
    'Groundnut': {
        'base_price': 78.0,
        'std': 10.5,
        'min_p': 52.0,
        'max_p': 115.0,
        # Harvest Oct-Nov, pre-monsoon oil milling in Apr-June
        'month_multipliers': [0.96, 0.98, 1.02, 1.08, 1.12, 1.10, 1.05, 1.00, 0.98, 0.86, 0.88, 0.92]
    },
    'Sugarcane': {
        'base_price': 32.0,
        'std': 5.0,
        'min_p': 20.0,
        'max_p': 50.0,
        # Crushing season Nov-March
        'month_multipliers': [0.90, 0.88, 0.92, 0.98, 1.04, 1.08, 1.12, 1.10, 1.06, 1.02, 0.94, 0.92]
    }
}

# State-specific price adjustment factor (due to local production strength & transport)
STATE_CROP_FACTOR = {
    'Maharashtra': {'Onion': 0.88, 'Pulses': 0.95, 'Sugarcane': 0.92, 'Potato': 1.05},
    'Uttar Pradesh': {'Potato': 0.84, 'Sugarcane': 0.90, 'Pulses': 0.98, 'Onion': 1.06},
    'Madhya Pradesh': {'Pulses': 0.86, 'Maize': 0.92, 'Potato': 1.02, 'Onion': 0.96},
    'Andhra Pradesh': {'Chilli': 0.85, 'Rice': 0.90, 'Groundnut': 0.92, 'Pulses': 1.04},
    'Punjab': {'Rice': 0.92, 'Potato': 0.90, 'Maize': 0.95, 'Pulses': 1.08},
    'Karnataka': {'Maize': 0.90, 'Pulses': 0.96, 'Onion': 0.92, 'Groundnut': 0.96},
    'Gujarat': {'Groundnut': 0.88, 'Potato': 0.95, 'Onion': 0.94, 'Pulses': 1.02},
    'West Bengal': {'Potato': 0.82, 'Rice': 0.88, 'Pulses': 1.10, 'Onion': 1.12},
    'Rajasthan': {'Pulses': 0.92, 'Maize': 0.96, 'Onion': 1.02, 'Potato': 1.08},
    'Bihar': {'Maize': 0.88, 'Potato': 0.92, 'Rice': 0.94, 'Pulses': 1.06}
}

records = []
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='MS')

# Inflation annual shift: 2019 to 2023
year_inflation = {2019: 0.92, 2020: 0.96, 2021: 1.00, 2022: 1.06, 2023: 1.12}

for date in dates:
    month = date.month
    year = date.year
    date_str = date.strftime('%b-%y')
    infl = year_inflation.get(year, 1.0)

    for state, clim in STATE_CLIMATE.items():
        # Seasonal weather variation
        # Monsoon in India: June-Sept (high rain, humidity)
        is_monsoon = month in [6, 7, 8, 9]
        is_summer = month in [4, 5]
        is_winter = month in [12, 1, 2]

        rain_seasonal = clim['rain_base'] * (1.6 if is_monsoon else (0.4 if is_winter else 0.8))
        temp_seasonal = clim['temp_base'] + (4.0 if is_summer else (-4.5 if is_winter else 0.5))
        hum_seasonal = clim['hum_base'] + (14.0 if is_monsoon else (-12.0 if is_summer else 0.0))

        for crop, spec in CROP_SPECS.items():
            # State affinity factor
            factor = STATE_CROP_FACTOR.get(state, {}).get(crop, 1.0)
            month_mult = spec['month_multipliers'][month - 1]

            # Generate 2 to 3 representative market samples per month-state-crop combination
            for _ in range(2):
                rain_val = int(np.clip(np.random.normal(rain_seasonal, clim['rain_std'] * 0.4), 350, 2200))
                temp_val = int(np.clip(np.random.normal(temp_seasonal, 2.0), 16, 42))
                hum_val = int(np.clip(np.random.normal(hum_seasonal, 4.0), 32, 94))

                # Climate impact on price:
                # Heavy excess rain or severe drought can increase vegetable/grain prices
                rain_anomaly = (rain_val - clim['rain_base']) / clim['rain_base']
                climate_price_mult = 1.0 + (0.08 * abs(rain_anomaly))

                base_p = spec['base_price'] * infl * factor * month_mult * climate_price_mult
                noise = np.random.normal(0, spec['std'] * 0.3)
                final_price = round(float(np.clip(base_p + noise, spec['min_p'], spec['max_p'])), 2)

                records.append({
                    'Location': state,
                    'Crop': crop,
                    'Rainfall': rain_val,
                    'Temperature': temp_val,
                    'Humidity': hum_val,
                    'Date': date_str,
                    'Crop_Price': final_price
                })

df = pd.DataFrame(records)
print(f"[*] Generated {len(df)} total dataset records.")
print(f"[*] States ({len(df['Location'].unique())}): {sorted(df['Location'].unique())}")
print(f"[*] Crops ({len(df['Crop'].unique())}): {sorted(df['Crop'].unique())}")

# Save to crops_dataset.csv in root and subfolder
df.to_csv('crops_dataset.csv', index=False)
sub_csv = os.path.join('Crop-Price-Prediction-Minor-Project-main', 'crops_dataset.csv')
if os.path.exists('Crop-Price-Prediction-Minor-Project-main'):
    df.to_csv(sub_csv, index=False)

print("[SUCCESS] crops_dataset.csv updated successfully!")
