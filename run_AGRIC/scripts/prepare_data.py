#!/usr/bin/env python3
"""
Prepare Agriculture dataset for ginn-lp training.
Creates separate CSV files for Sustainability Score (Y1) and Consumer Trend Index (Y2).
Data is NOT normalized - ginn-lp will handle preprocessing internally.

Matches the exact train/test split from codes_v2/Agriculture/mtr_ginn_agric_sym.py
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Agriculture Dataset Preparation for ginn-lp")
print("=" * 60)

# ================= Load Dataset (same as codes_v2) =================
possible_paths = [
    os.path.join(BASE_DIR, 'codes', 'Agriculture'),
    os.path.join(BASE_DIR, 'codes_v2', 'Agriculture'),
    os.path.join(BASE_DIR, 'ginn-lp_hex', 'original_data'),
]

base_path = None
for path in possible_paths:
    fa_path = os.path.join(path, 'farmer_advisor_dataset.csv')
    mr_path = os.path.join(path, 'market_researcher_dataset.csv')
    if os.path.exists(fa_path) and os.path.exists(mr_path):
        base_path = path
        break

if base_path is None:
    raise FileNotFoundError(
        f"Could not find Agriculture data files. Checked:\n" + 
        "\n".join(f"  - {p}" for p in possible_paths)
    )

print(f"\nFound data in: {base_path}")
print("Loading datasets...")

fa_ds = pd.read_csv(os.path.join(base_path, 'farmer_advisor_dataset.csv'))
mr_ds = pd.read_csv(os.path.join(base_path, 'market_researcher_dataset.csv'))

print(f"  Farmer Advisor: {fa_ds.shape}")
print(f"  Market Researcher: {mr_ds.shape}")

# Sort (same as codes_v2)
fa_ds.sort_values('Crop_Type', inplace=True)
mr_ds.sort_values('Product', inplace=True)

# Merge (same as codes_v2)
df_mrg = pd.merge(fa_ds, mr_ds, left_on='Crop_Type', right_on='Product', how='inner').drop(
    ['Farm_ID', 'Market_ID', 'Product'], axis=1
)

df_mrg = df_mrg[[
    'Crop_Type', 'Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
    'Fertilizer_Usage_kg', 'Pesticide_Usage_kg', 'Crop_Yield_ton',
    'Market_Price_per_ton', 'Demand_Index', 'Supply_Index', 'Competitor_Price_per_ton', 'Economic_Indicator',
    'Weather_Impact_Score', 'Seasonal_Factor', 'Sustainability_Score', 'Consumer_Trend_Index'
]]

# Cleanup
df_mrg.dropna(inplace=True)
for col in df_mrg.select_dtypes(include=['object']).columns:
    df_mrg[col] = df_mrg[col].astype('category')

# Encode categoricals (same as codes_v2)
df = df_mrg.copy()
label_encoders = {}
for c in ['Crop_Type', 'Seasonal_Factor']:
    if c in df.columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        label_encoders[c] = le

# Targets and features
y = df[['Sustainability_Score', 'Consumer_Trend_Index']].to_numpy()
X = df.drop(columns=['Sustainability_Score', 'Consumer_Trend_Index'])

print(f"\nFinal dataset shape: {X.shape}")
print(f"Targets shape: {y.shape}")

# Get feature column names
feature_cols = list(X.columns)

# ================= Split (same as codes_v2) =================
test_size = 0.2
seed = 42
print(f"\nSplitting data (test_size={test_size}, random_state={seed})...")
X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
    X, y, test_size=test_size, random_state=seed
)

print(f"Train: X={X_train_df.shape}, y={y_train_np.shape}")
print(f"Test:  X={X_test_df.shape}, y={y_test_np.shape}")

# Convert to numpy for consistency
X_train_np = X_train_df.values
X_test_np = X_test_df.values

# ================= Create Sustainability Score Dataset (Y1) =================
print("\n" + "=" * 60)
print("Creating Sustainability Score (Y1) datasets...")
print("=" * 60)

# Training set
sustainability_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
sustainability_train_df['target'] = y_train_np[:, 0]
sustainability_train_path = os.path.join(DATA_DIR, 'Sustainability_train.csv')
sustainability_train_df.to_csv(sustainability_train_path, index=False)
print(f"✅ Saved: {sustainability_train_path}")
print(f"   Shape: {sustainability_train_df.shape}")
print(f"   Target range: [{sustainability_train_df['target'].min():.2f}, {sustainability_train_df['target'].max():.2f}]")

# Test set
sustainability_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
sustainability_test_df['target'] = y_test_np[:, 0]
sustainability_test_path = os.path.join(DATA_DIR, 'Sustainability_test.csv')
sustainability_test_df.to_csv(sustainability_test_path, index=False)
print(f"✅ Saved: {sustainability_test_path}")
print(f"   Shape: {sustainability_test_df.shape}")
print(f"   Target range: [{sustainability_test_df['target'].min():.2f}, {sustainability_test_df['target'].max():.2f}]")

# ================= Create Consumer Trend Index Dataset (Y2) =================
print("\n" + "=" * 60)
print("Creating Consumer Trend Index (Y2) datasets...")
print("=" * 60)

# Training set
consumer_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
consumer_train_df['target'] = y_train_np[:, 1]
consumer_train_path = os.path.join(DATA_DIR, 'ConsumerTrend_train.csv')
consumer_train_df.to_csv(consumer_train_path, index=False)
print(f"✅ Saved: {consumer_train_path}")
print(f"   Shape: {consumer_train_df.shape}")
print(f"   Target range: [{consumer_train_df['target'].min():.2f}, {consumer_train_df['target'].max():.2f}]")

# Test set
consumer_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
consumer_test_df['target'] = y_test_np[:, 1]
consumer_test_path = os.path.join(DATA_DIR, 'ConsumerTrend_test.csv')
consumer_test_df.to_csv(consumer_test_path, index=False)
print(f"✅ Saved: {consumer_test_path}")
print(f"   Shape: {consumer_test_df.shape}")
print(f"   Target range: [{consumer_test_df['target'].min():.2f}, {consumer_test_df['target'].max():.2f}]")

# ================= Summary =================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Data directory: {DATA_DIR}")
print("\nFiles created:")
print("  📄 Sustainability_train.csv  - Training data for Sustainability Score (Y1)")
print("  📄 Sustainability_test.csv   - Test data for Sustainability Score (Y1)")
print("  📄 ConsumerTrend_train.csv  - Training data for Consumer Trend Index (Y2)")
print("  📄 ConsumerTrend_test.csv   - Test data for Consumer Trend Index (Y2)")
print("\n⚠️  Note: Data is RAW (not normalized). ginn-lp will handle preprocessing internally.")
print("⚠️  Note: Categorical features (Crop_Type, Seasonal_Factor) are label-encoded.")
print("\n" + "=" * 60)
print("Data preparation complete!")
print("=" * 60)
