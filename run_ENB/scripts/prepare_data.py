#!/usr/bin/env python3
"""
Prepare ENB dataset for ginn-lp training.
Creates separate CSV files for Heating (Y1) and Cooling (Y2) targets.
Data is NOT normalized - ginn-lp will handle preprocessing internally.

Matches the exact train/test split from codes_v2/ENB/mtr_ginn_sym.py
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("ENB Dataset Preparation for ginn-lp")
print("=" * 60)

# ================= Load Dataset (same as codes_v2) =================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
print(f"\nLoading ENB dataset from: {url}")
try:
    df = pd.read_excel(url)
    print(f"✅ Successfully loaded dataset!")
except Exception as e:
    print(f"❌ Error loading dataset from URL: {e}")
    print(f"\nTrying alternative: downloading to temp file first...")
    import urllib.request
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        df = pd.read_excel(tmp_file.name)
        os.unlink(tmp_file.name)
    print(f"✅ Successfully loaded dataset via temp file!")

# Features (X1...X8), Targets (Y1=Heating, Y2=Cooling)
X_np = df.iloc[:, :-2].values
y_np = df.iloc[:, -2:].values

# Validate data structure (should match codes_v2)
expected_shape = (768, 8)
if X_np.shape != expected_shape:
    raise ValueError(f"Unexpected X shape: {X_np.shape}, expected {expected_shape}")
if y_np.shape != (768, 2):
    raise ValueError(f"Unexpected y shape: {y_np.shape}, expected (768, 2)")

print(f"Dataset shape: {X_np.shape} ✓")
print(f"Targets shape: {y_np.shape} ✓")
print(f"Target names: Y1=Heating Load, Y2=Cooling Load")
print(f"Column names: {list(df.columns)}")

# ================= Split (same as codes_v2) =================
print("\nSplitting data (test_size=0.2, random_state=42)...")
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

print(f"Train: X={X_train_np.shape}, y={y_train_np.shape}")
print(f"Test:  X={X_test_np.shape}, y={y_test_np.shape}")

# Get feature column names
feature_cols = list(df.columns[:-2])  # X1, X2, ..., X8

# ================= Create Heating Dataset (Y1) =================
print("\n" + "=" * 60)
print("Creating Heating Load (Y1) datasets...")
print("=" * 60)

# Training set
heating_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
heating_train_df['target'] = y_train_np[:, 0]
heating_train_path = os.path.join(DATA_DIR, 'Heating_train.csv')
heating_train_df.to_csv(heating_train_path, index=False)
print(f"✅ Saved: {heating_train_path}")
print(f"   Shape: {heating_train_df.shape}")
print(f"   Target range: [{heating_train_df['target'].min():.2f}, {heating_train_df['target'].max():.2f}]")

# Test set
heating_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
heating_test_df['target'] = y_test_np[:, 0]
heating_test_path = os.path.join(DATA_DIR, 'Heating_test.csv')
heating_test_df.to_csv(heating_test_path, index=False)
print(f"✅ Saved: {heating_test_path}")
print(f"   Shape: {heating_test_df.shape}")
print(f"   Target range: [{heating_test_df['target'].min():.2f}, {heating_test_df['target'].max():.2f}]")

# ================= Create Cooling Dataset (Y2) =================
print("\n" + "=" * 60)
print("Creating Cooling Load (Y2) datasets...")
print("=" * 60)

# Training set
cooling_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
cooling_train_df['target'] = y_train_np[:, 1]
cooling_train_path = os.path.join(DATA_DIR, 'Cooling_train.csv')
cooling_train_df.to_csv(cooling_train_path, index=False)
print(f"✅ Saved: {cooling_train_path}")
print(f"   Shape: {cooling_train_df.shape}")
print(f"   Target range: [{cooling_train_df['target'].min():.2f}, {cooling_train_df['target'].max():.2f}]")

# Test set
cooling_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
cooling_test_df['target'] = y_test_np[:, 1]
cooling_test_path = os.path.join(DATA_DIR, 'Cooling_test.csv')
cooling_test_df.to_csv(cooling_test_path, index=False)
print(f"✅ Saved: {cooling_test_path}")
print(f"   Shape: {cooling_test_df.shape}")
print(f"   Target range: [{cooling_test_df['target'].min():.2f}, {cooling_test_df['target'].max():.2f}]")

# ================= Summary =================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Data directory: {DATA_DIR}")
print("\nFiles created:")
print("  📄 Heating_train.csv  - Training data for Heating Load (Y1)")
print("  📄 Heating_test.csv   - Test data for Heating Load (Y1)")
print("  📄 Cooling_train.csv  - Training data for Cooling Load (Y2)")
print("  📄 Cooling_test.csv   - Test data for Cooling Load (Y2)")
print("\n⚠️  Note: Data is RAW (not normalized). ginn-lp will handle preprocessing internally.")
print("\n" + "=" * 60)
print("Data preparation complete!")
print("=" * 60)
