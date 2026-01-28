import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from ginnlp.ginnlp import GINNLP
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_agriculture_like_codes(
    data_dir: str,
    target: str,
    sample_fraction: float | None,
    test_size: float,
    random_state: int,
):
    """
    Match the data processing from codes/Agriculture/mtr_ginn_agric_sym.py:
    - read farmer_advisor_dataset.csv + market_researcher_dataset.csv
    - optional sampling (random_state=42)
    - sort, merge on Crop_Type/Product
    - select exact columns, dropna
    - label-encode Crop_Type and Seasonal_Factor
    - split train/test with random_state
    - MinMaxScaler + 1e-6 on features
    - log-transform chosen target

    Returns:
      X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train_raw, y_test_raw, meta
    """
    fa_path = os.path.join(data_dir, "farmer_advisor_dataset.csv")
    mr_path = os.path.join(data_dir, "market_researcher_dataset.csv")
    if not os.path.exists(fa_path):
        raise FileNotFoundError(f"Missing: {fa_path}")
    if not os.path.exists(mr_path):
        raise FileNotFoundError(f"Missing: {mr_path}")

    fa_ds = pd.read_csv(fa_path)
    mr_ds = pd.read_csv(mr_path)

    if sample_fraction is not None and 0 < sample_fraction < 1:
        fa_ds = fa_ds.sample(frac=sample_fraction, random_state=42)
        mr_ds = mr_ds.sample(frac=sample_fraction, random_state=42)

    fa_ds.sort_values("Crop_Type", inplace=True)
    mr_ds.sort_values("Product", inplace=True)

    df_mrg = (
        pd.merge(fa_ds, mr_ds, left_on="Crop_Type", right_on="Product", how="inner")
        .drop(["Farm_ID", "Market_ID", "Product"], axis=1)
        .copy()
    )

    df_mrg = df_mrg[
        [
            "Crop_Type",
            "Soil_pH",
            "Soil_Moisture",
            "Temperature_C",
            "Rainfall_mm",
            "Fertilizer_Usage_kg",
            "Pesticide_Usage_kg",
            "Crop_Yield_ton",
            "Market_Price_per_ton",
            "Demand_Index",
            "Supply_Index",
            "Competitor_Price_per_ton",
            "Economic_Indicator",
            "Weather_Impact_Score",
            "Seasonal_Factor",
            "Sustainability_Score",
            "Consumer_Trend_Index",
        ]
    ]

    df_mrg.dropna(inplace=True)
    for col in df_mrg.select_dtypes(include=["object"]).columns:
        df_mrg[col] = df_mrg[col].astype("category")

    df = df_mrg.copy()
    for c in ["Crop_Type", "Seasonal_Factor"]:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])

    if target == "Y1":
        target_col = "Sustainability_Score"
        target_name = "Sustainability_Score (Y1)"
    elif target == "Y2":
        target_col = "Consumer_Trend_Index"
        target_name = "Consumer_Trend_Index (Y2)"
    else:
        raise ValueError('Invalid --target. Use "Y1" or "Y2".')

    y_raw = df[target_col].to_numpy(dtype=float)
    X_df = df.drop(columns=["Sustainability_Score", "Consumer_Trend_Index"])
    X = X_df.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=test_size, random_state=random_state
    )

    # Same as codes/Agriculture: MinMaxScaler + 1e-6
    MIN_POSITIVE = 1e-6
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train) + MIN_POSITIVE
    X_test_scaled = scaler.transform(X_test) + MIN_POSITIVE

    # Same as codes/Agriculture: log targets
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    meta = {
        "data_dir": data_dir,
        "target": target,
        "target_name": target_name,
        "target_col": target_col,
        "sample_fraction": sample_fraction,
        "min_positive": MIN_POSITIVE,
        "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": list(X_df.columns),
    }
    return X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train, y_test, meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train GINN-LP on Agriculture dataset (match codes/Agriculture/mtr_ginn_agric_sym.py preprocessing)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="run_AGRIC/data",
        help="Directory containing farmer_advisor_dataset.csv and market_researcher_dataset.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["Y1", "Y2"],
        required=True,
        help="Which single target to train: Y1=Sustainability_Score, Y2=Consumer_Trend_Index",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.1,
        help="Optional sampling fraction like codes (default 0.1). Use 1.0 for full data.",
    )
    parser.add_argument('--output_dir', type=str, default='run_AGRIC/outputs', help='Output directory for results')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--round_digits', type=int, default=3, help='Number of digits to round to')
    parser.add_argument('--start_ln_blocks', type=int, default=1, help='Number of starting blocks')
    parser.add_argument('--growth_steps', type=int, default=3, help='Number of growth steps')
    parser.add_argument('--l1_reg', type=float, default=1e-4, help='L1 regularization')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=1000, help='Decay steps')
    parser.add_argument('--reg_change', type=float, default=0.5,
                        help='Fraction of epochs to regularization change')
    parser.add_argument('--train_iter', type=int, default=4, help='Number of training iterations')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=100, help='Random state for train/test split (default: 100 for Agriculture)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GINN-LP Training on Agriculture Dataset")
    print("Preprocessing: same as codes/Agriculture/mtr_ginn_agric_sym.py")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Target: {args.target}")
    print(f"Sample fraction: {args.sample_fraction}")

    # Load + preprocess exactly like codes
    sample_fraction = None if args.sample_fraction is None else float(args.sample_fraction)
    if sample_fraction is not None and sample_fraction >= 1:
        sample_fraction = None

    (
        X_train_scaled,
        X_test_scaled,
        y_train_log,
        y_test_log,
        y_train_raw,
        y_test_raw,
        meta,
    ) = load_agriculture_like_codes(
        data_dir=args.data_dir,
        target=args.target,
        sample_fraction=sample_fraction,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"\nFeature shape: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
    print(
        f"Scaled feature ranges: train[min={X_train_scaled.min():.6f}, max={X_train_scaled.max():.6f}], "
        f"test[min={X_test_scaled.min():.6f}, max={X_test_scaled.max():.6f}]"
    )
    print(
        f"\nTarget raw range: train[{y_train_raw.min():.4f}, {y_train_raw.max():.4f}], "
        f"test[{y_test_raw.min():.4f}, {y_test_raw.max():.4f}]"
    )
    print(
        f"Target log range: train[{y_train_log.min():.4f}, {y_train_log.max():.4f}], "
        f"test[{y_test_log.min():.4f}, {y_test_log.max():.4f}]"
    )
    
    # ================= Train GINN-LP =================
    print("\n" + "=" * 60)
    print("Training GINN-LP...")
    print("=" * 60)
    
    model = GINNLP(
        num_epochs=args.num_epochs,
        round_digits=args.round_digits,
        start_ln_blocks=args.start_ln_blocks,
        growth_steps=args.growth_steps,
        l1_reg=args.l1_reg,
        l2_reg=args.l2_reg,
        init_lr=args.init_lr,
        decay_steps=args.decay_steps,
        reg_change=args.reg_change,
        train_iter=args.train_iter
    )
    
    model.fit(X_train_scaled, y_train_log)
    
    print(f"\nRecovered equation: {model.recovered_eq}")
    
    # ================= Evaluate on Test Set =================
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    
    # Predict on test set (using normalized features)
    y_pred_log = model.predict(X_test_scaled)
    
    # Flatten predictions if needed
    if y_pred_log.ndim > 1:
        y_pred_log = y_pred_log.flatten()
    
    # Convert back from log space for metrics (compare with original y_test)
    y_pred = np.exp(y_pred_log)
    
    # Calculate metrics on original scale
    mse = mean_squared_error(y_test_raw, y_pred)
    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_raw, y_pred) * 100
    
    print(f"\nTest Set Metrics (on original scale, after exp transform):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # Also calculate metrics in log space for reference
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    rmse_log = np.sqrt(mse_log)
    mape_log = mean_absolute_percentage_error(y_test_log, y_pred_log) * 100
    
    print(f"\nTest Set Metrics (in log space):")
    print(f"  MSE:  {mse_log:.6f}")
    print(f"  MAE:  {mae_log:.6f}")
    print(f"  RMSE: {rmse_log:.6f}")
    print(f"  MAPE: {mape_log:.4f}%")
    
    # ================= Save Results =================
    print("\n" + "=" * 60)
    print("Saving Results...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename with hyperparameters for easy identification
    data_basename = f"AGRIC_{meta['target']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include key hyperparameters in filename
    hyperparams_str = f"E{args.num_epochs}_B{args.start_ln_blocks}_G{args.growth_steps}"
    output_filename = f"ginnlp_AGRIC_{data_basename}_{hyperparams_str}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Prepare results dictionary
    results = {
        "experiment_info": {
            "timestamp": timestamp,
            "dataset": "Agriculture",
            "data_dir": args.data_dir,
            "dataset_name": data_basename,
            "target_id": meta["target"],
            "target_name": meta["target_name"],
            "sample_fraction": meta["sample_fraction"] if meta["sample_fraction"] is not None else 1.0,
            "num_epochs": args.num_epochs,
            "start_ln_blocks": args.start_ln_blocks,
            "growth_steps": args.growth_steps,
            "learning_rate": args.init_lr,
            "l1_reg": args.l1_reg,
            "l2_reg": args.l2_reg,
            "reg_change": args.reg_change,
            "train_iter": args.train_iter,
            "round_digits": args.round_digits,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "normalization": f"MinMaxScaler + 1e-6 (MIN_POSITIVE={meta['min_positive']}) (same as codes/Agriculture/mtr_ginn_agric_sym.py)",
            "target_transform": "log-transform (same as codes/Agriculture/mtr_ginn_agric_sym.py)",
        },
        "data_info": {
            "n_features": meta["n_features"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            "feature_columns": meta["feature_columns"],
            "target_range_train_original": [float(y_train_raw.min()), float(y_train_raw.max())],
            "target_mean_train_original": float(y_train_raw.mean()),
            "target_range_test_original": [float(y_test_raw.min()), float(y_test_raw.max())],
            "target_mean_test_original": float(y_test_raw.mean()),
            "target_range_train_log": [float(y_train_log.min()), float(y_train_log.max())],
            "target_mean_train_log": float(y_train_log.mean()),
            "target_range_test_log": [float(y_test_log.min()), float(y_test_log.max())],
            "target_mean_test_log": float(y_test_log.mean())
        },
        "metrics_original_scale": {
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "note": "Metrics computed after exp() transform back to original scale"
        },
        "metrics_log_scale": {
            "MSE": float(mse_log),
            "MAE": float(mae_log),
            "RMSE": float(rmse_log),
            "MAPE": float(mape_log),
            "note": "Metrics computed in log space (as model was trained)"
        },
        "recovered_equation": str(model.recovered_eq),
        "model_info": {
            "final_blocks": model.blk_count if hasattr(model, 'blk_count') else None,
            "note": "Equation is in log space. To get predictions in original scale, apply exp()"
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
