import pandas as pd
import numpy as np
import argparse
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ginnlp.ginnlp import GINNLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GINN-LP on Agriculture dataset (same normalization as codes/Agriculture/mtr_ginn_agric_sym.py)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--format', type=str, default='csv', help='Format of dataset file (csv or tsv)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for results')
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
    
    # Determine separator
    if args.format == 'csv':
        sep = ','
    elif args.format == 'tsv':
        sep = '\t'
    else:
        raise ValueError('Invalid format. Use "csv" or "tsv"')
    
    print("=" * 60)
    print("GINN-LP Training on Agriculture Dataset")
    print("Normalization: MinMaxScaler + 1e-6, log-transform targets")
    print("(same as codes/Agriculture/mtr_ginn_agric_sym.py)")
    print("=" * 60)
    print(f"Data file: {args.data}")
    print(f"Format: {args.format}")
    
    # Load data
    df = pd.read_csv(args.data, sep=sep)
    print(f"\nLoaded dataset shape: {df.shape}")
    
    # Extract features and target
    X = df.drop('target', axis=1).values.astype(float)
    y = df['target'].values.astype(float)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
    
    # ================= Split (same as codes/Agriculture/mtr_ginn_agric_sym.py) =================
    print(f"\nSplitting data (test_size={args.test_size}, random_state={args.random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    # ================= Normalize Features (same as codes/Agriculture) =================
    print("\nNormalizing features with MinMaxScaler + 1e-6 (same as codes)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train) + 1e-6
    X_test_scaled = scaler.transform(X_test) + 1e-6
    
    print(f"Scaled feature ranges:")
    print(f"  Train: min={X_train_scaled.min():.6f}, max={X_train_scaled.max():.6f}")
    print(f"  Test:  min={X_test_scaled.min():.6f}, max={X_test_scaled.max():.6f}")
    
    # ================= Log-transform Targets (same as codes/Agriculture) =================
    print(f"\nApplying log-transform to targets (Agriculture dataset - same as codes)...")
    print(f"  Before log-transform:")
    print(f"    Train: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
    print(f"    Test:  min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")
    
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    
    print(f"  After log-transform:")
    print(f"    Train: min={y_train_log.min():.2f}, max={y_train_log.max():.2f}, mean={y_train_log.mean():.2f}")
    print(f"    Test:  min={y_test_log.min():.2f}, max={y_test_log.max():.2f}, mean={y_test_log.mean():.2f}")
    
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
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
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
    data_basename = os.path.splitext(os.path.basename(args.data))[0]
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
            "data_file": args.data,
            "dataset_name": data_basename,
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
            "normalization": "MinMaxScaler + 1e-6 (same as codes/Agriculture/mtr_ginn_agric_sym.py)",
            "target_transform": "log-transform (same as codes)"
        },
        "data_info": {
            "n_features": X.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "target_range_train_original": [float(y_train.min()), float(y_train.max())],
            "target_mean_train_original": float(y_train.mean()),
            "target_range_test_original": [float(y_test.min()), float(y_test.max())],
            "target_mean_test_original": float(y_test.mean()),
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
