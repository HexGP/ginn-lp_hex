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
    parser = argparse.ArgumentParser(description='Train GINN-LP on ENB dataset (same normalization as codes/ENB/mtr_ginn_sym.py)')
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
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train/test split (default: 42)')
    
    args = parser.parse_args()
    
    # Determine separator
    if args.format == 'csv':
        sep = ','
    elif args.format == 'tsv':
        sep = '\t'
    else:
        raise ValueError('Invalid format. Use "csv" or "tsv"')
    
    print("=" * 60)
    print("GINN-LP Training on ENB Dataset")
    print("Normalization: MinMaxScaler + 1e-6 (same as codes/ENB/mtr_ginn_sym.py)")
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
    
    # ================= Split (same as codes/ENB/mtr_ginn_sym.py) =================
    print(f"\nSplitting data (test_size={args.test_size}, random_state={args.random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    # ================= Normalize Features for ginn-lp (higher minimum to avoid NaN) =================
    # Use higher minimum value (MIN_POSITIVE) instead of 1e-6 to ensure all values stay positive
    # during intermediate computations in log_activation
    MIN_POSITIVE = 1e-2  # Minimum positive value for log activation (same as ginn-lp_hex)
    
    print(f"\nNormalizing features with MinMaxScaler + MIN_POSITIVE (MIN_POSITIVE={MIN_POSITIVE})...")
    print(f"  Note: Using higher minimum than codes (1e-6) to prevent NaN in log_activation")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train) + MIN_POSITIVE
    X_test_scaled = scaler.transform(X_test) + MIN_POSITIVE
    
    # Ensure all values are strictly positive (safety check)
    X_train_scaled = np.maximum(X_train_scaled, MIN_POSITIVE)
    X_test_scaled = np.maximum(X_test_scaled, MIN_POSITIVE)
    
    print(f"Scaled feature ranges:")
    print(f"  Train: min={X_train_scaled.min():.6f}, max={X_train_scaled.max():.6f}")
    print(f"  Test:  min={X_test_scaled.min():.6f}, max={X_test_scaled.max():.6f}")
    
    # Targets are NOT normalized (same as codes/ENB)
    print(f"\nTargets NOT normalized (same as codes/ENB):")
    print(f"  Train: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
    print(f"  Test:  min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")
    print(f"  Note: Targets are already positive (ENB range: 6.01-43.10), so no transformation needed")
    
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
    
    model.fit(X_train_scaled, y_train)
    
    print(f"\nRecovered equation: {model.recovered_eq}")
    
    # ================= Evaluate on Test Set =================
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    
    # Predict on test set (using normalized features)
    y_pred = model.predict(X_test_scaled)
    
    # Flatten predictions if needed
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics on original scale (to compare with codes)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.4f}%")
    
    # ================= Save Results =================
    print("\n" + "=" * 60)
    print("Saving Results...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine Y1/Y2 from filename
    data_basename = os.path.splitext(os.path.basename(args.data))[0]
    if 'Heating' in data_basename or 'heating' in data_basename.lower():
        target_id = 'Y1'
        target_name = 'Heating_Load'
    elif 'Cooling' in data_basename or 'cooling' in data_basename.lower():
        target_id = 'Y2'
        target_name = 'Cooling_Load'
    else:
        target_id = 'Y?'
        target_name = data_basename
    
    # Generate filename with Y1/Y2 label
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include key hyperparameters in filename
    hyperparams_str = f"E{args.num_epochs}_B{args.start_ln_blocks}_G{args.growth_steps}"
    output_filename = f"ginnlp_ENB_ENB2012_{target_id}_{target_name}_{hyperparams_str}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Prepare results dictionary
    results = {
        "experiment_info": {
            "timestamp": timestamp,
            "dataset": "ENB",
            "target_id": target_id,
            "target_name": target_name,
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
            "normalization": f"MinMaxScaler + MIN_POSITIVE (MIN_POSITIVE={MIN_POSITIVE}) - higher than codes (1e-6) to prevent NaN in log_activation",
            "target_transform": "none (raw values, same as codes/ENB/mtr_ginn_sym.py)",
            "min_positive": float(MIN_POSITIVE),
            "note": "Only features use higher MIN_POSITIVE (1e-2 vs 1e-6). Targets unchanged (already positive). log_activation unchanged."
        },
        "data_info": {
            "n_features": X.shape[1],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "target_range_train": [float(y_train.min()), float(y_train.max())],
            "target_mean_train": float(y_train.mean()),
            "target_range_test": [float(y_test.min()), float(y_test.max())],
            "target_mean_test": float(y_test.mean())
        },
        "metrics": {
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape)
        },
        "recovered_equation": str(model.recovered_eq),
        "model_info": {
            "final_blocks": model.blk_count if hasattr(model, 'blk_count') else None
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
