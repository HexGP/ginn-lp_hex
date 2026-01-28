# Strategy: Running ginn-lp on ENB and Agriculture Datasets

## Overview
Compare `ginn-lp` (single-target) with `codes` (multi-target) on ENB and Agriculture datasets using **identical** data preprocessing.

## Key Differences to Handle

### 1. **Single vs Multi-Target**
- `ginn-lp`: Single target only → Run **twice** (once for Y1, once for Y2)
- `codes`: Multi-target (Y1 and Y2 simultaneously)

### 2. **Data Format**
- `ginn-lp` expects: `X` transposed (features as rows, samples as columns) - **handled internally**
- `codes` uses: Standard format (samples as rows, features as columns)

### 3. **Negative Values**
- `ginn-lp`: Cannot handle negatives (uses log activation)
- **Solution**: Shift data to be positive before feeding to ginn-lp
  - For features: Already positive after MinMaxScaler + 1e-6 ✓
  - For targets: May need shifting if negative after log-transform (Agriculture)

### 4. **Data Splitting**
- `codes`: Uses `train_test_split(test_size=0.2, random_state=42)` → **80/20 split**
- `ginn-lp`: Uses internal validation split (0.2) on training data
- **Strategy**: Use same 80/20 split as `codes`, then ginn-lp will further split the 80% into train/val

### 5. **Normalization**
- **ENB**: MinMaxScaler on features + 1e-6, targets unchanged
- **Agriculture**: MinMaxScaler on features + 1e-6, targets log-transformed

## Implementation Status

### ✅ Step 1: Data Preparation Scripts
- Created `run_ENB/scripts/prepare_data.py` - Prepares ENB data (Heating/Cooling split)
- Created `run_AGRIC/scripts/prepare_data.py` - Prepares Agriculture data (Sustainability/ConsumerTrend split)
- Both scripts match `codes` preprocessing exactly

### ✅ Step 2: Training Scripts
- Created `run_enb.py` - For ENB dataset (no log-transform, random_state=42)
- Created `run_agric.py` - For Agriculture dataset (log-transform, random_state=100)
- Both scripts handle normalization and save results as JSON

### ⏳ Step 3: Evaluation Scripts
- Metrics are automatically computed and saved in JSON files
- Comparison scripts can be created to extract and compare results

## How to Run

### Prerequisites
```bash
conda activate ginn
cd /raid/hussein/project/ginn-lp
```

### ENB Dataset

#### 1. Prepare Data (if not already done)
```bash
cd run_ENB/scripts
python prepare_data.py
```

This creates:
- `run_ENB/data/Heating_train.csv` and `Heating_test.csv`
- `run_ENB/data/Cooling_train.csv` and `Cooling_test.csv`

#### 2. Train on Heating Load (Y1)
```bash
python run_enb.py \
    --data run_ENB/data/ENB2012_Heating_Load.csv \
    --format csv \
    --num_epochs 500 \
    --start_ln_blocks 2 \
    --growth_steps 3 \
    --output_dir run_ENB/outputs
```

#### 3. Train on Cooling Load (Y2)
```bash
python run_enb.py \
    --data run_ENB/data/ENB2012_Cooling_Load.csv \
    --format csv \
    --num_epochs 500 \
    --start_ln_blocks 2 \
    --growth_steps 3 \
    --output_dir run_ENB/outputs
```

**Output Filename Format:**
```
ginnlp_ENB_ENB2012_Heating_Load_E500_B2_G3_20260128_142530.json
ginnlp_ENB_ENB2012_Cooling_Load_E500_B2_G3_20260128_142530.json
```

Format: `ginnlp_{DATASET}_{data_basename}_E{epochs}_B{blocks}_G{growth}_{timestamp}.json`

### Agriculture Dataset

#### 1. Prepare Data (if not already done)
```bash
cd run_AGRIC/scripts
python prepare_data.py
```

This creates:
- `run_AGRIC/data/Sustainability_train.csv` and `Sustainability_test.csv`
- `run_AGRIC/data/ConsumerTrend_train.csv` and `ConsumerTrend_test.csv`

#### 2. Train on Sustainability Score (Y1)
```bash
python run_agric.py \
    --data run_AGRIC/data/Sustainability_train.csv \
    --format csv \
    --num_epochs 500 \
    --start_ln_blocks 2 \
    --growth_steps 3 \
    --output_dir run_AGRIC/outputs
```

#### 3. Train on Consumer Trend Index (Y2)
```bash
python run_agric.py \
    --data run_AGRIC/data/ConsumerTrend_train.csv \
    --format csv \
    --num_epochs 500 \
    --start_ln_blocks 2 \
    --growth_steps 3 \
    --output_dir run_AGRIC/outputs
```

**Output Filename Format:**
```
ginnlp_AGRIC_Sustainability_train_E500_B2_G3_20260128_142530.json
ginnlp_AGRIC_ConsumerTrend_train_E500_B2_G3_20260128_142530.json
```

## Extracting Metrics from JSON Files

### Python Script to Extract Metrics

Create `extract_metrics.py`:

```python
import json
import glob
import os

def extract_metrics_from_json(json_file):
    """Extract metrics from a ginn-lp JSON output file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    dataset_name = data['experiment_info']['dataset_name']
    hyperparams = f"E{data['experiment_info']['num_epochs']}_B{data['experiment_info']['start_ln_blocks']}_G{data['experiment_info']['growth_steps']}"
    
    return {
        'dataset': dataset_name,
        'hyperparams': hyperparams,
        'MSE': metrics.get('MSE'),
        'MAE': metrics.get('MAE'),
        'RMSE': metrics.get('RMSE'),
        'MAPE': metrics.get('MAPE')
    }

# Example usage
output_dir = 'run_ENB/outputs'
json_files = glob.glob(os.path.join(output_dir, 'ginnlp_*.json'))

for json_file in json_files:
    print(f"\n{os.path.basename(json_file)}")
    metrics = extract_metrics_from_json(json_file)
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.4f}%")
```

## Comparison Table Template

### ENB Dataset Comparison

| Model | Target | Config | MSE | MAE | RMSE | MAPE |
|-------|--------|--------|-----|-----|------|------|
| **codes (multi-target)** | Heating (Y1) | 2 init, 8 max | 0.2598 | 0.4039 | 0.5097 | 1.95% |
| **codes (multi-target)** | Cooling (Y2) | 2 init, 8 max | 2.6200 | 1.0217 | 1.6186 | 3.54% |
| **ginn-lp (single-target)** | Heating (Y1) | E500_B2_G3 | [from JSON] | [from JSON] | [from JSON] | [from JSON] |
| **ginn-lp (single-target)** | Cooling (Y2) | E500_B2_G3 | [from JSON] | [from JSON] | [from JSON] | [from JSON] |

### Agriculture Dataset Comparison

| Model | Target | Config | MSE | MAE | RMSE | MAPE |
|-------|--------|--------|-----|-----|------|------|
| **codes (multi-target)** | Sustainability (Y1) | 3 init, 4 max | [from codes JSON] | [from codes JSON] | [from codes JSON] | [from codes JSON] |
| **codes (multi-target)** | ConsumerTrend (Y2) | 3 init, 4 max | [from codes JSON] | [from codes JSON] | [from codes JSON] | [from codes JSON] |
| **ginn-lp (single-target)** | Sustainability (Y1) | E500_B2_G3 | [from JSON] | [from JSON] | [from JSON] | [from JSON] |
| **ginn-lp (single-target)** | ConsumerTrend (Y2) | E500_B2_G3 | [from JSON] | [from JSON] | [from JSON] | [from JSON] |

**Note:** For Agriculture, `ginn-lp` metrics are in `metrics_original_scale` (after exp transform from log space).

## How to Build Comparison Table

### Method 1: Using the Extraction Script (Recommended)

Use the `extract_comparison.py` script to automatically extract metrics and create comparison tables:

#### For ENB Dataset:
```bash
python extract_comparison.py \
    --ginnlp_dir run_ENB/outputs \
    --codes_file ../codes/ENB/output/symbolic/mtr_ginn_2initial_8max_blocks_20250916.json \
    --dataset ENB
```

#### For Agriculture Dataset:
```bash
python extract_comparison.py \
    --ginnlp_dir run_AGRIC/outputs \
    --codes_file ../codes/Agriculture/output/findings_0.1/symbolic/mtr_ginn_agric_sym_*.json \
    --dataset Agriculture
```

The script will:
1. Load all ginn-lp JSON files from the specified directory
2. Load codes JSON file (if provided)
3. Extract metrics (MSE, MAE, RMSE, MAPE)
4. Create a markdown comparison table
5. Print detailed results

### Method 2: Manual Extraction

#### Step 1: Extract codes Results

For ENB, check:
- `codes/ENB/output/symbolic/mtr_ginn_2initial_8max_blocks_*.json`

For Agriculture, check:
- `codes/Agriculture/output/findings_0.1/symbolic/mtr_ginn_agric_sym_*.json`

Extract metrics from `neural_network_metrics` section:
```python
import json
with open('codes_json_file.json') as f:
    data = json.load(f)
    metrics = data['neural_network_metrics']
    # metrics['Heating Load (Y1)']['MSE'], etc.
```

#### Step 2: Extract ginn-lp Results

Run the extraction script or manually read JSON files:
- `run_ENB/outputs/ginnlp_ENB_*.json`
- `run_AGRIC/outputs/ginnlp_AGRIC_*.json`

Extract metrics from `metrics` section (or `metrics_original_scale` for Agriculture):
```python
import json
with open('ginnlp_json_file.json') as f:
    data = json.load(f)
    # For Agriculture, use data['metrics_original_scale']
    # For ENB, use data['metrics']
    metrics = data.get('metrics_original_scale', data.get('metrics', {}))
```

#### Step 3: Create Comparison Table

Use the template above and fill in values from both sources, or use the extraction script output.

### Step 4: Analysis

Compare:
1. **Neural Network Performance**: How well does each model predict?
2. **Single vs Multi-target**: Does training separately (ginn-lp) vs jointly (codes) affect performance?
3. **Equation Complexity**: Compare `recovered_equation` from both models
4. **Extraction Fidelity**: For codes, compare `neural_network_metrics` vs `symbolic_model_metrics`

## Folder Structure
```
ginn-lp/
├── run_enb.py           # Training script for ENB dataset
├── run_agric.py         # Training script for Agriculture dataset
├── run.py                # General training script (backup)
├── run_ENB/
│   ├── data/            # Preprocessed data (train/test splits for Y1, Y2)
│   │   ├── ENB2012_Heating_Load.csv
│   │   ├── ENB2012_Cooling_Load.csv
│   │   ├── Heating_train.csv
│   │   ├── Heating_test.csv
│   │   ├── Cooling_train.csv
│   │   └── Cooling_test.csv
│   ├── outputs/         # Models, equations, metrics JSON
│   └── scripts/         # Data prep scripts
├── run_AGRIC/
│   ├── data/            # Preprocessed data (train/test splits for Y1, Y2)
│   │   ├── Sustainability_train.csv
│   │   ├── Sustainability_test.csv
│   │   ├── ConsumerTrend_train.csv
│   │   └── ConsumerTrend_test.csv
│   ├── outputs/         # Models, equations, metrics JSON
│   └── scripts/         # Data prep scripts
└── STRATEGY.md          # This file
```

## Key Notes

1. **Same Preprocessing**: Both `codes` and `ginn-lp` use identical normalization (MinMaxScaler + 1e-6)
2. **Same Test Set**: Both use `test_size=0.2, random_state=42` for ENB, `random_state=100` for Agriculture
3. **Different Training**: 
   - `codes`: Multi-target (Y1 and Y2 together)
   - `ginn-lp`: Single-target (Y1 and Y2 separately)
4. **Metrics Location**:
   - `codes`: `neural_network_metrics` in JSON
   - `ginn-lp`: `metrics` in JSON (or `metrics_original_scale` for Agriculture)

## Next Steps

1. ✅ Create folder structure
2. ✅ Create data preparation scripts
3. ✅ Create training scripts
4. ⏳ Run experiments with matching hyperparameters
5. ⏳ Extract and compare results
6. ⏳ Document findings
