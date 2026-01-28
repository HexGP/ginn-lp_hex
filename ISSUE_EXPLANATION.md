# Issue with Running ginn-lp on ENB Dataset

## Problem Statement

We are attempting to compare `ginn-lp` (single-target) with our MTR-GINN-LP (multi-target) implementation on the ENB dataset. However, we encountered a critical issue related to the activation function used in `ginn-lp`.

## Technical Issue

**ginn-lp uses `log_activation` (`tf.math.log(x)`), which requires strictly positive inputs.** 

If any value ≤ 0 enters the log function, it produces NaN (Not a Number), which propagates through the entire training process, causing:
- Loss values to become NaN
- Model predictions to become NaN
- Training to fail completely

## The Dilemma

### Our MTR-GINN-LP Normalization (codes)
For fair comparison, we want to use **identical preprocessing** as our MTR-GINN-LP implementation:

```python
# Features: MinMaxScaler + 1e-6
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) + 1e-6
X_test_scaled = scaler.transform(X_test) + 1e-6

# Targets: No normalization (raw values)
# ENB targets range: [6.01, 43.10] (all positive)
y_train = y_train  # No transformation
y_test = y_test   # No transformation
```

### The Problem
Even though:
1. **Features are positive** (MinMaxScaler + 1e-6 ensures values in [1e-6, 1+1e-6])
2. **Targets are positive** (ENB Heating/Cooling Load values range from 6.01 to 43.10)

**ginn-lp still produces NaN during training** because:
- The `log_activation` function is applied to **intermediate values** in the network during forward/backward passes
- These intermediate values can become ≤ 0 due to:
  - Weight initialization
  - Gradient updates
  - Numerical precision issues
  - Even with positive inputs, intermediate computations can produce non-positive values

### Current Workaround
We implemented a workaround similar to `ginn-lp_hex`:

```python
# Shift targets to be strictly positive
MIN_POSITIVE = 1e-2
shift_amount = max(0, MIN_POSITIVE - y_train.min() + 1e-6)
y_train_shifted = y_train + shift_amount
y_test_shifted = y_test + shift_amount

# Train on shifted targets
model.fit(X_train_scaled, y_train_shifted)

# Reverse shift for metrics (to compare with codes)
y_pred = y_pred_shifted - shift_amount
```

**However, this breaks the "identical preprocessing" requirement** because:
- We're modifying the targets before training
- The shift affects the model's learned representation
- Metrics are computed after reversing the shift, but the model was trained on shifted data

## Question

**What should we do?**

1. **Option A**: Use the shift workaround and document it as a necessary modification for ginn-lp compatibility, acknowledging that preprocessing is not identical?

2. **Option B**: Modify ginn-lp's `log_activation` to handle non-positive values (e.g., `log(1 + abs(x))` or `sign(x) * log(1 + abs(x))`), which would allow identical preprocessing but changes ginn-lp's behavior?

3. **Option C**: Use a different normalization scheme that ensures all intermediate values remain positive (e.g., different scaling, different activation function, or different architecture)?

4. **Option D**: Accept that ginn-lp cannot be directly compared with our MTR-GINN-LP due to this fundamental architectural difference, and focus comparison on other aspects?

## Additional Context

- The ENB dataset targets are naturally positive (energy loads cannot be negative)
- Our MTR-GINN-LP uses Laurent blocks with `relu(x) + 1e-4`, which handles this naturally
- ginn-lp's log activation is a fundamental part of its architecture for learning multiplicative relationships
- The `ginn-lp_hex` repository uses similar shift-based workarounds for this issue

## Recommendation Needed

We need guidance on which approach to take to ensure a fair and scientifically valid comparison between the two methods.
