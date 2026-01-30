# MAMBA Tensor Shape Explanation

## Your Raw Data
```
Shape: (1565, 58)
- 1565 rows (timesteps/days)
- 58 columns = Date + 57 other columns (likely 55 commodities + 2 other features)
```

## What Happens During Processing

### Step 1: Select Commodity Columns
```python
data = prices_df[commodity_columns].dropna()
```
**Result**: `(1565, 55)` - Only commodity price columns, no Date column

### Step 2: Create Sliding Window Sequences
We need to transform the data into sequences for time series prediction.

**Concept**: Use last 21 days to predict day 22

```
Original data (1565 timesteps):
Day 1:  [Commodity_0=100, Commodity_1=50, ..., Commodity_54=75]
Day 2:  [Commodity_0=101, Commodity_1=51, ..., Commodity_54=76]
Day 3:  [Commodity_0=102, Commodity_1=52, ..., Commodity_54=77]
...
Day 21: [Commodity_0=120, Commodity_1=70, ..., Commodity_54=95]
Day 22: [Commodity_0=121, Commodity_1=71, ..., Commodity_54=96] <- TARGET
```

**Sliding Window Creates Training Samples**:

```
Sample 0:
  Input (X):  Days [1-21]   -> shape (21, 55)
  Target (y): Day 22        -> shape (55,)

Sample 1:
  Input (X):  Days [2-22]   -> shape (21, 55)
  Target (y): Day 23        -> shape (55,)

Sample 2:
  Input (X):  Days [3-23]   -> shape (21, 55)
  Target (y): Day 24        -> shape (55,)

...

Sample 1544:
  Input (X):  Days [1545-1565]  -> shape (21, 55)
  Target (y): Day 1566           -> shape (55,) [doesn't exist, so we stop at 1544]
```

### Step 3: Final Tensor Shapes

```python
X_tensor: (1544, 21, 55)
# ├─ 1544 samples (1565 - 21 = 1544)
# ├─ 21 timesteps (sequence_length)
# └─ 55 commodities (all at each timestep)

y_tensor: (1544, 55)
# ├─ 1544 samples (one prediction per input sequence)
# └─ 55 commodities (predicting all simultaneously)
```

## Why We "Lose" Data Points

**Original**: 1565 timesteps
**Training samples**: 1544 samples

**Lost**: 21 timesteps

**Why?** 
- Sample 0 needs days 1-21 to predict day 22
- We can't create a sample for day 1 (no history)
- We can't create a sample for day 2 (only 1 day of history, need 21)
- ...
- First valid sample uses days 1-21 to predict day 22
- Last valid sample uses days 1545-1565 to predict... day 1566 (doesn't exist)

So we get: `1565 - 21 = 1544` training samples

## Multivariate Advantage

Each training sample contains:
```
Shape: (21, 55)
      ↓    ↓
   21 days of history for ALL 55 commodities

This means at each timestep, the model sees:
  Day 1:  all 55 commodity prices
  Day 2:  all 55 commodity prices
  ...
  Day 21: all 55 commodity prices

Then predicts: Day 22's prices for ALL 55 commodities
```

## Why This is Better Than Univariate

**Univariate (old approach)**:
- 55 separate models
- Each sees only 1 commodity: (21, 1)
- Ignores relationships between commodities

**Multivariate (current approach)**:
- 1 unified model
- Sees all 55 commodities: (21, 55)
- Learns cross-commodity patterns (e.g., "when Isopropyl Alcohol rises, others follow")
- Leverages your 1,840 causality relationships automatically

## Summary

```
Raw data:           (1565, 58)     - timesteps × (date + features)
                           ↓
Select commodities: (1565, 55)     - timesteps × commodities
                           ↓
Create sequences:   (1544, 21, 55) - samples × sequence_length × commodities
                           ↓
Model learns from:  1544 training examples
Each example:       21 days of history for all 55 commodities
Predicts:           Next day's prices for all 55 commodities
```

**You ARE using all your data!** Just transformed for time series learning.
