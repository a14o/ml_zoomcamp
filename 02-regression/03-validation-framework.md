# Setting Up the Validation Framework

## Purpose

To evaluate model performance reliably by splitting the dataset into separate sets for training, validation, and testing. This helps prevent overfitting, enables hyperparameter tuning, and ensures generalization.

## 1. Data Splitting (Train/Validation/Test)

Example: 60/20/20 Split

Why: Divide the dataset into:

- Train set: Used to train the model.
- Validation set: Used to tune model parameters.
- Test set: Used only once to assess final model performance.

Tip: Always ensure the sum of splits equals `n` to avoid losing or duplicating records due to rounding.

```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
```

## 2. Create Subsets

### No Shuffle

When: Use this for time series or when data order is important.

Warning: Without shuffling, patterns in ordered data can lead to model bias (e.g., all new data in test set).

```python
df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train + n_val]
df_test = df.iloc[n_train + n_val:]
```

### Shuffle Records

Why: Random shuffling avoids ordering bias, making the dataset more representative.

Seed: Ensures reproducibility of results.

```python
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]
```

## 3. Reset Index

Why: Resets row numbering after slicing/shuffling. Prevents index misalignment during future operations.

```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
```

## 4. Extract and Transform Target Variable (`y`)

Why `log1p`: Used when the target is highly skewed (e.g., long tail prices).

What it does: Applies `log(1 + x)` to compress large values, improving model learning.

```python
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
```

## 5. Remove Target from Features

Why: Prevents data leakage. The target variable should never be part of the input features during training.

```python
del df_train["msrp"]
del df_val["msrp"]
del df_test["msrp"]
```

---

## Best Practices for Validation Framework

1. Use Stratified Splits for classification problems to maintain class balance.
2. Cross-Validation (e.g., K-Fold) for small datasets to reduce variance in evaluation.
3. Time Series Split for temporal data — never randomly shuffle time-based data.
4. Hyperparameter tuning only on validation set — test set must remain untouched until final evaluation.
5. Remove any form of data leakage between sets (e.g., features derived from the target).

### Scikit-learn Example (Tabular Regression)

Pros:

- Simple API: Easy to use (train_test_split, cross_val_score, etc.)
- Rich tools: Pipelines, grid search, metrics.
- Integration: Works well with NumPy, pandas.

Cons:

- Less control for large-scale or custom pipelines.
- Limited support for non-tabular data (e.g., images, text) compared to `PyTorch/TensorFlow`.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("your_data.csv")

# Log-transform target
df['msrp'] = np.log1p(df['msrp'])

# Split data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Separate features and target
y_train = df_train.pop("msrp").values
y_val = df_val.pop("msrp").values
y_test = df_test.pop("msrp").values

# Train a simple model
model = Ridge(alpha=1.0)
model.fit(df_train, y_train)

# Predict and evaluate
y_pred = model.predict(df_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("Validation RMSE:", rmse)
```

### PyTorch Example (Tabular Regression)

```python
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Load and preprocess
df = pd.read_csv("your_data.csv")
df['msrp'] = np.log1p(df['msrp'])

# Split
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df_train = df[:n_train]
df_val = df[n_train:n_train + n_val]
df_test = df[n_train + n_val:]

y_train = df_train.pop("msrp").values
y_val = df_val.pop("msrp").values
y_test = df_test.pop("msrp").values

# Convert to tensors
X_train = torch.tensor(df_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor(df_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Dataset and DataLoader
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Simple model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = torch.sqrt(loss_fn(val_pred, y_val))
        print(f"Epoch {epoch}, Val RMSE: {val_loss.item():.4f}")
```

### TensorFlow / Keras Example (Tabular Regression)

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load and preprocess
df = pd.read_csv("your_data.csv")
df['msrp'] = np.log1p(df['msrp'])

# Split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

y_train = df_train.pop("msrp").values
y_val = df_val.pop("msrp").values
y_test = df_test.pop("msrp").values

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(df_train.shape[1],)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train
model.fit(df_train, y_train, validation_data=(df_val, y_val), epochs=10, batch_size=64)

# Evaluate on test
test_loss, test_rmse = model.evaluate(df_test, y_test)
print("Test RMSE:", test_rmse)
```

### Summary

| **Framework**    | **Best For**                          | **Pros**                                           | **Cons**                                       |
| ---------------- | ------------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| **Scikit-learn** | Classic ML, tabular data              | Simple, fast prototyping, good metrics/utilities   | Not suitable for GPUs, deep learning           |
| **PyTorch**      | Research, deep learning, full control | Flexible, dynamic graphs, strong community         | More code-heavy than Keras                     |
| **TensorFlow**   | Production, scalability               | Scalable, many high-level APIs (Keras), deployment | Can feel more abstract or verbose at low level |
