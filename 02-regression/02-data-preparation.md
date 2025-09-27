# Data Preparation 

## 1. Download Data

How: Use `wget` command to download the dataset from a URL.

```bash
!wget https://example.com/data.csv -O data.csv
```

Why: To easily and quickly fetch raw data files to your working directory for analysis.

## 2. Inspect Data
How: Load data with pandas and look at the first and last few rows.

```python
import pandas as pd

df = pd.read_csv('data.csv')

print(df.head())  # Shows the first 5 rows
print(df.tail())  # Shows the last 5 rows
```

Why: To understand the structure and sample of your data, which helps spot obvious errors or missing info early.

## 3. Data Cleaning

### A. Standardize Column Names

How: Convert all column names to lowercase and replace spaces with underscores.

```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)
```

Why: Ensures consistent, easy-to-reference column names without spaces or case sensitivity issues.

### B. Clean String Columns

How: Find columns with string/object type, then convert entries to lowercase and replace spaces.

```python
# Find string columns
string_cols = df.dtypes == 'object'

# Loop through string columns and clean text
for col in df.columns[string_cols]:
    df[col] = df[col].str.lower().str.replace(' ', '_')

print(df[string_cols].head())
```
Why: Uniform text format avoids mismatches and improves downstream processing.

## Other Data Preparation Steps

### A. Handle Missing Values

When & Why to Handle Missing Values:

- **Dropping rows/columns:** Use when missing data is minimal or non-critical to avoid introducing bias.
- **Imputing with mean/median/mode:** Use when you want to fill missing values based on central tendency to preserve dataset size without complex assumptions.
- **Advanced imputation (KNN, regression, MICE):** Use when missingness is systematic and you want more accurate estimates by leveraging other feature relationships.

```python
# 1. Dropping rows with any missing value
data_dropped = data.dropna()

# 2. Imputing missing values with mean
imputer_mean = SimpleImputer(strategy='mean')
data_imputed_mean = pd.DataFrame(imputer_mean.fit_transform(data), columns=data.columns)

# 3. Imputing missing values with median (good for skewed data)
imputer_median = SimpleImputer(strategy='median')
data_imputed_median = pd.DataFrame(imputer_median.fit_transform(data), columns=data.columns)

# Print results
print("Original Data:\n", data)
print("\nDropped missing rows:\n", data_dropped)
print("\nMean Imputation:\n", data_imputed_mean)
print("\nMedian Imputation:\n", data_imputed_median)
```

### B. Convert Data Types

How: Change a column’s data type when needed.

```python
df['some_column'] = df['some_column'].astype('int')
```

Why: Correct data types ensure proper calculations and avoid unexpected errors.

### C. Feature Scaling

When and Why:

- **Normalization (Min-Max Scaling):** Use when you want to scale features to a fixed range [0,1], especially for algorithms sensitive to magnitude like neural networks.
- **Standardization (Z-score Scaling):** Use when data follows a roughly normal distribution and you want zero mean and unit variance, which benefits algorithms like SVM, logistic regression, and PCA.
- **Log Transformation:** Apply to highly skewed data (with a long tail) to reduce skewness and make the distribution more normal-like.

```python
# Sample skewed data with a long tail
data = np.array([1, 2, 3, 4, 5, 100, 200, 300])

# 1. Normalization: scales data between 0 and 1
scaler_norm = MinMaxScaler()
data_norm = scaler_norm.fit_transform(data.reshape(-1, 1))

# 2. Standardization: scales data to mean=0, std=1
scaler_std = StandardScaler()
data_std = scaler_std.fit_transform(data.reshape(-1, 1))

# 3. Log Transformation: reduces skewness of data
data_log = np.log1p(data)  # log1p handles zero values safely

# Plotting to visualize effect
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Original")
plt.hist(data, bins=10)

plt.subplot(1,4,2)
plt.title("Normalized")
plt.hist(data_norm, bins=10)

plt.subplot(1,4,3)
plt.title("Standardized")
plt.hist(data_std, bins=10)

plt.subplot(1,4,4)
plt.title("Log Transformed")
plt.hist(data_log, bins=10)

plt.show()
```

### D. Encode Categorical Variables

How: Convert categorical columns into dummy/one-hot encoded variables.

```python
df = pd.get_dummies(df, columns=['category_column'])
```

### E. Remove Duplicates

How: Drop duplicate rows from the dataset.

```python
df = pd.get_dummies(df, columns=['category_column'])
```

Why: Why: Duplicate data can bias results and inflate sample size.

---

## Local vs Global Data Preparation

### Local Preparation

How: Apply cleaning or transformation on specific columns or rows only.

```python
df['name'] = df['name'].str.lower()  # Lowercase just one column
```

Why: Use when only a small part of the dataset needs fixing.

### Global Preparation

How: Apply cleaning or transformation across the entire dataset or all columns of a certain type.

```python
df['name'] = df['name'].str.lower()  # Lowercase just one column
```
Why: Ensures consistent formatting and reduces code repetition.
