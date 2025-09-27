# Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) helps you understand the dataset's structure, patterns, and potential issues before modeling.

## 1. Inspect Columns

Purpose: Inspect the data types, unique values, and structure.

```python
for col in df.columns:
    print(f"{col}:")
    print(f"  Unique values: {df[col].unique()[:10]}")  # Show first 10
    print(f"  Number of unique values: {df[col].nunique()}")
    print()
```

Helps spot categorical variables, errors, or IDs with too many unique values.

## 2. Target Variable Distribution

Purpose: Understand the shape of the target variable, especially for regression.

```python
df['target'].hist(bins=50)
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.show()
```

## 3. Data Transformations

### Logarithmic Transformation

```python
df['target_log'] = np.log1p(df['target'])  # log(1 + x) handles zeros
```

Use when:
- Target or feature is right-skewed with a long tail
- You want to reduce the impact of outliers
- Data must be positive or non-zero

### Yeo-Johnson Transformation (via PowerTransformer)

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
df['feature_transformed'] = pt.fit_transform(df[['feature']])
```

Use when:
- Data is not normally distributed
- Contains zero or negative values
- You want to normalize the data shape before linear models
- Works with both positive and negative values — unlike log

### Normalization (Min-Max Scaling)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['feature']] = scaler.fit_transform(df[['feature']])
```

Use when:
- Features are on different scales
- You need values in a [0, 1] range (e.g., neural networks)
- Sensitive to outliers (since it scales based on min and max)

### Standardization (Z-score Scaling)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['feature']] = scaler.fit_transform(df[['feature']])
```

Use when:
- You want to center features (mean = 0) and standardize variance
- Many ML algorithms assume data is standard normal, like:
    - Linear regression
    - Logistic regression
    - SVMs
    - PCA
- Not sensitive to outliers as much as MinMax, but still affected

### Summary
| Technique  | Use When |
| ------------- |:-------------:|
| Log Transform      | Data is highly skewed, all values > 0     |
| Yeo-Johnson      | Data is skewed and includes 0 or negative values     |
| Normalization      | Input to neural nets or distance-based models (KNN), sensitive to scale     |
| Standardization      | For most ML models, especially linear models and SVMs     |

## 4. Missing Values

Purpose: Helps determine whether to drop, impute, or flag missing values.

```python
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
```

## 5. Data Types & Summary Stats

Purpose: Identify numerical vs. categorical features and spot issues like outliers or bad encoding.

```python
df.info()        # Data types, non-null counts
df.describe()    # Mean, std, min, max, quartiles
```

## 6. Correlation Matrix

Purpose: Helps identify multicollinearity or strong relationships between features.

```python
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
```

## 7. Categorical Feature Analysis

Purpose: Check class imbalance or rare categories.

```python
df['cat_feature'].value_counts()
sns.countplot(x='cat_feature', data=df)
```

## 8. Outlier Detection

Purpose: Useful before applying transformations or scaling.

### Box Plot

```python
sns.boxplot(x=df['feature'])
```

### IQR Method

```python
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['feature'] < Q1 - 1.5 * IQR) | (df['feature'] > Q3 + 1.5 * IQR)]
```

## 9. Pairwise Feature Relationships (Small Datasets)

Purpose: Helps find linear or nonlinear relationships between variables.

```python
sns.pairplot(df[['feature1', 'feature2', 'target']], diag_kind='kde')
```

## EDA Best Practices
- Start with `.info()` and `.describe()`
- Always visualize:
    - Distributions
    - Missing values
    - Correlations
- Choose transformations based on data shape and model needs
- Clean data before modeling:
    - Handle missing values
    - Encode categories
    - Scale numerical features appropriately