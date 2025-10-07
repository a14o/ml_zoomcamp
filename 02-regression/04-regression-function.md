# Linear Regression

Regression is a **function** $g(x)$ that maps input features $x$ to a **continuous output** $y$.

## Regression Function

### General form:

$$
g(x) = \mathbf{w}^\top \mathbf{x} + b
$$

- $\mathbf{x}$: Feature vector (input)
- $\mathbf{w}$: Weight vector (learned from data)
- $b$: Bias term (intercept)

This function produces predictions:

$$
g(x) \approx y
$$

## 1. Feature Matrix ($\mathbf{X}$):

- Contains all the input features for each data point.
- Dimensions: $m \times n$ (m = number of samples, n = number of features)

```python
X = [
  [Mileage],
  [Mileage],
  ...
]
```

## 2. Target Vector ($\mathbf{y}$):

- Contains the true output values for each example.

```python
y = [Price_1, Price_2, ..., Price_m]
```

## 3. Weights ($\mathbf{w}$):

- Each feature gets a weight $w_i$, which controls its influence on the prediction.

$$
g(x)=w_1​x_1​+w_2​x_2​+⋯+w_n​x_n​+b
$$

## 4. Bias Term ($w_0$ or $b$):

- Represents the baseline prediction when all features are 0.
- It's like the starting value or absolute base price.
