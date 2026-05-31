# <span style="font-size: 20px;">Concat and Correlate</span>

<span style="font-size: 14px;">Concatenation joins arrays along an axis, while correlation measures the linear relationship between variables. Combining these operations - concatenating multiple data sources and computing their pairwise correlations - is a common workflow in exploratory data analysis, multi-source data fusion, and feature selection.</span>

---

## <span style="font-size: 16px;">Array Concatenation</span>

### <span style="font-size: 14px;">np.concatenate()</span>

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.concatenate([a, b], axis=0)  # stack rows: shape (4, 2)
np.concatenate([a, b], axis=1)  # stack columns: shape (2, 4)
```

### <span style="font-size: 14px;">Convenience Functions</span>

```python
np.vstack([a, b])    # same as concatenate axis=0: vertical stack
np.hstack([a, b])    # same as concatenate axis=1: horizontal stack
np.row_stack([a, b]) # alias for vstack
np.column_stack([a, b])  # stacks 1D arrays as columns
```

### <span style="font-size: 14px;">Shape Requirements</span>

<span style="font-size: 14px;">Arrays must match in all dimensions except the concatenation axis. For `axis=0`, the number of columns must be equal. For `axis=1`, the number of rows must be equal.</span>

---

## <span style="font-size: 16px;">Pearson Correlation</span>

<span style="font-size: 14px;">The Pearson correlation coefficient between variables $x$ and $y$ is:</span>

$$r_{xy} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \sum_i (y_i - \bar{y})^2}}$$

<span style="font-size: 14px;">$r \in [-1, 1]$: 1 means perfect positive linear relationship, -1 means perfect negative, 0 means no linear relationship.</span>

### <span style="font-size: 14px;">np.corrcoef()</span>

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

np.corrcoef(x, y)
# [[1.   , 0.816],
#  [0.816, 1.   ]]
```

<span style="font-size: 14px;">`np.corrcoef()` returns a correlation matrix. For two variables, it is a $2 \times 2$ matrix with 1s on the diagonal.</span>

### <span style="font-size: 14px;">Correlation Matrix of a Data Matrix</span>

```python
data = np.random.randn(100, 5)  # 100 samples, 5 features
R = np.corrcoef(data, rowvar=False)  # 5x5 correlation matrix
```

<span style="font-size: 14px;">`rowvar=False` means columns are variables (rows are observations). This is the standard data science convention.</span>

---

## <span style="font-size: 16px;">Combined Workflow: Concat Then Correlate</span>

```python
# Two data sources with the same number of rows
source_a = np.random.randn(100, 3)  # 3 features
source_b = np.random.randn(100, 4)  # 4 features

# Concatenate into a single matrix
combined = np.concatenate([source_a, source_b], axis=0)

# This is wrong! We want column-wise concat for correlation
combined = np.concatenate([source_a, source_b[:, :3]], axis=0)

# Compute pairwise correlations
corr = np.corrcoef(source_a, source_b[:, :3], rowvar=False)
```

<span style="font-size: 14px;">For per-source and cross-source correlations:</span>

```python
R_a = np.corrcoef(source_a, rowvar=False)  # within source_a
R_b = np.corrcoef(source_b, rowvar=False)  # within source_b

# Cross-correlation
combined = np.hstack([source_a, source_b])
R_all = np.corrcoef(combined, rowvar=False)  # (7, 7) matrix
```

---

## <span style="font-size: 16px;">Stacking Correlation Matrices</span>

<span style="font-size: 14px;">To compare correlations across groups:</span>

```python
groups = [data_a, data_b, data_c]
corr_stack = np.array([np.corrcoef(g, rowvar=False) for g in groups])
# shape (3, n_features, n_features)
```

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Feature selection**: Remove highly correlated features ($|r| > 0.95$) to reduce multicollinearity</span>
* <span style="font-size: 14px;">**Portfolio analysis**: Correlation matrix of asset returns determines diversification benefit</span>
* <span style="font-size: 14px;">**Signal processing**: Cross-correlation detects time-lagged relationships between signals</span>
* <span style="font-size: 14px;">**Multi-source data**: Concatenate features from different sensors, then analyze their relationships</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**rowvar default**: `np.corrcoef` defaults to `rowvar=True` (rows are variables). For data matrices where rows are samples, pass `rowvar=False`.</span>
* <span style="font-size: 14px;">**Correlation is not causation**: A high correlation between features does not imply one causes the other.</span>
* <span style="font-size: 14px;">**Constant columns**: A column with zero variance produces NaN in the correlation matrix (division by zero in the formula).</span>
* <span style="font-size: 14px;">**Concatenation axis**: `axis=0` stacks rows (more observations); `axis=1` stacks columns (more features). Using the wrong axis produces a meaningless correlation matrix.</span>