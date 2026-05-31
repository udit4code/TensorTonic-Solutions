# <span style="font-size: 20px;">Winsorize</span>

<span style="font-size: 14px;">Winsorization clips extreme values in each column to percentile-based bounds. Unlike trimming (which removes outliers entirely), winsorization replaces extreme values with the boundary percentile value, preserving the array shape. This technique is widely used in financial data analysis, robust statistics, and data preprocessing for machine learning where outliers can distort model training.</span>

---

## <span style="font-size: 16px;">The Algorithm</span>

<span style="font-size: 14px;">For a given lower percentile $p_{\text{lo}}$ and upper percentile $p_{\text{hi}}$:</span>

1. <span style="font-size: 14px;">Compute $q_{\text{lo}} = \text{percentile}(x, p_{\text{lo}})$ and $q_{\text{hi}} = \text{percentile}(x, p_{\text{hi}})$ per column</span>
2. <span style="font-size: 14px;">Replace values below $q_{\text{lo}}$ with $q_{\text{lo}}$</span>
3. <span style="font-size: 14px;">Replace values above $q_{\text{hi}}$ with $q_{\text{hi}}$</span>

```python
def winsorize(arr, lower=5, upper=95):
    lo = np.percentile(arr, lower, axis=0)
    hi = np.percentile(arr, upper, axis=0)
    return np.clip(arr, lo, hi)
```

---

## <span style="font-size: 16px;">np.percentile() and np.quantile()</span>

```python
data = np.array([1, 2, 3, 4, 5, 100])

np.percentile(data, 5)    # 5th percentile (close to min)
np.percentile(data, 95)   # 95th percentile
np.quantile(data, 0.05)   # same as percentile(data, 5)
```

<span style="font-size: 14px;">`percentile` takes values in $[0, 100]$; `quantile` takes values in $[0, 1]$. They are otherwise identical.</span>

### <span style="font-size: 14px;">Per-Column Percentiles</span>

```python
arr = np.random.randn(1000, 5)
lo = np.percentile(arr, 5, axis=0)   # shape (5,) - one value per column
hi = np.percentile(arr, 95, axis=0)  # shape (5,)
```

---

## <span style="font-size: 16px;">np.clip() for Bounding</span>

```python
result = np.clip(arr, lo, hi)
```

<span style="font-size: 14px;">When `lo` and `hi` are arrays of shape $(n,)$ (one per column), broadcasting applies each column's bounds to that column.</span>

---

## <span style="font-size: 16px;">Winsorization vs. Trimming</span>

| Approach | Outlier handling | Shape preserved | Use case |
|----------|-----------------|----------------|----------|
| Winsorize | Replace with bound | Yes | Regression, robust statistics |
| Trim | Remove entirely | No (fewer rows) | Trimmed mean computation |

<span style="font-size: 14px;">Winsorization is preferred when you need a fixed-size array (e.g., for matrix operations or batch training). Trimming is preferred when extreme values should not influence any computation.</span>

---

## <span style="font-size: 16px;">Rounding Modes</span>

<span style="font-size: 14px;">NumPy provides multiple rounding functions that can be combined with winsorization:</span>

```python
np.floor(arr)      # round toward negative infinity
np.ceil(arr)       # round toward positive infinity
np.round(arr, 2)   # round to 2 decimal places
np.trunc(arr)      # round toward zero
```

<span style="font-size: 14px;">After winsorizing, rounding to a fixed precision can be useful for discretization.</span>

---

## <span style="font-size: 16px;">Padding with np.pad()</span>

<span style="font-size: 14px;">After processing, you may need to add a border of zeros around the result:</span>

```python
np.pad(arr, pad_width=1, mode='constant', constant_values=0)
# Adds 1 row/column of zeros on all sides
```

<span style="font-size: 14px;">Padding is used in image processing (convolution borders), signal processing (FFT zero-padding), and matrix construction.</span>

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Financial data**: Stock returns often have extreme outliers from data errors. Winsorizing at 1% and 99% removes these without deleting entire trading days.</span>
* <span style="font-size: 14px;">**Robust regression**: Winsorizing features reduces the influence of outliers on coefficient estimates.</span>
* <span style="font-size: 14px;">**Neural network training**: Clipping input features prevents extreme gradients from destabilizing training.</span>
* <span style="font-size: 14px;">**Percentile-based binning**: Winsorization is the first step before discretizing features into equal-frequency bins.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Percentile vs. quantile**: `percentile` uses $[0, 100]$; `quantile` uses $[0, 1]$. Passing 5 to quantile means the 500th percentile, not the 5th.</span>
* <span style="font-size: 14px;">**Broadcasting in clip**: When `lo` and `hi` have shape $(n,)$, they broadcast correctly for per-column clipping. If they have the wrong shape, all columns get the same bounds.</span>
* <span style="font-size: 14px;">**Winsorizing before splitting**: In ML, compute percentile bounds from training data only. Winsorizing the entire dataset before train/test split causes data leakage.</span>
* <span style="font-size: 14px;">**Symmetric vs. asymmetric bounds**: Winsorizing at (5, 95) clips both tails equally. For skewed distributions, asymmetric bounds (e.g., 1, 99) may be more appropriate.</span>