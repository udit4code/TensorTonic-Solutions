# <span style="font-size: 20px;">Normalized Difference</span>

<span style="font-size: 14px;">The normalized difference combines clipping, element-wise arithmetic, and safe division into a single pipeline. Given two arrays $a$ and $b$, the normalized difference $(a - b) / (a + b)$ produces a dimensionless ratio in $[-1, 1]$ that measures relative difference. This operation appears in remote sensing (NDVI vegetation index), audio processing, and feature engineering for machine learning.</span>

---

## <span style="font-size: 16px;">Mathematical Definition</span>

<span style="font-size: 14px;">For two arrays $a$ and $b$ of the same shape:</span>

$$\text{ND}(a, b) = \frac{a - b}{a + b}$$

<span style="font-size: 14px;">Properties:</span>

* <span style="font-size: 14px;">Range: $[-1, 1]$ when $a, b \geq 0$</span>
* <span style="font-size: 14px;">$\text{ND} = 0$ when $a = b$ (balanced)</span>
* <span style="font-size: 14px;">$\text{ND} = 1$ when $b = 0$ (only $a$ contributes)</span>
* <span style="font-size: 14px;">$\text{ND} = -1$ when $a = 0$ (only $b$ contributes)</span>
* <span style="font-size: 14px;">Undefined when $a = b = 0$ (division by zero)</span>

---

## <span style="font-size: 16px;">np.clip() for Range Bounding</span>

<span style="font-size: 14px;">Before computing the normalized difference, inputs often need to be clipped to a valid range:</span>

```python
a_clipped = np.clip(a, lo, hi)
b_clipped = np.clip(b, lo, hi)
```

<span style="font-size: 14px;">`np.clip(x, lo, hi)` clamps every element:</span>

$$\text{clip}(x_i) = \begin{cases} lo & \text{if } x_i < lo \\ x_i & \text{if } lo \leq x_i \leq hi \\ hi & \text{if } x_i > hi \end{cases}$$

<span style="font-size: 14px;">Clipping is essential for removing outliers or ensuring physical constraints (e.g., reflectance values must be in $[0, 1]$).</span>

---

## <span style="font-size: 16px;">Safe Division</span>

<span style="font-size: 14px;">The denominator $a + b$ can be zero, producing `inf` or `nan`. Handle this with `np.where` or `np.divide`:</span>

```python
numerator = a - b
denominator = a + b

# Method 1: np.where
result = np.where(denominator != 0, numerator / denominator, 0.0)

# Method 2: np.divide with where
result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
```

<span style="font-size: 14px;">Method 2 is preferred because it avoids computing the division where the denominator is zero (no runtime warning).</span>

---

## <span style="font-size: 16px;">Vectorized Pipeline</span>

<span style="font-size: 14px;">The complete operation as a vectorized pipeline:</span>

```python
def normalized_difference(a, b, lo=None, hi=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if lo is not None and hi is not None:
        a = np.clip(a, lo, hi)
        b = np.clip(b, lo, hi)
    num = a - b
    den = a + b
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)
```

<span style="font-size: 14px;">This runs entirely in vectorized C code with no Python loops.</span>

---

## <span style="font-size: 16px;">Applications</span>

### <span style="font-size: 14px;">NDVI (Vegetation Index)</span>

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

<span style="font-size: 14px;">Values near 1 indicate dense vegetation; values near 0 indicate bare soil; negative values indicate water.</span>

### <span style="font-size: 14px;">Feature Engineering</span>

<span style="font-size: 14px;">Normalized differences create scale-invariant features. If $a$ and $b$ are both doubled, the normalized difference remains unchanged. This invariance is valuable for models that need to be robust to absolute scale changes.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Division by zero**: When both inputs are zero at a position, the denominator is zero. Always handle this case.</span>
* <span style="font-size: 14px;">**Integer division**: If inputs are integer arrays, division produces integer results in Python 2 or with `//`. Cast to float first.</span>
* <span style="font-size: 14px;">**Clip order**: `np.clip(a, hi, lo)` with swapped bounds silently returns an array of NaN or the wrong value. Always ensure $lo \leq hi$.</span>
* <span style="font-size: 14px;">**Broadcasting shape mismatch**: $a$ and $b$ must have the same shape or be broadcastable. Mismatched shapes produce wrong results silently.</span>