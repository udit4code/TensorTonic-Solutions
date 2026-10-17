# SwiGLU: why it exists, and why we slice before casting

These notes follow the [Python implementation](./cs336-l03-parameter-matched-swiglu.py), including its parameter budget and `w_g[:, :h].to(torch.float32)` line. All equations below omit biases, just as the implementation does.

## 1. What problem does a feed-forward layer solve?

A token enters a transformer layer as a vector of `d` numbers. Attention mixes information across token positions. The feed-forward network (FFN) then transforms the features within each token independently, using the same weights at every position.

A matrix multiplication creates learned combinations of input features. For a single input vector `x`, output feature `j` is:

```text
(x W)[j] = x[0] W[0,j] + x[1] W[1,j] + ... + x[d-1] W[d-1,j]
```

But two linear transformations alone collapse into one: `(x W1) W2 = x (W1 W2)`. A hidden layer needs a nonlinear operation to express more than a single linear map.

A conventional FFN uses `activation(x W1) W2`: it projects into a hidden feature space, applies a nonlinearity, and projects back to width `d`.

SwiGLU gives the hidden computation two learned branches. One produces values; the other produces input-dependent multipliers for those values. Its goal is to learn useful interactions between features before projecting them back into the token representation.

SwiGLU is an architectural choice, not a requirement for a working transformer. The original GLU-variants study reported quality improvements for some gated variants over ReLU/GELU FFNs in its experiments; that is empirical motivation, not a guarantee for every model. See [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

## 2. Build SwiGLU from its parts

The sigmoid function smoothly maps a number into `(0, 1)`:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

SiLU, also called Swish with parameter 1, multiplies a number by its own sigmoid:

```text
SiLU(z) = z * sigmoid(z)
```

For example, `SiLU(-2) ≈ -0.2384`, `SiLU(0) = 0`, and `SiLU(2) ≈ 1.7616`. Large positive inputs pass approximately unchanged; large negative inputs approach zero from below. Unlike ReLU, it does not set every negative input to exactly zero.

The implementation computes:

```text
a     = x W_g
gate  = SiLU(a)
value = x W_v
z     = gate ⊙ value
y     = z W_o
```

Here `⊙` means element-by-element multiplication. Each hidden value gets its own learned, input-dependent multiplier. For example, a value of `3` becomes approximately `5.2848` when its gate input is `2`, but `-0.7152` when its gate input is `-2`.

Despite the name “gate,” **SiLU's output is not a probability or a switch restricted to `[0, 1]`**. It can attenuate, amplify, or reverse the sign of the value. The branches learn these roles jointly; their features need not correspond to human-readable concepts.

```mermaid
flowchart LR
    X["Token features x: (..., d)"] --&gt; G["Gate projection: x W_g"]
    X --&gt; V["Value projection: x W_v"]
    G --&gt; S["SiLU: a * sigmoid(a)"]
    S --&gt; M["Elementwise multiply: (..., h)"]
    V --&gt; M
    M --&gt; O["Output projection: multiply by W_o"]
    O --&gt; Y["Output: (..., d)"]
```

The `...` can represent batch and sequence dimensions. No operation here mixes token positions. The final width `d` makes the result compatible with a transformer residual stream, though this function itself does not add a residual connection.

## 3. Why “parameter matched”?

Let `H` be the available hidden width in the supplied matrices and `h` the width actually selected for this call.

| Matrix | Supplied shape | Selected shape | Active weights |
| --- | --- | --- | --- |
| `w_g` | `(d, H)` | `(d, h)` | `d * h` |
| `w_v` | `(d, H)` | `(d, h)` | `d * h` |
| `w_o` | `(H, d)` | `(h, d)` | `h * d` |

There are three matrices, so the selected computation uses `3 * d * h` scalar weights. A conventional two-matrix FFN of width `m` uses `2 * d * m`. To compare them at similar parameter counts:

```text
3 d h ≈ 2 d m
    h ≈ (2/3) m
```

For a conventional width `m = 4d`, this gives `h ≈ 8d/3`. Comparing both at the same hidden width would give SwiGLU 50% more matrix weights.

This implementation accepts an arbitrary target `base_params` and selects:

```python
h = min(H, max(1, math.floor(base_params / (3 * d) + 0.5)))
parameter_count = 3 * d * h
```

Divide by the cost of one hidden channel, round to the nearest integer with half values rounded upward, then clamp to `[1, H]`. This assumes positive `d`, available `H &gt;= 1`, and compatible matrix shapes.

**Matching is approximate; this is not a strict upper-budget rule.** For `d=2`, `H=4`, and `base_params=10`, the rule chooses `h=2` and uses 12 weights. A very small budget still selects one channel; a very large budget stops at `H`.

Also, `parameter_count` counts selected weights, not all allocated storage. The full supplied tensors still exist. They are arguments to `forward`, not `nn.Parameter` objects registered by this module.

## 4. Why exactly `w_g[:, :h].to(torch.float32)`?

Read it as two operations:

```python
selected_gate_weights = w_g[:, :h]                  # Choose hidden channels.
w_g_f32 = selected_gate_weights.to(torch.float32)  # Choose numeric dtype.
```

### Why slice the columns?

In `x @ w_g`, each **column** of `w_g` defines one hidden feature. The first colon keeps every input coordinate; `:h` keeps columns `0` through `h-1`.

With `d=2`, `H=4`, and `h=2`:

```text
                      hidden channels
                       0   1 │  2   3
                     ┌───────┼───────┐
input coordinate 0   │  1   0 │  9   9│
input coordinate 1   │  0   1 │  9   9│
                     └───────┴───────┘
                        keep   unused

w_g[:, :2] = [[1, 0],
              [0, 1]]        shape: (2, 2)

x: (..., 2) @ selected w_g: (2, 2) -&gt; (..., 2)
```

We select matching columns from `w_v` because the gate and value for hidden channel `j` must multiply together. We select matching **rows** from `w_o` because it consumes hidden channels:

```text
x (..., d) -- W_g[:, :h] (d, h) --&gt; gate  (..., h)
x (..., d) -- W_v[:, :h] (d, h) --&gt; value (..., h)
                                      │
                               gate * value
                                      │
                       W_o[:h, :] (h, d)
                                      │
                               output (..., d)
```

Keeping the first `h` channels is this exercise's selection convention. It does not identify the “best” neurons or make arbitrary pruning of a trained network harmless. If selecting different indices, all three matrices must use the same hidden-channel indices.

### Why cast to float32?

Shape determines which arithmetic is possible; dtype determines how numbers are represented. FP16 and BF16 use 16 bits per element, while FP32 uses 32. FP32 has finer precision than either, and a much larger finite range than FP16. BF16 already has a range similar to FP32 but fewer precision bits.

Dot products, SiLU, and the gate-value product introduce intermediate results that can round or overflow. This implementation explicitly converts **both inputs and selected weights** to float32 before computing. This also avoids a dtype mismatch between, for example, float32 `x_f32` and FP16 weights in ordinary matrix multiplication.

Casting cannot recover information already lost when weights were stored in a lower precision. If an input was float64, converting to float32 reduces its precision. Float32 is the computation policy chosen here, not a universal requirement of SwiGLU.

The function converts the final result back to `x.dtype`; that final conversion can still round or overflow. The examples below run without autocast. An enclosing autocast context can change matrix-multiplication precision despite explicit float32 operands, so this function does not by itself enforce full FP32 execution under every external configuration.

### Why slice *before* casting?

These produce the same selected numerical values:

```python
a = w_g[:, :h].to(torch.float32)
b = w_g.to(torch.float32)[:, :h]
```

Their allocation costs can differ. Basic slicing creates a view sharing the original storage. When a dtype conversion is needed, `.to()` creates a converted copy; if the dtype already matches, it can return the existing tensor. See PyTorch's [tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html) and [Tensor.to documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html).

```mermaid
flowchart TD
    W["Original FP16 weights: d by H"] --&gt; S["Slice: view of d by h"]
    S --&gt; C["Cast: copy d*h values into FP32"]
    W --&gt; F["Cast first: copy d*H values into FP32"]
    F --&gt; V["Slice: view retaining the full FP32 storage"]
```

For `d=4096`, `H=16384`, `h=8192`, a dense FP32 copy of the selected gate weights contains 128 MiB of data; a full FP32 copy contains 256 MiB. This counts the converted weight data only, excluding original tensors, activations, and allocator overhead. Slicing first avoids converting unused values. It does not free the original weights.

If `h == H`, there are no unused columns to avoid. If `w_g` is already float32, neither ordering needs a dtype-conversion copy. Slicing is not a promise of contiguous storage, and neither operation here modifies the supplied matrix in place.

## 5. A complete numerical example

Take `d=2`, `H=4`, and `base_params=12`. Then `h=2` and the selected parameter count is `3 * 2 * 2 = 12`.

The following standalone example can be run from this directory. The large unused weights make it easy to see why selecting the intended channels matters.

```python
import runpy
import torch

solve = runpy.run_path("cs336-l03-parameter-matched-swiglu.py")[
    "parameter_matched_swiglu"
]
x = torch.tensor([[1., 2.]], dtype=torch.float16)
w_g = torch.tensor([[1., 0., 9., 9.],
                    [0., 1., 9., 9.]], dtype=torch.float16)
w_v = torch.tensor([[2., 0., 9., 9.],
                    [0., 3., 9., 9.]], dtype=torch.float16)
w_o = torch.tensor([[1., 0.],
                    [0., 1.],
                    [9., 9.],
                    [9., 9.]], dtype=torch.float16)

result = solve(x, w_g, w_v, w_o, base_params=12)
print(result)

h = result["hidden_width"]
a = w_g[:, :h].float()
b = w_g.float()[:, :h]
assert torch.equal(a, b)
print(a.untyped_storage().nbytes())  # 16: four FP32 values
print(b.untyped_storage().nbytes())  # 32: retains eight FP32 values
```

Follow the selected matrices by hand:

```text
x                 = [1, 2]
x W_g_selected    = [1, 2]
SiLU(...)         ≈ [0.731059, 1.761594]
x W_v_selected    = [2, 6]
gate * value      ≈ [1.462117, 10.569565]
W_o_selected      = identity matrix
FP32 output       ≈ [1.462117, 10.569565]
returned FP16     = [1.4619140625, 10.5703125]
```

The returned dictionary has `hidden_width=2`, `parameter_count=12`, and an FP16 output of shape `(1, 2)`.

## 6. What happens if we leave something out?

| Change | Consequence |
| --- | --- |
| Remove only the gate slice | The gate has width `H`, while the value has width `h`. Usually the elementwise product fails. A singleton dimension can broadcast instead, silently producing unintended behavior. |
| Remove all three slices | The computation uses all `H` channels. It may run, but it generally changes the answer and uses `3*d*H` active weights instead of the reported `3*d*h`. |
| Slice rows of `w_g` instead of columns | You remove input coordinates instead of hidden features; the matrix-multiplication dimensions generally stop matching. |
| Remove only the gate cast | With lower-precision gate weights and float32 `x_f32`, ordinary matmul raises a dtype mismatch error. Already-float32 weights need no conversion. |
| Remove all casts consistently | Matching low-precision tensors may compute successfully, but intermediates have different rounding/range behavior and no longer follow this implementation's FP32 policy. |
| Cast first, then slice | Selected values and shapes are correct, but conversion may allocate and process unused columns. |

In the worked example, using all four channels adds two gate inputs of `27` and two values of `27`. Each extra gated feature is approximately `729`. The final two rows of `w_o` add approximately `13,122` to **each** output coordinate! The active parameter count also doubles from 12 to 24. Slicing changes the actual function being evaluated.

### A separate example: why cast before arithmetic?

FP16's largest finite value is 65,504. Consider one hidden channel with gate input `300`, value `300`, and output weight `0.5`. Since `sigmoid(300)` is effectively 1, the intermediate product is approximately 90,000, but the final mathematical answer is 45,000.

```python
import torch

gate_input = torch.tensor([300.], dtype=torch.float16)
value = torch.tensor([300.], dtype=torch.float16)
output_weight = torch.tensor([0.5], dtype=torch.float16)

low_precision = (gate_input * torch.sigmoid(gate_input)) * value
too_late = low_precision.float() * output_weight.float()

g = gate_input.float()
fp32_path = (g * torch.sigmoid(g)) * value.float() * output_weight.float()

print(low_precision)       # inf: 90,000 cannot fit in FP16
print(too_late)            # inf: casting afterward cannot repair overflow
print(fp32_path)           # 45000 in FP32
print(fp32_path.half())    # 44992: finite, rounded when returned as FP16
```

This isolates an intermediate overflow even though the final result fits in FP16. It does not imply that every low-precision SwiGLU evaluation overflows, or that BF16 has the same range limitation.

The three decisions serve different purposes: **SwiGLU defines the feature transformation; slicing selects its width; casting chooses the representation used for the computation.**