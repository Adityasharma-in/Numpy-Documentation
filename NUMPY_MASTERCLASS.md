# NumPy Masterclass

> *"In the world of data science, mastering NumPy is like learning to count before doing arithmetic."*

---

## Module 1: Foundational Core

---

### 1.1 Introduction to NumPy

> **Lecture Core Insight:** NumPy is a Python library for working with numbers. Lots of them. Fast. Normal Python lists are slow with math because each element is checked one by one. NumPy does everything in one go using C code in the background. That makes it 10–50× faster. It is the foundation of data science in Python — Pandas, Scikit-learn, TensorFlow all use NumPy underneath.

#### 1. Engine Rules & Mechanics

| Feature | Python List | NumPy Array |
|---|---|---|
| Data types | Can mix types | One type only |
| Speed | Slow (Python loop) | Fast (C loop) |
| Memory | Scattered in RAM | One contiguous block |
| Math | Need for loop | Direct operators |

#### 2. Live Lecture Code Blueprint

```python
import numpy as np
import time

# Python list — slow, element by element
py_list = list(range(10_000_000))
start = time.perf_counter()
result = [x * 2 for x in py_list]
print(f"Python: {time.perf_counter() - start:.3f}s")

# NumPy array — fast, runs C code in background
np_arr = np.arange(10_000_000, dtype=np.float64)
start = time.perf_counter()
result = np_arr * 2
print(f"NumPy:  {time.perf_counter() - start:.3f}s")

# Python: ~0.48s  |  NumPy: ~0.02s  (20-25x faster)
```

---

### 1.2 ndarray & Data Types

> **Lecture Core Insight:** An ndarray (n-dimensional array) is a grid of values, all of the same type. The type is called `dtype`. It decides how much memory each element takes. If you have a million integers, using `int8` (1 byte each) uses way less memory than `int64` (8 bytes each).

#### 1. Engine Rules & Mechanics

| dtype | Bytes | Range |
|---|---|---|
| `int8` | 1 | -128 to 127 |
| `int32` | 4 | -2 billion to 2 billion |
| `int64` | 8 | Very large numbers |
| `float32` | 4 | ~7 digits precision |
| `float64` | 8 | ~15 digits precision |
| `bool` | 1 | True or False |

- NumPy picks type automatically. `np.array([1, 2, 3])` gives `int64`. Add a decimal like `2.5` and it becomes `float64`.
- Use `arr.astype(np.float32)` to change type. Going from float64 to float32 loses precision silently.
- When mixing types, NumPy promotes to the safest one: `bool → int → float → complex`.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

# NumPy decides the type
a = np.array([1, 2, 3])
print("int array:", a.dtype)  # int64

# Mix int and float → becomes float
b = np.array([1, 2.5, 3])
print("mixed:", b.dtype)      # float64

# Force a specific type
c = np.array([1, 2, 3], dtype=np.int8)
print("int8:", c.dtype, "memory:", c.nbytes, "bytes")

# Convert type
d = c.astype(np.float32)
print("converted:", d.dtype)  # float32
```

---

### 1.3 Array Attributes

> **Lecture Core Insight:** Every array has built-in properties: its shape, number of dimensions, total elements, memory usage. These are useful when debugging or managing memory.

#### 1. Engine Rules & Mechanics

| Attribute | What it means | Example (3×4 float64) |
|---|---|---|
| `.shape` | Size of each dimension | `(3, 4)` |
| `.ndim` | How many dimensions | `2` |
| `.dtype` | Type of elements | `float64` |
| `.size` | Total elements | `12` |
| `.itemsize` | Bytes per element | `8` |
| `.nbytes` | Total memory used | `96` |

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]], dtype=np.float64)

print("shape:", arr.shape)       # (3, 4)
print("dimensions:", arr.ndim)    # 2
print("type:", arr.dtype)         # float64
print("elements:", arr.size)       # 12
print("bytes per element:", arr.itemsize)  # 8
print("total memory:", arr.nbytes, "bytes")  # 96
```

---

## Module 2: Initialization & Structure

---

### 2.1 Creating Arrays

> **Lecture Core Insight:** NumPy has many ways to create arrays. You can create from a Python list, generate sequences, make arrays of zeros or ones, or create identity matrices. Some methods initialize memory (zeros, ones) and some don't (empty — which can have garbage data).

#### 1. Engine Rules & Mechanics

| Function | What it creates |
|---|---|
| `np.array([1, 2, 3])` | From a Python list |
| `np.arange(0, 10, 2)` | Sequence like Python range() |
| `np.linspace(0, 1, 5)` | 5 evenly spaced numbers from 0 to 1 |
| `np.zeros((3, 4))` | All zeros |
| `np.ones((3, 4))` | All ones |
| `np.full((2, 3), 7)` | All elements = 7 |
| `np.eye(4)` | Identity matrix (1s on diagonal) |
| `np.empty((3, 3))` | Garbage data — use with caution |

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

# From a Python list
a = np.array([[1, 2], [3, 4]])

# Sequences
b = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
c = np.linspace(0, 1, 5)   # [0.0, 0.25, 0.5, 0.75, 1.0]

# Filled with constants
d = np.zeros((2, 3))       # all 0
e = np.ones((2, 3))        # all 1
f = np.full((2, 3), 42)    # all 42

# Identity matrix
g = np.eye(3)               # 3x3 with 1s on diagonal

# WARNING: empty has garbage values!
h = np.empty((3, 3))
print("empty:", h)
```

---

### 2.2 Changing Shape

> **Lecture Core Insight:** What if your data is in the wrong shape? You can reshape it. `reshape` and `T` change the view without copying data. `flatten` always makes a copy. Understanding this saves memory.

#### 1. Engine Rules & Mechanics

- `reshape(a, b)` — change to a × b shape. Total elements must stay same. Returns a view.
- `ravel()` — flatten to 1D. Returns a view when possible.
- `flatten()` — flatten to 1D. Always returns a new copy.
- `.T` — transpose (swap rows and columns). Returns a view.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

arr = np.arange(12).reshape(3, 4)
print("original shape:", arr.shape)

# reshape — shares memory with original
r = arr.reshape(2, 6)
r[0, 0] = 999
print("original changed:", arr[0, 0])  # 999 (proof it's a view)

# ravel — 1D view
v = arr.ravel()
print("ravel shares memory:", v.base is arr)  # True

# flatten — always a copy
f = arr.flatten()
f[0] = 0
print("original after flatten change:", arr[0, 0])  # 999 (unchanged)

# transpose
t = arr.T
print("transposed shape:", t.shape)    # (4, 3)
```

---

### 2.3 Views vs. Copies

> **Lecture Core Insight:** Does your array share memory or not? Some operations create a view (shares data with original). Others create a copy (independent data). Change a view = changes original. Change a copy = original stays same. Use `.base` to check: if `base is None`, it owns its data.

#### 1. Engine Rules & Mechanics

- **View:** slicing `arr[:]`, `reshape()`, `ravel()`, `.T`
- **Copy:** `.copy()`, `.flatten()`, `.astype()`, `arr[[0, 2]]` (fancy indexing), boolean masking

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

original = np.array([1, 2, 3, 4, 5])

# Slicing gives a VIEW — changes affect original
view = original[1:4]
view[0] = 99
print("original changed:", original)  # [1, 99, 3, 4, 5]

# .copy() gives a COPY — changes do not affect original
safe = original.copy()
safe[0] = -1
print("original unchanged:", original)  # [1, 99, 3, 4, 5]

# Check with .base
print("view.base is original?", view.base is original)  # True
print("copy.base is None?", safe.base is None)        # True
```

---

## Module 3: Selection & Traversal

---

### 3.1 Indexing & Slicing

> **Lecture Core Insight:** Use square brackets. `arr[0, 2]` gets element at row 0, column 2. Slicing `[start:stop:step]` gets a range. Slices are views — they don't copy data.

#### 1. Engine Rules & Mechanics

| Code | What it does |
|---|---|
| `arr[i, j]` | Element at row i, column j |
| `arr[1:4]` | Elements at index 1, 2, 3 |
| `arr[::2]` | Every other element |
| `arr[::-1]` | Reverse the array |
| `arr[-1]` | Last element |

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

arr = np.arange(20).reshape(4, 5)
print("original:")
print(arr)

# One element
print("row 0, col 2:", arr[0, 2])

# Slice a row and some columns
print("row 0, cols 1 to 3:", arr[0, 1:4])

# Every other row, reversed columns
print("every other row, reversed:")
print(arr[::2, ::-1])

# Last row, last column
print("last element:", arr[-1, -1])

# Modify a slice (affects original because it's a view)
arr[1:3, 2:4] = [[-1, -2], [-3, -4]]
print("after mutation:")
print(arr)
```

---

### 3.2 Filtering & Masking

> **Lecture Core Insight:** Use comparisons like `arr > 5`. It gives a boolean array. Use that as a mask: `arr[mask]` gives matching elements. Combine conditions with `&` (AND), `|` (OR), `~` (NOT).

#### 1. Engine Rules & Mechanics

- `arr > 5` — boolean array (True/False per element)
- `arr[mask]` — returns matching values as a 1D array (copy)
- Use `(arr > 3) & (arr < 10)` for combined conditions
- Do NOT use Python `and`/`or` — use `&`/`|`

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

data = np.array([[3, 15, -2, 7],
                 [0, -5, 12, 8],
                 [9, 4, -1, 6]])

# Find all values > 5
mask = data > 5
print("values > 5:", data[mask])  # [15, 7, 12, 8, 9, 6]

# Values between 0 and 10
filtered = data[(data > 0) & (data < 10)]
print("0 < x < 10:", filtered)

# Replace all negatives with 0
data[data < 0] = 0
print("negatives replaced:")
print(data)

# np.where — pick from two options
scores = np.array([42, 85, 33, 91, 67])
result = np.where(scores >= 60, "pass", "fail")
print("results:", result)
```

---

### 3.3 Fancy Indexing

> **Lecture Core Insight:** Use a list of indices: `arr[[0, 2, 4]]`. This is called fancy indexing. Unlike slicing, it always returns a copy.

#### 1. Engine Rules & Mechanics

- `arr[[0, 2, 4]]` — picks rows at those indices (returns a copy)
- `arr[[0, 1], [2, 3]]` — picks elements at (0,2) and (1,3)
- Duplicates are allowed: `arr[[0, 0, 1]]`
- Fancy indexing is slower than slicing because it copies data element by element.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

arr = np.arange(20).reshape(4, 5)
print("original:")
print(arr)

# Pick specific rows
rows = arr[[0, 2, 3]]
print("rows 0, 2, 3:")
print(rows)

# Pick by coordinates: (0,0), (1,2), (2,4)
pairs = arr[[0, 1, 2], [0, 2, 4]]
print("at specific coords:", pairs)  # [0, 7, 14]

# Fancy indexing always copies
original = np.array([10, 20, 30, 40])
selected = original[[0, 2]]
selected[0] = 999
print("original unchanged:", original)  # [10, 20, 30, 40]
```

---

## Module 4: Math & Broadcasting

---

### 4.1 Element-wise Operations

> **Lecture Core Insight:** Just use normal operators: `+`, `-`, `*`, `/`. They work on every element automatically. Add a number and it adds to every element. Compare two arrays and you get True/False for each position.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

a = np.array([10, 20, 30])
b = np.array([3, 4, 5])

# Math happens on each element
print("a + b:", a + b)   # [13, 24, 35]
print("a * b:", a * b)   # [30, 80, 150]
print("a / b:", a / b)   # [3.33, 5, 6]
print("a ** 2:", a ** 2) # [100, 400, 900]

# Add a number to every element
print("a + 100:", a + 100)  # [110, 120, 130]

# Comparisons give boolean arrays
print("a > 15:", a > 15)     # [False, True, True]
print("a == b:", a == b)    # [False, False, False]

# In-place: changes the array directly
a += 5
print("after += 5:", a)       # [15, 25, 35]
```

---

### 4.2 Broadcasting

> **Lecture Core Insight:** What if the shapes don't match? Broadcasting stretches the smaller array to fit the bigger one, without copying data. Rules: compare shapes from the right. A dimension works if both have the same size or one is 1.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

# Normalize each row (subtract row average)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])  # shape (4, 3)

# Row means — keepdims keeps it as (4, 1) instead of (4,)
row_avg = data.mean(axis=1, keepdims=True)
normalized = data - row_avg  # (4,3) - (4,1) works because of broadcasting
print("normalized:")
print(normalized)
print("each row avg:", normalized.mean(axis=1))  # all ~0

# A (3,4) array times a (4,) array — broadcast works
A = np.ones((3, 4))
B = np.array([1, 2, 3, 4])
print("broadcast result:")
print(A * B)  # B stretched from (4,) to (3,4)

# This would fail (uncomment to see):
# np.ones((3,4)) + np.ones((3,5))  # shapes don't match
```

---

### 4.3 Universal Functions (ufuncs)

> **Lecture Core Insight:** Need math like sin, exp, log? NumPy has them as ufuncs (universal functions). They work element by element, run in C (fast). Includes trig, exponentials, rounding, and more.

#### 1. Engine Rules & Mechanics

| Category | Functions |
|---|---|
| Trig | `sin`, `cos`, `tan` |
| Exp/Log | `exp`, `log`, `log2`, `log10` |
| Rounding | `floor`, `ceil`, `round` |
| Other | `abs`, `sqrt`, `sign` |

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

vals = np.array([1.0, 2.0, 3.0, 4.0])

# Math functions
print("sqrt:", np.sqrt(vals))
print("exp:", np.exp(vals))
print("log:", np.log(vals))

# Rounding
prices = np.array([19.975, 20.001, 3.4567])
print("floor:", np.floor(prices))
print("ceil:", np.ceil(prices))
print("round:", np.round(prices, 2))

# Special ufunc methods
print("sum:", np.add.reduce(vals))         # 10
print("cumulative:", np.add.accumulate(vals))  # [1, 3, 6, 10]

# Outer product
a = np.array([1, 2, 3])
b = np.array([4, 5])
print("outer:")
print(np.add.outer(a, b))
```

---

## Module 5: Reduction & Manipulation

---

### 5.1 Aggregation (Min, Max, Mean, etc.)

> **Lecture Core Insight:** Need one number from the whole array? Use `.min()`, `.max()`, `.mean()`, `.std()`. By default they use all elements. With `axis` you can summarize along one dimension.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

data = np.array([[22.5, 23.1, 21.8],
                 [19.4, 20.7, 21.2],
                 [25.0, 24.3, 26.1],
                 [18.9, 19.5, 20.0]])

print("min:", data.min())     # 18.9
print("max:", data.max())     # 26.1
print("mean:", data.mean())  # ~21.85
print("std:", data.std())    # ~2.28
print("sum:", data.sum())    # 262.4
print("median:", np.median(data))  # 21.5

# Find index of minimum value
idx = data.argmin()
print("min at flat index:", idx)
print("unraveled:", np.unravel_index(idx, data.shape))
```

---

### 5.2 Axis-Based Operations

> **Lecture Core Insight:** Use `axis=0` for columns (collapse rows) and `axis=1` for rows (collapse columns). Think: axis=0 goes down, axis=1 goes across.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

arr = np.arange(12).reshape(3, 4)
print("array:")
print(arr)

# axis=0 — collapse rows (work column by column)
print("sum per column:", arr.sum(axis=0))   # [12, 15, 18, 21]

# axis=1 — collapse columns (work row by row)
print("sum per row:", arr.sum(axis=1))    # [6, 22, 38]

# keepdims preserves shape for broadcasting
row_means = arr.mean(axis=1, keepdims=True)
print("row means (kept shape):")
print(row_means)
print("shape:", row_means.shape)  # (3, 1)
```

---

### 5.3 Sorting & Searching

> **Lecture Core Insight:** `np.sort()` returns a sorted copy. `argsort()` gives the indices that would sort. `unique()` finds unique values.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

scores = np.array([[85, 92, 78],
                   [91, 88, 95],
                   [73, 80, 84]])

# Sort each row
print("sorted rows:")
print(np.sort(scores))

# argsort — which position in ranking
print("ranking:")
print(np.argsort(scores, axis=1))

# Find where condition is true
print("positions where > 90:")
print(np.argwhere(scores > 90))

# Unique values
vals, counts = np.unique(scores, return_counts=True)
print("unique:", vals)
print("counts:", counts)

# Set operations
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7])
print("in both:", np.intersect1d(a, b))   # [4, 5]
print("in a not b:", np.setdiff1d(a, b)) # [1, 2, 3]
```

---

### 5.4 Combining & Splitting Arrays

> **Lecture Core Insight:** `vstack` stacks vertically (adds rows), `hstack` stacks horizontally (adds columns). `vsplit`/`hsplit` split arrays into pieces.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

train = np.array([[1, 2], [3, 4]])
test  = np.array([[5, 6], [7, 8]])

# Stack vertically (add rows)
all_data = np.vstack((train, test))
print("vstack:")
print(all_data)

# Stack horizontally (add columns)
labels = np.array([[0], [1]])
combined = np.hstack((train, labels))
print("hstack:")
print(combined)

# Split
big = np.arange(20).reshape(4, 5)
parts = np.vsplit(big, 2)
print("split shapes:", [p.shape for p in parts])
```

---

## Module 6: Real-World Pipelines

---

### 6.1 Missing Data (NaN)

> **Lecture Core Insight:** NaN (Not a Number) marks missing data. Any math with NaN gives NaN. Use `np.nanmean()`, `np.nanstd()` to skip NaN. Detect with `np.isnan()`.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

data = np.array([[1.2, 3.4, np.nan],
                 [np.nan, 5.6, 7.8],
                 [9.0, np.nan, 2.1]])

# Count missing
print("missing:", np.isnan(data).sum())  # 3

# NaN-safe functions
print("mean (skip nan):", np.nanmean(data))
print("sum (skip nan):", np.nansum(data))

# Replace NaN with column average
col_avg = np.nanmean(data, axis=0, keepdims=True)
cleaned = np.where(np.isnan(data), col_avg, data)
print("cleaned:")
print(cleaned)
```

---

### 6.2 Random Numbers

> **Lecture Core Insight:** Use `np.random.default_rng(seed)` to create a generator. Get random floats, integers, or normally distributed values. Set a seed for reproducible results.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

# Create generator with seed for reproducibility
rng = np.random.default_rng(seed=42)

# Random floats between 0 and 1
print("uniform:", rng.random(size=5))

# Random integers
print("dice rolls:", rng.integers(1, 7, size=10))

# Normal distribution
heights = rng.normal(loc=170, scale=7.5, size=1000)
print("heights:", f"mean={heights.mean():.1f}", f"std={heights.std():.1f}")

# Same seed = same numbers
r1 = np.random.default_rng(123)
r2 = np.random.default_rng(123)
assert np.array_equal(r1.random(5), r2.random(5))
print("reproducible: yes")
```

---

### 6.3 Linear Algebra

> **Lecture Core Insight:** Use `@` for matrix multiply, `np.linalg.inv()` for inverse, `np.linalg.det()` for determinant, `np.linalg.eig()` for eigenvalues.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

# Solve A @ x = b
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])

x = np.linalg.inv(A) @ b
print("x:", x)        # [2., 3.]
print("check:", A @ x)  # [9., 8.]

# Determinant
print("det:", np.linalg.det(A))   # 5.0

# Norm (vector length)
v = np.array([3.0, 4.0])
print("norm:", np.linalg.norm(v))  # 5.0

# Eigenvalues
vals, vecs = np.linalg.eig(A)
print("eigenvalues:", vals)
```

---

### 6.4 Data Science Workflows (Scaling Data)

> **Lecture Core Insight:** Min-Max scaling pushes values to [0, 1]. Z-score centers at 0 with std=1. Both are one-liners in NumPy. Compute stats on training data only, then apply to test data.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np

X = np.array([[10.0, 2.0, 100],
              [15.0, 3.5, 200],
              [12.0, 2.8, 150],
              [18.0, 4.2, 300],
              [11.0, 2.1, 120]])

# Min-Max: each feature scaled to [0, 1]
X_min = X.min(axis=0, keepdims=True)
X_max = X.max(axis=0, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min)
print("min-max:")
print(X_norm)

# Z-score: center at 0, std = 1
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True)
X_z = (X - X_mean) / X_std
print("z-score:")
print(X_z)
print("mean ~0:", X_z.mean(axis=0))
print("std ~1:", X_z.std(axis=0))

# Apply same transform to new data
new = np.array([[14.0, 3.0, 180]])
print("new (z-score):", (new - X_mean) / X_std)
```

---

### 6.5 Performance & Vectorization

> **Lecture Core Insight:** NumPy runs C code, not Python loops. `float32` uses half the memory of `float64` and can be 2x faster. Keep arrays contiguous for best performance.

#### 2. Live Lecture Code Blueprint

```python
import numpy as np
import time

N = 10_000_000

# Python loop vs NumPy
py_list = list(range(N))
start = time.perf_counter()
result = [x * 2 for x in py_list]
print(f"Python: {time.perf_counter() - start:.3f}s")

np_arr = np.arange(N, dtype=np.float64)
start = time.perf_counter()
result = np_arr * 2
print(f"NumPy:  {time.perf_counter() - start:.3f}s")

# float32 is faster than float64
f32 = np.ones(N, dtype=np.float32)
f64 = np.ones(N, dtype=np.float64)

start = time.perf_counter(); f32.mean(); t32 = time.perf_counter() - start
start = time.perf_counter(); f64.mean(); t64 = time.perf_counter() - start

print(f"float32: {t32:.4f}s")
print(f"float64: {t64:.4f}s")
print(f"float32 is {t64/t32:.1f}x faster")
```

---

### 6.6 Applied Use Cases

> **Lecture Core Insight:** Putting it all together. Three real examples: cleaning server logs (NaN + scaling), processing images (channel math), and smoothing time-series (sliding windows).

#### 1. Clean Server Logs

```python
import numpy as np

logs = np.array([
    [1, 45.2, 62.1, 12.5],
    [2, 78.9, 81.3, 45.2],
    [3, np.nan, 55.0, 33.1],
    [4, 62.3, np.nan, 28.7],
    [5, 55.6, 59.8, np.nan]
], dtype=np.float64)

means = np.nanmean(logs, axis=0, keepdims=True)
cleaned = np.where(np.isnan(logs), means, logs)
features = cleaned[:, 1:]
z = (features - features.mean(axis=0)) / features.std(axis=0)
print("cleaned & scaled:")
print(z)
```

#### 2. Process RGB Image

```python
import numpy as np

rng = np.random.default_rng(42)
img = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
img_f = img.astype(np.float32) / 255.0

gray = 0.299 * img_f[:,:,0] + 0.587 * img_f[:,:,1] + 0.114 * img_f[:,:,2]
print("grayscale:")
print(gray)

flipped = img_f[:, ::-1, :]
print("flipped shape:", flipped.shape)
```

#### 3. Smooth Time-Series

```python
import numpy as np

rng = np.random.default_rng(42)
t = np.linspace(0, 10, 1000)
s1 = np.sin(t) + 0.1 * rng.normal(size=t.shape)
s2 = np.cos(t * 2) + 0.1 * rng.normal(size=t.shape)
s3 = np.sin(t * 3) + 0.1 * rng.normal(size=t.shape)

ts = np.column_stack((s1, s2, s3))
w = 50
idx = np.arange(w)[None, :] + np.arange(len(ts) - w + 1)[:, None]
smoothed = ts[idx].mean(axis=1)
print("smoothed shape:", smoothed.shape)  # (951, 3)
```
