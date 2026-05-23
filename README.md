# NumPy Masterclass — Complete Reference

A comprehensive documentation covering **NumPy** from array foundations through real-world data pipelines. Built from a full lecture transcript, organized into 6 modules with 22 topics, each with runnable code examples.

## Content Structure

Every topic follows a consistent blueprint:

1. **Concept Anchor** — core intuition and what problem this tool solves
2. **Engine Rules** — how it works under the hood, edge cases, and behavioral rules
3. **Code** — realistic example with sensor, analytics, or scientific data
4. **Visual Memory** — mnemonic or mental model for retention

## Modules

| Module | Topics |
|---|---|
| **1 — Foundational Core** | Introduction & Architecture, ndarray & dtypes, Array Attributes |
| **2 — Initialization & Structure** | Creating Arrays, Changing Shape, Views vs Copies |
| **3 — Selection & Traversal** | Indexing & Slicing, Filtering & Masking, Fancy Indexing |
| **4 — Math & Broadcasting** | Element-wise Operations, Broadcasting, Universal Functions (ufuncs) |
| **5 — Reduction & Manipulation** | Aggregation (min, max, mean), Axis-Based Operations, Sorting & Searching, Combining & Splitting |
| **6 — Real-World Pipelines** | Missing Data (NaN), Random Numbers, Linear Algebra, Data Science Workflows, Performance & Vectorization, Applied Use Cases |

## Topics

| # | Topic | Key Ideas |
|---|---|---|
| 01 | Introduction to NumPy | What NumPy is, why arrays beat Python lists, memory layout (C vs Fortran order) |
| 02 | ndarray & Data Types | `np.array()`, dtype system, type coercion, memory footprint per dtype |
| 03 | Array Attributes | `.ndim`, `.shape`, `.size`, `.dtype`, `.itemsize`, `.nbytes` — the metadata toolbox |
| 04 | Creating Arrays | `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.eye()`, `np.full()` |
| 05 | Changing Shape | `.reshape()`, `.ravel()`, `.flatten()`, `.resize()`, `.T` — shape manipulation mechanics |
| 06 | Views vs Copies | Memory model: views share data, copies own data. When each operation returns which |
| 07 | Indexing & Slicing | Basic indexing `arr[i]`, slice semantics `arr[1:5:2]`, negative steps, axis selection |
| 08 | Filtering & Masking | Boolean masks, `np.where()`, multi-condition filtering with `&` and `|` |
| 09 | Fancy Indexing | Integer array indexing, selecting arbitrary rows/columns, vs slicing behavior |
| 10 | Element-wise Operations | Vectorized `+`, `-`, `*`, `/`, `**`, `//`, comparison ops — all element-wise |
| 11 | Broadcasting | Shape alignment rules, dimension expansion, scalar broadcasting, common pitfalls |
| 12 | Universal Functions | `np.sqrt()`, `np.exp()`, `np.log()`, `np.sin()`, ufunc attributes, `reduce()`/`accumulate()` |
| 13 | Aggregation | `np.sum()`, `np.mean()`, `np.std()`, `np.var()`, `np.min()`, `np.max()`, `np.argmin()`, `np.argmax()` |
| 14 | Axis-Based Operations | `axis` parameter semantics, keeping dimensions with `keepdims` |
| 15 | Sorting & Searching | `np.sort()`, `np.argsort()`, `np.searchsorted()`, `np.partition()` |
| 16 | Combining & Splitting | `np.concatenate()`, `np.stack()`, `np.hstack()`/`vstack()`, `np.split()` |
| 17 | Missing Data (NaN) | `np.nan`, `np.isnan()`, `np.nan_to_num()`, nan-safe aggregations |
| 18 | Random Numbers | `np.random.default_rng()`, `rng.random()`, `.integers()`, `.choice()`, `.shuffle()`, `.normal()` |
| 19 | Linear Algebra | `np.dot()`, `@` operator, `np.linalg.inv()`, `linalg.det()`, `linalg.eig()`, `linalg.solve()` |
| 20 | Data Science Workflows | Scaling/normalization, one-hot encoding, train-test splitting using base NumPy |
| 21 | Performance & Vectorization | Loop avoidance, explicit vectorization, memory layout for cache efficiency |
| 22 | Applied Use Cases | Rolling statistics, distance matrices, image processing basics, Monte Carlo simulation |

## Data Philosophy

All examples use realistic contexts — sensor readings, financial data, scientific measurements. No `foo`/`bar` placeholders.

## Files

| File | Description |
|---|---|
| `numpy-masterclass.html` | Full HTML reference (22 topics, 6 modules) |

## License

MIT
