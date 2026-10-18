# When to Use GRF vs KL vs Piecewise for logE(x) Generation

## Quick Answer
**Use GRF as the default baseline** for most inverse problems involving continuous material parameters. Use piecewise for layered materials, and KL for fast prototyping.

---

## Detailed Comparison

### 1. Gaussian Random Field (GRF) - **RECOMMENDED**

**Use when:**
- Material parameters exhibit **spatial correlation** (most realistic scenarios)
- You need **statistical rigor** and well-defined correlation structure
- You want to control the **smoothness** of E(x) via kernel choice
- Training data should represent **manufacturing variations**, **geological media**, or **continuous heterogeneous materials**

**Pros:**
- `corr_len` directly controls correlation length scale (intuitive)
- Statistically rigorous: proper covariance structure
- Kernel choice controls smoothness:
  - `exp`: Rough texture, continuous but not differentiable
  - `rbf`: Ultra-smooth, infinitely differentiable
- Well-studied in uncertainty quantification and inverse problems

**Cons:**
- O(n³) cost due to Cholesky (acceptable for 1D with n<500)
- Requires scipy for Cholesky decomposition

**Parameters:**
```python
'GRF': {
    'kernel': 'exp',      # 'exp' (rough) or 'rbf' (smooth)
    'corr_len': 0.2,      # Correlation length (0.05-0.5 typical)
    'sigma': 0.5,         # Standard deviation of logE
    'mean_logE': 0.0      # Mean of logE
}
```

**Example use cases:**
- Composite materials with spatial variations
- Metal alloys with grain structure
- Biological tissues with heterogeneous properties
- Geological media (rock, soil)

---

### 2. KL/Fourier Expansion - Fast Approximation

**Use when:**
- You need **fast data generation** (no Cholesky)
- You want **smooth, low-frequency** variations
- Prototyping or testing code before full-scale training

**Pros:**
- O(n·n_modes) cost - much faster than GRF
- No scipy dependency
- Good for smooth material distributions

**Cons:**
- `lengthscale` is NOT a true correlation length (just modal decay rate)
- Less statistically rigorous than GRF
- Limited control over correlation structure

**Parameters:**
```python
'KL': {
    'n_modes': 10,        # Number of Fourier modes (5-20 typical)
    'lengthscale': 0.2,   # Modal decay rate (0.1-0.5)
    'sigma': 0.5,         # Std of logE
    'mean_logE': 0.0
}
```

**When to prefer over GRF:**
- Quick prototyping / debugging
- Very large n_node (>1000) where GRF is too slow
- You only care about "smooth variation" without strict correlation

---

### 3. Piecewise Constant - Discontinuous Materials

**Use when:**
- Materials have **sharp interfaces** (layered structures)
- Testing robustness to **discontinuities**
- Simulating **defects**, **cracks**, or **phase transitions**

**Pros:**
- O(n) cost - very fast
- Realistic for layered composites, coatings, geological strata
- Tests network's ability to handle sharp gradients

**Cons:**
- Not realistic for most continuous materials
- May be too harsh for initial training (causes large gradients)

**Parameters:**
```python
'piecewise': {
    'n_segments': 4,               # Number of layers (2-10 typical)
    'logE_range': (-0.5, 0.5)      # (min_logE, max_logE)
}
```

**Example use cases:**
- Multi-layer coatings
- Sedimentary rock layers
- Functionally graded materials (discrete approximation)
- Testing generalization to distribution shift

---

## Recommended Strategy

### For Training
**Option A: Single distribution (recommended for most cases)**
```python
# Use GRF with exp kernel (good balance)
logE_type='GRF'
logE_params={'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5}
```

**Option B: Mixed distribution (advanced)**
Generate datasets with multiple types and mix:
- 70% GRF (exp kernel, corr_len=0.1-0.3)
- 20% GRF (rbf kernel, corr_len=0.2-0.4) - smoother
- 10% piecewise (n_segments=2-6) - test robustness

Or use separate datasets for training vs testing to evaluate **distribution shift**.

### For Testing
Create test sets with **different parameters** than training:
```python
# Training: GRF with corr_len=0.2
train: logE_params={'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5}

# Test 1: Different correlation length
test1: logE_params={'kernel': 'exp', 'corr_len': 0.1, 'sigma': 0.5}

# Test 2: Different kernel (smoothness)
test2: logE_params={'kernel': 'rbf', 'corr_len': 0.2, 'sigma': 0.5}

# Test 3: Different distribution type
test3: logE_type='piecewise', logE_params={'n_segments': 4}
```

This tests **generalization** beyond training distribution.

---

## Correlation Length Guide (for GRF)

| `corr_len` | Behavior | When to Use |
|------------|----------|-------------|
| 0.05       | Very rough, almost uncorrelated | High-frequency variations |
| 0.1-0.2    | Moderate correlation (typical) | General-purpose, realistic |
| 0.3-0.5    | Smooth, long-range correlation | Very smooth materials |
| >0.5       | Nearly constant field | Not recommended (trivial) |

**Rule of thumb:** `corr_len = 0.2` means roughly **5 oscillations** across the domain (1/0.2 = 5).

---

## Code Example: Mixed Dataset

```python
# Generate 70% GRF + 30% piecewise
n_samples_grf = 700
n_samples_piece = 300

# GRF samples
data_grf = generate_dataset(
    n_samples=n_samples_grf,
    logE_type='GRF',
    logE_params={'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5},
    seed_base=0
)

# Piecewise samples
data_piece = generate_dataset(
    n_samples=n_samples_piece,
    logE_type='piecewise',
    logE_params={'n_segments': 4, 'logE_range': (-0.5, 0.5)},
    seed_base=10000
)

# Combine
mixed_data = {
    'u': np.concatenate([data_grf['u'], data_piece['u']]),
    'x': np.concatenate([data_grf['x'], data_piece['x']]),
    'F': np.concatenate([data_grf['F'], data_piece['F']]),
    'logE_true': np.concatenate([data_grf['logE_true'], data_piece['logE_true']]),
    'E_true': np.concatenate([data_grf['E_true'], data_piece['E_true']])
}

# Shuffle
idx = np.random.permutation(len(mixed_data['F']))
for key in mixed_data:
    mixed_data[key] = mixed_data[key][idx]
```

---

## Summary

| Method | Speed | Realism | Control | Recommended |
|--------|-------|---------|---------|-------------|
| GRF    | Slow  | High    | Excellent | ✅ Yes (default) |
| KL     | Fast  | Medium  | Limited | For prototyping |
| Piecewise | Very fast | Low (for continuous) | Good | For testing robustness |

**Bottom line:** Unless you have a specific reason (speed, layered materials), **use GRF with exp kernel and corr_len=0.2** as your baseline.
