# FNO for 1D Bar Inverse Problem

This directory contains code for training a Fourier Neural Operator (FNO) to solve the inverse problem of identifying Young's modulus distribution E(x) from displacement measurements.

## Problem Setup

- **Forward Problem**: 1D bar element, length L, fixed at x=0, loaded with force F at x=L
- **Inverse Problem**: Given displacement field u(x) → Predict E(x)
- **Network Design**: FNO predicts logE(x), then E = exp(logE) ensures positivity

## File Structure

```
fno/
├── fem1d.py          # 1D FEM solver (mesh, assembly, solve, residual)
├── data_gen.py       # Generate random E(x) and compute u(x) via FEM
├── dataset.py        # PyTorch Dataset/DataLoader with normalization
├── model.py          # FNO model (neuralop)
├── train.py          # Training with supervised + physics loss
├── eval.py           # Evaluation and visualization
└── README.md         # This file
```

## Installation

```bash
pip install torch numpy scipy matplotlib tensorboard tqdm
pip install neuralop  # Fourier Neural Operator library
```

## Usage

### 1. Generate Data

```bash
cd fno
python data_gen.py
```

This generates:
- `train_data.pt` (1000 samples)
- `val_data.pt` (200 samples)
- `test_data.pt` (200 samples)

Each sample contains:
- Random E(x) field (via KL expansion or piecewise constant)
- Random force F
- FEM-computed displacement u(x)

### 2. Train Model

```bash
python train.py
```

Training uses three loss components:
- **Supervised loss**: ||logE_pred - logE_true||²
- **Physics loss**: ||K(E_pred)·u - f||² / ||f||² (FEM residual)
- **Smoothness loss**: ||∂logE/∂x||² (regularization)

Default weights: λ_sup=1.0, λ_phys=1.0, λ_smooth=0.01

Checkpoints saved to `checkpoints/`

Monitor with TensorBoard:
```bash
tensorboard --logdir=runs
```

### 3. Evaluate

```bash
python eval.py
```

Computes:
- Relative L2 error for E and logE
- Physics residual (FEM consistency)
- Point-wise MAE/MAPE

Generates plots in `plots/`:
- Example predictions
- Error distributions
- Best/worst cases

## Key Design Choices

### Network Architecture
- **Input**: 3 channels [u, x/L, F_broadcast] (batch, 3, n_node)
- **Output**: 1 channel logE (batch, 1, n_node)
- **FNO**: 16 Fourier modes, 64 hidden channels, 4 layers
- **Inference**: E = exp(clamp(logE, -20, 20))

### Data Normalization
- u: standardized (zero mean, unit variance)
- F: standardized
- x/L: already in [0,1]
- logE: standardized (for supervised loss)

### Physics Loss
- Assemble K(E_pred) for each sample
- Compute residual r = K·u - f on free DOFs (excluding fixed BC)
- Normalized by ||f|| for scale invariance

### Training Strategy
- Optimizer: AdamW with weight decay 1e-5
- Scheduler: Cosine annealing over 100 epochs
- Initial LR: 1e-3
- Batch size: 32

## Customization

### Change E(x) distribution type

Edit `data_gen.py`:
```python
logE_type='KL'  # or 'piecewise'
logE_params={'n_modes': 10, 'lengthscale': 0.2, 'sigma': 0.5}
```

### Adjust loss weights

Edit `train.py`:
```python
'lambda_sup': 1.0,    # Supervised term
'lambda_phys': 1.0,   # Physics consistency
'lambda_smooth': 0.01 # Smoothness regularization
```

### Change network size

Edit `train.py` or `model.py`:
```python
'n_modes': 16,          # Fourier modes (higher = more capacity)
'hidden_channels': 64,  # Hidden dimension
'n_layers': 4           # Number of FNO layers
```

### Resolution generalization test

Generate test data with different n_node:
```python
test_data = generate_dataset(..., n_node=201)  # Double resolution
```

## Expected Performance

On default settings (KL expansion, 101 nodes, noise-free):
- **E relative L2 error**: ~1-5%
- **Physics residual**: ~1e-3 to 1e-5
- **Training time**: ~10-30 min (GPU), ~2-4 hours (CPU)

## Notes

1. **Positivity enforcement**: Network outputs logE (unbounded), then E = exp(logE) ≥ 0 always
2. **Numerical stability**: Clamp logE ∈ [-20, 20] before exp to avoid overflow
3. **Dirichlet BC handling**: Physics loss computed only on free DOFs (u[1:])
4. **Batch assembly**: FEM residual computed per sample (can be slow for large batches)

## References

- Fourier Neural Operator: https://arxiv.org/abs/2010.08895
- neuralop library: https://github.com/neuraloperator/neuralop
