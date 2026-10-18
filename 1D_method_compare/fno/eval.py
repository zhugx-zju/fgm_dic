"""
Evaluation Script
Load trained model and evaluate on test set
Compute: L2 error, physics residual, visualizations
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import FNOModel
from dataset import FNODataset
from data_gen import load_dataset
from fem1d import FEM1DBar


def compute_metrics(E_pred, E_true, logE_pred, logE_true):
    """
    Compute accuracy metrics

    Parameters:
    -----------
    E_pred, E_true : array (n_samples, n_node)
    logE_pred, logE_true : array (n_samples, n_node)

    Returns:
    --------
    metrics : dict
    """
    # Relative L2 error
    E_diff = E_pred - E_true
    E_rel_L2 = np.linalg.norm(E_diff, axis=1) / (np.linalg.norm(E_true, axis=1) + 1e-10)

    logE_diff = logE_pred - logE_true
    logE_rel_L2 = np.linalg.norm(logE_diff, axis=1) / (np.linalg.norm(logE_true, axis=1) + 1e-10)

    # Point-wise metrics
    E_MAE = np.abs(E_diff).mean()
    E_MAPE = (np.abs(E_diff) / (np.abs(E_true) + 1e-10)).mean() * 100

    logE_MAE = np.abs(logE_diff).mean()

    metrics = {
        'E_rel_L2_mean': E_rel_L2.mean(),
        'E_rel_L2_std': E_rel_L2.std(),
        'E_rel_L2_max': E_rel_L2.max(),
        'logE_rel_L2_mean': logE_rel_L2.mean(),
        'logE_rel_L2_std': logE_rel_L2.std(),
        'E_MAE': E_MAE,
        'E_MAPE': E_MAPE,
        'logE_MAE': logE_MAE
    }

    return metrics


def evaluate_model(model, test_loader, device, stats, L, A, n_node):
    """
    Evaluate model on test set

    Returns:
    --------
    results : dict
        Contains predictions, errors, metrics
    """
    model.eval()

    E_pred_all = []
    E_true_all = []
    logE_pred_all = []
    logE_true_all = []
    u_obs_all = []
    F_all = []
    phys_residuals = []

    fem = FEM1DBar(L, A, n_node)

    with torch.no_grad():
        for inputs, targets, metadata in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Predict
            logE_pred_norm = model(inputs)  # (B, 1, N)

            # Denormalize
            logE_pred_denorm = logE_pred_norm * stats['logE_std'].to(device) + stats['logE_mean'].to(device)
            logE_pred = logE_pred_denorm.squeeze(1).cpu().numpy()  # (B, N)

            E_pred = np.exp(np.clip(logE_pred, -20, 20))

            # True values
            logE_true = metadata['logE_true'].cpu().numpy()
            E_true = np.exp(logE_true)

            # Observed u and F
            u_obs = metadata['u'].cpu().numpy()
            F_batch = metadata['F'].cpu().numpy()

            # Compute physics residuals
            batch_size = E_pred.shape[0]
            for i in range(batch_size):
                residual = fem.compute_residual(E_pred[i], u_obs[i], F_batch[i])
                phys_residuals.append(residual)

            E_pred_all.append(E_pred)
            E_true_all.append(E_true)
            logE_pred_all.append(logE_pred)
            logE_true_all.append(logE_true)
            u_obs_all.append(u_obs)
            F_all.append(F_batch)

    # Concatenate
    E_pred_all = np.concatenate(E_pred_all, axis=0)
    E_true_all = np.concatenate(E_true_all, axis=0)
    logE_pred_all = np.concatenate(logE_pred_all, axis=0)
    logE_true_all = np.concatenate(logE_true_all, axis=0)
    u_obs_all = np.concatenate(u_obs_all, axis=0)
    F_all = np.concatenate(F_all, axis=0)
    phys_residuals = np.array(phys_residuals)

    # Compute metrics
    metrics = compute_metrics(E_pred_all, E_true_all, logE_pred_all, logE_true_all)
    metrics['phys_residual_mean'] = phys_residuals.mean()
    metrics['phys_residual_std'] = phys_residuals.std()
    metrics['phys_residual_max'] = phys_residuals.max()

    results = {
        'E_pred': E_pred_all,
        'E_true': E_true_all,
        'logE_pred': logE_pred_all,
        'logE_true': logE_true_all,
        'u_obs': u_obs_all,
        'F': F_all,
        'phys_residuals': phys_residuals,
        'metrics': metrics
    }

    return results


def plot_results(results, x_coord, save_dir='plots', n_examples=5):
    """
    Visualize results

    Parameters:
    -----------
    results : dict
        Evaluation results
    x_coord : array (n_node,)
        Spatial coordinates
    save_dir : str
        Directory to save plots
    n_examples : int
        Number of example predictions to plot
    """
    os.makedirs(save_dir, exist_ok=True)

    E_pred = results['E_pred']
    E_true = results['E_true']
    logE_pred = results['logE_pred']
    logE_true = results['logE_true']
    metrics = results['metrics']

    n_samples = E_pred.shape[0]

    # Plot 1: Example predictions (E)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(min(n_examples, len(axes))):
        idx = np.random.randint(0, n_samples)
        axes[i].plot(x_coord, E_true[idx], 'k-', label='True', linewidth=2)
        axes[i].plot(x_coord, E_pred[idx], 'r--', label='Predicted', linewidth=2)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('E')
        axes[i].set_title(f'Sample {idx}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide last subplot if n_examples < 6
    if n_examples < 6:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'E_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Error distribution (E)
    E_rel_L2 = np.linalg.norm(E_pred - E_true, axis=1) / (np.linalg.norm(E_true, axis=1) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(E_rel_L2, bins=50, edgecolor='black')
    axes[0].set_xlabel('Relative L2 Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title('E Relative Error Distribution')
    axes[0].axvline(E_rel_L2.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean={E_rel_L2.mean():.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(results['phys_residuals'], bins=50, edgecolor='black')
    axes[1].set_xlabel('Physics Residual ||K(E)*u - f|| / ||f||')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Physics Residual Distribution')
    axes[1].axvline(results['phys_residuals'].mean(), color='r', linestyle='--',
                   linewidth=2, label=f'Mean={results["phys_residuals"].mean():.4e}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Worst/Best cases
    E_rel_L2_sorted_idx = np.argsort(E_rel_L2)
    best_idx = E_rel_L2_sorted_idx[0]
    worst_idx = E_rel_L2_sorted_idx[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Best case
    axes[0].plot(x_coord, E_true[best_idx], 'k-', label='True', linewidth=2.5)
    axes[0].plot(x_coord, E_pred[best_idx], 'g--', label='Predicted', linewidth=2)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('E', fontsize=12)
    axes[0].set_title(f'Best Case (Rel. Error={E_rel_L2[best_idx]:.4f})', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Worst case
    axes[1].plot(x_coord, E_true[worst_idx], 'k-', label='True', linewidth=2.5)
    axes[1].plot(x_coord, E_pred[worst_idx], 'r--', label='Predicted', linewidth=2)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('E', fontsize=12)
    axes[1].set_title(f'Worst Case (Rel. Error={E_rel_L2[worst_idx]:.4f})', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_worst_cases.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {save_dir}/")


def print_metrics(metrics):
    """Print evaluation metrics"""
    print("\n" + "="*80)
    print("Evaluation Metrics")
    print("="*80)
    print(f"E Relative L2 Error:")
    print(f"  Mean: {metrics['E_rel_L2_mean']:.6f}")
    print(f"  Std:  {metrics['E_rel_L2_std']:.6f}")
    print(f"  Max:  {metrics['E_rel_L2_max']:.6f}")
    print(f"\nlogE Relative L2 Error:")
    print(f"  Mean: {metrics['logE_rel_L2_mean']:.6f}")
    print(f"  Std:  {metrics['logE_rel_L2_std']:.6f}")
    print(f"\nPoint-wise Metrics:")
    print(f"  E MAE:   {metrics['E_MAE']:.6f}")
    print(f"  E MAPE:  {metrics['E_MAPE']:.2f}%")
    print(f"  logE MAE: {metrics['logE_MAE']:.6f}")
    print(f"\nPhysics Residual (||K(E)*u - f|| / ||f||):")
    print(f"  Mean: {metrics['phys_residual_mean']:.6e}")
    print(f"  Std:  {metrics['phys_residual_std']:.6e}")
    print(f"  Max:  {metrics['phys_residual_max']:.6e}")
    print("="*80)


def main():
    # Configuration
    checkpoint_path = 'checkpoints/best_model.pt'
    test_data_path = 'test_data.pt'
    save_dir = 'plots'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("FNO Evaluation")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    stats = checkpoint['stats']

    # Load model
    print("Loading model...")
    model = FNOModel(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        n_layers=config['n_layers']
    ).to(device)

    # Load state dict with strict=False to ignore metadata
    state_dict = checkpoint['model_state_dict']
    if '_metadata' in state_dict:
        del state_dict['_metadata']
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # Load test data
    print("\nLoading test data...")
    test_data = load_dataset(test_data_path)

    # Create dataset
    test_dataset = FNODataset(test_data, normalize=True, stats=stats)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(
        model, test_loader, device, stats,
        config['L'], config['A'], config['n_node']
    )

    # Print metrics
    print_metrics(results['metrics'])

    # Plot results
    print("\nGenerating plots...")
    x_coord = np.linspace(0, config['L'], config['n_node'])
    plot_results(results, x_coord, save_dir=save_dir, n_examples=6)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
