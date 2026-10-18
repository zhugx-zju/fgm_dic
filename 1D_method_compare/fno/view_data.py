"""
Quick visualization of generated dataset
View E(x) fields from saved .pt files
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os


def visualize_dataset(data_file, n_samples=12, save_fig=None):
    """
    Visualize E(x) fields from dataset

    Parameters:
    -----------
    data_file : str
        Path to .pt file (e.g., 'train_data.pt')
    n_samples : int
        Number of samples to display
    save_fig : str or None
        If provided, save figure to this path
    """
    import torch

    # Load data
    print(f"Loading {data_file}...")
    data = torch.load(data_file)

    E_true = data['E_true']
    logE_true = data['logE_true']
    u = data['u']
    F = data['F']
    x = data['x'][0]  # All samples have same x

    n_total = len(F)
    n_samples = min(n_samples, n_total)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Plot samples
    for i in range(min(12, n_samples)):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        ax.plot(x, E_true[i], 'b-', linewidth=2)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('E(x)', fontsize=9)
        ax.set_title(f'Sample {i}: F={F[i]:.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Add statistics text
    fig.text(0.5, 0.02,
             f'Dataset: {os.path.basename(data_file)} | '
             f'Total samples: {n_total} | '
             f'E range: [{E_true.min():.3f}, {E_true.max():.3f}] | '
             f'logE range: [{logE_true.min():.3f}, {logE_true.max():.3f}]',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'E(x) Fields from {os.path.basename(data_file)}', fontsize=14, y=0.98)

    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_fig}")

    return fig


def compare_datasets(train_file='train_data.pt', val_file='val_data.pt', test_file='test_data.pt'):
    """
    Compare statistics across train/val/test datasets
    """
    import torch

    datasets = {
        'Train': train_file,
        'Val': val_file,
        'Test': test_file
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    stats_text = []

    for idx, (name, file) in enumerate(datasets.items()):
        if not os.path.exists(file):
            print(f"Warning: {file} not found, skipping...")
            continue

        data = torch.load(file)
        E_true = data['E_true']
        F = data['F']

        # Plot 1: E distribution (histogram)
        ax = axes[idx]
        ax.hist(E_true.flatten(), bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('E', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name}: E Distribution', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 2: F distribution
        ax = axes[idx + 3]
        ax.hist(F, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax.set_xlabel('F', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}: F Distribution', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Collect statistics
        stats_text.append(
            f"{name}: n={len(F)}, "
            f"E∈[{E_true.min():.3f},{E_true.max():.3f}], "
            f"F∈[{F.min():.4f},{F.max():.4f}]"
        )

    # Add statistics
    stats_str = '\n'.join(stats_text)
    fig.text(0.5, 0.02, stats_str, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('Dataset Statistics Comparison', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    return fig


def visualize_input_output(data_file, sample_idx=0):
    """
    Visualize one sample: input (u, F) and output (E)

    Parameters:
    -----------
    data_file : str
        Dataset file
    sample_idx : int
        Sample index to visualize
    """
    import torch

    data = torch.load(data_file)

    E_true = data['E_true'][sample_idx]
    logE_true = data['logE_true'][sample_idx]
    u = data['u'][sample_idx]
    F = data['F'][sample_idx]
    x = data['x'][sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: E(x)
    axes[0].plot(x, E_true, 'b-', linewidth=2.5)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('E(x)', fontsize=12)
    axes[0].set_title('Output: Young\'s Modulus', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: u(x)
    axes[1].plot(x, u, 'r-', linewidth=2.5)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('u(x)', fontsize=12)
    axes[1].set_title('Input: Displacement', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: logE(x)
    axes[2].plot(x, logE_true, 'g-', linewidth=2.5)
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('logE(x)', fontsize=12)
    axes[2].set_title('Output (log scale): logE', fontsize=13)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Sample {sample_idx} from {os.path.basename(data_file)} | F = {F:.4f}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize generated datasets')
    parser.add_argument('--file', type=str, default='train_data.pt',
                       help='Dataset file to visualize')
    parser.add_argument('--mode', type=str, default='samples',
                       choices=['samples', 'compare', 'detail'],
                       help='Visualization mode')
    parser.add_argument('--n', type=int, default=12,
                       help='Number of samples to show')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index for detail mode')
    parser.add_argument('--save', action='store_true',
                       help='Save figure')

    args = parser.parse_args()

    if args.mode == 'samples':
        # Show multiple samples
        save_path = f'view_{os.path.splitext(os.path.basename(args.file))[0]}.png' if args.save else None
        fig = visualize_dataset(args.file, n_samples=args.n, save_fig=save_path)

    elif args.mode == 'compare':
        # Compare train/val/test statistics
        fig = compare_datasets()
        if args.save:
            fig.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved: dataset_comparison.png")

    elif args.mode == 'detail':
        # Detailed view of one sample
        fig = visualize_input_output(args.file, sample_idx=args.sample_idx)
        if args.save:
            fig.savefig(f'sample_{args.sample_idx}_detail.png', dpi=300, bbox_inches='tight')
            print(f"Saved: sample_{args.sample_idx}_detail.png")

    if not args.save:
        plt.show()

    print("Done!")
