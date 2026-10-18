"""
Visualize Generated E(x) Fields
Compare different generators (KL, piecewise, GRF) and their characteristics
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_gen import (
    generate_random_logE_KL,
    generate_random_logE_piecewise,
    generate_random_logE_GRF
)


def visualize_single_type(logE_type, logE_params, n_samples=6, n_node=101, L=1.0, seed_base=0):
    """
    Visualize samples from a single generator type

    Parameters:
    -----------
    logE_type : str
        'KL', 'piecewise', or 'GRF'
    logE_params : dict
        Generator parameters
    n_samples : int
        Number of samples to show
    n_node : int
        Number of nodes
    L : float
        Domain length
    seed_base : int
        Base seed
    """
    x = np.linspace(0, L, n_node)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(n_samples):
        rng = np.random.default_rng(seed_base + i)

        if logE_type == 'KL':
            logE = generate_random_logE_KL(n_node, L, rng, **logE_params)
        elif logE_type == 'piecewise':
            logE = generate_random_logE_piecewise(n_node, L, rng, **logE_params)
        elif logE_type == 'GRF':
            logE = generate_random_logE_GRF(n_node, L, rng, **logE_params)
        else:
            raise ValueError(f"Unknown type: {logE_type}")

        E = np.exp(logE)

        axes[i].plot(x, E, 'b-', linewidth=2)
        axes[i].set_xlabel('x', fontsize=11)
        axes[i].set_ylabel('E(x)', fontsize=11)
        axes[i].set_title(f'Sample {i}', fontsize=12)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'{logE_type} Generator (params: {logE_params})', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def visualize_comparison(n_samples=4, n_node=101, L=1.0, seed_base=0):
    """
    Compare all three generator types side-by-side

    Parameters:
    -----------
    n_samples : int
        Number of samples per generator
    n_node : int
        Number of nodes
    L : float
        Domain length
    seed_base : int
        Base seed
    """
    x = np.linspace(0, L, n_node)

    # Generator configurations
    configs = {
        'KL': {'n_modes': 10, 'lengthscale': 0.2, 'sigma': 0.5, 'mean_logE': 0.0},
        'piecewise': {'n_segments': 4, 'logE_range': (-0.5, 0.5)},
        'GRF (exp)': {'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5, 'mean_logE': 0.0},
        'GRF (rbf)': {'kernel': 'rbf', 'corr_len': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
    }

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, n_samples, figure=fig, hspace=0.35, wspace=0.3)

    row_idx = 0
    for name, params in configs.items():
        for col in range(n_samples):
            ax = fig.add_subplot(gs[row_idx, col])

            rng = np.random.default_rng(seed_base + col)

            if name == 'KL':
                logE = generate_random_logE_KL(n_node, L, rng, **params)
            elif name == 'piecewise':
                logE = generate_random_logE_piecewise(n_node, L, rng, **params)
            elif name.startswith('GRF'):
                logE = generate_random_logE_GRF(n_node, L, rng, **params)

            E = np.exp(logE)

            ax.plot(x, E, 'b-', linewidth=2)
            ax.set_ylim([0.4, 2.5])  # Fixed y-axis for comparison

            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight='bold')
            if row_idx == 0:
                ax.set_title(f'Sample {col}', fontsize=11)
            if row_idx == 3:
                ax.set_xlabel('x', fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

        row_idx += 1

    plt.suptitle('Comparison of E(x) Generators', fontsize=16, y=0.995)

    return fig


def visualize_statistics(logE_type, logE_params, n_samples=500, n_node=101, L=1.0, seed_base=0):
    """
    Statistical analysis of generated fields

    Parameters:
    -----------
    logE_type : str
        Generator type
    logE_params : dict
        Generator parameters
    n_samples : int
        Number of samples for statistics
    n_node : int
        Number of nodes
    L : float
        Domain length
    seed_base : int
        Base seed
    """
    x = np.linspace(0, L, n_node)

    # Generate many samples
    E_samples = np.zeros((n_samples, n_node))

    for i in range(n_samples):
        rng = np.random.default_rng(seed_base + i)

        if logE_type == 'KL':
            logE = generate_random_logE_KL(n_node, L, rng, **logE_params)
        elif logE_type == 'piecewise':
            logE = generate_random_logE_piecewise(n_node, L, rng, **logE_params)
        elif logE_type == 'GRF':
            logE = generate_random_logE_GRF(n_node, L, rng, **logE_params)

        E_samples[i] = np.exp(logE)

    # Compute statistics
    E_mean = E_samples.mean(axis=0)
    E_std = E_samples.std(axis=0)
    E_min = E_samples.min(axis=0)
    E_max = E_samples.max(axis=0)
    E_median = np.median(E_samples, axis=0)

    # Compute correlation (spatial)
    if logE_type != 'piecewise':
        mid_idx = n_node // 2
        correlations = np.zeros(n_node)
        for i in range(n_node):
            correlations[i] = np.corrcoef(E_samples[:, mid_idx], E_samples[:, i])[0, 1]

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Mean ± std
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, E_mean, 'b-', linewidth=2.5, label='Mean')
    ax1.fill_between(x, E_mean - E_std, E_mean + E_std,
                      alpha=0.3, color='b', label='±1 std')
    ax1.fill_between(x, E_min, E_max,
                      alpha=0.15, color='gray', label='Min-Max envelope')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('E(x)', fontsize=12)
    ax1.set_title(f'{logE_type}: Mean and Variability ({n_samples} samples)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sample trajectories
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(min(50, n_samples)):
        ax2.plot(x, E_samples[i], 'b-', alpha=0.1, linewidth=0.5)
    ax2.plot(x, E_mean, 'r-', linewidth=2, label='Mean')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('E(x)', fontsize=11)
    ax2.set_title('Sample Trajectories', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Histogram at x=0.5
    ax3 = fig.add_subplot(gs[1, 1])
    mid_idx = n_node // 2
    ax3.hist(E_samples[:, mid_idx], bins=50, density=True, alpha=0.7, edgecolor='black')
    ax3.axvline(E_mean[mid_idx], color='r', linestyle='--', linewidth=2, label=f'Mean={E_mean[mid_idx]:.3f}')
    ax3.set_xlabel('E(x=0.5)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Marginal Distribution at x=0.5', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spatial correlation (if applicable)
    if logE_type != 'piecewise':
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(x, correlations, 'b-', linewidth=2)
        ax4.axhline(0, color='k', linestyle='--', linewidth=1)
        ax4.set_xlabel('x', fontsize=11)
        ax4.set_ylabel('Correlation with E(x=0.5)', fontsize=11)
        ax4.set_title('Spatial Correlation Structure', fontsize=12)
        ax4.grid(True, alpha=0.3)
    else:
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.text(0.5, 0.5, 'Correlation not meaningful\nfor piecewise constant',
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.axis('off')

    # Plot 5: Pointwise std
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(x, E_std, 'g-', linewidth=2)
    ax5.set_xlabel('x', fontsize=11)
    ax5.set_ylabel('Std(E(x))', fontsize=11)
    ax5.set_title('Pointwise Standard Deviation', fontsize=12)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f'{logE_type} Generator Statistics (params: {logE_params})',
                 fontsize=14, y=0.995)

    return fig


def visualize_correlation_length_effect(n_samples=4, n_node=101, L=1.0):
    """
    Show effect of correlation length in GRF

    Parameters:
    -----------
    n_samples : int
        Number of samples per corr_len
    n_node : int
        Number of nodes
    L : float
        Domain length
    """
    x = np.linspace(0, L, n_node)

    corr_lens = [0.05, 0.1, 0.2, 0.4]
    fig, axes = plt.subplots(len(corr_lens), n_samples, figsize=(14, 10))

    for row, corr_len in enumerate(corr_lens):
        for col in range(n_samples):
            ax = axes[row, col]

            rng = np.random.default_rng(col)
            logE = generate_random_logE_GRF(
                n_node, L, rng,
                kernel='exp',
                corr_len=corr_len,
                sigma=0.5,
                mean_logE=0.0
            )
            E = np.exp(logE)

            ax.plot(x, E, 'b-', linewidth=2)
            ax.set_ylim([0.3, 2.5])

            if col == 0:
                ax.set_ylabel(f'corr_len={corr_len}', fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f'Sample {col}', fontsize=10)
            if row == len(corr_lens) - 1:
                ax.set_xlabel('x', fontsize=9)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle('Effect of Correlation Length (GRF, exp kernel)', fontsize=14, y=0.995)
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize E(x) generators')
    parser.add_argument('--type', type=str, default='comparison',
                       choices=['comparison', 'KL', 'piecewise', 'GRF', 'stats', 'corr_effect'],
                       help='Visualization type')
    parser.add_argument('--save', action='store_true', help='Save figures')

    args = parser.parse_args()

    print("Generating visualizations...")

    if args.type == 'comparison':
        # Compare all types
        fig = visualize_comparison(n_samples=4, seed_base=0)
        if args.save:
            fig.savefig('generator_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved: generator_comparison.png")

    elif args.type == 'KL':
        params = {'n_modes': 10, 'lengthscale': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
        fig = visualize_single_type('KL', params, n_samples=6, seed_base=0)
        if args.save:
            fig.savefig('KL_samples.png', dpi=300, bbox_inches='tight')
            print("Saved: KL_samples.png")

    elif args.type == 'piecewise':
        params = {'n_segments': 4, 'logE_range': (-0.5, 0.5)}
        fig = visualize_single_type('piecewise', params, n_samples=6, seed_base=0)
        if args.save:
            fig.savefig('piecewise_samples.png', dpi=300, bbox_inches='tight')
            print("Saved: piecewise_samples.png")

    elif args.type == 'GRF':
        params = {'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
        fig = visualize_single_type('GRF', params, n_samples=6, seed_base=0)
        if args.save:
            fig.savefig('GRF_samples.png', dpi=300, bbox_inches='tight')
            print("Saved: GRF_samples.png")

    elif args.type == 'stats':
        # Statistical analysis (takes longer)
        print("Computing statistics (this may take a few seconds)...")
        params = {'kernel': 'exp', 'corr_len': 0.2, 'sigma': 0.5, 'mean_logE': 0.0}
        fig = visualize_statistics('GRF', params, n_samples=500, seed_base=0)
        if args.save:
            fig.savefig('GRF_statistics.png', dpi=300, bbox_inches='tight')
            print("Saved: GRF_statistics.png")

    elif args.type == 'corr_effect':
        # Show correlation length effect
        fig = visualize_correlation_length_effect(n_samples=4)
        if args.save:
            fig.savefig('correlation_length_effect.png', dpi=300, bbox_inches='tight')
            print("Saved: correlation_length_effect.png")

    if not args.save:
        plt.show()

    print("Done!")
