"""
PyTorch Dataset and DataLoader
Handles data loading, normalization, and batch preparation
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FNODataset(Dataset):
    """
    Dataset for FNO training
    Input: [u, x/L, F] (3 channels)
    Output: logE (1 channel)
    """

    def __init__(self, data, normalize=True, stats=None):
        """
        Parameters:
        -----------
        data : dict
            Contains 'u', 'x', 'F', 'logE_true'
        normalize : bool
            Whether to normalize inputs
        stats : dict or None
            Normalization statistics (mean, std)
            If None and normalize=True, compute from data
        """
        self.u = torch.tensor(data['u'], dtype=torch.float32)
        self.x = torch.tensor(data['x'], dtype=torch.float32)
        self.F = torch.tensor(data['F'], dtype=torch.float32)
        self.logE_true = torch.tensor(data['logE_true'], dtype=torch.float32)

        self.n_samples = len(self.F)
        self.n_node = self.u.shape[1]

        self.normalize = normalize
        if normalize:
            if stats is None:
                # Compute statistics from data
                self.stats = {
                    'u_mean': self.u.mean(dim=(0, 1), keepdim=True),
                    'u_std': self.u.std(dim=(0, 1), keepdim=True) + 1e-8,
                    'F_mean': self.F.mean(),
                    'F_std': self.F.std() + 1e-8,
                    'logE_mean': self.logE_true.mean(dim=(0, 1), keepdim=True),
                    'logE_std': self.logE_true.std(dim=(0, 1), keepdim=True) + 1e-8
                }
            else:
                self.stats = stats

            # Normalize (ensure stats tensors are on CPU for compatibility)
            u_mean = self.stats['u_mean'].cpu() if torch.is_tensor(self.stats['u_mean']) else self.stats['u_mean']
            u_std = self.stats['u_std'].cpu() if torch.is_tensor(self.stats['u_std']) else self.stats['u_std']
            F_mean = self.stats['F_mean'].cpu() if torch.is_tensor(self.stats['F_mean']) else self.stats['F_mean']
            F_std = self.stats['F_std'].cpu() if torch.is_tensor(self.stats['F_std']) else self.stats['F_std']
            logE_mean = self.stats['logE_mean'].cpu() if torch.is_tensor(self.stats['logE_mean']) else self.stats['logE_mean']
            logE_std = self.stats['logE_std'].cpu() if torch.is_tensor(self.stats['logE_std']) else self.stats['logE_std']

            self.u_norm = (self.u - u_mean) / u_std
            self.F_norm = (self.F - F_mean) / F_std
            self.logE_norm = (self.logE_true - logE_mean) / logE_std
        else:
            self.stats = None
            self.u_norm = self.u
            self.F_norm = self.F
            self.logE_norm = self.logE_true

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
        --------
        input_tensor : (3, n_node)
            [u_norm, x/L, F_norm_broadcast]
        target : (1, n_node)
            logE_norm or logE_true
        metadata : dict
            Original u, F, logE for physics loss
        """
        # Input channels: [u, x/L, F]
        u_channel = self.u_norm[idx]  # (n_node,)
        x_channel = self.x[idx]  # (n_node,)
        F_channel = self.F_norm[idx].expand(self.n_node)  # Broadcast to (n_node,)

        # Stack to (3, n_node)
        input_tensor = torch.stack([u_channel, x_channel, F_channel], dim=0)

        # Target: (1, n_node)
        target = self.logE_norm[idx].unsqueeze(0)

        # Metadata for physics loss (original values)
        metadata = {
            'u': self.u[idx],  # Original (unnormalized) u
            'F': self.F[idx],  # Original F
            'logE_true': self.logE_true[idx],  # Original logE
            'u_norm': self.u_norm[idx],
            'F_norm': self.F_norm[idx]
        }

        return input_tensor, target, metadata

    def get_stats(self):
        """Return normalization statistics"""
        return self.stats

    def denormalize_logE(self, logE_norm):
        """
        Denormalize predicted logE

        Parameters:
        -----------
        logE_norm : tensor (..., n_node)
            Normalized logE

        Returns:
        --------
        logE : tensor (..., n_node)
            Denormalized logE
        """
        if self.stats is not None:
            return logE_norm * self.stats['logE_std'] + self.stats['logE_mean']
        else:
            return logE_norm


def create_dataloaders(train_data, val_data, test_data, batch_size=32, num_workers=0):
    """
    Create train/val/test dataloaders with consistent normalization

    Parameters:
    -----------
    train_data : dict
        Training data
    val_data : dict
        Validation data
    test_data : dict
        Test data
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for DataLoader

    Returns:
    --------
    train_loader, val_loader, test_loader, stats
    """
    # Create training dataset and compute normalization stats
    train_dataset = FNODataset(train_data, normalize=True, stats=None)
    stats = train_dataset.get_stats()

    # Create val/test datasets with training stats
    val_dataset = FNODataset(val_data, normalize=True, stats=stats)
    test_dataset = FNODataset(test_data, normalize=True, stats=stats)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, stats


if __name__ == '__main__':
    # Test dataset loading
    from data_gen import load_dataset

    print("Loading datasets...")
    train_data = load_dataset('train_data.pt')
    val_data = load_dataset('val_data.pt')
    test_data = load_dataset('test_data.pt')

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, stats = create_dataloaders(
        train_data, val_data, test_data, batch_size=32
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    print("\nTesting one batch...")
    for inputs, targets, metadata in train_loader:
        print(f"Input shape: {inputs.shape}")  # (batch, 3, n_node)
        print(f"Target shape: {targets.shape}")  # (batch, 1, n_node)
        print(f"Metadata keys: {metadata.keys()}")
        break

    print("\nNormalization stats:")
    for key, val in stats.items():
        if torch.is_tensor(val):
            print(f"  {key}: {val.item():.6f}")
        else:
            print(f"  {key}: {val}")
