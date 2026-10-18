"""
Training Script with Supervised + Physics Loss
Loss = λ_sup * ||logE_pred - logE_true||^2
     + λ_phys * ||K(E_pred)*u - f||^2 / ||f||^2
     + λ_smooth * ||∂logE/∂x||^2
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import FNOModel, count_parameters
from dataset import create_dataloaders
from data_gen import load_dataset
from physics_loss_torch import PhysicsLossPyTorch, SmoothnessLossPyTorch


def train_epoch(model, train_loader, optimizer, device,
                lambda_sup, lambda_phys, lambda_smooth,
                physics_loss, smoothness_loss, stats):
    """
    Train for one epoch

    Returns:
    --------
    losses : dict
        Average losses
    """
    model.train()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_phys_loss = 0.0
    total_smooth_loss = 0.0
    n_batches = 0

    for inputs, targets, metadata in tqdm(train_loader, desc='Training'):
        inputs = inputs.to(device)  # (B, 3, N)
        targets = targets.to(device)  # (B, 1, N)

        # Forward pass
        logE_pred = model(inputs)  # (B, 1, N)

        # Supervised loss (on normalized logE)
        sup_loss = nn.functional.mse_loss(logE_pred, targets)

        # Denormalize for physics loss
        logE_pred_denorm = logE_pred * stats['logE_std'].to(device) + stats['logE_mean'].to(device)
        E_pred = torch.exp(torch.clamp(logE_pred_denorm, -20, 20))

        # Physics loss (original u and F) - NOW DIFFERENTIABLE!
        u_obs = metadata['u'].to(device)  # (B, N)
        F_batch = metadata['F'].to(device)  # (B,)
        phys_loss = physics_loss(E_pred, u_obs, F_batch)

        # Smoothness loss (on denormalized logE)
        smooth_loss = smoothness_loss(logE_pred_denorm)

        # Total loss
        loss = lambda_sup * sup_loss + lambda_phys * phys_loss + lambda_smooth * smooth_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_phys_loss += phys_loss.item()
        total_smooth_loss += smooth_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'sup_loss': total_sup_loss / n_batches,
        'phys_loss': total_phys_loss / n_batches,
        'smooth_loss': total_smooth_loss / n_batches
    }



def validate(model, val_loader, device,
             lambda_sup, lambda_phys, lambda_smooth,
             physics_loss, smoothness_loss, stats):
    """
    Validate

    Returns:
    --------
    losses : dict
        Average losses
    """
    model.eval()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_phys_loss = 0.0
    total_smooth_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets, metadata in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logE_pred = model(inputs)

            sup_loss = nn.functional.mse_loss(logE_pred, targets)

            logE_pred_denorm = logE_pred * stats['logE_std'].to(device) + stats['logE_mean'].to(device)
            E_pred = torch.exp(torch.clamp(logE_pred_denorm, -20, 20))

            u_obs = metadata['u'].to(device)
            F_batch = metadata['F'].to(device)
            phys_loss = physics_loss(E_pred, u_obs, F_batch)

            smooth_loss = smoothness_loss(logE_pred_denorm)

            loss = lambda_sup * sup_loss + lambda_phys * phys_loss + lambda_smooth * smooth_loss

            total_loss += loss.item()
            total_sup_loss += sup_loss.item()
            total_phys_loss += phys_loss.item()
            total_smooth_loss += smooth_loss.item()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'sup_loss': total_sup_loss / n_batches,
        'phys_loss': total_phys_loss / n_batches,
        'smooth_loss': total_smooth_loss / n_batches
    }


def main():
    # Configuration
    config = {
        # Model
        'n_modes': 16,
        'hidden_channels': 64,
        'n_layers': 4,

        # Training
        'batch_size': 32,
        'n_epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-5,

        # Loss weights
        'lambda_sup': 1.0,
        'lambda_phys': 1.0,
        'lambda_smooth': 0.01,

        # Data
        'L': 1.0,
        'A': 1.0,
        'n_node': 101,

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Checkpoint
        'save_dir': 'checkpoints',
        'log_dir': 'runs'
    }

    print("="*80)
    print("FNO Training for 1D Bar Inverse Problem")
    print("="*80)
    print(f"Device: {config['device']}")
    print(f"Loss weights: λ_sup={config['lambda_sup']}, λ_phys={config['lambda_phys']}, λ_smooth={config['lambda_smooth']}")
    print("="*80)

    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Load data
    print("\nLoading datasets...")
    train_data = load_dataset('train_data.pt')
    val_data = load_dataset('val_data.pt')

    # Create dataloaders
    train_loader, val_loader, _, stats = create_dataloaders(
        train_data, val_data, val_data,  # Use val as dummy test
        batch_size=config['batch_size']
    )

    # Model
    print("\nInitializing model...")
    model = FNOModel(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        n_layers=config['n_layers']
    ).to(config['device'])

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['n_epochs']
    )

    # Loss functions (PyTorch-based, fully differentiable)
    h = config['L'] / (config['n_node'] - 1)
    physics_loss = PhysicsLossPyTorch(config['L'], config['A'], config['n_node']).to(config['device'])
    smoothness_loss = SmoothnessLossPyTorch(h).to(config['device'])

    # TensorBoard
    writer = SummaryWriter(config['log_dir'])

    # Training loop
    best_val_loss = float('inf')

    print("\nStarting training...")
    for epoch in range(config['n_epochs']):
        print(f"\nEpoch {epoch+1}/{config['n_epochs']}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, config['device'],
            config['lambda_sup'], config['lambda_phys'], config['lambda_smooth'],
            physics_loss, smoothness_loss, stats
        )

        # Validate
        val_losses = validate(
            model, val_loader, config['device'],
            config['lambda_sup'], config['lambda_phys'], config['lambda_smooth'],
            physics_loss, smoothness_loss, stats
        )

        # Scheduler step
        scheduler.step()

        # Logging
        print(f"Train - Loss: {train_losses['loss']:.6f}, "
              f"Sup: {train_losses['sup_loss']:.6f}, "
              f"Phys: {train_losses['phys_loss']:.6f}, "
              f"Smooth: {train_losses['smooth_loss']:.6f}")
        print(f"Val   - Loss: {val_losses['loss']:.6f}, "
              f"Sup: {val_losses['sup_loss']:.6f}, "
              f"Phys: {val_losses['phys_loss']:.6f}, "
              f"Smooth: {val_losses['smooth_loss']:.6f}")

        writer.add_scalar('Train/Loss', train_losses['loss'], epoch)
        writer.add_scalar('Train/Sup', train_losses['sup_loss'], epoch)
        writer.add_scalar('Train/Phys', train_losses['phys_loss'], epoch)
        writer.add_scalar('Train/Smooth', train_losses['smooth_loss'], epoch)

        writer.add_scalar('Val/Loss', val_losses['loss'], epoch)
        writer.add_scalar('Val/Sup', val_losses['sup_loss'], epoch)
        writer.add_scalar('Val/Phys', val_losses['phys_loss'], epoch)
        writer.add_scalar('Val/Smooth', val_losses['smooth_loss'], epoch)

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['loss'],
                'config': config,
                'stats': stats
            }, os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"Saved best model (val_loss={best_val_loss:.6f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'stats': stats
            }, os.path.join(config['save_dir'], f'checkpoint_epoch{epoch+1}.pt'))

    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
