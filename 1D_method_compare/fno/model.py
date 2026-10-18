"""
FNO Model for 1D Bar Inverse Problem
Input: [u, x/L, F] (3 channels)
Output: logE (1 channel, no constraints during training)
Inference: E = exp(logE) to ensure positivity
"""
import torch
import torch.nn as nn
from neuralop.models import FNO


class FNOModel(nn.Module):
    """
    1D FNO for predicting logE from displacement field
    """

    def __init__(self, n_modes=16, hidden_channels=64, n_layers=4,
                 in_channels=3, out_channels=1):
        """
        Parameters:
        -----------
        n_modes : int
            Number of Fourier modes to keep
        hidden_channels : int
            Hidden dimension
        n_layers : int
            Number of FNO layers
        in_channels : int
            Input channels (u, x/L, F)
        out_channels : int
            Output channels (logE)
        """
        super(FNOModel, self).__init__()

        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # FNO expects input (batch, channels, spatial_dims)
        # For 1D: (B, C, N)
        self.fno = FNO(
            n_modes=(n_modes,),  # 1D modes
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            # Use default lifting/projection
        )

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x : tensor (B, 3, N)
            Input [u, x/L, F]

        Returns:
        --------
        logE_pred : tensor (B, 1, N)
            Predicted logE (unbounded)
        """
        logE_pred = self.fno(x)
        return logE_pred

    def predict_E(self, x, clamp_logE=None):
        """
        Predict E = exp(logE) for inference

        Parameters:
        -----------
        x : tensor (B, 3, N)
            Input
        clamp_logE : tuple or None
            (min, max) to clamp logE before exp (for numerical stability)

        Returns:
        --------
        E_pred : tensor (B, 1, N)
            Predicted E (strictly positive)
        logE_pred : tensor (B, 1, N)
            Predicted logE
        """
        logE_pred = self.forward(x)

        if clamp_logE is not None:
            logE_pred = torch.clamp(logE_pred, clamp_logE[0], clamp_logE[1])

        E_pred = torch.exp(logE_pred)
        return E_pred, logE_pred


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    print("Testing FNO model...")

    model = FNOModel(
        n_modes=16,
        hidden_channels=64,
        n_layers=4,
        in_channels=3,
        out_channels=1
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    n_node = 101
    x = torch.randn(batch_size, 3, n_node)

    print(f"\nInput shape: {x.shape}")

    logE_pred = model(x)
    print(f"Output logE shape: {logE_pred.shape}")

    E_pred, _ = model.predict_E(x, clamp_logE=(-20, 20))
    print(f"Output E shape: {E_pred.shape}")
    print(f"E range: [{E_pred.min().item():.6f}, {E_pred.max().item():.6f}]")

    print("\nModel test passed!")
